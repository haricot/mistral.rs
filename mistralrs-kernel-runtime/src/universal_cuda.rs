use crate::{
    cuda::compute_capability,
    universal::{KernelCatalog, KernelManifest, LaunchRequest, VerifiedKernel},
    image_policy::ImageMode,
};
use anyhow::{anyhow, ensure, Result};
use dkr_kernel_registry::{telemetry::{self,Event,Handle},TYPED_ABI};
use candle_core::{
    cuda_backend::cudarc::{driver::{CudaFunction, CudaModule, LaunchConfig}, nvrtc::Ptx},
    Device,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

const MAX_PREPARED_KERNELS: usize = 128;
const SHARED_CATALOG_ENV: &str = "MISTRALRS_DKR_SHARED_CATALOG";
const MAX_CUBIN_MODULES: usize = 512;

// Physical, context-bound CUBIN cache. The PTX branch uses Candle's shared
// artifact cache; CUDA binary images do not yet have that transport entrypoint.
struct CubinCacheEntry {
    device: Device,
    sha256: String,
    module: Arc<CudaModule>,
}
static CUBIN_MODULES: OnceLock<Mutex<Vec<CubinCacheEntry>>> = OnceLock::new();

#[derive(Debug)]
enum CubinLoadFailure {
    // Never silently fall back from a host/runtime invariant failure.
    Invariant(anyhow::Error),
    // CUDA driver's module/function admission may fall back before a launch.
    Admission(anyhow::Error),
}
impl CubinLoadFailure {
    fn into_error(self) -> anyhow::Error {
        match self { Self::Invariant(error) | Self::Admission(error) => error }
    }
}


struct PreparedEntry {
    request: LaunchRequest,
    device: Device,
    kernel: Arc<PreparedKernel>,
}

pub struct KernelExecutionCache {
    catalog: Arc<KernelCatalog>,
    prepared: Mutex<Vec<PreparedEntry>>,
}

impl KernelExecutionCache {
    pub fn package_root() -> Option<PathBuf> {
        std::env::var_os("DKR_KERNEL_PACKAGES").map(PathBuf::from)
            .or_else(|| crate::embedded::ROOT.map(PathBuf::from))
    }

    pub fn discover(root: &Path) -> Result<Self> {
        let source = std::env::var("MISTRALRS_DKR_ARTIFACT_SOURCE")
            .unwrap_or_else(|_| if crate::embedded::ROOT.is_some() { "embedded".into() } else { "filesystem".into() });
        let catalog = if source == "embedded" {
            ensure!(crate::embedded::ROOT.is_some_and(|path| root == Path::new(path)),
                    "typed directory differs from the embedded build input");
            crate::embedded::catalog()?
        } else if source != "filesystem" {
            anyhow::bail!("MISTRALRS_DKR_ARTIFACT_SOURCE must be embedded or filesystem");
        } else if std::env::var_os(SHARED_CATALOG_ENV).is_some_and(|value| value == "0") {
            Arc::new(KernelCatalog::discover(root)?)
        } else {
            KernelCatalog::discover_shared(root)?
        };
        Ok(Self {
            catalog,
            prepared: Mutex::new(Vec::new()),
        })
    }

    pub fn prepare_for_device(
        &self,
        request: &mut LaunchRequest,
        device: &Device,
    ) -> Result<Arc<PreparedKernel>> {
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| anyhow!("dynamic kernel preparation cache poisoned"))?;
        if let Some(entry) = prepared
            .iter()
            .find(|entry| {
                entry.device.same_device(device)
                    && entry.request.operation == request.operation
                    && entry.request.args == request.args
                    && entry.request.grid == request.grid
                    && entry.request.max_dynamic_shared_mem_bytes
                        == request.max_dynamic_shared_mem_bytes
            })
        {
            request.cc = entry.request.cc;
            telemetry::handle(TYPED_ABI,&request.operation,"prepared_cache","",request.cc).record(Event::PlanHit);
            return Ok(Arc::clone(&entry.kernel));
        }
        request.cc = compute_capability(device)?;
        let ev=telemetry::handle(TYPED_ABI,&request.operation,"prepared_cache","",request.cc);
        ev.record(Event::PlanMiss);
        let kernel = match PreparedKernel::load_auto(&self.catalog, request, device) {
            Ok(kernel)=>Arc::new(kernel),Err(e)=>{ev.record(Event::PrepareError);return Err(e)}
        };
        if prepared.len() == MAX_PREPARED_KERNELS {
            prepared.swap_remove(0);
            ev.record(Event::PlanEvicted);
        }
        prepared.push(PreparedEntry {
            request: request.clone(),
            device: device.clone(),
            kernel: Arc::clone(&kernel),
        });
        Ok(kernel)
    }

    pub fn prepared_count(&self) -> Result<usize> {
        Ok(self
            .prepared
            .lock()
            .map_err(|_| anyhow!("dynamic kernel preparation cache poisoned"))?
            .len())
    }
}

pub struct PreparedKernel {
    pub manifest: KernelManifest,
    function: CudaFunction,
    device: Device,
    cc: u32,
    evidence: Handle,
}

pub struct PrevalidatedLaunchPlan {
    kernel: Arc<PreparedKernel>,
    config: LaunchConfig,
}

impl PreparedKernel {
    pub fn load_auto(
        catalog: &KernelCatalog,
        request: &LaunchRequest,
        device: &Device,
    ) -> Result<Self> {
        let cc = compute_capability(device)?;
        ensure!(
            request.cc == cc,
            "request architecture differs from CUDA device"
        );
        let selected = catalog.resolve(request)?;
        Self::load(selected, device)
    }

    pub fn load(module: &VerifiedKernel, device: &Device) -> Result<Self> {
        let cc = compute_capability(device)?;
        module.manifest.validate()?;
        ensure!(
            (module.manifest.min_cc..=module.manifest.max_cc).contains(&cc),
            "GPU architecture outside kernel contract"
        );
        let mode = ImageMode::current()?;
        let binary = if cc == module.manifest.target_cc {
            module.cubin.as_ref().map(|image| image.as_ref())
        } else {
            None
        };
        let function = match mode {
            ImageMode::Ptx => Self::load_ptx(module, device)?,
            ImageMode::Cubin => {
                let image = binary.ok_or_else(|| anyhow!(
                    "strict CUBIN mode requires an exact-architecture verified image for {} sm{}",
                    module.manifest.operation, cc
                ))?;
                Self::load_cubin(module, device, image).map_err(CubinLoadFailure::into_error)?
            }
            ImageMode::Auto => match binary {
                Some(image) => match Self::load_cubin(module, device, image) {
                    Ok(function) => function,
                    Err(CubinLoadFailure::Invariant(error)) => return Err(error),
                    Err(CubinLoadFailure::Admission(error)) => {
                        // Only the cold CUDA module/function admission failure is recoverable.
                        // Once launched, any CUDA execution fault must propagate unchanged.
                        eprintln!(
                            "MISTRAL_DKR_CUBIN_FALLBACK operation={} cc={} error={} selected=ptx reason=module_admission_failure",
                            module.manifest.operation, cc, error
                        );
                        Self::load_ptx(module, device)?
                    }
                },
                None => Self::load_ptx(module, device)?,
            },
        };
        let evidence=telemetry::handle(TYPED_ABI,&module.manifest.operation,&module.manifest.entry,&module.manifest.sha256,module.manifest.target_cc);
        evidence.record(Event::Resolved);
        Ok(Self {
            manifest: module.manifest.clone(),
            function,
            device: device.clone(),
            cc,
            evidence,
        })
    }

    fn load_ptx(module: &VerifiedKernel, device: &Device) -> Result<CudaFunction> {
        #[cfg(feature = "shared-registry")]
        {
            let artifact = match &module.ptx {
                std::borrow::Cow::Borrowed(ptx) => dkr_kernel_registry::VerifiedArtifact::from_build_verified(
                    ptx, module.manifest.target_cc, &module.manifest.sha256, [module.manifest.entry.clone()]),
                std::borrow::Cow::Owned(ptx) => dkr_kernel_registry::VerifiedArtifact::new(ptx.clone(),
                    module.manifest.target_cc, &module.manifest.sha256).map_err(anyhow::Error::msg)?,
            };
            let (func, first) = device.as_cuda_device()?
                .get_or_load_dkr_func(&artifact, &module.manifest.entry)?;
            if first && std::env::var_os("CANDLE_DKR_TRACE").is_some() {
                eprintln!("MISTRAL_DKR_IMAGE_LOADED operation={} target=sm{} format=ptx",
                    module.manifest.operation, module.manifest.target_cc);
            }
            Ok(func.into_cuda_function())
        }
        #[cfg(not(feature = "shared-registry"))]
        {
            let func = device.as_cuda_device()?.get_or_load_custom_func(
                &module.manifest.entry, &module.module_key(), &module.ptx)?;
            Ok(func.into_cuda_function())
        }
    }

    fn load_cubin(module: &VerifiedKernel, device: &Device, image: &[u8]) -> std::result::Result<CudaFunction, CubinLoadFailure> {
        let sha = module.manifest.cubin_sha256.as_deref()
            .ok_or_else(|| CubinLoadFailure::Invariant(anyhow!("CUBIN has no verified digest")))?;
        let cache = CUBIN_MODULES.get_or_init(|| Mutex::new(Vec::new()));
        let mut cache = cache.lock().map_err(|_| CubinLoadFailure::Invariant(anyhow!("CUBIN module cache poisoned")))?;
        if let Some(existing) = cache.iter().find(|row|
            row.device.same_device(device) && row.sha256 == sha
        ) {
            return existing.module.load_function(&module.manifest.entry)
                .map_err(|error| CubinLoadFailure::Admission(error.into()));
        }
        // cudarc accepts a byte image through the existing Candle CUDA context;
        // module admission and entry-symbol lookup happen before any inference.
        let stream = device.as_cuda_device().map_err(|error|
            CubinLoadFailure::Invariant(error.into()))?.cuda_stream();
        let loaded = stream.context().load_module(Ptx::from_binary(image.to_vec()))
            .map_err(|error| CubinLoadFailure::Admission(error.into()))?;
        let func = loaded.load_function(&module.manifest.entry)
            .map_err(|error| CubinLoadFailure::Admission(error.into()))?;
        if cache.len() >= MAX_CUBIN_MODULES {
            cache.remove(0);
        }
        cache.push(CubinCacheEntry {
            device: device.clone(),
            sha256: sha.to_owned(),
            module: loaded,
        });
        if std::env::var_os("CANDLE_DKR_TRACE").is_some() {
            eprintln!("MISTRAL_DKR_IMAGE_LOADED operation={} target=sm{} format=cubin cubin_sha256={} ptx_sha256={}",
                module.manifest.operation, module.manifest.target_cc, sha, module.manifest.sha256);
        }
        Ok(func)
    }

    pub fn launch_config(&self, request: &LaunchRequest, device: &Device) -> Result<LaunchConfig> {
        ensure!(
            device.same_device(&self.device),
            "CUDA context mismatch for dynamic kernel"
        );
        ensure!(
            request.cc == self.cc,
            "caller GPU architecture differs from loaded module"
        );
        self.manifest.admit(request)?;
        Ok(LaunchConfig {
            grid_dim: (request.grid[0], request.grid[1], request.grid[2]),
            block_dim: (
                self.manifest.block[0],
                self.manifest.block[1],
                self.manifest.block[2],
            ),
            shared_mem_bytes: self.manifest.shared_mem_bytes,
        })
    }

    pub fn prevalidate_launch(
        self: &Arc<Self>,
        request: &LaunchRequest,
        device: &Device,
    ) -> Result<PrevalidatedLaunchPlan> {
        let config = self.launch_config(request, device)?;
        Ok(PrevalidatedLaunchPlan {
            kernel: Arc::clone(self),
            config,
        })
    }

    // The adapter must supply actual CUDA buffers/scalars in the declared argument order.
    pub unsafe fn with_verified_launch<T>(
        &self,
        request: &LaunchRequest,
        device: &Device,
        launch: impl FnOnce(&CudaFunction, LaunchConfig) -> Result<T>,
    ) -> Result<T> {
        let config = match self.launch_config(request, device) {
            Ok(c)=>c, Err(e)=>{self.evidence.record(Event::Rejected);return Err(e)}
        };
        let result=launch(&self.function, config);
        self.evidence.record(if result.is_ok(){Event::AdapterOk}else{Event::AdapterError});
        result
    }
}

impl PrevalidatedLaunchPlan {
    // The caller must retain the validated tensor layout and CUDA device invariants for every replay.
    pub unsafe fn with_prevalidated_launch<T>(
        &self,
        launch: impl FnOnce(&CudaFunction, LaunchConfig) -> Result<T>,
    ) -> Result<T> {
        let config = LaunchConfig {
            grid_dim: self.config.grid_dim,
            block_dim: self.config.block_dim,
            shared_mem_bytes: self.config.shared_mem_bytes,
        };
        let result=launch(&self.kernel.function, config);
        self.kernel.evidence.record(if result.is_ok(){Event::AdapterOk}else{Event::AdapterError});
        result
    }
}
