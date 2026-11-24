use crate::{
    cuda::compute_capability,
    universal::{KernelCatalog, KernelManifest, LaunchRequest, VerifiedKernel},
};
use anyhow::{anyhow, ensure, Context, Result};
use dkr_kernel_registry::{telemetry::{self,Event,Handle},TYPED_ABI};
use candle_core::{
    cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig},
    Device,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const MAX_PREPARED_KERNELS: usize = 128;
const SHARED_CATALOG_ENV: &str = "MISTRALRS_DKR_SHARED_CATALOG";
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
        #[cfg(feature = "shared-registry")]
        let function = {
            let artifact = match &module.ptx {
                crate::universal::SharedImage::Embedded(ptx) => dkr_kernel_registry::VerifiedArtifact::from_build_verified(
                    ptx, module.manifest.target_cc, &module.manifest.sha256,
                    [module.manifest.entry.clone()]),
                crate::universal::SharedImage::External(ptx) => dkr_kernel_registry::VerifiedArtifact::new(
                    ptx.to_string(), module.manifest.target_cc, &module.manifest.sha256)
                    .map_err(anyhow::Error::msg)?,
            };
            let image = match (&module.cubin, module.manifest.cubin_sha256.as_deref()) {
                (Some(bytes), Some(sha)) => Some((bytes.as_ref(), sha)),
                (None, None) => None,
                _ => anyhow::bail!("typed manifest must attest both CUBIN bytes and digest"),
            };
            // Candle owns the CUDA context, image policy, verification and
            // the physical image cache; Mistral only supplies the typed ABI.
            let (func, _) = device.as_cuda_device()?
                .get_or_load_dkr_typed_image_func(&artifact, &module.manifest.entry, image)?;
            func.into_cuda_function()
        };
        #[cfg(not(feature = "shared-registry"))]
        let function = device.as_cuda_device()?.get_or_load_custom_func(
            &module.manifest.entry, &module.module_key(), &module.ptx)?
            .into_cuda_function();
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

    /// Device-aware persistent grid, computed on plan miss only.
    /// Matches the historical FlashInfer occupancy policy without a CUDA host shim.
    pub fn persistent_grid_x(&self, work_items: u32, device: &Device) -> Result<u32> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
        ensure!(device.same_device(&self.device) && work_items > 0,
            "invalid persistent launch device or work count");
        let stream = device.as_cuda_device()?.cuda_stream();
        let context = stream.context();
        context.bind_to_thread()?;
        let sms = u32::try_from(context.attribute(
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?;
        ensure!(sms > 0, "CUDA device has no multiprocessors");
        let threads = self.manifest.block.iter().product::<u32>();
        let resident = self.function.occupancy_max_active_blocks_per_multiprocessor(
            threads, self.manifest.shared_mem_bytes as usize, None)?;
        ensure!(resident > 0, "persistent kernel has zero occupancy");
        sms.checked_mul(resident.min(work_items.div_ceil(sms)))
            .context("persistent grid overflow")
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
