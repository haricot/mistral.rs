use crate::{
    cuda::compute_capability,
    cooperative::{requires_cooperative_launch, resident_group_grid},
    model_scope::{read_scope_receipt, ScopeLatch, ScopeReceipt},
    universal::{KernelCatalog, KernelManifest, LaunchRequest, VerifiedKernel},
};
use anyhow::{anyhow, ensure, Context, Result};
use dkr_kernel_registry::{telemetry::{self,Event,Handle},TYPED_ABI};
use candle_core::{
    cuda_backend::cudarc::driver::{CudaFunction, CudaStream, LaunchArgs, LaunchConfig},
    Device,
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
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
    scope: Option<ScopeReceipt>,
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
        let scope: Option<ScopeReceipt> = if source == "embedded" {
            crate::embedded::SCOPE_RECEIPT.map(serde_json::from_str).transpose()?
        } else {
            read_scope_receipt(root)?
        };
        if std::env::var_os("DKR_REQUIRE_MODEL_SCOPE").is_some_and(|v| v == "1") {
            ensure!(scope.is_some(), "required DKR model scope receipt missing");
        }
        if let Some(receipt) = &scope {
            receipt.verify(&catalog)?;
        }
        static PROCESS_SCOPE: OnceLock<ScopeLatch> = OnceLock::new();
        PROCESS_SCOPE.get_or_init(ScopeLatch::default).admit(scope.as_ref())?;
        Ok(Self {
            catalog,
            scope,
            prepared: Mutex::new(Vec::new()),
        })
    }

    pub fn scope_id(&self) -> Option<&str> {
        self.scope.as_ref().map(|scope| scope.scope_id.as_str())
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
        if let Some(scope) = &self.scope {
            ensure!(request.cc == scope.device_cc, "device CC differs from admitted model scope");
        }
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

/// No ordinary-launch callback or CudaFunction accessor is exposed here.
pub struct PrevalidatedCooperativeLaunchPlan {
    kernel: Arc<PreparedKernel>,
    config: LaunchConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct CooperativeDeviceLimits {
    pub sms: u32,
    pub max_shared_bytes: u32,
}

pub fn cooperative_device_limits(device: &Device) -> Result<Option<CooperativeDeviceLimits>> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    let stream = device.as_cuda_device()?.cuda_stream();
    let context = stream.context();
    context.bind_to_thread()?;
    if context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)? == 0 {
        return Ok(None);
    }
    let sms = u32::try_from(context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?)?;
    let max_shared_bytes = u32::try_from(context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?)?;
    if sms == 0 || max_shared_bytes == 0 { return Ok(None); }
    Ok(Some(CooperativeDeviceLimits { sms, max_shared_bytes }))
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
        ensure!(!requires_cooperative_launch(&self.manifest.operation),
            "cooperative kernel requires the cooperative launch API");
        self.launch_config_internal(request, device)
    }

    fn launch_config_internal(&self, request: &LaunchRequest, device: &Device) -> Result<LaunchConfig> {
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
        ensure!(!requires_cooperative_launch(&self.manifest.operation), "cooperative kernel requires grouped residency");
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

    /// For a typed kernel whose manifest declares a dynamic shared-memory
    /// ceiling, bind the shape-derived launch size rather than the ceiling.
    /// Safety: the caller must prove the kernel's actual shared-memory
    /// accesses fit in `bytes` for this exact validated launch signature.
    pub unsafe fn prevalidate_launch_with_shared(
        self: &Arc<Self>, request: &LaunchRequest, device: &Device, bytes: u32,
    ) -> Result<PrevalidatedLaunchPlan> {
        let mut config = self.launch_config(request, device)?;
        ensure!(bytes <= self.manifest.shared_mem_bytes, "dynamic shared size exceeds manifest ceiling");
        config.shared_mem_bytes = bytes;
        Ok(PrevalidatedLaunchPlan {kernel:Arc::clone(self),config})
    }

    /// Shape-derived shared memory using the ordinary (non-opt-in) launch
    /// budget, including the function's static allocation. Checked on plan
    /// miss only. The caller must prove the device accesses fit in bytes.
    pub unsafe fn prevalidate_launch_with_default_shared(
        self: &Arc<Self>, request: &LaunchRequest, device: &Device, bytes: u32,
    ) -> Result<PrevalidatedLaunchPlan> {
        use candle_core::cuda_backend::cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
        let plan = unsafe { self.prevalidate_launch_with_shared(request, device, bytes)? };
        let stream = device.as_cuda_device()?.cuda_stream();
        let context = stream.context();
        context.bind_to_thread()?;
        let limit = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let static_bytes = self.function.get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
        let total = static_bytes.checked_add(i32::try_from(bytes)?).context("shared memory overflow")?;
        ensure!(static_bytes >= 0 && total <= limit, "kernel shared memory exceeds default device limit");
        Ok(plan)
    }

    /// Construct a fully resident cooperative grid, using actual function
    /// occupancy. None means the cooperative algorithm is not eligible BEFORE
    /// any launch. Driver errors are not a fallback signal.
    ///
    /// Safety: the caller proves that each group has exactly ctas_per_group
    /// CTAs and that all shared and global accesses fit the declared buffers.
    pub unsafe fn prevalidate_cooperative_grouped(
        self: &Arc<Self>, request: &LaunchRequest, device: &Device,
        bytes: u32, rows: u32, ctas_per_group: u32,
    ) -> Result<Option<PrevalidatedCooperativeLaunchPlan>> {
        use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute;
        ensure!(requires_cooperative_launch(&self.manifest.operation),
            "ordinary kernel cannot be admitted as a cooperative kernel");
        ensure!(request.grid == [1,1,1] && rows > 0 && ctas_per_group > 0,
            "invalid cooperative preparation request");
        let mut config = self.launch_config_internal(request, device)?;
        ensure!(bytes <= self.manifest.shared_mem_bytes, "cooperative shared-memory ceiling exceeded");
        let Some(limits) = cooperative_device_limits(device)? else { return Ok(None); };
        let static_bytes = u32::try_from(self.function.get_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?)?;
        let Some(available) = limits.max_shared_bytes.checked_sub(static_bytes) else { return Ok(None); };
        // Always set the same maximum for a given function/device, not the
        // per-shape size: another cached plan must not have its cap reduced.
        let ceiling = available.min(self.manifest.shared_mem_bytes);
        if bytes > ceiling { return Ok(None); }
        self.function.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            i32::try_from(ceiling)?)?;
        let threads = self.manifest.block.iter().product::<u32>();
        let active = self.function.occupancy_max_active_blocks_per_multiprocessor(
            threads, usize::try_from(bytes)?, None)?;
        let Some(grid_x) = resident_group_grid(rows, ctas_per_group, active, limits.sms)? else {
            return Ok(None);
        };
        let mut actual_request = request.clone();
        actual_request.grid = [grid_x,1,1];
        self.manifest.admit(&actual_request)?;
        config.grid_dim = (grid_x,1,1);
        config.shared_mem_bytes = bytes;
        Ok(Some(PrevalidatedCooperativeLaunchPlan { kernel: Arc::clone(self), config }))
    }

    /// Resource-safe fixed dynamic shared-memory plan. Opt-in is performed
    /// only on a plan miss, before graph capture; no architecture-specific policy.
    pub fn prevalidate_launch_with_optin(
        self: &Arc<Self>, request: &LaunchRequest, device: &Device,
    ) -> Result<PrevalidatedLaunchPlan> {
        use candle_core::cuda_backend::cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
        let config = self.launch_config(request, device)?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let context = stream.context();
        context.bind_to_thread()?;
        let default = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?;
        let static_bytes = self.function.get_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?;
        let dynamic = i32::try_from(config.shared_mem_bytes)?;
        let total = static_bytes.checked_add(dynamic).context("shared memory overflow")?;
        if total > default {
            let optin = context.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?;
            ensure!(total <= optin, "kernel shared memory exceeds opt-in device limit");
            self.function.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,dynamic)?;
        }
        Ok(PrevalidatedLaunchPlan {kernel:Arc::clone(self),config})
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


impl PrevalidatedCooperativeLaunchPlan {
    /// Submit only through the cooperative driver entry point. This method
    /// deliberately never returns the underlying function to an adapter.
    /// Safety: bindings must match the admitted ABI, and buffer/stream
    /// lifetimes and group geometry must remain valid for every replay.
    pub unsafe fn launch_cooperative<'a>(
        &'a self, stream: &'a CudaStream,
        bind: impl FnOnce(&mut LaunchArgs<'a>),
    ) -> Result<()> {
        let expected = self.kernel.device.as_cuda_device()?.cuda_stream();
        ensure!(Arc::ptr_eq(expected.context(), stream.context()) &&
            expected.cu_stream() == stream.cu_stream(), "cooperative launch stream mismatch");
        let config = LaunchConfig { grid_dim: self.config.grid_dim,
            block_dim: self.config.block_dim, shared_mem_bytes: self.config.shared_mem_bytes };
        let mut args = stream.launch_builder(&self.kernel.function);
        bind(&mut args);
        let result = unsafe { args.launch_cooperative(config) }.map(|_| ()).map_err(anyhow::Error::from);
        self.kernel.evidence.record(if result.is_ok() { Event::AdapterOk } else { Event::AdapterError });
        result
    }
}
