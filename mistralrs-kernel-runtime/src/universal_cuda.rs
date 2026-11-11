use crate::{
    cuda::compute_capability,
    universal::{KernelCatalog, KernelManifest, LaunchRequest, VerifiedKernel},
};
use anyhow::{anyhow, ensure, Result};
use candle_core::{
    cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig},
    Device,
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};

const MAX_PREPARED_KERNELS: usize = 128;

struct PreparedEntry {
    request: LaunchRequest,
    device: Device,
    kernel: Arc<PreparedKernel>,
}

pub struct KernelExecutionCache {
    catalog: KernelCatalog,
    prepared: Mutex<Vec<PreparedEntry>>,
}

impl KernelExecutionCache {
    pub fn discover(root: &Path) -> Result<Self> {
        Ok(Self {
            catalog: KernelCatalog::discover(root)?,
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
            return Ok(Arc::clone(&entry.kernel));
        }
        request.cc = compute_capability(device)?;
        let kernel = Arc::new(PreparedKernel::load_auto(&self.catalog, request, device)?);
        if prepared.len() == MAX_PREPARED_KERNELS {
            prepared.swap_remove(0);
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
        let function = device
            .as_cuda_device()?
            .get_or_load_custom_func(&module.manifest.entry, &module.module_key(), &module.ptx)?
            .into_cuda_function();
        Ok(Self {
            manifest: module.manifest.clone(),
            function,
            device: device.clone(),
            cc,
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
        let config = self.launch_config(request, device)?;
        launch(&self.function, config)
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
        launch(&self.kernel.function, config)
    }
}
