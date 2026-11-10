use crate::{
    cuda::compute_capability,
    universal::{KernelCatalog, KernelManifest, LaunchRequest, VerifiedKernel},
};
use anyhow::{ensure, Result};
use candle_core::{
    cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig},
    Device,
};

pub struct PreparedKernel {
    pub manifest: KernelManifest,
    function: CudaFunction,
    device: Device,
    cc: u32,
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
