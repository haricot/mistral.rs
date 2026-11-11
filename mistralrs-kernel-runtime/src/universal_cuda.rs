use crate::{
    cuda::compute_capability,
    universal::{KernelManifest, LaunchRequest, VerifiedKernel},
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
    module_key: String,
}

#[derive(Clone, Debug)]
pub struct PrevalidatedLaunchPlan {
    module_key: String,
    cc: u32,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem_bytes: u32,
}

impl PreparedKernel {
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
            module_key: module.module_key(),
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

    pub fn prepare_launch(
        &self,
        module: &VerifiedKernel,
        request: &LaunchRequest,
        device: &Device,
    ) -> Result<PrevalidatedLaunchPlan> {
        ensure!(
            module.module_key() == self.module_key,
            "launch plan uses a different verified kernel"
        );
        let config = self.launch_config(request, device)?;
        Ok(PrevalidatedLaunchPlan {
            module_key: self.module_key.clone(),
            cc: self.cc,
            grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
            block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
            shared_mem_bytes: config.shared_mem_bytes,
        })
    }

    // The caller still owns argument values, buffer lifetimes, stream order and immutable shapes.
    pub unsafe fn with_prevalidated_launch<T>(
        &self,
        plan: &PrevalidatedLaunchPlan,
        device: &Device,
        launch: impl FnOnce(&CudaFunction, LaunchConfig) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            device.same_device(&self.device),
            "CUDA context mismatch for prevalidated launch"
        );
        ensure!(
            plan.cc == self.cc && plan.module_key == self.module_key,
            "prevalidated launch plan does not match prepared kernel"
        );
        let config = LaunchConfig {
            grid_dim: (plan.grid[0], plan.grid[1], plan.grid[2]),
            block_dim: (plan.block[0], plan.block[1], plan.block[2]),
            shared_mem_bytes: plan.shared_mem_bytes,
        };
        launch(&self.function, config)
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
