use crate::{tensor_contract, Manifest, VerifiedModule};
use anyhow::{ensure, Result};
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use candle_core::{CpuStorage, CudaStorage, CustomOp2, Device, Layout, Shape, Tensor};

pub fn compute_capability(device: &Device) -> Result<u32> {
    let stream = device.as_cuda_device()?.cuda_stream();
    let (major, minor) = stream.context().compute_capability()?;
    Ok((major * 10 + minor) as u32)
}

pub struct PreparedLinear {
    pub manifest: Manifest,
    function: CudaFunction,
    device: Device,
    cc: u32,
}

impl PreparedLinear {
    pub fn load(module: VerifiedModule, device: &Device) -> Result<Self> {
        let cc = compute_capability(device)?;
        module.manifest.admit(cc, [1, 1, 1])?;
        let function = device
            .as_cuda_device()?
            .get_or_load_custom_func(
                &module.manifest.entry,
                &module.manifest.module_key(),
                &module.ptx,
            )?
            .into_cuda_function();
        Ok(Self {
            manifest: module.manifest,
            function,
            device: device.clone(),
            cc,
        })
    }

    pub fn run(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        ensure!(
            x.device().same_device(&self.device) && w.device().same_device(&self.device),
            "provider and tensors must share the Candle CUDA device/context"
        );
        let (m, k) = tensor_contract(x.dtype(), x.layout())?;
        let (n, wk) = tensor_contract(w.dtype(), w.layout())?;
        ensure!(k == wk, "linear inner dimension mismatch");
        self.manifest.admit(self.cc, [m, n, k])?;
        Ok(x.apply_op2_no_bwd(w, self)?)
    }
}

impl CustomOp2 for PreparedLinear {
    fn name(&self) -> &'static str {
        crate::ABI
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("external CUDA provider has no CPU fallback")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let check = || -> Result<[usize; 3]> {
            let (m, k) = tensor_contract(x.dtype(), xl)?;
            let (n, wk) = tensor_contract(w.dtype(), wl)?;
            ensure!(k == wk, "linear inner dimension mismatch");
            let expected = self.device.as_cuda_device()?.id();
            ensure!(
                x.device().id() == expected && w.device().id() == expected,
                "CUDA context mismatch"
            );
            self.manifest.admit(self.cc, [m, n, k])?;
            Ok([m, n, k])
        };
        let [m, n, k] = check().map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let device = x.device();
        let xs = x.as_cuda_slice::<f32>()?;
        let ws = w.as_cuda_slice::<f32>()?;
        // NaNs reveal unwritten outputs; initialization is on the same Candle stream.
        let mut output = device.clone_htod(&vec![f32::NAN; m * n])?;
        let [bx, by, bz] = self.manifest.block;
        let config = LaunchConfig {
            grid_dim: ((n as u32).div_ceil(bx), (m as u32).div_ceil(by), 1),
            block_dim: (bx, by, bz),
            shared_mem_bytes: 0,
        };
        let stream = device.cuda_stream();
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(xs)
            .arg(ws)
            .arg(&mut output)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        // The manifest binds the trusted external kernel to this pointer/scalar ABI.
        unsafe { launch.launch(config) }.map_err(candle_core::Error::wrap)?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from((m, n)),
        ))
    }
}
