use anyhow::{ensure, Result};
use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::PushKernelArg, CpuStorage, CudaStorage,
    CustomOp1, DType, Device, Layout, Shape, Tensor,
};
use mistralrs_kernel_runtime::{
    cuda::compute_capability,
    universal::{Argument, ElementType, KernelCatalog, LaunchRequest},
    universal_cuda::PreparedKernel,
};
use std::{path::PathBuf, sync::Arc};

const OPERATION: &str = "tensor.elementwise.scale.v1";
const ALPHA: f32 = 1.75;
const LENGTH: usize = 257;

struct ScaleOp {
    prepared: Arc<PreparedKernel>,
    request: LaunchRequest,
    device: Device,
}

impl CustomOp1 for ScaleOp {
    fn name(&self) -> &'static str {
        OPERATION
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("dynamic CUDA provider has no implicit CPU fallback")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let verify = || -> Result<usize> {
            ensure!(input.dtype() == DType::F32, "expected F32 input");
            ensure!(
                layout.is_contiguous() && layout.start_offset() == 0,
                "invalid input layout"
            );
            let len = layout.shape().elem_count();
            ensure!(len == LENGTH, "unexpected input length");
            ensure!(
                input.device().id() == self.device.as_cuda_device()?.id(),
                "CUDA context mismatch"
            );
            Ok(len)
        };
        let count = verify().map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        let cuda = input.device();
        let xs = input.as_cuda_slice::<f32>()?;
        let mut output = cuda.alloc_zeros::<f32>(count)?;
        let length = count as u32;
        unsafe {
            self.prepared
                .with_verified_launch(&self.request, &self.device, |function, config| {
                    let stream = cuda.cuda_stream();
                    let mut builder = stream.launch_builder(function);
                    builder.arg(xs).arg(&ALPHA).arg(&mut output).arg(&length);
                    builder.launch(config)?;
                    Ok(())
                })
        }
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        Ok((
            CudaStorage::wrap_cuda_slice(output, cuda.clone()),
            layout.shape().clone(),
        ))
    }
}

fn main() -> Result<()> {
    let root = std::env::var_os("DKR_KERNEL_PACKAGES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("kernel-packages/cuda"));
    let catalog = KernelCatalog::discover(&root)?;
    let device = Device::new_cuda(0)?;
    let cc = compute_capability(&device)?;
    let tensor = Argument::Tensor {
        dtype: ElementType::F32,
        shape: vec![LENGTH],
        contiguous: true,
        start_offset: 0,
        representation: None,
    };
    let mut request = LaunchRequest {
        operation: OPERATION.into(),
        cc,
        args: vec![
            tensor.clone(),
            Argument::Scalar {
                dtype: ElementType::F32,
            },
            tensor,
            Argument::Scalar {
                dtype: ElementType::U32,
            },
        ],
        grid: [1, 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let selected = catalog.resolve(&request)?;
    request.grid = [(LENGTH as u32).div_ceil(selected.manifest.block[0]), 1, 1];
    let prepared = Arc::new(PreparedKernel::load_auto(&catalog, &request, &device)?);
    let target_cc = prepared.manifest.target_cc;
    let source: Vec<f32> = (0..LENGTH).map(|i| (i as f32 - 13.0) / 17.0).collect();
    let input = Tensor::new(source.as_slice(), &device)?;
    let output = input.apply_op1_no_bwd(&ScaleOp {
        prepared,
        request,
        device,
    })?;
    let values = output.to_vec1::<f32>()?;
    let max_error = source
        .iter()
        .zip(&values)
        .map(|(x, y)| (x * ALPHA - y).abs())
        .fold(0f32, f32::max);
    ensure!(
        max_error <= 1e-5,
        "dynamic CUDA result differs from scalar oracle"
    );
    println!("DKR_AUTO_SCALE_GPU=PASS device_cc={cc} selected_target_cc={target_cc} max_error={max_error}");
    Ok(())
}
