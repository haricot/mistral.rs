use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{Device, Result};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct RmsPlanKey {
    dtype: ElementType,
    rows: u32,
    columns: u32,
    scaled: bool,
    second_norm: bool,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<RmsPlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

fn tensor(dtype: ElementType, shape: Vec<usize>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn scalar(dtype: ElementType) -> Argument {
    Argument::Scalar { dtype }
}

fn prepare(key: RmsPlanKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let rows = key.rows as usize;
    let columns = key.columns as usize;
    let x = || tensor(key.dtype, vec![rows, columns]);
    let w = || tensor(key.dtype, vec![columns]);
    let mut args = vec![x(), x(), w()];
    if key.scaled {
        args.push(tensor(key.dtype, vec![1]));
    }
    if key.second_norm {
        args.extend([w(), x(), x()]);
    } else {
        args.push(x());
    }
    args.extend([scalar(ElementType::U32), scalar(ElementType::U32)]);
    args.push(scalar(ElementType::F32));
    if key.second_norm {
        args.push(scalar(ElementType::F32));
    }
    let operation = format!(
        "tensor.{}.{}.v1",
        if key.second_norm {
            "rms_residual_then_rms"
        } else {
            "rms_residual"
        },
        if key.scaled { "scaled" } else { "plain" }
    );
    let mut request = LaunchRequest {
        operation,
        cc: 0,
        args,
        grid: [key.rows, 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = kernel_cache()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] {
        candle_core::bail!("RMSNorm variant external block must be 256 threads");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

pub(crate) fn with_rms_plan<T>(
    device: &Device,
    dtype: ElementType,
    rows: usize,
    columns: usize,
    scaled: bool,
    second_norm: bool,
    launch: impl FnOnce(&PrevalidatedLaunchPlan) -> Result<T>,
) -> Result<T> {
    let rows = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let columns = u32::try_from(columns).map_err(candle_core::Error::wrap)?;
    if rows == 0 || columns == 0 {
        candle_core::bail!("RMSNorm requires nonempty dimensions");
    }
    let key = RmsPlanKey {
        dtype,
        rows,
        columns,
        scaled,
        second_norm,
    };
    HOT_PLANS.with(|hot| {
        hot.borrow_mut()
            .with_plan(key, device, || prepare(key, device), launch)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};

    #[test]
    #[ignore = "requires SM61 GPU and externally built RMSNorm variant packages"]
    fn dkr_rms_variants_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for width in [7, 16, 520, 1024] {
                let input = Tensor::ones((2, width), dtype, &device)?;
                let residual = Tensor::ones((2, width), dtype, &device)?;
                let weight = Tensor::ones(width, dtype, &device)?;
                let scale = Tensor::from_vec(vec![0.5f32], 1, &device)?.to_dtype(dtype)?;
                for optional in [None, Some(&scale)] {
                    let result = crate::ops::cuda_rms_norm_residual(
                        &input, &residual, &weight, optional, 1e-6,
                    )?;
                    let (_, again) = crate::ops::cuda_rms_norm_residual_then_rms_norm(
                        &input, &residual, &weight, optional, &weight, 1e-6, 1e-6,
                    )?;
                    let expect = if optional.is_some() { 1.0f32 } else { 2.0 };
                    let actual = result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    for v in actual {
                        assert!((v - expect).abs() <= 0.02, "{v} != {expect}");
                    }
                    let values = again.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    for v in values {
                        assert!((v - 1.0).abs() <= 0.02, "{v} != 1.0");
                    }
                }
            }
        }
        Ok(())
    }
}
