use crate::dkr_cuda::{catalog, dtype, tensor, LaunchPlanCache};
use candle_core::{
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

#[derive(Clone, Copy, Eq, PartialEq)]
struct PlanKey {
    dtype: DType,
    batch: usize,
    heads: usize,
    queries: usize,
    columns: usize,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<PlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn operation(columns: usize) -> &'static str {
    match columns {
        0..=64 => "attention.softmax_sinks.b64.v1",
        65..=128 => "attention.softmax_sinks.b128.v1",
        129..=256 => "attention.softmax_sinks.b256.v1",
        257..=512 => "attention.softmax_sinks.b512.v1",
        _ => "attention.softmax_sinks.b1024.v1",
    }
}

pub(crate) fn cuda_fwd(
    storage: &CudaStorage,
    layout: &Layout,
    sinks: &Tensor,
    heads: usize,
    queries: usize,
    columns: usize,
) -> Result<(CudaStorage, Shape)> {
    let dtype_value = storage.dtype();
    let element_type = dtype(dtype_value)?;
    let shape = layout.shape().clone();
    let [batch, shape_heads, shape_queries, shape_columns] = shape.dims() else {
        candle_core::bail!("softmax with sinks requires four-dimensional logits")
    };
    if *shape_heads != heads || *shape_queries != queries || *shape_columns != columns {
        candle_core::bail!("softmax with sinks shape differs from prepared dimensions")
    }
    if !layout.is_contiguous() || sinks.dims() != [heads] || sinks.dtype() != dtype_value {
        candle_core::bail!("softmax with sinks requires contiguous, same-dtype inputs")
    }
    if *batch == 0 || heads == 0 || queries == 0 || columns == 0 {
        candle_core::bail!("softmax with sinks requires nonempty dimensions")
    }
    let rows = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(queries))
        .ok_or_else(|| candle_core::Error::msg("softmax with sinks row count overflow"))?;
    let rows_u32 = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let heads_u32 = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let queries_u32 = u32::try_from(queries).map_err(candle_core::Error::wrap)?;
    let columns_u32 = u32::try_from(columns).map_err(candle_core::Error::wrap)?;
    let elements = rows
        .checked_mul(columns)
        .ok_or_else(|| candle_core::Error::msg("softmax with sinks element count overflow"))?;
    let cuda = storage.device();
    let device = Device::Cuda(cuda.clone());
    if !device.same_device(sinks.device()) {
        candle_core::bail!("softmax with sinks requires one CUDA device")
    }
    let stream = cuda.cuda_stream();
    let (sinks_storage, sinks_layout) = sinks.storage_and_layout();
    let Storage::Cuda(sinks_storage) = &*sinks_storage else {
        candle_core::bail!("softmax with sinks requires CUDA sinks")
    };
    let key = PlanKey {
        dtype: dtype_value,
        batch: *batch,
        heads,
        queries,
        columns,
    };

    THREAD_LAUNCH_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(
            key,
            &device,
            || {
                let mut request = LaunchRequest {
                    operation: operation(columns).into(),
                    cc: 0,
                    args: vec![
                        tensor(element_type, vec![*batch, heads, queries, columns]),
                        tensor(element_type, vec![heads]),
                        tensor(element_type, vec![*batch, heads, queries, columns]),
                        Argument::Scalar {
                            dtype: ElementType::U32,
                        },
                        Argument::Scalar {
                            dtype: ElementType::U32,
                        },
                        Argument::Scalar {
                            dtype: ElementType::U32,
                        },
                    ],
                    grid: [1, 1, 1],
                    max_dynamic_shared_mem_bytes: 0,
                };
                let prepared = catalog()?
                    .prepare_for_device(&mut request, &device)
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                let expected_block = match columns {
                    0..=64 => 64,
                    65..=128 => 128,
                    129..=256 => 256,
                    257..=512 => 512,
                    _ => 1024,
                };
                if prepared.manifest.block != [expected_block, 1, 1] {
                    candle_core::bail!("softmax package block does not match reduction geometry")
                }
                request.grid[0] = rows_u32;
                prepared
                    .prevalidate_launch(&request, &device)
                    .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (logits_base, _logits_guard) =
                            storage.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        let (sinks_base, _sinks_guard) =
                            sinks_storage.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        let logits_ptr = logits_base
                            + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64;
                        let sinks_ptr = sinks_base
                            + (sinks_layout.start_offset() * std::mem::size_of::<$ty>()) as u64;
                        let mut output = unsafe { cuda.alloc::<$ty>(elements)? };
                        {
                            let (out_ptr, _out_guard) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder
                                        .arg(&logits_ptr)
                                        .arg(&sinks_ptr)
                                        .arg(&out_ptr)
                                        .arg(&heads_u32)
                                        .arg(&queries_u32)
                                        .arg(&columns_u32);
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        Ok((CudaStorage::wrap_cuda_slice(output, cuda.clone()), shape))
                    }};
                }
                match dtype_value {
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
                    DType::F32 => launch!(f32),
                    _ => unreachable!(),
                }
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA and externally built reduction packages"]
    fn dkr_softmax_sinks_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for columns in [3, 64, 65, 128, 129, 256, 257, 512, 513] {
                let logits = Tensor::ones((2, 2, 2, columns), dtype, &device)?;
                let sinks = Tensor::from_vec(vec![0.0f32, 1.0], (2,), &device)?
                    .to_dtype(dtype)?;
                let result = crate::softmax_with_sinks(&logits, &sinks, None)?;
                let cached = THREAD_LAUNCH_PLANS.with(|hot| hot.borrow().len());
                let repeated = crate::softmax_with_sinks(&logits, &sinks, None)?;
                assert_eq!(cached, THREAD_LAUNCH_PLANS.with(|hot| hot.borrow().len()));
                let flattened = result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (index, &value) in flattened.iter().enumerate() {
                    let head = (index / (columns * 2)) % 2;
                    let sink_exp = if head == 0 { 1.0 } else { std::f32::consts::E };
                    let expected = std::f32::consts::E
                        / (columns as f32 * std::f32::consts::E + sink_exp);
                    assert!((value - expected).abs() < 0.005, "{value} != {expected}");
                }
                assert_eq!(
                    result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    repeated.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        let logits = Tensor::zeros((1, 1, 1, 4), DType::F32, &device)?;
        let mask = Tensor::from_vec(
            vec![f32::NEG_INFINITY, 0.0, 0.0, 0.0],
            (1, 1, 1, 4),
            &device,
        )?;
        let sinks = Tensor::zeros((1,), DType::F32, &device)?;
        let masked = crate::softmax_with_sinks(&logits, &sinks, Some(&mask))?;
        assert_eq!(
            masked.flatten_all()?.to_vec1::<f32>()?,
            vec![0.0, 0.25, 0.25, 0.25]
        );
        Ok(())
    }
}
