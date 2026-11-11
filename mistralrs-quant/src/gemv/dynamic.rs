//! DKR-R1.15: external SIMT dense GEMV, contiguous CUDA tensors only.
//! Typed, fixed, separate plain/bias ABI; no nullable launch argument.
//! Keep the legacy GEMV dispatcher for noncontiguous and unsupported layouts.

use crate::dkr_cuda::{catalog, dtype, tensor, LaunchPlanCache};
use crate::utils::slice_ptr_on_stream;
use candle_core::{
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const BLOCK: u32 = 128;

#[derive(Clone, Copy, Eq, PartialEq)]
struct PlanKey {
    dtype: DType,
    batch: usize,
    rows: usize,
    columns: usize,
    has_bias: bool,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<PlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn prepare_plan(key: PlanKey, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let dtype = dtype(key.dtype)?;
    let mode = if key.has_bias { "bias" } else { "plain" };
    let mut args = vec![
        tensor(dtype, vec![key.rows, key.columns]),
        tensor(dtype, vec![key.batch, key.columns]),
    ];
    if key.has_bias {
        args.push(tensor(dtype, vec![key.rows]));
    }
    args.extend([
        tensor(dtype, vec![key.batch, key.rows]),
        Argument::Scalar { dtype: ElementType::U32 },
        Argument::Scalar { dtype: ElementType::U32 },
        Argument::Scalar { dtype: ElementType::U32 },
    ]);
    let mut request = LaunchRequest {
        operation: format!("linear.gemv.{mode}.b128.v1"),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.rows).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.batch).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] {
        candle_core::bail!("DKR GEMV: unexpected CUDA block geometry");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Returns None only for layouts handled by the existing static implementation.
/// A missing/invalid DKR package is a hard error when the external route is chosen.
pub(super) fn try_gemv(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    batch: usize,
    rows: usize,
    columns: usize,
    output_shape: &[usize],
) -> Result<Option<Tensor>> {
    if !x.is_contiguous()
        || !weight.is_contiguous()
        || bias.is_some_and(|bias| !bias.is_contiguous())
        || !x.device().same_device(weight.device())
        || bias.is_some_and(|bias| !x.device().same_device(bias.device()))
        || weight.dtype() != x.dtype()
        || bias.is_some_and(|bias| bias.dtype() != x.dtype() || bias.dims() != [rows])
        || rows == 0 || batch == 0 || columns == 0
        || rows > u32::MAX as usize || columns > u32::MAX as usize
        || batch > u32::MAX as usize
    {
        return Ok(None);
    }
    if !matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32) {
        return Ok(None);
    }
    let output_elements = batch
        .checked_mul(rows)
        .ok_or_else(|| candle_core::Error::msg("DKR GEMV: output size overflow"))?;
    let device = x.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    let key = PlanKey { dtype: x.dtype(), batch, rows, columns, has_bias: bias.is_some() };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (x_storage, x_layout) = x.storage_and_layout();
    let (Storage::Cuda(weight_storage), Storage::Cuda(x_storage)) =
        (&*weight_storage, &*x_storage)
    else {
        candle_core::bail!("DKR GEMV requires CUDA storage");
    };
    let bias_storage = bias.map(|value| value.storage_and_layout());
    let rows_u32 = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let columns_u32 = u32::try_from(columns).map_err(candle_core::Error::wrap)?;
    let batch_u32 = u32::try_from(batch).map_err(candle_core::Error::wrap)?;
    THREAD_LAUNCH_PLANS.with(|cache| {
        cache.borrow_mut().with_plan(
            key,
            x.device(),
            || prepare_plan(key, x.device()),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (weight_ptr, _weight_guard) = slice_ptr_on_stream(
                            weight_storage.as_cuda_slice::<$ty>()?,
                            weight_layout.start_offset(),
                            &stream,
                        );
                        let (input_ptr, _input_guard) = slice_ptr_on_stream(
                            x_storage.as_cuda_slice::<$ty>()?,
                            x_layout.start_offset(),
                            &stream,
                        );
                        let (bias_ptr, _bias_guard) =
                            if let Some((ref storage, layout)) = bias_storage {
                                let Storage::Cuda(storage) = &**storage else {
                                    candle_core::bail!("DKR GEMV requires CUDA bias");
                                };
                                let (ptr, guard) = slice_ptr_on_stream(
                                    storage.as_cuda_slice::<$ty>()?,
                                    layout.start_offset(),
                                    &stream,
                                );
                                (ptr, Some(guard))
                            } else {
                                (0u64, None)
                            };
                        let mut output = unsafe { device.alloc::<$ty>(output_elements)? };
                        {
                            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder.arg(&weight_ptr).arg(&input_ptr);
                                    if key.has_bias {
                                        builder.arg(&bias_ptr);
                                    }
                                    builder.arg(&output_ptr)
                                        .arg(&rows_u32).arg(&columns_u32).arg(&batch_u32);
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        let storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                        Ok(Some(Tensor::from((
                            Storage::Cuda(storage),
                            Shape::from(output_shape),
                        ))))
                    }};
                }
                match key.dtype {
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
    use candle_core::Device;

    #[test]
    #[ignore = "requires externally built DKR dense-GEMV PTX and an SM61 CUDA device"]
    fn dkr_dense_gemv_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for batch in [1, 3, 8] {
                let (rows, columns) = (17usize, 65usize);
                let x_values: Vec<f32> = (0..batch * columns)
                    .map(|i| (i as i32 % 11 - 5) as f32 / 16.0)
                    .collect();
                let w_values: Vec<f32> = (0..rows * columns)
                    .map(|i| (i as i32 % 13 - 6) as f32 / 16.0)
                    .collect();
                let bias_values: Vec<f32> = (0..rows)
                    .map(|i| (i as i32 % 7 - 3) as f32 / 16.0)
                    .collect();
                let x = Tensor::from_vec(x_values.clone(), (batch, columns), &device)?
                    .to_dtype(dtype)?;
                let w = Tensor::from_vec(w_values.clone(), (rows, columns), &device)?
                    .to_dtype(dtype)?;
                let bias = Tensor::from_vec(bias_values.clone(), rows, &device)?
                    .to_dtype(dtype)?;
                for with_bias in [false, true] {
                    let b = with_bias.then_some(&bias);
                    let output = super::super::gemv(&x, &w, b)?;
                    let actual = output.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    for row in 0..batch {
                        for col in 0..rows {
                            let expected = (0..columns).map(|idx| {
                                x_values[row * columns + idx] * w_values[col * columns + idx]
                            }).sum::<f32>()
                                + if with_bias { bias_values[col] } else { 0.0 };
                            let tolerance = if dtype == DType::BF16 { 0.025 } else { 0.008 };
                            assert!(
                                (actual[row][col] - expected).abs()
                                    < tolerance * (1.0 + expected.abs()),
                                "dtype={dtype:?}, batch={batch}, bias={with_bias}, \
                                 output={row},{col}: actual={} expected={expected}",
                                actual[row][col],
                            );
                        }
                    }
                    let cached = THREAD_LAUNCH_PLANS.with(|cache| cache.borrow().len());
                    let repeated = super::super::gemv(&x, &w, b)?;
                    assert_eq!(
                        cached,
                        THREAD_LAUNCH_PLANS.with(|cache| cache.borrow().len())
                    );
                    assert_eq!(
                        actual,
                        repeated.to_dtype(DType::F32)?.to_vec2::<f32>()?
                    );
                }
            }
        }
        Ok(())
    }
}
