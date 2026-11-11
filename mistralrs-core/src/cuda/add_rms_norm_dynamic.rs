use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Storage, Tensor, D,
};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

#[derive(Clone, Copy, Eq, PartialEq)]
struct AddRmsNormKey {
    dtype: ElementType,
    rows: u32,
    columns: u32,
    vectorized: bool,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<AddRmsNormKey, PrevalidatedLaunchPlan>> =
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

fn shifted_ptr(ptr: u64, offset: usize, size: usize) -> Result<u64> {
    let bytes = u64::try_from(offset)
        .ok()
        .and_then(|offset| offset.checked_mul(size as u64))
        .ok_or_else(|| candle_core::Error::msg("add RMSNorm CUDA offset overflow"))?;
    ptr.checked_add(bytes)
        .ok_or_else(|| candle_core::Error::msg("add RMSNorm CUDA pointer overflow"))
}

pub fn cuda_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_add_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_add_rms_norm dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_add_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_add_rms_norm tensors must be on the same CUDA device");
    }
    let columns = input.dim(D::Minus1)?;
    if weight.dims1()? != columns {
        candle_core::bail!(
            "cuda_add_rms_norm weight size {} does not match last dim {columns}",
            weight.dims1()?
        );
    }
    let elements = input.elem_count();
    if columns == 0 || elements == 0 {
        candle_core::bail!("cuda_add_rms_norm got empty input");
    }
    let rows = elements / columns;
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| candle_core::Error::msg("add RMSNorm row count exceeds u32"))?;
    let columns_u32 = u32::try_from(columns)
        .map_err(|_| candle_core::Error::msg("add RMSNorm column count exceeds u32"))?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(residual_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*residual_storage, &*weight_storage)
    else {
        candle_core::bail!("cuda_add_rms_norm requires CUDA input, residual and weight");
    };
    let cuda = input_storage.device();
    let stream = cuda.cuda_stream();
    let shape = input.shape().clone();

    macro_rules! launch {
        ($type:ty, $element:expr) => {{
            let (src, _src_guard) = input_storage
                .as_cuda_slice::<$type>()?
                .device_ptr(&stream);
            let (res, _res_guard) = residual_storage
                .as_cuda_slice::<$type>()?
                .device_ptr(&stream);
            let (wgt, _wgt_guard) = weight_storage
                .as_cuda_slice::<$type>()?
                .device_ptr(&stream);
            let mut residual_out = unsafe { cuda.alloc::<$type>(elements)? };
            let mut norm_out = unsafe { cuda.alloc::<$type>(elements)? };
            let (sum_dst, _sum_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_dst, _norm_guard) = norm_out.device_ptr_mut(&stream);
            let size = std::mem::size_of::<$type>();
            let src = shifted_ptr(src, input_layout.start_offset(), size)?;
            let res = shifted_ptr(res, residual_layout.start_offset(), size)?;
            let wgt = shifted_ptr(wgt, weight_layout.start_offset(), size)?;
            // The legacy CUDA wrapper requires 16-byte alignment of all five
            // pointers and eight columns per vector. Preserve that boundary.
            let vectorized = size == 2
                && columns % 8 == 0
                && [src, res, wgt, sum_dst, norm_dst]
                    .iter()
                    .all(|ptr| ptr % 16 == 0);
            let block = if vectorized {
                (columns / 8).div_ceil(32).saturating_mul(32).clamp(32, 1024)
            } else if columns < 1024 {
                32
            } else {
                1024
            };
            let key = AddRmsNormKey {
                dtype: $element,
                rows: rows_u32,
                columns: columns_u32,
                vectorized,
            };
            HOT_PLANS.with(|hot| {
                hot.borrow_mut().with_plan(
                    key,
                    input.device(),
                    || {
                        let mut request = LaunchRequest {
                            operation: format!(
                                "tensor.add_rms_norm.{}.b{block}.v1",
                                if vectorized { "vec8" } else { "scalar" }
                            ),
                            cc: 0,
                            args: vec![
                                tensor($element, vec![rows, columns]),
                                tensor($element, vec![rows, columns]),
                                tensor($element, vec![columns]),
                                tensor($element, vec![rows, columns]),
                                tensor($element, vec![rows, columns]),
                                Argument::Scalar {
                                    dtype: ElementType::U32,
                                },
                                Argument::Scalar {
                                    dtype: ElementType::F32,
                                },
                            ],
                            grid: [rows_u32, 1, 1],
                            max_dynamic_shared_mem_bytes: 0,
                        };
                        let prepared = kernel_cache()?
                            .prepare_for_device(&mut request, input.device())
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        if prepared.manifest.block != [block as u32, 1, 1] {
                            candle_core::bail!("add RMSNorm external block does not match legacy");
                        }
                        prepared
                            .prevalidate_launch(&request, input.device())
                            .map_err(|error| candle_core::Error::msg(error.to_string()))
                    },
                    |plan| {
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut builder = stream.launch_builder(function);
                                builder
                                    .arg(&src)
                                    .arg(&res)
                                    .arg(&wgt)
                                    .arg(&sum_dst)
                                    .arg(&norm_dst)
                                    .arg(&columns_u32)
                                    .arg(&eps);
                                builder.launch(config)?;
                                Ok(())
                            })
                        }
                        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        Ok(())
                    },
                )
            })?;
            drop(_sum_guard);
            drop(_norm_guard);
            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(residual_out, cuda.clone())),
                    shape.clone(),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(norm_out, cuda.clone())),
                    shape.clone(),
                )),
            ))
        }};
    }

    match input.dtype() {
        DType::F16 => launch!(half::f16, ElementType::F16),
        DType::BF16 => launch!(half::bf16, ElementType::Bf16),
        DType::F32 => launch!(f32, ElementType::F32),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires CUDA and separately built external add-RMSNorm packages"]
    fn dkr_add_rms_norm_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            // Exercises scalar 32, scalar 1024 and aligned vector8 sizes.
            for columns in [3, 16, 520, 1024] {
                let input = Tensor::ones((2, columns), dtype, &device)?;
                let residual = Tensor::ones((2, columns), dtype, &device)?;
                let weight = Tensor::ones(columns, dtype, &device)?;
                let (sum, norm) = cuda_add_rms_norm(&input, &residual, &weight, 1e-6)?;
                let cached = HOT_PLANS.with(|hot| hot.borrow().len());
                let (sum_again, norm_again) =
                    cuda_add_rms_norm(&input, &residual, &weight, 1e-6)?;
                assert_eq!(cached, HOT_PLANS.with(|hot| hot.borrow().len()));
                let expected = (&input + &residual)?;
                assert_eq!(
                    sum.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    expected.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
                );
                let oracle = candle_nn::ops::rms_norm(&expected.contiguous()?, &weight, 1e-6)?;
                let actual = norm.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let reference = oracle.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let tolerance = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
                for (got, want) in actual.iter().zip(reference.iter()) {
                    assert!((got - want).abs() <= tolerance, "{dtype:?}: {got} != {want}");
                }
                assert_eq!(
                    sum.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    sum_again.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
                );
                assert_eq!(
                    norm.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                    norm_again.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
                );
            }
        }
        Ok(())
    }
}
