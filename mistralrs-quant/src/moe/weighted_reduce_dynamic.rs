use crate::dkr_cuda::{catalog, dtype, tensor, HotPlanCache};
use candle_core::{
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const OPERATION: &str = "moe.weighted_reduce.v1";

#[derive(Clone, Copy, Eq, PartialEq)]
struct PlanKey {
    input: DType,
    output: DType,
    tokens: usize,
    topk: usize,
    hidden: usize,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<PlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

pub fn moe_weighted_reduce(input: &Tensor, weights: &Tensor, output_dtype: DType) -> Result<Tensor> {
    let (tokens, topk, hidden) = input.dims3()?;
    if weights.dims() != [tokens, topk]
        || weights.dtype() != DType::F32
        || !input.device().same_device(weights.device())
    {
        candle_core::bail!(
            "MoE weighted reduction requires F32 [tokens, topk] weights on the input device"
        );
    }
    let input_dtype = dtype(input.dtype())?;
    let output_element = dtype(output_dtype)?;
    if tokens == 0 || hidden == 0 || topk == 0 {
        return Tensor::zeros((tokens, hidden), output_dtype, input.device());
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let device = input.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weights_storage)) =
        (&*input_storage, &*weights_storage)
    else {
        candle_core::bail!("MoE weighted reduction requires CUDA tensors")
    };
    let tokens_u64 = u64::try_from(tokens).map_err(candle_core::Error::wrap)?;
    let topk_u64 = u64::try_from(topk).map_err(candle_core::Error::wrap)?;
    let hidden_u64 = u64::try_from(hidden).map_err(candle_core::Error::wrap)?;
    let elements = tokens
        .checked_mul(hidden)
        .ok_or_else(|| candle_core::Error::msg("weighted reduction element count overflow"))?;
    let key = PlanKey {
        input: input.dtype(),
        output: output_dtype,
        tokens,
        topk,
        hidden,
    };

    HOT_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(
            key,
            input.device(),
            || {
                let mut request = LaunchRequest {
                    operation: OPERATION.into(),
                    cc: 0,
                    args: vec![
                        tensor(input_dtype, vec![tokens, topk, hidden]),
                        tensor(ElementType::F32, vec![tokens, topk]),
                        tensor(output_element, vec![tokens, hidden]),
                        Argument::Scalar {
                            dtype: ElementType::U64,
                        },
                        Argument::Scalar {
                            dtype: ElementType::U64,
                        },
                        Argument::Scalar {
                            dtype: ElementType::U64,
                        },
                    ],
                    grid: [1, 1, 1],
                    max_dynamic_shared_mem_bytes: 0,
                };
                let prepared = catalog()?
                    .prepare_for_device(&mut request, input.device())
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                if prepared.manifest.block[1..] != [1, 1] {
                    candle_core::bail!("weighted reduction requires a one-dimensional CUDA block")
                }
                let blocks = elements.div_ceil(prepared.manifest.block[0] as usize);
                request.grid[0] = u32::try_from(blocks).map_err(candle_core::Error::wrap)?;
                prepared
                    .prevalidate_launch(&request, input.device())
                    .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
            |plan| {
                macro_rules! launch {
                    ($input_ty:ty, $output_ty:ty) => {{
                        let (input_base, _input_guard) = input_storage
                            .as_cuda_slice::<$input_ty>()?
                            .device_ptr(&stream);
                        let (weights_base, _weights_guard) = weights_storage
                            .as_cuda_slice::<f32>()?
                            .device_ptr(&stream);
                        let input_ptr = input_base
                            + (input_layout.start_offset() * std::mem::size_of::<$input_ty>()) as u64;
                        let weights_ptr = weights_base + (weights_layout.start_offset() * 4) as u64;
                        let mut output = unsafe { device.alloc::<$output_ty>(elements)? };
                        {
                            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder
                                        .arg(&input_ptr)
                                        .arg(&weights_ptr)
                                        .arg(&output_ptr)
                                        .arg(&tokens_u64)
                                        .arg(&hidden_u64)
                                        .arg(&topk_u64);
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        let output = CudaStorage::wrap_cuda_slice(output, device.clone());
                        Ok(Tensor::from((Storage::Cuda(output), (tokens, hidden))))
                    }};
                }
                match (input.dtype(), output_dtype) {
                    (DType::F16, DType::F16) => launch!(half::f16, half::f16),
                    (DType::F16, DType::BF16) => launch!(half::f16, half::bf16),
                    (DType::F16, DType::F32) => launch!(half::f16, f32),
                    (DType::BF16, DType::F16) => launch!(half::bf16, half::f16),
                    (DType::BF16, DType::BF16) => launch!(half::bf16, half::bf16),
                    (DType::BF16, DType::F32) => launch!(half::bf16, f32),
                    (DType::F32, DType::F16) => launch!(f32, half::f16),
                    (DType::F32, DType::BF16) => launch!(f32, half::bf16),
                    (DType::F32, DType::F32) => launch!(f32, f32),
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
    #[ignore = "requires external PTX and a CUDA device"]
    fn dkr_weighted_reduce_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values: Vec<f32> = (0..42)
            .map(|i| ((i * 17 % 31) as i32 - 15) as f32 / 8.0)
            .collect();
        let route_weights = [0.25f32, 0.25, 0.5, 0.5, 0.25, 0.25];
        let expected: Vec<f32> = (0..2)
            .flat_map(|token| {
                let values = &values;
                (0..7).map(move |column| {
                    (0..3)
                        .map(|route| {
                            let row = token * 3 + route;
                            values[row * 7 + column] * route_weights[row]
                        })
                        .sum::<f32>()
                })
            })
            .collect();
        for input_dtype in [DType::F16, DType::BF16, DType::F32] {
            for output_dtype in [DType::F16, DType::BF16, DType::F32] {
                let input = Tensor::from_vec(values.clone(), (2, 3, 7), &device)?
                    .to_dtype(input_dtype)?;
                let weights = Tensor::from_vec(route_weights.to_vec(), (2, 3), &device)?;
                let result = moe_weighted_reduce(&input, &weights, output_dtype)?;
                let cached = HOT_PLANS.with(|hot| hot.borrow().len());
                let repeated = moe_weighted_reduce(&input, &weights, output_dtype)?;
                assert_eq!(cached, HOT_PLANS.with(|hot| hot.borrow().len()));
                assert_eq!(result.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, expected);
                assert_eq!(
                    result.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    repeated.to_dtype(DType::F32)?.to_vec2::<f32>()?
                );
            }
        }
        Ok(())
    }
}
