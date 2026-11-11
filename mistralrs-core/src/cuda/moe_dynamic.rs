use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{DType, Result, Tensor};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const OPERATION: &str = "moe.grouped_gemm.v1";
const MAX_ROWS: usize = u32::MAX as usize;

#[derive(Clone, Copy, Eq, PartialEq)]
struct MoePlanKey {
    dtype: ElementType,
    input_m: usize,
    size_k: usize,
    num_experts: usize,
    size_n: usize,
    topk: usize,
    has_gating: bool,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<MoePlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

macro_rules! check {
    ($condition:expr, $message:literal) => {
        if !$condition {
            candle_core::bail!($message)
        }
    };
}

fn tensor_arg(dtype: ElementType, shape: Vec<usize>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn scalar_arg() -> Argument {
    Argument::Scalar {
        dtype: ElementType::U32,
    }
}

fn validate_tensor(tensor: &Tensor, dtype: DType, device: &candle_core::Device) -> Result<()> {
    check!(tensor.dtype() == dtype, "MoE tensor dtype mismatch");
    check!(
        tensor.device().same_device(device),
        "MoE CUDA context mismatch"
    );
    let (_, layout) = tensor.storage_and_layout();
    check!(
        layout.is_contiguous() && layout.start_offset() == 0,
        "DKR MoE requires contiguous, zero-offset tensors"
    );
    Ok(())
}

fn forward<T>(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor>
where
    T: candle_core::cuda_backend::CudaDType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr
        + candle_core::cuda_backend::cudarc::driver::ValidAsZeroBits,
{
    use candle_core::{cuda_backend::cudarc::driver::PushKernelArg, Storage};

    let device = input.device();
    let dtype = input.dtype();
    validate_tensor(input, dtype, device)?;
    validate_tensor(weights, dtype, device)?;
    validate_tensor(sorted_token_ids, DType::U32, device)?;
    validate_tensor(experts_ids, DType::U32, device)?;
    if let Some(gating) = topk_weights {
        validate_tensor(gating, DType::F32, device)?;
    }

    let (input_m, size_k) = input.dims2()?;
    let (num_experts, size_n, weight_k) = weights.dims3()?;
    check!(size_k == weight_k, "MoE input and weight shape mismatch");
    check!(
        topk > 0 && num_experts > 0,
        "invalid MoE routing dimensions"
    );
    let has_gating = topk_weights.is_some();
    let size_m = if has_gating {
        input_m
    } else {
        input_m
            .checked_mul(topk)
            .ok_or_else(|| candle_core::Error::Msg("MoE row count overflow".into()))?
    };
    check!(
        size_m > 0 && size_n > 0 && size_k > 0 && size_m <= MAX_ROWS,
        "invalid MoE shape"
    );
    check!(
        sorted_token_ids.dims1()? == size_m && experts_ids.dims1()? == size_m,
        "MoE routing index length mismatch"
    );
    if let Some(gating) = topk_weights {
        check!(gating.dims1()? == size_m, "MoE gating length mismatch");
    }

    let to_u32 = |value: usize| {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("MoE dimension exceeds u32".into()))
    };
    let num_experts_u32 = to_u32(num_experts)?;
    let topk_u32 = to_u32(topk)?;
    let m_u32 = to_u32(size_m)?;
    let n_u32 = to_u32(size_n)?;
    let k_u32 = to_u32(size_k)?;
    let gating_u32: u32 = has_gating.into();
    let output_count = size_m
        .checked_mul(size_n)
        .ok_or_else(|| candle_core::Error::Msg("MoE output size overflow".into()))?;
    let output_count_u32 = to_u32(output_count)?;
    let element_type = match dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        _ => candle_core::bail!("DKR MoE currently supports F16/BF16"),
    };
    let dev = device.as_cuda_device()?;
    let plan_key = MoePlanKey {
        dtype: element_type,
        input_m,
        size_k,
        num_experts,
        size_n,
        topk,
        has_gating,
    };

    let (input_storage, _) = input.storage_and_layout();
    let (weight_storage, _) = weights.storage_and_layout();
    let (sorted_storage, _) = sorted_token_ids.storage_and_layout();
    let (experts_storage, _) = experts_ids.storage_and_layout();
    let gating_storage = topk_weights.as_ref().map(Tensor::storage_and_layout);
    let input_slice = match &*input_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("MoE input must be CUDA"),
    };
    let weight_slice = match &*weight_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("MoE weights must be CUDA"),
    };
    let sorted_slice = match &*sorted_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("MoE routed tokens must be CUDA"),
    };
    let experts_slice = match &*experts_storage {
        Storage::Cuda(storage) => storage.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("MoE expert indices must be CUDA"),
    };
    let gating_slice = if let Some((storage, _)) = &gating_storage {
        match &**storage {
            Storage::Cuda(storage) => Some(storage.as_cuda_slice::<f32>()?),
            _ => candle_core::bail!("MoE gating weights must be CUDA"),
        }
    } else {
        None
    };
    let scratch = if has_gating {
        None
    } else {
        Some(dev.alloc_zeros::<f32>(size_m)?)
    };
    let mut output = dev.alloc_zeros::<T>(output_count)?;

    HOT_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(
            plan_key,
            device,
            || {
                let mut request = LaunchRequest {
                    operation: OPERATION.into(),
                    cc: 0,
                    args: vec![
                        tensor_arg(element_type, vec![input_m, size_k]),
                        tensor_arg(element_type, vec![num_experts, size_n, size_k]),
                        tensor_arg(ElementType::U32, vec![size_m]),
                        tensor_arg(ElementType::U32, vec![size_m]),
                        tensor_arg(ElementType::F32, vec![size_m]),
                        tensor_arg(element_type, vec![size_m, size_n]),
                        scalar_arg(),
                        scalar_arg(),
                        scalar_arg(),
                        scalar_arg(),
                        scalar_arg(),
                        scalar_arg(),
                    ],
                    grid: [1, 1, 1],
                    max_dynamic_shared_mem_bytes: 0,
                };
                let prepared = kernel_cache()?
                    .prepare_for_device(&mut request, device)
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                check!(
                    prepared.manifest.block[1..] == [1, 1],
                    "DKR MoE requires a one-dimensional launch block"
                );
                request.grid = [
                    output_count_u32.div_ceil(prepared.manifest.block[0]),
                    1,
                    1,
                ];
                prepared
                    .prevalidate_launch(&request, device)
                    .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
            |plan| {
                unsafe {
                    plan.with_prevalidated_launch(|function, config| {
                        let stream = dev.cuda_stream();
                        let mut builder = stream.launch_builder(function);
                        builder
                            .arg(input_slice)
                            .arg(weight_slice)
                            .arg(sorted_slice)
                            .arg(experts_slice);
                        if let Some(slice) = gating_slice {
                            builder.arg(slice);
                        } else {
                            builder.arg(scratch.as_ref().expect("missing dummy gating buffer"));
                        }
                        builder
                            .arg(&mut output)
                            .arg(&num_experts_u32)
                            .arg(&topk_u32)
                            .arg(&m_u32)
                            .arg(&n_u32)
                            .arg(&k_u32)
                            .arg(&gating_u32);
                        builder.launch(config)?;
                        Ok(())
                    })
                }
                .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
        )
    })?;

    let output = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from((Storage::Cuda(output), (size_m, size_n))))
}

pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    _is_prefill: bool,
) -> Result<Tensor> {
    use half::{bf16, f16};

    match input.dtype() {
        DType::F16 => forward::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
        ),
        DType::BF16 => forward::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
        ),
        _ => candle_core::bail!("DKR MoE currently supports F16/BF16"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires CUDA and separately built external MoE PTX packages"]
    fn dkr_moe_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            let input = Tensor::ones((2, 16), dtype, &device)?;
            let weights = Tensor::ones((2, 3, 16), dtype, &device)?;
            let sorted = Tensor::new(&[1u32, 0], &device)?;
            let experts = Tensor::new(&[1u32, 0], &device)?;
            let gating = Some(Tensor::new(&[2.0f32, 0.5], &device)?);
            let output = moe_gemm(&input, &weights, &gating, &sorted, &experts, 1, true)?;
            let values = output
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            for (row, expected) in values.iter().zip([32.0f32, 8.0]) {
                for &value in row {
                    assert!((value - expected).abs() < 0.5, "{dtype:?}: {value}");
                }
            }
            let prepared_before = kernel_cache()?
                .prepared_count()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            let plans_before = HOT_PLANS.with(|hot| hot.borrow().len());
            let repeated = moe_gemm(&input, &weights, &gating, &sorted, &experts, 1, true)?;
            let prepared_after = kernel_cache()?
                .prepared_count()
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            assert_eq!(prepared_before, prepared_after);
            assert_eq!(plans_before, HOT_PLANS.with(|hot| hot.borrow().len()));
            let repeated = repeated
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            assert_eq!(values, repeated);

            let sorted = Tensor::new(&[0u32, 1, 2, 3], &device)?;
            let experts = Tensor::new(&[0u32, 1, 0, 1], &device)?;
            let output = moe_gemm(&input, &weights, &None, &sorted, &experts, 2, false)?;
            let values = output
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?;
            for row in values {
                for value in row {
                    assert!((value - 16.0).abs() < 0.5, "{dtype:?}: {value}");
                }
            }
        }
        Ok(())
    }
}
