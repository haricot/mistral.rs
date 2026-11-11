use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    Device, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const WARP: u32 = 32;
const LARGE_BLOCK: u32 = 1024;

#[derive(Clone, Copy, Eq, PartialEq)]
struct RopeKey {
    dtype: ElementType,
    batch: u32,
    sequence: u32,
    heads: u32,
    columns: u32,
    half_rot: u32,
    block: u32,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<RopeKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
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

fn prepare(key: RopeKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let (b, s, h, c, half) = (
        key.batch as usize,
        key.sequence as usize,
        key.heads as usize,
        key.columns as usize,
        key.half_rot as usize,
    );
    let rows = key.batch.checked_mul(key.sequence)
        .and_then(|x| x.checked_mul(key.heads))
        .ok_or_else(|| candle_core::Error::msg("RMSNorm+RoPE launch grid overflow"))?;
    let mut request = LaunchRequest {
        operation: format!("rotary.rounded_rms_rope.b{}.v1", key.block),
        cc: 0,
        args: vec![
            tensor(key.dtype, vec![b, s, h, c]),
            tensor(key.dtype, vec![c]),
            tensor(key.dtype, vec![b * s, half]),
            tensor(key.dtype, vec![b * s, half]),
            tensor(key.dtype, vec![b, s, h, c]),
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::F32 },
        ],
        grid: [rows, 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = kernel_cache()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [key.block, 1, 1] {
        candle_core::bail!("RMSNorm+RoPE external block does not match prevalidated plan");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Called only after the established host shape/device checks.
/// Noncontiguous input continues to use the legacy CUDA stride-aware entry.
pub(crate) fn run(
    input: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    epsilon: f32,
    [batch, sequence, heads, columns]: [usize; 4],
    half_rot: usize,
) -> Result<Option<Tensor>> {
    if !input.is_contiguous() {
        return Ok(None);
    }
    let rows = batch.checked_mul(sequence).and_then(|n| n.checked_mul(heads))
        .ok_or_else(|| candle_core::Error::msg("RMSNorm+RoPE row overflow"))?;
    if rows == 0 || half_rot == 0 || half_rot > columns / 2
        || rows > u32::MAX as usize
        || [batch, sequence, heads, columns, half_rot]
            .iter().any(|&n| n > i32::MAX as usize)
    {
        candle_core::bail!("invalid contiguous RMSNorm+RoPE launch geometry");
    }
    let dtype = match input.dtype() {
        DType::F32 => ElementType::F32,
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        _ => return Ok(None),
    };
    let key = RopeKey {
        dtype,
        batch: u32::try_from(batch).map_err(candle_core::Error::wrap)?,
        sequence: u32::try_from(sequence).map_err(candle_core::Error::wrap)?,
        heads: u32::try_from(heads).map_err(candle_core::Error::wrap)?,
        columns: u32::try_from(columns).map_err(candle_core::Error::wrap)?,
        half_rot: u32::try_from(half_rot).map_err(candle_core::Error::wrap)?,
        block: if columns < LARGE_BLOCK as usize { WARP } else { LARGE_BLOCK },
    };

    // These are pure contiguous views, not copies. They make the typed ABI
    // input rank exactly match the manifest for 2D, 3D or 4D callers.
    let original_shape = input.shape().clone();
    let input = input.reshape((batch, sequence, heads, columns))?;
    let cos = cos.reshape((batch * sequence, half_rot))?;
    let sin = sin.reshape((batch * sequence, half_rot))?;
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let cuda_device = Device::Cuda(dev.clone());

    macro_rules! launch {
        ($type:ty) => {{
            let (input_storage, input_layout) = input.storage_and_layout();
            let (weight_storage, weight_layout) = weight.storage_and_layout();
            let (cos_storage, cos_layout) = cos.storage_and_layout();
            let (sin_storage, sin_layout) = sin.storage_and_layout();
            let Storage::Cuda(input_storage) = &*input_storage else {
                candle_core::bail!("RMSNorm+RoPE requires CUDA input")
            };
            let Storage::Cuda(weight_storage) = &*weight_storage else {
                candle_core::bail!("RMSNorm+RoPE requires CUDA weight")
            };
            let Storage::Cuda(cos_storage) = &*cos_storage else {
                candle_core::bail!("RMSNorm+RoPE requires CUDA cos")
            };
            let Storage::Cuda(sin_storage) = &*sin_storage else {
                candle_core::bail!("RMSNorm+RoPE requires CUDA sin")
            };
            let input_slice = input_storage.as_cuda_slice::<$type>()?;
            let weight_slice = weight_storage.as_cuda_slice::<$type>()?;
            let cos_slice = cos_storage.as_cuda_slice::<$type>()?;
            let sin_slice = sin_storage.as_cuda_slice::<$type>()?;
            let (input_ptr, _input_guard) = input_slice.device_ptr(&stream);
            let (weight_ptr, _weight_guard) = weight_slice.device_ptr(&stream);
            let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
            let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
            let ptr_offset = |address: u64, elements: usize| -> Result<u64> {
                let bytes = elements
                    .checked_mul(std::mem::size_of::<$type>())
                    .ok_or_else(|| candle_core::Error::msg("RMSNorm+RoPE offset overflow"))?;
                address.checked_add(bytes as u64)
                    .ok_or_else(|| candle_core::Error::msg("RMSNorm+RoPE address overflow"))
            };
            let input_addr = ptr_offset(input_ptr, input_layout.start_offset())?;
            let weight_addr = ptr_offset(weight_ptr, weight_layout.start_offset())?;
            let cos_addr = ptr_offset(cos_ptr, cos_layout.start_offset())?;
            let sin_addr = ptr_offset(sin_ptr, sin_layout.start_offset())?;
            let mut out = unsafe { dev.alloc::<$type>(input.elem_count())? };
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
            let out_addr = out_ptr as u64;
            let batch = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
            let sequence = i32::try_from(sequence).map_err(candle_core::Error::wrap)?;
            let heads = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
            let columns = i32::try_from(columns).map_err(candle_core::Error::wrap)?;
            let half_rot = i32::try_from(half_rot).map_err(candle_core::Error::wrap)?;
            THREAD_LAUNCH_PLANS.with(|cache| {
                cache.borrow_mut().with_plan(
                    key,
                    &cuda_device,
                    || prepare(key, &cuda_device),
                    |plan| {
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut builder = stream.launch_builder(function);
                                builder
                                    .arg(&input_addr).arg(&weight_addr)
                                    .arg(&cos_addr).arg(&sin_addr).arg(&out_addr)
                                    .arg(&batch).arg(&sequence).arg(&heads)
                                    .arg(&columns).arg(&half_rot).arg(&epsilon);
                                builder.launch(config)?;
                                Ok(())
                            })
                        }
                        .map_err(|error| candle_core::Error::msg(error.to_string()))
                    },
                )
            })?;
            drop(out_guard);
            Ok(Some(Tensor::from((
                Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone())),
                original_shape.clone(),
            ))))
        }};
    }

    match dtype {
        ElementType::F32 => launch!(f32),
        ElementType::F16 => launch!(half::f16),
        ElementType::Bf16 => launch!(half::bf16),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires external rounded RMSNorm+RoPE CUDA PTX and a GPU"]
    fn dkr_rounded_rms_rope_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for columns in [16, 1024] {
                let x = Tensor::ones((1, 2, 3, columns), dtype, &device)?;
                let weight = Tensor::ones(columns, dtype, &device)?;
                let cos = Tensor::ones((1, 2, columns / 2), dtype, &device)?;
                let sin = Tensor::zeros((1, 2, columns / 2), dtype, &device)?;
                let actual = super::super::rounded_rope::try_rms_norm_partial_rotary_neox(
                    &x, &weight, &cos, &sin, 1e-6,
                )?.expect("eligible contiguous RMSNorm+RoPE tensor");
                let vals = actual.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for v in vals {
                    assert!((v - 1.0).abs() < 0.02, "unexpected {v}");
                }
            }
        }
        Ok(())
    }
}
