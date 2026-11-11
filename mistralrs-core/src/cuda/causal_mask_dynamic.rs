use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CpuStorage, CudaStorage, Device, InplaceOp1, Layout, Result, Tensor,
};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const OPERATION: &str = "attention.causal_mask_f32.inplace.v1";
const BLOCK: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct MaskPlanKey {
    batch_heads: i32,
    q_len: i32,
    kv_len: i32,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<MaskPlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

struct CausalMaskF32 {
    q_offset: usize,
    prefix_len: usize,
}

impl InplaceOp1 for CausalMaskF32 {
    fn name(&self) -> &'static str {
        "causal-mask-f32"
    }

    fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
        candle_core::bail!("causal-mask-f32 requires CUDA storage")
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        let (batch, queries, columns) = layout.shape().dims3()?;
        let batch_heads = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
        let q_len = i32::try_from(queries).map_err(candle_core::Error::wrap)?;
        let kv_len = i32::try_from(columns).map_err(candle_core::Error::wrap)?;
        let q_offset = i32::try_from(self.q_offset).map_err(candle_core::Error::wrap)?;
        let prefix_len = i32::try_from(self.prefix_len).map_err(candle_core::Error::wrap)?;
        if !layout.is_contiguous() {
            candle_core::bail!("causal-mask-f32 requires contiguous scores");
        }
        if batch == 0 || queries == 0 || columns == 0 {
            return Ok(());
        }
        let elements = batch
            .checked_mul(queries)
            .and_then(|total| total.checked_mul(columns))
            .ok_or_else(|| candle_core::Error::msg("causal mask element count overflow"))?;
        // The original kernel indexes with signed int32; never launch an
        // overflowing index space on its external equivalent.
        if elements > i32::MAX as usize {
            candle_core::bail!("causal mask F32 index count exceeds i32");
        }
        let grids = u32::try_from(elements.div_ceil(BLOCK))
            .map_err(|_| candle_core::Error::msg("causal mask launch grid exceeds u32"))?;
        let device = Device::Cuda(storage.device().clone());
        let stream = storage.device().cuda_stream();
        let scores = storage.as_cuda_slice_mut::<f32>()?;
        let (pointer, _guard) = scores.device_ptr_mut(&stream);
        let bytes = u64::try_from(layout.start_offset())
            .ok()
            .and_then(|offset| offset.checked_mul(4))
            .ok_or_else(|| candle_core::Error::msg("causal mask CUDA offset overflow"))?;
        let pointer = pointer
            .checked_add(bytes)
            .ok_or_else(|| candle_core::Error::msg("causal mask CUDA pointer overflow"))?;
        let key = MaskPlanKey {
            batch_heads,
            q_len,
            kv_len,
        };
        HOT_PLANS.with(|hot| {
            hot.borrow_mut().with_plan(
                key,
                &device,
                || {
                    let mut request = LaunchRequest {
                        operation: OPERATION.into(),
                        cc: 0,
                        args: vec![
                            Argument::Tensor {
                                dtype: ElementType::F32,
                                shape: vec![batch, queries, columns],
                                contiguous: true,
                                start_offset: 0,
                                representation: None,
                            },
                            Argument::Scalar {
                                dtype: ElementType::I32,
                            },
                            Argument::Scalar {
                                dtype: ElementType::I32,
                            },
                            Argument::Scalar {
                                dtype: ElementType::I32,
                            },
                            Argument::Scalar {
                                dtype: ElementType::I32,
                            },
                            Argument::Scalar {
                                dtype: ElementType::I32,
                            },
                        ],
                        grid: [grids, 1, 1],
                        max_dynamic_shared_mem_bytes: 0,
                    };
                    let prepared = kernel_cache()?
                        .prepare_for_device(&mut request, &device)
                        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                    if prepared.manifest.block != [BLOCK as u32, 1, 1] {
                        candle_core::bail!("causal mask external block mismatch");
                    }
                    prepared
                        .prevalidate_launch(&request, &device)
                        .map_err(|error| candle_core::Error::msg(error.to_string()))
                },
                |plan| {
                    unsafe {
                        plan.with_prevalidated_launch(|function, config| {
                            let mut builder = stream.launch_builder(function);
                            builder
                                .arg(&pointer)
                                .arg(&batch_heads)
                                .arg(&q_len)
                                .arg(&kv_len)
                                .arg(&q_offset)
                                .arg(&prefix_len);
                            builder.launch(config)?;
                            Ok(())
                        })
                    }
                    .map_err(|error| candle_core::Error::msg(error.to_string()))
                },
            )
        })
    }
}

pub(crate) fn cuda_apply_causal_mask_f32(
    scores: &Tensor,
    q_offset: usize,
    prefix_len: usize,
) -> Result<()> {
    scores.inplace_op1(&CausalMaskF32 {
        q_offset,
        prefix_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    #[ignore = "requires CUDA and externally built causal-mask PTX"]
    fn dkr_causal_mask_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for (q_offset, prefix_len) in [(0, 0), (1, 2)] {
            let scores = Tensor::zeros((2, 3, 5), DType::F32, &device)?;
            cuda_apply_causal_mask_f32(&scores, q_offset, prefix_len)?;
            let cached = HOT_PLANS.with(|hot| hot.borrow().len());
            let repeated = Tensor::zeros((2, 3, 5), DType::F32, &device)?;
            cuda_apply_causal_mask_f32(&repeated, q_offset, prefix_len)?;
            assert_eq!(cached, HOT_PLANS.with(|hot| hot.borrow().len()));
            let actual = scores.flatten_all()?.to_vec1::<f32>()?;
            let repeated_values = repeated.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(actual, repeated_values);
            for (index, value) in actual.iter().enumerate() {
                let q = (index / 5) % 3;
                let kv = index % 5;
                let expected = if kv > prefix_len + q_offset + q {
                    f32::NEG_INFINITY
                } else {
                    0.0
                };
                assert_eq!(*value, expected);
            }
        }
        Ok(())
    }
}
