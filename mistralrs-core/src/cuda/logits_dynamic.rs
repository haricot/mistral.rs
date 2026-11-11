use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{Device, Result};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const BLOCK: u32 = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
enum LogitsStep {
    Copy,
    Penalties,
    Bias,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct LogitsPlanKey {
    step: LogitsStep,
    vocab: u32,
    entries: u32,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<LogitsPlanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

fn tensor(dtype: ElementType, length: usize) -> Argument {
    Argument::Tensor {
        dtype,
        shape: vec![length],
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn prepare(key: LogitsPlanKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let n = key.vocab as usize;
    let entries = key.entries as usize;
    let (operation, args, work) = match key.step {
        LogitsStep::Copy => (
            "tensor.copy.contiguous.f32.v1",
            vec![
                tensor(ElementType::F32, n),
                tensor(ElementType::F32, n),
                Argument::Scalar { dtype: ElementType::U32 },
            ],
            key.vocab,
        ),
        LogitsStep::Penalties => (
            "sampling.logits.penalties_sparse.f32.v1",
            vec![
                tensor(ElementType::F32, n),
                tensor(ElementType::U32, entries),
                tensor(ElementType::F32, entries),
                Argument::Scalar { dtype: ElementType::U32 },
                Argument::Scalar { dtype: ElementType::U32 },
                Argument::Scalar { dtype: ElementType::F32 },
                Argument::Scalar { dtype: ElementType::F32 },
                Argument::Scalar { dtype: ElementType::F32 },
            ],
            key.entries,
        ),
        LogitsStep::Bias => (
            "sampling.logits.bias_sparse.f32.v1",
            vec![
                tensor(ElementType::F32, n),
                tensor(ElementType::U32, entries),
                tensor(ElementType::F32, entries),
                Argument::Scalar { dtype: ElementType::U32 },
                Argument::Scalar { dtype: ElementType::U32 },
            ],
            key.entries,
        ),
    };
    let mut request = LaunchRequest {
        operation: operation.into(),
        cc: 0,
        args,
        grid: [work.div_ceil(BLOCK), 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = kernel_cache()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] {
        candle_core::bail!("sparse logits external block must be 256 threads");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

pub(crate) fn with_logits_plans(
    device: &Device,
    vocab: usize,
    entries: usize,
    is_bias: bool,
    launch_copy: impl FnOnce(&PrevalidatedLaunchPlan) -> Result<()>,
    launch_update: impl FnOnce(&PrevalidatedLaunchPlan) -> Result<()>,
) -> Result<()> {
    let vocab = u32::try_from(vocab).map_err(candle_core::Error::wrap)?;
    let entries = u32::try_from(entries).map_err(candle_core::Error::wrap)?;
    if vocab == 0 {
        candle_core::bail!("sparse logits require nonempty vocabulary");
    }
    HOT_PLANS.with(|hot| {
        let mut cache = hot.borrow_mut();
        let copy_key = LogitsPlanKey { step: LogitsStep::Copy, vocab, entries: 0 };
        cache.with_plan(copy_key, device, || prepare(copy_key, device), launch_copy)?;
        if entries != 0 {
            let key = LogitsPlanKey {
                step: if is_bias { LogitsStep::Bias } else { LogitsStep::Penalties },
                vocab,
                entries,
            };
            cache.with_plan(key, device, || prepare(key, device), launch_update)?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    #[ignore = "requires CUDA and the final external logits package build"]
    fn dkr_logits_penalties_and_bias_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.5], 4, &device)?;
        let ids = Tensor::from_vec(vec![0u32, 2], 2, &device)?;
        let counts = Tensor::from_vec(vec![2.0f32, 1.0], 2, &device)?;
        let output = crate::ops::cuda_apply_sparse_penalties_f32(
            &input, &ids, &counts, 0.1, 0.2, 1.1,
        )?;
        let expected = [0.6f32 / 1.1, -2.0, 2.7 / 1.1, 0.5];
        for (got, want) in output.to_vec1::<f32>()?.iter().zip(expected) {
            assert!((*got - want).abs() <= 1e-5, "{got} != {want}");
        }
        let ids = Tensor::from_vec(vec![0u32, 3], 2, &device)?;
        let biases = Tensor::from_vec(vec![0.25f32, -0.5], 2, &device)?;
        let biased = crate::ops::cuda_apply_sparse_logits_bias_f32(&input, &ids, &biases)?;
        assert_eq!(biased.to_vec1::<f32>()?, vec![1.25, -2.0, 3.0, 0.0]);
        let count = HOT_PLANS.with(|hot| hot.borrow().len());
        let repeated = crate::ops::cuda_apply_sparse_logits_bias_f32(&input, &ids, &biases)?;
        assert_eq!(biased.to_vec1::<f32>()?, repeated.to_vec1::<f32>()?);
        assert_eq!(count, HOT_PLANS.with(|hot| hot.borrow().len()));
        Ok(())
    }
}
