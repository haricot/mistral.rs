use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{Device, Result};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const BLOCK: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct PadKey {
    input_rows: u32,
    output_rows: u32,
    width: u32,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<PadKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn tensor(rows: usize, width: usize) -> Argument {
    Argument::Tensor {
        dtype: ElementType::U32,
        shape: vec![rows, width],
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn prepare(key: PadKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let (i, o, w) = (key.input_rows as usize, key.output_rows as usize, key.width as usize);
    let elements = o.checked_mul(w)
        .ok_or_else(|| candle_core::Error::msg("decode input pad grid overflow"))?;
    let mut request = LaunchRequest {
        operation: "layout.pad_decode_u32.v1".into(),
        cc: 0,
        args: vec![
            tensor(i, w), tensor(o, w),
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
            Argument::Scalar { dtype: ElementType::I32 },
        ],
        grid: [u32::try_from(elements.div_ceil(BLOCK))
            .map_err(candle_core::Error::wrap)?, 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let kernel = kernel_cache()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if kernel.manifest.block != [BLOCK as u32, 1, 1] {
        candle_core::bail!("input pad external block mismatch");
    }
    kernel
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

pub(crate) fn with_pad_plan<T>(
    device: &Device, input_rows: usize, output_rows: usize, width: usize,
    launch: impl FnOnce(&PrevalidatedLaunchPlan) -> Result<T>,
) -> Result<T> {
    if input_rows == 0 || width == 0 || output_rows < input_rows {
        candle_core::bail!("invalid decode input padding dimensions");
    }
    let key = PadKey {
        input_rows: u32::try_from(input_rows).map_err(candle_core::Error::wrap)?,
        output_rows: u32::try_from(output_rows).map_err(candle_core::Error::wrap)?,
        width: u32::try_from(width).map_err(candle_core::Error::wrap)?,
    };
    THREAD_LAUNCH_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(key, device, || prepare(key, device), launch)
    })
}
