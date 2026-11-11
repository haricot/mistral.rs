use crate::cuda::kernel_runtime::{kernel_cache, HotPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    Device, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const WARP: u32 = 32;
const MAX_D_STATE: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct ScanKey {
    batch: u32,
    seq_len: u32,
    heads: u32,
    head_dim: u32,
    d_state: u32,
    items: u32,
}

thread_local! {
    static HOT_PLANS: RefCell<HotPlanCache<ScanKey, PrevalidatedLaunchPlan>> =
        RefCell::new(HotPlanCache::new());
}

fn tensor(shape: Vec<usize>) -> Argument {
    Argument::Tensor {
        dtype: ElementType::F32,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn scalar(dtype: ElementType) -> Argument {
    Argument::Scalar { dtype }
}

fn prepare(key: ScanKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let (b, s, h, c, d) = (
        key.batch as usize, key.seq_len as usize, key.heads as usize,
        key.head_dim as usize, key.d_state as usize,
    );
    let grid_x = key.heads.checked_mul(key.head_dim)
        .ok_or_else(|| candle_core::Error::msg("SSM scan grid overflow"))?;
    let mut request = LaunchRequest {
        operation: format!("state_space.ssm_scan_f32.cf{}.v1", key.items),
        cc: 0,
        args: vec![
            tensor(vec![b, s, h, c]),
            tensor(vec![b, s, h]),
            tensor(vec![h]),
            tensor(vec![b, s, h, d]),
            tensor(vec![b, s, h, d]),
            tensor(vec![h]),
            tensor(vec![h]),
            tensor(vec![b, h, c, d]),
            tensor(vec![b, s, h, c]),
            scalar(ElementType::I32),
            scalar(ElementType::I32),
            scalar(ElementType::I32),
            scalar(ElementType::I32),
            scalar(ElementType::F32),
            scalar(ElementType::F32),
        ],
        grid: [grid_x, key.batch, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = kernel_cache()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [WARP, 1, 1] {
        candle_core::bail!("SSM external package must launch exactly one warp per block");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn selective_scan_cuda(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    dt_min: f32,
    dt_max: f32,
) -> Result<Tensor> {
    if !dt_min.is_finite() || !dt_max.is_finite() || dt_min > dt_max {
        candle_core::bail!("SSM scan requires finite ordered dt clamp bounds");
    }
    let x = x.contiguous()?;
    let dt = dt.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    let d = d.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let (batch, seq_len, heads, head_dim) = x.dims4()?;
    let (_, _, _, d_state) = b.dims4()?;
    if batch == 0 || seq_len == 0 || heads == 0 || head_dim == 0
        || d_state == 0 || d_state > MAX_D_STATE
        || dt.dims() != [batch, seq_len, heads]
        || b.dims() != [batch, seq_len, heads, d_state]
        || c.dims() != b.dims()
        || a.dims() != [heads]
        || d.dims() != [heads]
        || dt_bias.dims() != [heads]
        || state.dims() != [batch, heads, head_dim, d_state]
        || !state.is_contiguous()
    {
        candle_core::bail!("SSM scan tensor geometry or state layout mismatch");
    }
    for tensor in [&x, &dt, &a, &b, &c, &d, &dt_bias, &*state] {
        if tensor.dtype() != DType::F32 || !tensor.device().same_device(x.device()) {
            candle_core::bail!("SSM scan requires F32 tensors on the same CUDA device");
        }
        if tensor.elem_count() > i32::MAX as usize {
            candle_core::bail!("SSM scan index space exceeds signed 32-bit limit");
        }
    }
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    macro_rules! pointer {
        ($tensor:expr) => {{
            let (storage, layout) = $tensor.storage_and_layout();
            let Storage::Cuda(storage) = &*storage else {
                candle_core::bail!("SSM scan requires CUDA storage");
            };
            let values = storage.as_cuda_slice::<f32>()?;
            let (raw, guard) = values.device_ptr(&stream);
            let offset = layout.start_offset()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| candle_core::Error::msg("SSM tensor pointer overflow"))?;
            let address = raw.checked_add(offset as u64)
                .ok_or_else(|| candle_core::Error::msg("SSM tensor pointer overflow"))?;
            drop(guard);
            address
        }};
    }
    let x_ptr = pointer!(&x);
    let dt_ptr = pointer!(&dt);
    let a_ptr = pointer!(&a);
    let b_ptr = pointer!(&b);
    let c_ptr = pointer!(&c);
    let d_ptr = pointer!(&d);
    let dt_bias_ptr = pointer!(&dt_bias);
    let state_ptr = pointer!(&*state);

    let mut output = unsafe { dev.alloc::<f32>(x.elem_count())? };
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let output_ptr = output_ptr as u64;
    let rows = u32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let seq = u32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let h = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let c = u32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let d_state_u32 = u32::try_from(d_state).map_err(candle_core::Error::wrap)?;
    let items = ((d_state_u32 + WARP - 1) / WARP).next_power_of_two();
    let key = ScanKey {
        batch: rows, seq_len: seq, heads: h, head_dim: c,
        d_state: d_state_u32, items,
    };
    let device = Device::Cuda(dev.clone());
    HOT_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(
            key,
            &device,
            || prepare(key, &device),
            |plan| {
                let n_heads = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
                let head_dim = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
                let d_state = i32::try_from(d_state).map_err(candle_core::Error::wrap)?;
                let seq_len = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
                unsafe {
                    plan.with_prevalidated_launch(|function, config| {
                        let mut builder = stream.launch_builder(function);
                        builder
                            .arg(&x_ptr).arg(&dt_ptr).arg(&a_ptr).arg(&b_ptr).arg(&c_ptr)
                            .arg(&d_ptr).arg(&dt_bias_ptr).arg(&state_ptr).arg(&output_ptr)
                            .arg(&n_heads).arg(&head_dim).arg(&d_state).arg(&seq_len)
                            .arg(&dt_min).arg(&dt_max);
                        builder.launch(config)?;
                        Ok(())
                    })
                }
                .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
        )
    })?;
    drop(output_guard);
    Ok(Tensor::from((
        Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone())),
        (batch, seq_len, heads, head_dim),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA and external F32 SSM scan packages"]
    fn dkr_ssm_scan_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for d_state in [16, 32, 64, 128, 256] {
            let x = Tensor::ones((1, 2, 1, 1), DType::F32, &device)?;
            let dt = Tensor::zeros((1, 2, 1), DType::F32, &device)?;
            let a = Tensor::from_vec(vec![-1.0f32], 1, &device)?;
            let b = Tensor::ones((1, 2, 1, d_state), DType::F32, &device)?;
            let c = Tensor::ones((1, 2, 1, d_state), DType::F32, &device)?;
            let d = Tensor::zeros(1, DType::F32, &device)?;
            let bias = Tensor::zeros(1, DType::F32, &device)?;
            let mut state = Tensor::zeros((1, 1, 1, d_state), DType::F32, &device)?;
            let output = selective_scan_cuda(
                &x, &dt, &a, &b, &c, &d, &bias, &mut state, 0.0, 10.0,
            )?;
            let values = output.flatten_all()?.to_vec1::<f32>()?;
            let step = std::f32::consts::LN_2;
            for (got, want) in values.iter().zip([
                d_state as f32 * step, d_state as f32 * 1.5 * step,
            ]) {
                assert!((*got - want).abs() < 1e-2, "state={d_state}, {got} != {want}");
            }
        }
        Ok(())
    }
}
