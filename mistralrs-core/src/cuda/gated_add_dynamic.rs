use crate::cuda::kernel_runtime::{kernel_cache, LaunchPlanCache};
use candle_core::{
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};
use mistralrs_kernel_runtime::universal_cuda::PrevalidatedLaunchPlan;
use std::cell::RefCell;

const OPERATION_FLAT: &str = "tensor.sigmoid_gated_add.flat.v1";
const OPERATION_ROWS: &str = "tensor.sigmoid_gated_add.rows.v1";
const ROW_TILE_MIN_COLUMNS: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct GatedAddPlanKey {
    dtype: ElementType,
    elements: usize,
    columns: usize,
    tiled_rows: bool,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<GatedAddPlanKey, PrevalidatedLaunchPlan>> =
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

pub(crate) fn try_sigmoid_gated_add(
    residual: &Tensor,
    value: &Tensor,
    gate: &Tensor,
) -> Result<Option<Tensor>> {
    if !residual.device().is_cuda()
        || residual.rank() == 0
        || residual.shape() != value.shape()
        || residual.dtype() != value.dtype()
        || residual.dtype() != gate.dtype()
        || !residual.device().same_device(value.device())
        || !residual.device().same_device(gate.device())
        || !residual.is_contiguous()
        || !value.is_contiguous()
        || !gate.is_contiguous()
        || !matches!(residual.dtype(), DType::BF16 | DType::F16 | DType::F32)
    {
        return Ok(None);
    }
    let columns = residual.dim(candle_core::D::Minus1)?;
    let elements = residual.elem_count();
    if columns == 0
        || elements == 0
        || gate.rank() != residual.rank()
        || gate.dims().last() != Some(&1)
        || gate.dims()[..gate.rank() - 1] != residual.dims()[..residual.rank() - 1]
    {
        return Ok(None);
    }

    let dtype = match residual.dtype() {
        DType::BF16 => ElementType::Bf16,
        DType::F16 => ElementType::F16,
        DType::F32 => ElementType::F32,
        _ => unreachable!(),
    };
    let tiled_rows = residual.dtype() == DType::F32 && columns >= ROW_TILE_MIN_COLUMNS;
    let key = GatedAddPlanKey {
        dtype,
        elements,
        columns,
        tiled_rows,
    };
    let rows = elements / columns;
    let elements_u64 = u64::try_from(elements)
        .map_err(|_| candle_core::Error::msg("gated add element count exceeds u64"))?;
    let columns_u64 = u64::try_from(columns)
        .map_err(|_| candle_core::Error::msg("gated add column count exceeds u64"))?;

    let device = residual.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (
        Storage::Cuda(residual_storage),
        Storage::Cuda(value_storage),
        Storage::Cuda(gate_storage),
    ) = (&*residual_storage, &*value_storage, &*gate_storage)
    else {
        unreachable!()
    };

    THREAD_LAUNCH_PLANS.with(|hot| {
        hot.borrow_mut().with_plan(
            key,
            residual.device(),
            || {
                let mut request = LaunchRequest {
                    operation: if tiled_rows {
                        OPERATION_ROWS.into()
                    } else {
                        OPERATION_FLAT.into()
                    },
                    cc: 0,
                    args: vec![
                        tensor(dtype, vec![rows, columns]),
                        tensor(dtype, vec![rows, columns]),
                        tensor(dtype, vec![rows]),
                        tensor(dtype, vec![rows, columns]),
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
                let prepared = kernel_cache()?
                    .prepare_for_device(&mut request, residual.device())
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                if prepared.manifest.block[1..] != [1, 1] {
                    candle_core::bail!("gated add requires a one-dimensional launch block")
                }
                let block = prepared.manifest.block[0] as usize;
                let blocks = if tiled_rows {
                    rows.checked_mul(columns.div_ceil(block))
                        .ok_or_else(|| candle_core::Error::msg("gated add grid overflow"))?
                } else {
                    elements.div_ceil(block)
                };
                request.grid[0] = u32::try_from(blocks)
                    .map_err(|_| candle_core::Error::msg("gated add grid exceeds u32"))?;
                prepared
                    .prevalidate_launch(&request, residual.device())
                    .map_err(|error| candle_core::Error::msg(error.to_string()))
            },
            |plan| {
                macro_rules! launch {
                    ($type:ty) => {{
                        let (r, _rg) = residual_storage
                            .as_cuda_slice::<$type>()?
                            .device_ptr(&stream);
                        let (v, _vg) = value_storage.as_cuda_slice::<$type>()?.device_ptr(&stream);
                        let (g, _gg) = gate_storage.as_cuda_slice::<$type>()?.device_ptr(&stream);
                        let mut output = unsafe { device.alloc::<$type>(elements)? };
                        {
                            let (o, _og) = output.device_ptr_mut(&stream);
                            let bytes = std::mem::size_of::<$type>() as u64;
                            let r_ptr = r + bytes * residual_layout.start_offset() as u64;
                            let v_ptr = v + bytes * value_layout.start_offset() as u64;
                            let g_ptr = g + bytes * gate_layout.start_offset() as u64;
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder
                                        .arg(&r_ptr)
                                        .arg(&v_ptr)
                                        .arg(&g_ptr)
                                        .arg(&o)
                                        .arg(&elements_u64)
                                        .arg(&columns_u64);
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        Ok(Some(Tensor::from((
                            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                            residual.shape().clone(),
                        ))))
                    }};
                }
                match residual.dtype() {
                    DType::F32 => launch!(f32),
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
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
    #[ignore = "requires CUDA and separately built external gated-add packages"]
    fn dkr_gated_add_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for columns in [3, 513] {
                let residual = Tensor::ones((2, columns), dtype, &device)?;
                let value = Tensor::ones((2, columns), dtype, &device)?;
                let gate = Tensor::zeros((2, 1), dtype, &device)?;
                let output = try_sigmoid_gated_add(&residual, &value, &gate)?
                    .expect("eligible gated add must use DKR");
                let prepared_before = THREAD_LAUNCH_PLANS.with(|hot| hot.borrow().len());
                let repeated = try_sigmoid_gated_add(&residual, &value, &gate)?
                    .expect("eligible gated add must use DKR");
                assert_eq!(prepared_before, THREAD_LAUNCH_PLANS.with(|hot| hot.borrow().len()));
                let expected = &residual + value.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?;
                assert_eq!(
                    output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    expected?.to_dtype(DType::F32)?.to_vec2::<f32>()?
                );
                assert_eq!(
                    output.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                    repeated.to_dtype(DType::F32)?.to_vec2::<f32>()?
                );
            }
        }
        Ok(())
    }
}
