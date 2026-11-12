//! DKR-R1.21/S1: direct Pascal SIMT FP8-blockwise and MXFP4 decode GEMV.
//! No intermediate FP8/FP4 dequantized tensor. A static FFI fallback remains
//! for every unsupported shape/layout and the normal production prefill route.

use crate::{
    dkr_cuda::{catalog, LaunchPlanCache},
    utils::slice_ptr_on_stream,
};
use candle_core::{
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const BLOCK: usize = 128;
const MAX_BATCH: usize = 8;

#[derive(Clone, Copy, Eq, PartialEq)]
enum Format {
    Fp8Blockwise,
    Mxfp4,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    fmt: Format,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
    block_y: usize,
    block_x: usize,
    has_bias: bool,
}

thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn tensor(dtype: ElementType, dims: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype,
        shape: dims,
        contiguous: true,
        start_offset: 0,
        representation: repr.map(str::to_owned),
    }
}

fn prepare(key: Key, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let dtype = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        other => candle_core::bail!("DKR quantized GEMV dtype unsupported: {other:?}"),
    };
    let (operation, weight_dtype, weight_shape, weight_repr, scale_dtype, scale_repr) =
        match key.fmt {
            Format::Fp8Blockwise => (
                "quant.fp8.blockwise.gemv.b128.v1",
                ElementType::Fp8E4m3,
                vec![key.n, key.k],
                "fp8_e4m3_blockwise_weight.v1",
                ElementType::F32,
                "fp8_blockwise_f32.v1",
            ),
            Format::Mxfp4 => (
                if key.has_bias {
                    "quant.mxfp4.gemv.bias.b128.v1"
                } else {
                    "quant.mxfp4.gemv.plain.b128.v1"
                },
                ElementType::U8,
                vec![key.n, key.k / 2],
                "mxfp4_e2m1_group32.v1",
                ElementType::U8,
                "mxfp4_e8m0_group32.v1",
            ),
        };
    let mut args = vec![
        tensor(dtype, vec![key.m, key.k], None),
        tensor(weight_dtype, weight_shape, Some(weight_repr)),
        tensor(
            scale_dtype, vec![key.scale_rows, key.scale_cols], Some(scale_repr),
        ),
    ];
    if key.has_bias {
        args.push(tensor(dtype, vec![key.n], None));
    }
    args.push(tensor(dtype, vec![key.m, key.n], None));
    args.extend((0..if key.fmt == Format::Fp8Blockwise { 6 } else { 3 })
        .map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut request = LaunchRequest {
        operation: operation.to_owned(),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.n).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.m).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK as u32, 1, 1] {
        candle_core::bail!("unexpected quantized GEMV block geometry");
    }
    prepared.prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

fn try_gemv(
    fmt: Format,
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    bias: Option<&Tensor>,
    block_y: usize,
    block_x: usize,
) -> Result<Option<Tensor>> {
    if !matches!(input.dtype(), DType::F16 | DType::BF16)
        || input.rank() != 2 || weight.rank() != 2 || scales.rank() != 2
        || !input.is_contiguous() || !weight.is_contiguous() || !scales.is_contiguous()
        || !input.device().is_cuda()
        || !input.device().same_device(weight.device())
        || !input.device().same_device(scales.device())
        || bias.is_some_and(|b| {
            b.dtype() != input.dtype() || b.rank() != 1 || !b.is_contiguous()
                || !input.device().same_device(b.device())
        })
    {
        return Ok(None);
    }
    let (m, k) = input.dims2()?;
    let (n, weight_width) = weight.dims2()?;
    if m == 0 || m > MAX_BATCH || k == 0 || n == 0
        || block_x == 0 || block_y == 0
        || [m, n, k, block_x, block_y].iter().any(|&x| x > u32::MAX as usize)
        || bias.is_some_and(|b| b.dims() != [n])
    {
        return Ok(None);
    }
    let (scale_rows, scale_cols) = match fmt {
        Format::Fp8Blockwise => {
            if weight.dtype() != DType::F8E4M3
                || scales.dtype() != DType::F32
                || weight_width != k || bias.is_some()
            {
                return Ok(None);
            }
            (n.div_ceil(block_y), k.div_ceil(block_x))
        }
        Format::Mxfp4 => {
            if weight.dtype() != DType::U8
                || scales.dtype() != DType::U8
                || k % 32 != 0 || weight_width != k / 2
            {
                return Ok(None);
            }
            (n, k / 32)
        }
    };
    if scales.dims() != [scale_rows, scale_cols]
        || scale_cols > u32::MAX as usize
        || m.checked_mul(n).is_none()
    {
        return Ok(None);
    }

    let key = Key {
        fmt, dtype: input.dtype(), m, n, k, scale_rows, scale_cols,
        block_y, block_x, has_bias: bias.is_some(),
    };
    let cuda = input.device().as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let mi = u32::try_from(m).map_err(candle_core::Error::wrap)?;
    let ni = u32::try_from(n).map_err(candle_core::Error::wrap)?;
    let ki = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    let sy = u32::try_from(block_y).map_err(candle_core::Error::wrap)?;
    let sx = u32::try_from(block_x).map_err(candle_core::Error::wrap)?;
    let ss = u32::try_from(scale_cols).map_err(candle_core::Error::wrap)?;
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ss_storage, sl) = scales.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss_storage)) =
        (&*xs, &*ws, &*ss_storage)
    else {
        candle_core::bail!("DKR quantized GEMV requires CUDA storages");
    };
    let bias_storage = bias.map(|b| b.storage_and_layout());
    HOT.with(|plans| {
        plans.borrow_mut().with_plan(
            key,
            input.device(),
            || prepare(key, input.device()),
            |plan| {
                macro_rules! launch {
                    ($ty:ty, $wt:ty, $st:ty) => {{
                        let (xp, _xg) = slice_ptr_on_stream(
                            xs.as_cuda_slice::<$ty>()?, xl.start_offset(), &stream,
                        );
                        let (wp, _wg) = slice_ptr_on_stream(
                            ws.as_cuda_slice::<$wt>()?, wl.start_offset(), &stream,
                        );
                        let (sp, _sg) = slice_ptr_on_stream(
                            ss_storage.as_cuda_slice::<$st>()?, sl.start_offset(), &stream,
                        );
                        let (bp, _bg) = if let Some((ref bs, bl)) = bias_storage {
                            let Storage::Cuda(bs) = &**bs else {
                                candle_core::bail!("DKR quantized GEMV requires CUDA bias");
                            };
                            let (ptr, guard) = slice_ptr_on_stream(
                                bs.as_cuda_slice::<$ty>()?, bl.start_offset(), &stream,
                            );
                            (ptr, Some(guard))
                        } else {
                            (0u64, None)
                        };
                        let count = m.checked_mul(n)
                            .ok_or_else(|| candle_core::Error::msg("GEMV count overflow"))?;
                        let mut output = unsafe { cuda.alloc::<$ty>(count)? };
                        {
                            let (yp, _yg) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&xp).arg(&wp).arg(&sp);
                                    if key.has_bias {
                                        launch.arg(&bp);
                                    }
                                    launch.arg(&yp).arg(&mi).arg(&ni).arg(&ki);
                                    if key.fmt == Format::Fp8Blockwise {
                                        launch.arg(&ss).arg(&sy).arg(&sx);
                                    }
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        let output = CudaStorage::wrap_cuda_slice(output, cuda.clone());
                        Ok(Some(Tensor::from((Storage::Cuda(output), (m, n)))))
                    }};
                }
                match (key.fmt, key.dtype) {
                    (Format::Fp8Blockwise, DType::F16) =>
                        launch!(half::f16, float8::F8E4M3, f32),
                    (Format::Fp8Blockwise, DType::BF16) =>
                        launch!(half::bf16, float8::F8E4M3, f32),
                    (Format::Mxfp4, DType::F16) =>
                        launch!(half::f16, u8, u8),
                    (Format::Mxfp4, DType::BF16) =>
                        launch!(half::bf16, u8, u8),
                    _ => unreachable!(),
                }
            },
        )
    })
}

pub(crate) fn try_fp8(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    weight_block_size: &[usize],
) -> Result<Option<Tensor>> {
    let [by, bx] = weight_block_size else {
        return Ok(None);
    };
    try_gemv(Format::Fp8Blockwise, input, weight, scales, None, *by, *bx)
}

pub(crate) fn try_mxfp4(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    try_gemv(Format::Mxfp4, input, weight, scales, bias, 1, 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires SM61 DKR quant-gemv PTX and a CUDA GPU"]
    fn dkr_quantized_gemv_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for m in [1usize, 3] {
                let n = 7usize;
                let k = 64usize;
                let x_f32: Vec<f32> =
                    (0..m*k).map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
                let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev)?.to_dtype(dtype)?;
                let fp8_weight_f32: Vec<f32> =
                    (0..n*k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
                let fp8_weight = Tensor::from_vec(
                    fp8_weight_f32.clone(), (n, k), &dev,
                )?.to_dtype(DType::F8E4M3)?;
                let fp8_scale = Tensor::from_vec(
                    vec![0.5f32; n * k/16], (n, k/16), &dev,
                )?;
                let fp8_y = try_fp8(
                    &x, &fp8_weight, &fp8_scale, &[1, 16],
                )?.ok_or_else(|| candle_core::Error::msg("DKR FP8 GEMV not selected"))?;
                let fp8_actual = fp8_y.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                for row in 0..m {
                    for col in 0..n {
                        let expected: f32 = (0..k).map(|i|
                            x_f32[row*k+i] * fp8_weight_f32[col*k+i] * 0.5
                        ).sum();
                        assert!(
                            (fp8_actual[row][col] - expected).abs()
                                < if dtype == DType::BF16 { 0.025 } else { 0.009 },
                            "FP8 {dtype:?} m={m} row={row} col={col}"
                        );
                    }
                }

                // Packed low E2M1 code 4 = 2.0, high code 2 = 1.0.
                let mxfp4_weight = Tensor::from_vec(
                    vec![0x24u8; n * k/2], (n, k/2), &dev,
                )?;
                let scales = Tensor::from_vec(
                    vec![127u8; n*k/32], (n, k/32), &dev,
                )?;
                let bias = Tensor::from_vec(vec![0.25f32; n], n, &dev)?
                    .to_dtype(dtype)?;
                for with_bias in [false, true] {
                    let selected_bias = with_bias.then_some(&bias);
                    let actual = try_mxfp4(
                        &x, &mxfp4_weight, &scales, selected_bias,
                    )?.ok_or_else(|| candle_core::Error::msg("DKR MXFP4 GEMV not selected"))?
                        .to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    for row in 0..m {
                        for col in 0..n {
                            let expected = (0..k).map(|i|
                                x_f32[row*k+i] * if i % 2 == 0 {2.0} else {1.0}
                            ).sum::<f32>() + if with_bias {0.25} else {0.0};
                            assert!(
                                (actual[row][col] - expected).abs()
                                    < if dtype == DType::BF16 {0.04} else {0.01},
                                "MXFP4 {dtype:?} bias={with_bias} row={row} col={col}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
