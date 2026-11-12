//! DKR-R1.21/S4: opt-in fused FP8/MXFP4 prefill GEMM on Pascal.
//! Reference SIMT path; preserves FFI GEMM as performance fallback.
//! Distinct plain/bias ABIs and no dequantized weight tensor allocation.

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
use std::{cell::RefCell, sync::OnceLock};

const TILE: usize = 16;
const GEMV_BOUNDARY: usize = 8;
static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("MISTRALRS_DKR_QUANT_GEMM")
        .map_or(true, |value| value != "0"))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Format { Fp8, Mxfp4 }

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    fmt: Format,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    by: usize,
    bx: usize,
    sr: usize,
    sc: usize,
    has_bias: bool,
}

thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn tensor(dtype: ElementType, shape: Vec<usize>, representation: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype, shape, contiguous: true, start_offset: 0,
        representation: representation.map(str::to_owned),
    }
}

fn prepare(key: Key, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let dt = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        _ => candle_core::bail!("DKR quant GEMM requires F16/BF16"),
    };
    let (op, wt, weight_shape, scales_dtype, wt_repr, s_repr) = match key.fmt {
        Format::Fp8 => (
            "quant.fp8.blockwise.gemm.t16.v1",
            ElementType::Fp8E4m3,
            vec![key.n, key.k],
            ElementType::F32,
            "fp8_e4m3_blockwise_weight.v1",
            "fp8_blockwise_f32.v1",
        ),
        Format::Mxfp4 => (
            if key.has_bias {
                "quant.mxfp4.gemm.bias.t16.v1"
            } else {
                "quant.mxfp4.gemm.plain.t16.v1"
            },
            ElementType::U8,
            vec![key.n, key.k / 2],
            ElementType::U8,
            "mxfp4_e2m1_group32.v1",
            "mxfp4_e8m0_group32.v1",
        ),
    };
    let mut args = vec![
        tensor(dt, vec![key.m, key.k], None),
        tensor(wt, weight_shape, Some(wt_repr)),
        tensor(scales_dtype, vec![key.sr, key.sc], Some(s_repr)),
    ];
    if key.has_bias { args.push(tensor(dt, vec![key.n], None)); }
    args.push(tensor(dt, vec![key.m, key.n], None));
    args.extend((0..if key.fmt == Format::Fp8 {6} else {3})
        .map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut request = LaunchRequest {
        operation: op.into(), cc: 0, args,
        grid: [
            u32::try_from(key.n.div_ceil(TILE)).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.m.div_ceil(TILE)).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?.prepare_for_device(&mut request, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))?;
    if prepared.manifest.block != [TILE as u32, TILE as u32, 1] {
        candle_core::bail!("DKR quant GEMM: invalid block geometry");
    }
    prepared.prevalidate_launch(&request, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))
}

fn try_gemm(
    fmt: Format,
    input: &Tensor, weight: &Tensor, scales: &Tensor,
    bias: Option<&Tensor>, block_y: usize, block_x: usize,
) -> Result<Option<Tensor>> {
    if !enabled()
        || !input.device().is_cuda()
        || !matches!(input.dtype(), DType::F16 | DType::BF16)
        || input.rank() != 2 || weight.rank() != 2 || scales.rank() != 2
        || !input.is_contiguous() || !weight.is_contiguous() || !scales.is_contiguous()
        || !input.device().same_device(weight.device())
        || !input.device().same_device(scales.device())
        || bias.is_some_and(|b| {
            b.dtype() != input.dtype() || b.rank() != 1 || !b.is_contiguous()
                || !input.device().same_device(b.device())
        })
        || block_x == 0 || block_y == 0
    {
        return Ok(None);
    }
    let (m, k) = input.dims2()?;
    let (n, weight_width) = weight.dims2()?;
    if m <= GEMV_BOUNDARY || n == 0 || k == 0
        || bias.is_some_and(|b| b.dims() != [n])
        || [m, n, k, block_x, block_y].iter().any(|&x| x > u32::MAX as usize)
    {
        return Ok(None);
    }
    let (sr, sc) = match fmt {
        Format::Fp8 => {
            if weight.dtype() != DType::F8E4M3
                || scales.dtype() != DType::F32
                || weight_width != k || bias.is_some()
            {
                return Ok(None);
            }
            (n.div_ceil(block_y), k.div_ceil(block_x))
        }
        Format::Mxfp4 => {
            if weight.dtype() != DType::U8 || scales.dtype() != DType::U8
                || k % 32 != 0 || weight_width != k/2
            {
                return Ok(None);
            }
            (n, k/32)
        }
    };
    if scales.dims() != [sr, sc] || sc > u32::MAX as usize {
        return Ok(None);
    }
    let count = m.checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("quant GEMM output overflow"))?;
    let key = Key {fmt, dtype: input.dtype(), m, n, k,
                   by: block_y, bx: block_x, sr, sc, has_bias: bias.is_some()};
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (xs, xl) = input.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ss, sl) = scales.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss)) = (&*xs, &*ws, &*ss)
    else {
        candle_core::bail!("DKR quant GEMM requires CUDA storages");
    };
    let bias_storage = bias.map(|b| b.storage_and_layout());
    let mi = u32::try_from(m).map_err(candle_core::Error::wrap)?;
    let ni = u32::try_from(n).map_err(candle_core::Error::wrap)?;
    let ki = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    let by = u32::try_from(block_y).map_err(candle_core::Error::wrap)?;
    let bx = u32::try_from(block_x).map_err(candle_core::Error::wrap)?;
    let sc = u32::try_from(sc).map_err(candle_core::Error::wrap)?;
    HOT.with(|plans| {
        plans.borrow_mut().with_plan(key, input.device(),
            || prepare(key, input.device()),
            |plan| {
                macro_rules! launch {
                    ($ty:ty, $wt:ty, $st:ty) => {{
                        let (xp, _xg) = slice_ptr_on_stream(
                            xs.as_cuda_slice::<$ty>()?, xl.start_offset(), &stream);
                        let (wp, _wg) = slice_ptr_on_stream(
                            ws.as_cuda_slice::<$wt>()?, wl.start_offset(), &stream);
                        let (sp, _sg) = slice_ptr_on_stream(
                            ss.as_cuda_slice::<$st>()?, sl.start_offset(), &stream);
                        let (bp, _bg) = if let Some((ref storage, layout)) = bias_storage {
                            let Storage::Cuda(storage) = &**storage else {
                                candle_core::bail!("DKR quant GEMM bias must be CUDA");
                            };
                            let (pointer, guard) = slice_ptr_on_stream(
                                storage.as_cuda_slice::<$ty>()?, layout.start_offset(), &stream);
                            (pointer, Some(guard))
                        } else { (0u64, None) };
                        let mut output = unsafe { dev.alloc::<$ty>(count)? };
                        {
                            let (yp, _yg) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder.arg(&xp).arg(&wp).arg(&sp);
                                    if key.has_bias { builder.arg(&bp); }
                                    builder.arg(&yp).arg(&mi).arg(&ni).arg(&ki);
                                    if key.fmt == Format::Fp8 {
                                        builder.arg(&sc).arg(&by).arg(&bx);
                                    }
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        }
                        Ok(Some(Tensor::from((
                            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                            (m, n),
                        ))))
                    }};
                }
                match (key.fmt, key.dtype) {
                    (Format::Fp8, DType::F16) => launch!(half::f16, float8::F8E4M3, f32),
                    (Format::Fp8, DType::BF16) => launch!(half::bf16, float8::F8E4M3, f32),
                    (Format::Mxfp4, DType::F16) => launch!(half::f16, u8, u8),
                    (Format::Mxfp4, DType::BF16) => launch!(half::bf16, u8, u8),
                    _ => unreachable!(),
                }
            },
        )
    })
}

pub(crate) fn try_fp8(
    input: &Tensor, weight: &Tensor, scales: &Tensor, weight_block_size: &[usize],
) -> Result<Option<Tensor>> {
    let [by, bx] = weight_block_size else { return Ok(None); };
    try_gemm(Format::Fp8, input, weight, scales, None, *by, *bx)
}
pub(crate) fn try_mxfp4(
    input: &Tensor, weight: &Tensor, scales: &Tensor, bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    try_gemm(Format::Mxfp4, input, weight, scales, bias, 1, 32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires prebuilt SM61 DKR quantized GEMM PTX"]
    fn dkr_quantized_gemm_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (m, n, k) = (17usize, 19usize, 64usize);
        for dtype in [DType::F16, DType::BF16] {
            let xs: Vec<f32> = (0..m*k)
                .map(|i| (i as i32%13-6) as f32 / 8.0).collect();
            let ws: Vec<f32> = (0..n*k)
                .map(|i| (i as i32%11-5) as f32 / 4.0).collect();
            let x = Tensor::from_vec(xs.clone(), (m,k), &dev)?.to_dtype(dtype)?;
            let fp8w = Tensor::from_vec(ws.clone(), (n,k), &dev)?
                .to_dtype(DType::F8E4M3)?;
            let scale = Tensor::from_vec(
                vec![0.5f32; n.div_ceil(2)*(k/16)], (n.div_ceil(2), k/16), &dev)?;
            let out = try_fp8(&x, &fp8w, &scale, &[2,16])?
                .ok_or_else(|| candle_core::Error::msg("DKR FP8 prefill GEMM not selected"))?
                .to_dtype(DType::F32)?.to_vec2::<f32>()?;
            for row in 0..m {
                for col in 0..n {
                    let expected = (0..k).map(|j| xs[row*k+j]*ws[col*k+j]*0.5).sum::<f32>();
                    assert!((out[row][col]-expected).abs() < 0.05,
                            "FP8 prefill {dtype:?} row={row} col={col}");
                }
            }
            let w = Tensor::from_vec(vec![0x24u8; n*k/2], (n,k/2), &dev)?;
            let s = Tensor::from_vec(vec![127u8; n*k/32], (n,k/32), &dev)?;
            let bias = Tensor::from_vec(vec![0.25f32;n],n,&dev)?.to_dtype(dtype)?;
            for has_bias in [false,true] {
                let out = try_mxfp4(&x,&w,&s,has_bias.then_some(&bias))?
                    .ok_or_else(|| candle_core::Error::msg("DKR MXFP4 prefill GEMM not selected"))?
                    .to_dtype(DType::F32)?.to_vec2::<f32>()?;
                for row in 0..m {
                    for col in 0..n {
                        let expected = (0..k).map(|j| {
                            xs[row*k+j]*(if j%2==0 {2.0} else {1.0})
                        }).sum::<f32>() + if has_bias {0.25} else {0.0};
                        assert!((out[row][col]-expected).abs() < 0.06,
                            "MXFP4 prefill {dtype:?},bias={has_bias}, row={row} col={col}");
                    }
                }
            }
            let cached = HOT.with(|hot| hot.borrow().len());
            try_mxfp4(&x,&w,&s,None)?;
            try_fp8(&x,&fp8w,&scale,&[2,16])?;
            assert_eq!(cached,HOT.with(|hot| hot.borrow().len()));
        }
        Ok(())
    }
}
