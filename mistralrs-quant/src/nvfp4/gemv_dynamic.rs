//! DKR-R1.21/S3 SM61 NVFP4 W4A16 fused GEMV, packed E2M1 weights.
//! E4M3 block16 scales + F32 global per weight row. CUDA E7 remains fallback.

use crate::{
    dkr_cuda::LaunchPlanCache,
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    dtype: DType,
    rows: usize,
    cols: usize,
    k: usize,
    has_bias: bool,
}

thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn tensor(dtype: ElementType, dims: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype, shape: dims, contiguous: true, start_offset: 0,
        representation: repr.map(str::to_owned),
    }
}

fn prepare(key: Key, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let dtype = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        _ => candle_core::bail!("DKR NVFP4 GEMV requires F16/BF16"),
    };
    let mode = if key.has_bias { "bias" } else { "plain" };
    let mut args = vec![
        tensor(dtype, vec![key.rows, key.k], None),
        tensor(ElementType::U8, vec![key.cols, key.k / 2],
               Some("nvfp4_e2m1_group16.v1")),
        tensor(ElementType::Fp8E4m3, vec![key.cols, key.k / 16],
               Some("nvfp4_e4m3_group16.v1")),
        tensor(ElementType::F32, vec![key.cols], None),
    ];
    if key.has_bias {
        args.push(tensor(dtype, vec![key.cols], None));
    }
    args.push(tensor(dtype, vec![key.rows, key.cols], None));
    args.extend((0..3).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut req = LaunchRequest {
        operation: format!("quant.nvfp4.gemv.{mode}.b128.v1"),
        cc: 0, args,
        grid: [
            u32::try_from(key.cols).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.rows).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = crate::dkr_cuda::catalog()?
        .prepare_for_device(&mut req, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))?;
    if prepared.manifest.block != [BLOCK as u32, 1, 1] {
        candle_core::bail!("DKR NVFP4 GEMV: invalid block configuration");
    }
    prepared.prevalidate_launch(&req, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))
}

pub(super) fn try_gemv(
    x: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    global: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    if !matches!(x.dtype(), DType::F16 | DType::BF16)
        || x.rank() != 2 || weight.rank() != 2
        || !x.device().is_cuda()
        || weight.dtype() != DType::U8 || scales.dtype() != DType::F8E4M3
        || global.dtype() != DType::F32
        || !x.is_contiguous() || !weight.is_contiguous()
        || !scales.is_contiguous() || !global.is_contiguous()
        || !x.device().same_device(weight.device())
        || !x.device().same_device(scales.device())
        || !x.device().same_device(global.device())
        || bias.is_some_and(|b| {
            b.dtype() != x.dtype() || !b.is_contiguous()
                || !x.device().same_device(b.device())
        })
    {
        return Ok(None);
    }
    let (rows, k) = x.dims2()?;
    let (cols, packed) = weight.dims2()?;
    if rows == 0 || rows > MAX_BATCH || cols == 0 || k == 0
        || k % 16 != 0 || packed != k/2
        || scales.dims() != [cols, k/16] || global.dims() != [cols]
        || bias.is_some_and(|b| b.dims() != [cols])
        || [rows, cols, k].iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let count = rows.checked_mul(cols)
        .ok_or_else(|| candle_core::Error::msg("DKR NVFP4 output overflow"))?;
    let key = Key { dtype: x.dtype(), rows, cols, k, has_bias: bias.is_some() };
    let cuda = x.device().as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (ss, sl) = scales.storage_and_layout();
    let (gs, gl) = global.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws),
         Storage::Cuda(ss), Storage::Cuda(gs)) = (&*xs, &*ws, &*ss, &*gs)
    else {
        candle_core::bail!("DKR NVFP4 GEMV requires CUDA buffers");
    };
    let bias_storage = bias.map(|b| b.storage_and_layout());
    let (wp, _wg) = slice_ptr_on_stream(
        ws.as_cuda_slice::<u8>()?, wl.start_offset(), &stream);
    let (sp, _sg) = slice_ptr_on_stream(
        ss.as_cuda_slice::<float8::F8E4M3>()?, sl.start_offset(), &stream);
    let (gp, _gg) = slice_ptr_on_stream(
        gs.as_cuda_slice::<f32>()?, gl.start_offset(), &stream);
    let mi = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let ni = u32::try_from(cols).map_err(candle_core::Error::wrap)?;
    let ki = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    PLANS.with(|plans| {
        plans.borrow_mut().with_plan(
            key, x.device(), || prepare(key, x.device()),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (xp, _xg) = slice_ptr_on_stream(
                            xs.as_cuda_slice::<$ty>()?, xl.start_offset(), &stream);
                        let (bp, _bg) = if let Some((ref bs, bl)) = bias_storage {
                            let Storage::Cuda(bs) = &**bs else {
                                candle_core::bail!("NVFP4 GEMV bias requires CUDA");
                            };
                            let (pointer, guard) = slice_ptr_on_stream(
                                bs.as_cuda_slice::<$ty>()?, bl.start_offset(), &stream);
                            (pointer, Some(guard))
                        } else {
                            (0u64, None)
                        };
                        let mut output = unsafe { cuda.alloc::<$ty>(count)? };
                        {
                            let (yp, _yg) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&xp).arg(&wp).arg(&sp).arg(&gp);
                                    if key.has_bias {
                                        launch.arg(&bp);
                                    }
                                    launch.arg(&yp).arg(&mi).arg(&ni).arg(&ki);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        }
                        Ok(Some(Tensor::from((
                            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, cuda.clone())),
                            (rows, cols),
                        ))))
                    }};
                }
                match key.dtype {
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
    #[ignore = "requires SM61 NVFP4 GEMV PTX and CUDA GPU"]
    fn dkr_nvfp4_gemv_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let k = 32usize;
        let n = 5usize;
        for dtype in [DType::F16, DType::BF16] {
            for rows in [1usize, 3] {
                let input: Vec<f32> = (0..rows*k)
                    .map(|i| (i as i32 % 7 - 3) as f32 / 16.0).collect();
                let x = Tensor::from_vec(input.clone(), (rows, k), &dev)?
                    .to_dtype(dtype)?;
                let weights = Tensor::from_vec(
                    vec![0x24u8; n*k/2], (n, k/2), &dev)?;
                let scales = Tensor::from_vec(
                    vec![float8::F8E4M3::from_f32(1.0); n*k/16],
                    (n, k/16), &dev)?;
                let globals = Tensor::from_vec(
                    vec![0.5f32, 1.0, 1.5, 2.0, 2.5], n, &dev)?;
                let bias = Tensor::from_vec(vec![0.25f32; n], n, &dev)?
                    .to_dtype(dtype)?;
                for with_bias in [false, true] {
                    let output = try_gemv(
                        &x, &weights, &scales, &globals,
                        with_bias.then_some(&bias),
                    )?.ok_or_else(|| candle_core::Error::msg("DKR NVFP4 not selected"))?;
                    let actual = output.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    for row in 0..rows {
                        for col in 0..n {
                            let dot: f32 = (0..k).map(|i|
                                input[row*k+i] * if i%2 == 0 {2.0} else {1.0}
                            ).sum();
                            let expected = dot * (0.5 + 0.5 * col as f32)
                                + if with_bias {0.25} else {0.0};
                            assert!(
                                (actual[row][col] - expected).abs()
                                    < if dtype == DType::BF16 {0.04} else {0.01},
                                "NVFP4 {dtype:?} rows={rows} col={col}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
