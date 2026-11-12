//! DKR-R1.19 experimental dense GEMM for prefill on SM61.
//! The ordinary cuBLAS/candle matmul remains the default unless opt-in
//! dkr-gemm is explicitly enabled. No cache-miss work on the hot path.

use crate::{
    dkr_cuda::{catalog, dtype, tensor, LaunchPlanCache},
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

const TILE: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    bias: bool,
}

thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn prepare(key: Key, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let dt = dtype(key.dtype)?;
    let mut args = vec![
        tensor(dt, vec![key.m, key.k]),
        tensor(dt, vec![key.n, key.k]),
    ];
    if key.bias {
        args.push(tensor(dt, vec![key.n]));
    }
    args.extend([
        tensor(dt, vec![key.m, key.n]),
        Argument::Scalar { dtype: ElementType::U32 },
        Argument::Scalar { dtype: ElementType::U32 },
        Argument::Scalar { dtype: ElementType::U32 },
    ]);
    let mut req = LaunchRequest {
        operation: format!(
            "linear.gemm.{}.t16.v1",
            if key.bias { "bias" } else { "plain" }
        ),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.n.div_ceil(TILE)).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.m.div_ceil(TILE)).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [TILE as u32, TILE as u32, 1] {
        candle_core::bail!("DKR GEMM: invalid 16x16 thread geometry");
    }
    prepared.prevalidate_launch(&req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Y = X[M,K] @ Weight[N,K]^T + optional Bias[N].
/// Returns None only if original matmul should handle this layout/dtype.
pub(super) fn try_gemm(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    if !x.is_contiguous()
        || !weight.is_contiguous()
        || bias.is_some_and(|tensor| !tensor.is_contiguous())
        || x.rank() != 2
        || weight.rank() != 2
        || !matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
        || x.dtype() != weight.dtype()
        || !x.device().same_device(weight.device())
        || bias.is_some_and(|tensor| {
            tensor.dtype() != x.dtype() || !tensor.device().same_device(x.device())
        })
    {
        return Ok(None);
    }
    let (m, k) = x.dims2()?;
    let (n, wk) = weight.dims2()?;
    if m == 0 || n == 0 || k == 0 || wk != k
        || bias.is_some_and(|tensor| tensor.dims() != [n])
        || [m, n, k].iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let elements = m.checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("DKR GEMM: output allocation overflow"))?;
    let cuda = x.device().as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws)) = (&*xs, &*ws) else {
        candle_core::bail!("DKR GEMM requires CUDA storage");
    };
    let bias_storage = bias.map(|tensor| tensor.storage_and_layout());
    let key = Key { dtype: x.dtype(), m, n, k, bias: bias.is_some() };
    let mi = u32::try_from(m).map_err(candle_core::Error::wrap)?;
    let ni = u32::try_from(n).map_err(candle_core::Error::wrap)?;
    let ki = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    HOT.with(|hot| {
        hot.borrow_mut().with_plan(
            key, x.device(),
            || prepare(key, x.device()),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (xp, _xg) = slice_ptr_on_stream(
                            xs.as_cuda_slice::<$ty>()?, xl.start_offset(), &stream);
                        let (wp, _wg) = slice_ptr_on_stream(
                            ws.as_cuda_slice::<$ty>()?, wl.start_offset(), &stream);
                        let (bp, _bg) = if let Some((ref bs, bl)) = bias_storage {
                            let Storage::Cuda(bs) = &**bs else {
                                candle_core::bail!("DKR GEMM bias must be on CUDA");
                            };
                            let (bp, guard) = slice_ptr_on_stream(
                                bs.as_cuda_slice::<$ty>()?, bl.start_offset(), &stream);
                            (bp, Some(guard))
                        } else {
                            (0u64, None)
                        };
                        let mut output = unsafe { cuda.alloc::<$ty>(elements)? };
                        {
                            let (yp, _yg) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&xp).arg(&wp);
                                    if key.bias {
                                        launch.arg(&bp);
                                    }
                                    launch.arg(&yp).arg(&mi).arg(&ni).arg(&ki);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        }
                        let output = CudaStorage::wrap_cuda_slice(output, cuda.clone());
                        Ok(Some(Tensor::from((Storage::Cuda(output), (m, n)))))
                    }};
                }
                match key.dtype {
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
                    DType::F32 => launch!(f32),
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
    #[ignore = "requires external SM61 DKR dense GEMM PTX"]
    fn dkr_dense_gemm_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let (m, n, k) = (3usize, 19usize, 65usize);
            let xs: Vec<f32> = (0..m*k)
                .map(|i| (i as i32 % 19 - 9) as f32 / 32.0).collect();
            let ws: Vec<f32> = (0..n*k)
                .map(|i| (i as i32 % 23 - 11) as f32 / 32.0).collect();
            let bs: Vec<f32> = (0..n)
                .map(|i| (i as i32 % 7 - 3) as f32 / 32.0).collect();
            let x = Tensor::from_vec(xs.clone(), (m, k), &dev)?.to_dtype(dtype)?;
            let w = Tensor::from_vec(ws.clone(), (n, k), &dev)?.to_dtype(dtype)?;
            let b = Tensor::from_vec(bs.clone(), n, &dev)?.to_dtype(dtype)?;
            for has_bias in [false, true] {
                let selected_bias = has_bias.then_some(&b);
                let out = try_gemm(&x, &w, selected_bias)?
                    .ok_or_else(|| candle_core::Error::msg("DKR GEMM route not selected"))?;
                let actual = out.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                for row in 0..m {
                    for col in 0..n {
                        let mut expected = (0..k)
                            .map(|idx| xs[row*k+idx] * ws[col*k+idx])
                            .sum::<f32>();
                        if has_bias { expected += bs[col]; }
                        let tolerance = if dtype == DType::BF16 { 0.03 } else { 0.01 };
                        assert!(
                            (actual[row][col] - expected).abs() <= tolerance * (1.0 + expected.abs()),
                            "dtype={dtype:?}, bias={has_bias} actual={} expected={expected}",
                            actual[row][col]
                        );
                    }
                }
                let cached = HOT.with(|hot| hot.borrow().len());
                let repeated = try_gemm(&x, &w, selected_bias)?.unwrap();
                assert_eq!(cached, HOT.with(|hot| hot.borrow().len()));
                assert_eq!(actual, repeated.to_dtype(DType::F32)?.to_vec2::<f32>()?);
            }
        }
        Ok(())
    }
}
