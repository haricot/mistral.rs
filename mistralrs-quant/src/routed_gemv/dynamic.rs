//! DKR-R1.21/S8 — experimental directly routed FP8/FP4 GEMV and GEMM.
//!
//! Restricted to plain expert projections, no bias/GLU/top-k reduction.
//! The existing fused routed-gemv and sorted grouped-MMQ kernels remain
//! available, including the original FFN/LoRA dispatch.
//! Runtime opt-in requires MISTRALRS_DKR_ROUTED_QUANT=1 (feature alone
//! never changes production paths). CUDA and Rust compile only in S9.

use super::{RoutedCodec, RoutedWeights};
use crate::{
    dkr_cuda::{catalog, LaunchPlanCache},
    utils::{slice_ptr_mut_on_stream, CudaReadView},
};
use candle_core::{
    cuda::cudarc::driver::PushKernelArg,
    CudaStorage, DType, Device, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::{cell::RefCell, sync::OnceLock};

const GEMV_MAX_TOKENS: usize = 16;
const GEMV_BLOCK: [u32; 3] = [128, 1, 1];
const GEMM_TILE: usize = 16;
const GEMM_BLOCK: [u32; 3] = [16, 16, 1];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Codec {
    Fp8,
    Mxfp8,
    Mxfp4,
    Nvfp4,
}
impl Codec {
    fn tag(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Mxfp8 => "mxfp8",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Gemv,
    Gemm,
}
impl Stage {
    fn tag(self) -> &'static str {
        match self {
            Self::Gemv => "gemv",
            Self::Gemm => "gemm",
        }
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    codec: Codec,
    stage: Stage,
    dtype: DType,
    tokens: usize,
    topk: usize,
    input_routes: usize,
    experts: usize,
    n: usize,
    k: usize,
    scale_rows: usize,
    scale_cols: usize,
    block_y: usize,
    block_x: usize,
}
thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MISTRALRS_DKR_ROUTED_QUANT")
            .is_ok_and(|value| value == "1")
    })
}
fn tensor(dtype: ElementType, shape: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: repr.map(str::to_owned),
    }
}
fn dtype(dtype: DType) -> Result<ElementType> {
    match dtype {
        DType::F16 => Ok(ElementType::F16),
        DType::BF16 => Ok(ElementType::Bf16),
        DType::F32 => Ok(ElementType::F32),
        _ => candle_core::bail!("DKR routed quant only accepts F16/BF16/F32 activations"),
    }
}

fn prepare(key: Key, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dt = dtype(key.dtype)?;
    let (weight_dtype, scale_dtype, weight_repr, scale_repr, packed) = match key.codec {
        Codec::Fp8 => (
            ElementType::Fp8E4m3, ElementType::F32,
            "fp8_e4m3_blockwise_weight.v1", "fp8_blockwise_f32.v1", false,
        ),
        Codec::Mxfp8 => (
            ElementType::Fp8E4m3, ElementType::U8,
            "mxfp8_e4m3_group32.v1", "mxfp8_e8m0_group32.v1", false,
        ),
        Codec::Mxfp4 => (
            ElementType::U8, ElementType::U8,
            "mxfp4_e2m1_group32.v1", "mxfp4_e8m0_group32.v1", true,
        ),
        Codec::Nvfp4 => (
            ElementType::U8, ElementType::Fp8E4m3,
            "nvfp4_e2m1_group16.v1", "nvfp4_e4m3_group16.v1", true,
        ),
    };
    let weight_k = if packed { key.k / 2 } else { key.k };
    let mut arguments = vec![
        tensor(dt, vec![key.tokens, key.input_routes, key.k], None),
        tensor(
            weight_dtype,
            vec![key.experts, key.n, weight_k],
            Some(weight_repr),
        ),
        tensor(
            scale_dtype,
            vec![key.experts, key.scale_rows, key.scale_cols],
            Some(scale_repr),
        ),
    ];
    if key.codec == Codec::Nvfp4 {
        arguments.push(tensor(
            ElementType::F32, vec![key.experts, key.n],
            Some("nvfp4_f32_row_global.v1"),
        ));
    }
    arguments.push(tensor(ElementType::U32, vec![key.tokens, key.topk], None));
    arguments.push(tensor(dt, vec![key.tokens, key.topk, key.n], None));
    arguments.extend((0..6).map(|_| Argument::Scalar {
        dtype: ElementType::U32,
    }));
    if key.codec == Codec::Fp8 {
        arguments.extend((0..3).map(|_| Argument::Scalar {
            dtype: ElementType::U32,
        }));
    }
    let assignments = key.tokens.checked_mul(key.topk)
        .ok_or_else(|| candle_core::Error::msg("DKR routed assignments overflow"))?;
    let grid = match key.stage {
        Stage::Gemv => [
            u32::try_from(key.n).map_err(candle_core::Error::wrap)?,
            u32::try_from(assignments).map_err(candle_core::Error::wrap)?,
            1,
        ],
        Stage::Gemm => [
            u32::try_from(key.n.div_ceil(GEMM_TILE)).map_err(candle_core::Error::wrap)?,
            u32::try_from(assignments.div_ceil(GEMM_TILE))
                .map_err(candle_core::Error::wrap)?,
            1,
        ],
    };
    let mut request = LaunchRequest {
        operation: format!(
            "moe.routed.{}.{}.plain.v1",
            key.codec.tag(),
            key.stage.tag(),
        ),
        cc: 0,
        args: arguments,
        grid,
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?.prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    let expected = match key.stage {
        Stage::Gemv => GEMV_BLOCK,
        Stage::Gemm => GEMM_BLOCK,
    };
    if prepared.manifest.block != expected
        || prepared.manifest.shared_mem_bytes != 0
    {
        candle_core::bail!("DKR routed quant: unexpected CUDA launch geometry");
    }
    prepared.prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

pub(super) fn try_gather(
    x: &Tensor,
    ids: &Tensor,
    weights: &RoutedWeights,
) -> Result<Option<Tensor>> {
    if !enabled() {
        return Ok(None);
    }
    dispatch(x, ids, weights)
}

// Called by the ignored GPU qualification without altering global env
// or racing another nextest test.
fn dispatch(x: &Tensor, ids: &Tensor, weights: &RoutedWeights)
    -> Result<Option<Tensor>>
{
    if !x.device().is_cuda()
        || !matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
        || !x.is_contiguous() || !ids.is_contiguous()
        || ids.dtype() != DType::U32 || ids.rank() != 2
        || weights.bias.is_some()
        || !weights.data.is_contiguous() || !weights.scales.is_contiguous()
        || !x.device().same_device(ids.device())
        || !x.device().same_device(weights.data.device())
        || !x.device().same_device(weights.scales.device())
    {
        return Ok(None);
    }
    let (tokens, topk) = ids.dims2()?;
    let (input_tokens, input_routes, input_k) = match x.dims() {
        [t, k] => (*t, 1, *k),
        [t, routes, k] => (*t, *routes, *k),
        _ => return Ok(None),
    };
    let (experts, n, k) = (weights.experts, weights.n, weights.k);
    let assignments = match tokens.checked_mul(topk) {
        Some(count) => count,
        None => return Ok(None),
    };
    if tokens == 0 || topk == 0 || n == 0 || k == 0 || experts == 0
        || tokens != input_tokens || input_k != k
        || !(input_routes == 1 || input_routes == topk)
        || assignments == 0
        || assignments > 65_535usize * GEMM_TILE
        || [tokens, topk, n, k, experts, input_routes]
            .iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let stage = if tokens <= GEMV_MAX_TOKENS {
        if assignments > 65_535 { return Ok(None); }
        Stage::Gemv
    } else {
        Stage::Gemm
    };
    let (codec, data_k, scale_rows, scale_cols, by, bx) = match weights.codec {
        RoutedCodec::Fp8 { block_rows, block_cols } => {
            if block_rows == 0 || block_cols == 0
                || weights.data.dtype() != DType::F8E4M3
                || weights.scales.dtype() != DType::F32
                || weights.extra.is_some()
                || [block_rows, block_cols].iter().any(|&v| v > u32::MAX as usize)
            { return Ok(None); }
            (
                Codec::Fp8, k, n.div_ceil(block_rows), k.div_ceil(block_cols),
                block_rows, block_cols,
            )
        },
        RoutedCodec::Mxfp8 => {
            if k % 32 != 0 || weights.data.dtype() != DType::F8E4M3
                || weights.scales.dtype() != DType::U8
                || weights.extra.is_some()
            { return Ok(None); }
            (Codec::Mxfp8, k, n, k / 32, 1, 32)
        },
        RoutedCodec::Mxfp4 => {
            if k % 32 != 0 || weights.data.dtype() != DType::U8
                || weights.scales.dtype() != DType::U8
                || weights.extra.is_some()
            { return Ok(None); }
            (Codec::Mxfp4, k / 2, n, k / 32, 1, 32)
        },
        RoutedCodec::Nvfp4 => {
            if k % 16 != 0 || weights.data.dtype() != DType::U8
                || weights.scales.dtype() != DType::F8E4M3
                || weights.extra.as_ref().is_none_or(|extra| {
                    extra.dtype() != DType::F32 || extra.dims() != [experts, n]
                        || !extra.is_contiguous()
                        || !x.device().same_device(extra.device())
                })
            { return Ok(None); }
            (Codec::Nvfp4, k / 2, n, k / 16, 1, 16)
        },
        RoutedCodec::Afq { .. } => return Ok(None),
    };
    if weights.data.dims() != [experts, n, data_k]
        || weights.scales.dims() != [experts, scale_rows, scale_cols]
        || scale_cols == 0 || scale_cols > u32::MAX as usize
        || scale_rows > u32::MAX as usize
    {
        return Ok(None);
    }
    let Some(output_len) = assignments.checked_mul(n) else {
        return Ok(None);
    };
    let key = Key {
        codec, stage, dtype: x.dtype(),
        tokens, topk, input_routes, experts, n, k,
        scale_rows, scale_cols, block_y: by, block_x: bx,
    };
    let device = x.device();
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let input_view = CudaReadView::new(x)?;
    let data_view = CudaReadView::new(&weights.data)?;
    let scale_view = CudaReadView::new(&weights.scales)?;
    let ids_view = CudaReadView::new(ids)?;
    let extra_view = CudaReadView::optional(weights.extra.as_ref())?;
    let (input, _input_guard) = input_view.read(&stream)?;
    let (data, _data_guard) = data_view.read(&stream)?;
    let (scales, _scale_guard) = scale_view.read(&stream)?;
    let (indices, _indices_guard) = ids_view.read(&stream)?;
    let (global, _global_guard) = CudaReadView::read_optional(extra_view.as_ref(), &stream)?;

    let tokens32 = u32::try_from(tokens).map_err(candle_core::Error::wrap)?;
    let topk32 = u32::try_from(topk).map_err(candle_core::Error::wrap)?;
    let n32 = u32::try_from(n).map_err(candle_core::Error::wrap)?;
    let k32 = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    let experts32 = u32::try_from(experts).map_err(candle_core::Error::wrap)?;
    let routes32 = u32::try_from(input_routes).map_err(candle_core::Error::wrap)?;
    let by32 = u32::try_from(by).map_err(candle_core::Error::wrap)?;
    let bx32 = u32::try_from(bx).map_err(candle_core::Error::wrap)?;
    let sc32 = u32::try_from(scale_cols).map_err(candle_core::Error::wrap)?;
    HOT.with(|plans| {
        plans.borrow_mut().with_plan(
            key, device, || prepare(key, device),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let mut output = unsafe { cuda.alloc::<$ty>(output_len)? };
                        let (out_ptr, _output_guard) =
                            slice_ptr_mut_on_stream(&mut output, 0, &stream);
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut run = stream.launch_builder(function);
                                run.arg(&input).arg(&data).arg(&scales);
                                if codec == Codec::Nvfp4 {
                                    run.arg(&global);
                                }
                                run.arg(&indices).arg(&out_ptr)
                                    .arg(&tokens32).arg(&topk32).arg(&n32).arg(&k32)
                                    .arg(&experts32).arg(&routes32);
                                if codec == Codec::Fp8 {
                                    run.arg(&by32).arg(&bx32).arg(&sc32);
                                }
                                run.launch(config)?;
                                Ok(())
                            })
                        }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        Ok(Some(Tensor::from((
                            Storage::Cuda(CudaStorage::wrap_cuda_slice(
                                output, cuda.clone(),
                            )),
                            (tokens, topk, n),
                        ))))
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

    const EXPERTS: usize = 3;
    const N: usize = 5;
    const K: usize = 32;
    const TOPK: usize = 2;

    fn weights(codec: Codec, device: &Device) -> Result<RoutedWeights> {
        let fp8 = float8::F8E4M3::from_f32(1.0);
        let (routed, data, scales, extra) = match codec {
            Codec::Fp8 => (
                RoutedCodec::Fp8 { block_rows: 2, block_cols: 16 },
                Tensor::from_vec(
                    vec![fp8; EXPERTS * N * K], (EXPERTS, N, K), device,
                )?,
                Tensor::from_vec(
                    vec![0.5f32; EXPERTS * N.div_ceil(2) * K.div_ceil(16)],
                    (EXPERTS, N.div_ceil(2), K.div_ceil(16)), device,
                )?,
                None,
            ),
            Codec::Mxfp8 => (
                RoutedCodec::Mxfp8,
                Tensor::from_vec(
                    vec![fp8; EXPERTS * N * K], (EXPERTS, N, K), device,
                )?,
                Tensor::from_vec(
                    vec![127u8; EXPERTS * N * (K/32)],
                    (EXPERTS, N, K/32), device,
                )?,
                None,
            ),
            Codec::Mxfp4 => (
                RoutedCodec::Mxfp4,
                Tensor::from_vec(
                    vec![0x24u8; EXPERTS * N * (K/2)],
                    (EXPERTS, N, K/2), device,
                )?,
                Tensor::from_vec(
                    vec![127u8; EXPERTS * N * (K/32)],
                    (EXPERTS, N, K/32), device,
                )?,
                None,
            ),
            Codec::Nvfp4 => (
                RoutedCodec::Nvfp4,
                Tensor::from_vec(
                    vec![0x24u8; EXPERTS * N * (K/2)],
                    (EXPERTS, N, K/2), device,
                )?,
                Tensor::from_vec(
                    vec![fp8; EXPERTS * N * (K/16)],
                    (EXPERTS, N, K/16), device,
                )?,
                Some(Tensor::from_vec(
                    (0..EXPERTS).flat_map(|e|
                        (0..N).map(move |_| 0.5 + e as f32 * 0.25)
                    ).collect(), (EXPERTS, N), device,
                )?),
            ),
        };
        Ok(RoutedWeights::new(
            routed, data, scales, extra, None, [EXPERTS, N, K],
        ))
    }

    fn check_stage(tokens: usize, input_routes: usize) -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for codec in [Codec::Fp8, Codec::Mxfp8, Codec::Mxfp4, Codec::Nvfp4] {
                let weight = weights(codec, &device)?;
                let ids: Vec<u32> = (0..tokens * TOPK)
                    .map(|i| ((i * 2 + 1) % EXPERTS) as u32).collect();
                let input: Vec<f32> = (0..tokens * input_routes * K)
                    .map(|i| ((i % 7) as f32 - 3.0) * 0.125).collect();
                let x = Tensor::from_vec(
                    input.clone(), (tokens, input_routes, K), &device,
                )?.to_dtype(dtype)?;
                let ids_tensor = Tensor::from_vec(
                    ids.clone(), (tokens, TOPK), &device,
                )?;
                let result = dispatch(&x, &ids_tensor, &weight)?
                    .ok_or_else(|| candle_core::Error::msg(
                        "DKR routed quant projection was not selected"
                    ))?;
                let actual = result.to_dtype(DType::F32)?.flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(result.dims(), [tokens, TOPK, N]);
                for route in 0..(tokens * TOPK) {
                    let expert = ids[route] as usize;
                    let token = route / TOPK;
                    let source_route =
                        if input_routes == 1 { 0 } else { route % TOPK };
                    let source_base = (token * input_routes + source_route) * K;
                    for col in 0..N {
                        let mut expected = 0.0f32;
                        for k in 0..K {
                            let weight = match codec {
                                Codec::Fp8 => 0.5,
                                Codec::Mxfp8 => 1.0,
                                Codec::Mxfp4 | Codec::Nvfp4 =>
                                    if k % 2 == 0 { 2.0 } else { 1.0 },
                            };
                            expected += input[source_base + k] * weight;
                        }
                        if codec == Codec::Nvfp4 {
                            expected *= 0.5 + expert as f32 * 0.25;
                        }
                        let got = actual[route * N + col];
                        let tolerance = if dtype == DType::BF16 { 0.05 } else { 0.012 };
                        assert!(
                            (got - expected).abs() <= tolerance,
                            "routed {codec:?} {dtype:?} route={route} col={col}: \
                             got={got}, expected={expected}",
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires SM61 routed-quant PTX and a CUDA GPU"]
    fn dkr_routed_quant_gemv_gpu() -> Result<()> {
        check_stage(2, 1)
    }

    #[test]
    #[ignore = "requires SM61 routed-quant PTX and a CUDA GPU"]
    fn dkr_routed_quant_gemm_gpu() -> Result<()> {
        check_stage(17, TOPK)
    }

    #[test]
    fn dkr_routed_quant_refuses_biased_or_affine_weights() -> Result<()> {
        let device = Device::Cpu;
        let ids = Tensor::zeros((1, 1), DType::U32, &device)?;
        let x = Tensor::zeros((1, 1, K), DType::F32, &device)?;
        let w = RoutedWeights::new(
            RoutedCodec::Afq { bits: 4, group: 32 },
            Tensor::zeros((1, N, K/2), DType::U8, &device)?,
            Tensor::ones((1, N, 1), DType::F32, &device)?,
            None, None, [1, N, K],
        );
        assert!(dispatch(&x, &ids, &w)?.is_none());
        Ok(())
    }
}
