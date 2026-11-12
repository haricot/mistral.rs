//! DKR-R1.21/S5: scaled FP8 E4M3 K/V short-context decode on Pascal.
//! FP8 K/V and explicit F32 scalar scales, contiguous and uncapped.
//! All other cases retain existing v1/v2 and FlashInfer FFI backends.

use super::slice_ptr_on_stream;
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, path::PathBuf, sync::OnceLock};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MISTRALRS_DKR_PAGED_FP8").map_or(true, |value| value != "0")
    })
}

const BLOCK: u32 = 128;
const MAX_CONTEXT: usize = 256;
const ENV: &str = "DKR_KERNEL_PACKAGES";
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    dtype: DType,
    seqs: usize,
    heads: usize,
    kv_heads: usize,
    head_size: usize,
    blocks: usize,
    block_size: usize,
    x: usize,
    table_stride: usize,
    max_context: usize,
    scale_bits: u32,
}

thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| {
        std::env::var_os(ENV).map(PathBuf::from)
            .ok_or_else(|| format!("{ENV} not set for external paged-decode"))
            .and_then(|root| KernelExecutionCache::discover(&root)
                .map_err(|err| err.to_string()))
    }).as_ref().map_err(|err| candle_core::Error::Msg(err.clone()))
}

fn tensor(dtype: ElementType, shape: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype, shape,
        contiguous: true, start_offset: 0,
        representation: repr.map(str::to_owned),
    }
}

fn prepare(key: Key, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dt = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("invalid DKR paged-FP8 decode dtype"),
    };
    let mut args = vec![
        tensor(dt, vec![key.seqs, key.heads, key.head_size], None),
        tensor(ElementType::Fp8E4m3, vec![
            key.blocks, key.kv_heads, key.head_size/key.x, key.block_size, key.x
        ], Some("vllm_k_blocked_x.v1")),
        tensor(ElementType::Fp8E4m3, vec![
            key.blocks, key.kv_heads, key.head_size, key.block_size
        ], Some("vllm_v_token_strided.v1")),
        tensor(ElementType::U32, vec![key.seqs, key.table_stride], None),
        tensor(ElementType::U32, vec![key.seqs], None),
        tensor(ElementType::F32, vec![1], None),
        tensor(ElementType::F32, vec![1], None),
        tensor(dt, vec![key.seqs, key.heads, key.head_size], None),
    ];
    args.extend((0..8).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    args.push(Argument::Scalar { dtype: ElementType::F32 });
    let mut req = LaunchRequest {
        operation: "attention.paged.fp8_decode.online.v1".into(),
        cc: 0, args,
        grid: [
            u32::try_from(key.heads).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.seqs).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut req, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] {
        candle_core::bail!("DKR paged-FP8 decode: invalid block size");
    }
    prepared.prevalidate_launch(&req, device)
        .map_err(|err| candle_core::Error::msg(err.to_string()))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_decode(
    q: &CudaStorage,
    q_layout: &Layout,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    k_scale: &Tensor,
    v_scale: &Tensor,
    softmax_scale: f32,
    softcapping: f32,
    max_context_len: usize,
    complex_options: bool,
) -> Result<Option<(CudaStorage, Shape)>> {
    if !enabled() {
        return Ok(None);
    }
    let dtype = q.dtype();
    if complex_options
        || softcapping != 1.0 || !softmax_scale.is_finite()
        || max_context_len == 0 || max_context_len > MAX_CONTEXT
        || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || key_cache.dtype() != DType::F8E4M3
        || value_cache.dtype() != DType::F8E4M3
        || k_scale.dtype() != DType::F32 || v_scale.dtype() != DType::F32
        || k_scale.dims() != [1] || v_scale.dims() != [1]
        || !k_scale.is_contiguous() || !v_scale.is_contiguous()
        || block_table.dtype() != DType::U32
        || context_lens.dtype() != DType::U32
        || !q_layout.is_contiguous()
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !block_table.is_contiguous() || !context_lens.is_contiguous()
        || q_layout.shape().rank() != 3 || key_cache.rank() != 5
        || value_cache.rank() != 4
        || block_table.rank() != 2 || context_lens.rank() != 1
    {
        return Ok(None);
    }
    let (seqs, heads, head_size) = q_layout.shape().dims3()?;
    let (blocks, kv_heads, head_width, block_size, x) = key_cache.dims5()?;
    let (value_blocks, value_heads, value_width, value_block) = value_cache.dims4()?;
    let (table_seqs, table_stride) = block_table.dims2()?;
    if seqs == 0 || heads == 0 || kv_heads == 0 || blocks == 0
        || x == 0 || block_size == 0 || table_stride == 0
        || heads % kv_heads != 0 || head_width.checked_mul(x) != Some(head_size)
        || blocks != value_blocks || kv_heads != value_heads
        || head_size != value_width || block_size != value_block
        || table_seqs != seqs || context_lens.dims() != [seqs]
        || max_context_len > table_stride.saturating_mul(block_size)
        || !matches!(head_size, 64 | 80 | 96 | 112 | 128 | 192 | 256 | 512)
        || [seqs, heads, kv_heads, head_size, block_size, x, table_stride, max_context_len]
            .iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let cuda = q.device();
    let dev = Device::Cuda(cuda.clone());
    if ![key_cache, value_cache, block_table, context_lens].iter()
        .all(|t| dev.same_device(t.device()))
        || !dev.same_device(k_scale.device())
        || !dev.same_device(v_scale.device())
    {
        return Ok(None);
    }
    let output_elements = seqs.checked_mul(heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| candle_core::Error::msg("DKR paged-FP8 decode count overflow"))?;
    let (ks, kl) = key_cache.storage_and_layout();
    let (vs, vl) = value_cache.storage_and_layout();
    let (ts, tl) = block_table.storage_and_layout();
    let (cs, cl) = context_lens.storage_and_layout();
    let (kss, ksl) = k_scale.storage_and_layout();
    let (vss, vsl) = v_scale.storage_and_layout();
    let (Storage::Cuda(ks), Storage::Cuda(vs),
        Storage::Cuda(ts), Storage::Cuda(cs),
        Storage::Cuda(kss), Storage::Cuda(vss)) =
        (&*ks, &*vs, &*ts, &*cs, &*kss, &*vss)
    else {
        candle_core::bail!("DKR paged-FP8 decode requires CUDA buffers");
    };
    let stream = cuda.cuda_stream();
    let key = Key {
        dtype, seqs, heads, kv_heads, head_size, blocks, block_size, x,
        table_stride, max_context: max_context_len,
        scale_bits: softmax_scale.to_bits(),
    };
    let seqs32 = u32::try_from(seqs).map_err(candle_core::Error::wrap)?;
    let heads32 = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let kv_heads32 = u32::try_from(kv_heads).map_err(candle_core::Error::wrap)?;
    let head_size32 = u32::try_from(head_size).map_err(candle_core::Error::wrap)?;
    let block_size32 = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let x32 = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    let stride32 = u32::try_from(table_stride).map_err(candle_core::Error::wrap)?;
    let max32 = u32::try_from(max_context_len).map_err(candle_core::Error::wrap)?;
    PLANS.with(|plans| {
        plans.borrow_mut().with_plan(
            key, &dev, || prepare(key, &dev),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (qp, _qg) = slice_ptr_on_stream(
                            q.as_cuda_slice::<$ty>()?, q_layout.start_offset(), &stream,
                        );
                        let (kp, _kg) = slice_ptr_on_stream(
                            ks.as_cuda_slice::<float8::F8E4M3>()?, kl.start_offset(), &stream,
                        );
                        let (vp, _vg) = slice_ptr_on_stream(
                            vs.as_cuda_slice::<float8::F8E4M3>()?, vl.start_offset(), &stream,
                        );
                        let (tp, _tg) = slice_ptr_on_stream(
                            ts.as_cuda_slice::<u32>()?, tl.start_offset(), &stream,
                        );
                        let (cp, _cg) = slice_ptr_on_stream(
                            cs.as_cuda_slice::<u32>()?, cl.start_offset(), &stream,
                        );
                        let (ksp, _ksg) = slice_ptr_on_stream(
                            kss.as_cuda_slice::<f32>()?, ksl.start_offset(), &stream,
                        );
                        let (vsp, _vsg) = slice_ptr_on_stream(
                            vss.as_cuda_slice::<f32>()?, vsl.start_offset(), &stream,
                        );
                        let mut output = unsafe { cuda.alloc::<$ty>(output_elements)? };
                        {
                            let (op, _og) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&qp).arg(&kp).arg(&vp)
                                        .arg(&tp).arg(&cp).arg(&ksp).arg(&vsp).arg(&op)
                                        .arg(&seqs32).arg(&heads32).arg(&kv_heads32)
                                        .arg(&head_size32).arg(&block_size32).arg(&x32)
                                        .arg(&stride32).arg(&max32).arg(&softmax_scale);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        }
                        Ok(Some((
                            CudaStorage::wrap_cuda_slice(output, cuda.clone()),
                            Shape::from((seqs, heads, head_size)),
                        )))
                    }};
                }
                match dtype {
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
    use candle_core::Tensor;

    #[test]
    #[ignore = "requires SM61 DKR paged-FP8 PTX and CUDA GPU"]
    fn dkr_paged_fp8_decode_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let hd = 64usize;
            let x = 16usize;
            let mut q = vec![0f32; hd];
            q[0] = 2.0;
            let mut keys = vec![0f32; hd*2];
            keys[0] = 1.0; // key at first token, first dimension.
            let mut values = vec![0f32; hd*2];
            for d in 0..hd {
                values[d * 2] = 1.0;
                values[d * 2 + 1] = 3.0;
            }
            let q = Tensor::from_vec(q, (1, 1, hd), &device)?.to_dtype(dtype)?;
            let kc = Tensor::from_vec(keys, (1, 1, hd/x, 2, x), &device)?
                .to_dtype(DType::F8E4M3)?;
            let vc = Tensor::from_vec(values, (1, 1, hd, 2), &device)?
                .to_dtype(DType::F8E4M3)?;
            let table = Tensor::from_vec(vec![0u32], (1, 1), &device)?;
            let lengths = Tensor::from_vec(vec![2u32], 1, &device)?;
            let ks = Tensor::from_vec(vec![0.5f32], 1, &device)?;
            let vs = Tensor::from_vec(vec![0.5f32], 1, &device)?;
            let out = super::super::paged_attention::paged_attention(
                &q, Some(&ks), Some(&vs), &kc, &vc, &table, &lengths,
                None, 2, 1.0, 1.0, None,
            )?;
            let actual = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let exp1 = 1f32.exp();
            let expected = (0.5 * exp1 + 1.5) / (exp1 + 1.0);
            let tolerance = if dtype == DType::BF16 { 0.02 } else { 0.004 };
            for value in actual {
                assert!((value - expected).abs() < tolerance,
                    "paged FP8 decode {dtype:?} actual={value} expected={expected}");
            }
            let cached = PLANS.with(|cache| cache.borrow().len());
            super::super::paged_attention::paged_attention(
                &q, Some(&ks), Some(&vs), &kc, &vc, &table, &lengths,
                None, 2, 1.0, 1.0, None,
            )?;
            assert_eq!(cached, PLANS.with(|cache| cache.borrow().len()));
        }
        Ok(())
    }
}
