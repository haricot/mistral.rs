//! DKR-R1.21/S6: two-stage partitioned VLLM paged attention, SM61.
//!
//! Stage 1: online softmax on 256-token partitions, writing F32 normalized
//! partial vectors and partition log-normalizers.
//! Stage 2: stable log-sum-exp merge, writing the original Q dtype.
//! This experimental route is only for contiguous, non-FP8 K/V without
//! ALiBi, sinks, softcapping or windows; all other cases use static v1/v2.
//! The long-lived static-v2 workspace pool is reused, not reallocated per
//! token. No static NVCC flags or CUDA_COMPUTE_CAP changes are introduced.

use super::{paged_attention::workspace_ensure, slice_ptr_on_stream};
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

const ENV: &str = "DKR_KERNEL_PACKAGES";
const PARTITION: usize = 256;
const MAX_CONTEXT: usize = 32_768;
const BLOCK: u32 = 128;
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();
static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("MISTRALRS_DKR_PAGED_V2")
            .map_or(true, |value| value != "0")
    })
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG
        .get_or_init(|| {
            std::env::var_os(ENV).map(PathBuf::from)
                .ok_or_else(|| format!("{ENV} not set for external paged-v2"))
                .and_then(|path| {
                    KernelExecutionCache::discover(&path)
                        .map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Stage {
    Partition,
    Merge,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Partition => "partition",
            Self::Merge => "merge",
        }
    }
}

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
    parts: usize,
    scale_bits: u32,
}

thread_local! {
    static PLANS: RefCell<LaunchPlanCache<(Key, Stage), PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
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

fn prepare(
    key: Key,
    stage: Stage,
    device: &Device,
) -> Result<PrevalidatedLaunchPlan> {
    let dt = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("unsupported DKR paged-v2 activation dtype"),
    };
    let partial_shape = vec![key.seqs, key.heads, key.parts, key.head_size];
    let lse_shape = vec![key.seqs, key.heads, key.parts];
    let (args, grid) = match stage {
        Stage::Partition => {
            let mut args = vec![
                tensor(dt, vec![key.seqs, key.heads, key.head_size], None),
                tensor(dt, vec![
                    key.blocks, key.kv_heads, key.head_size / key.x,
                    key.block_size, key.x,
                ], Some("vllm_k_blocked_x.v1")),
                tensor(dt, vec![
                    key.blocks, key.kv_heads, key.head_size, key.block_size,
                ], Some("vllm_v_token_strided.v1")),
                tensor(ElementType::U32, vec![key.seqs, key.table_stride], None),
                tensor(ElementType::U32, vec![key.seqs], None),
                tensor(ElementType::F32, partial_shape, None),
                tensor(ElementType::F32, lse_shape, None),
            ];
            args.extend(
                (0..8).map(|_| Argument::Scalar { dtype: ElementType::U32 }),
            );
            args.push(Argument::Scalar { dtype: ElementType::F32 });
            (
                args,
                [
                    u32::try_from(key.heads).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.seqs).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.parts).map_err(candle_core::Error::wrap)?,
                ],
            )
        }
        Stage::Merge => {
            let mut args = vec![
                tensor(ElementType::F32, partial_shape, None),
                tensor(ElementType::F32, lse_shape, None),
                tensor(dt, vec![key.seqs, key.heads, key.head_size], None),
            ];
            args.extend(
                (0..4).map(|_| Argument::Scalar { dtype: ElementType::U32 }),
            );
            (
                args,
                [
                    u32::try_from(key.heads).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.seqs).map_err(candle_core::Error::wrap)?,
                    1,
                ],
            )
        }
    };
    let mut request = LaunchRequest {
        operation: format!("attention.paged.v2.{}.p256.v1", stage.name()),
        cc: 0,
        args,
        grid,
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] ||
        prepared.manifest.shared_mem_bytes != 0
    {
        candle_core::bail!("DKR paged-v2: unexpected block/shared-memory geometry");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

fn workspace_layout(key: Key) -> Result<(usize, usize)> {
    let count = key.seqs.checked_mul(key.heads)
        .and_then(|count| count.checked_mul(key.parts))
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partition count overflow"))?;
    let partial_elems = count.checked_mul(key.head_size)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partial count overflow"))?;
    let partial_bytes = partial_elems.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partial allocation overflow"))?;
    let lse_offset = partial_bytes.checked_add(15)
        .map(|value| value & !15)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 offset overflow"))?;
    let lse_bytes = count.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 lse allocation overflow"))?;
    let total_bytes = lse_offset.checked_add(lse_bytes)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 workspace overflow"))?;
    Ok((lse_offset, total_bytes))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_decode(
    q: &CudaStorage,
    q_layout: &Layout,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
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
        || softcapping != 1.0
        || !softmax_scale.is_finite()
        || max_context_len <= PARTITION || max_context_len > MAX_CONTEXT
        || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || key_cache.dtype() != dtype || value_cache.dtype() != dtype
        || block_table.dtype() != DType::U32
        || context_lens.dtype() != DType::U32
        || !q_layout.is_contiguous()
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !block_table.is_contiguous() || !context_lens.is_contiguous()
        || q_layout.shape().rank() != 3
        || key_cache.rank() != 5 || value_cache.rank() != 4
        || block_table.rank() != 2 || context_lens.rank() != 1
    {
        return Ok(None);
    }
    let (seqs, heads, head_size) = q_layout.shape().dims3()?;
    let (blocks, kv_heads, head_width, block_size, x) = key_cache.dims5()?;
    let (vblocks, vheads, vhead_size, vblock_size) = value_cache.dims4()?;
    let (table_seqs, table_stride) = block_table.dims2()?;
    let parts = max_context_len.div_ceil(PARTITION);
    if seqs == 0 || seqs > 65_535
        || heads == 0 || kv_heads == 0 || blocks == 0
        || block_size == 0 || x == 0 || table_stride == 0
        || heads % kv_heads != 0
        || head_width.checked_mul(x) != Some(head_size)
        || blocks != vblocks || kv_heads != vheads
        || head_size != vhead_size || block_size != vblock_size
        || table_seqs != seqs || context_lens.dims() != [seqs]
        || max_context_len > table_stride.saturating_mul(block_size)
        || !matches!(head_size, 64 | 80 | 96 | 112 | 128 | 192 | 256 | 512)
        || [heads, kv_heads, head_size, block_size, x, table_stride, parts]
            .iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let cuda = q.device();
    let device = Device::Cuda(cuda.clone());
    if ![key_cache, value_cache, block_table, context_lens]
        .iter().all(|tensor| device.same_device(tensor.device()))
    {
        return Ok(None);
    }
    let output_elements = seqs.checked_mul(heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 output overflow"))?;
    let key = Key {
        dtype, seqs, heads, kv_heads, head_size,
        blocks, block_size, x, table_stride,
        max_context: max_context_len, parts,
        scale_bits: softmax_scale.to_bits(),
    };
    let (lse_offset, workspace_bytes) = workspace_layout(key)?;
    let (ks, kl) = key_cache.storage_and_layout();
    let (vs, vl) = value_cache.storage_and_layout();
    let (ts, tl) = block_table.storage_and_layout();
    let (ls, ll) = context_lens.storage_and_layout();
    let (Storage::Cuda(ks), Storage::Cuda(vs),
        Storage::Cuda(ts), Storage::Cuda(ls)) =
        (&*ks, &*vs, &*ts, &*ls)
    else {
        candle_core::bail!("DKR paged-v2 requires CUDA page and metadata tensors");
    };
    let stream = cuda.cuda_stream();
    // Shared with the historical v2 backend. Its guard serializes host
    // dispatches on a device while both kernels enqueue; capacity grows
    // geometrically, and old buffers survive for graph capture.
    let (workspace_ptr, _workspace_guard) = workspace_ensure(cuda, workspace_bytes)?;
    let partial_ptr = workspace_ptr;
    let lse_ptr = workspace_ptr
        .checked_add(u64::try_from(lse_offset).map_err(candle_core::Error::wrap)?)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 GPU offset overflow"))?;
    let num_seqs = u32::try_from(seqs).map_err(candle_core::Error::wrap)?;
    let num_heads = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let num_kv_heads = u32::try_from(kv_heads).map_err(candle_core::Error::wrap)?;
    let head_size_u32 = u32::try_from(head_size).map_err(candle_core::Error::wrap)?;
    let block_size_u32 = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let x_u32 = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    let table_stride_u32 = u32::try_from(table_stride).map_err(candle_core::Error::wrap)?;
    let max_context_u32 = u32::try_from(max_context_len).map_err(candle_core::Error::wrap)?;
    let parts_u32 = u32::try_from(parts).map_err(candle_core::Error::wrap)?;
    let mut output = None;
    let out_shape = q_layout.shape().clone();

    macro_rules! launch_stages {
        ($ty:ty) => {{
            let (qp, _qg) = slice_ptr_on_stream(
                q.as_cuda_slice::<$ty>()?, q_layout.start_offset(), &stream,
            );
            let (kp, _kg) = slice_ptr_on_stream(
                ks.as_cuda_slice::<$ty>()?, kl.start_offset(), &stream,
            );
            let (vp, _vg) = slice_ptr_on_stream(
                vs.as_cuda_slice::<$ty>()?, vl.start_offset(), &stream,
            );
            let (tp, _tg) = slice_ptr_on_stream(
                ts.as_cuda_slice::<u32>()?, tl.start_offset(), &stream,
            );
            let (lp, _lg) = slice_ptr_on_stream(
                ls.as_cuda_slice::<u32>()?, ll.start_offset(), &stream,
            );
            let mut result = unsafe { cuda.alloc::<$ty>(output_elements)? };
            {
                let (op, _og) = result.device_ptr_mut(&stream);
                PLANS.with(|plans| {
                    let mut plans = plans.borrow_mut();
                    plans.with_plan(
                        (key, Stage::Partition), &device,
                        || prepare(key, Stage::Partition, &device),
                        |plan| {
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&qp).arg(&kp).arg(&vp)
                                        .arg(&tp).arg(&lp)
                                        .arg(&partial_ptr).arg(&lse_ptr)
                                        .arg(&num_seqs).arg(&num_heads).arg(&num_kv_heads)
                                        .arg(&head_size_u32).arg(&block_size_u32).arg(&x_u32)
                                        .arg(&table_stride_u32).arg(&max_context_u32)
                                        .arg(&softmax_scale);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|error| candle_core::Error::msg(error.to_string()))
                        },
                    )?;
                    plans.with_plan(
                        (key, Stage::Merge), &device,
                        || prepare(key, Stage::Merge, &device),
                        |plan| {
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&partial_ptr).arg(&lse_ptr).arg(&op)
                                        .arg(&num_seqs).arg(&num_heads).arg(&head_size_u32)
                                        .arg(&parts_u32);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|error| candle_core::Error::msg(error.to_string()))
                        },
                    )
                })?;
            }
            output = Some(CudaStorage::wrap_cuda_slice(result, cuda.clone()));
        }};
    }
    match dtype {
        DType::F16 => launch_stages!(half::f16),
        DType::BF16 => launch_stages!(half::bf16),
        DType::F32 => launch_stages!(f32),
        _ => unreachable!(),
    }
    Ok(output.map(|storage| (storage, out_shape)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn token_key(i: usize) -> f32 {
        (i % 5) as f32 - 2.0
    }

    fn token_value(i: usize) -> f32 {
        ((i % 13) as f32 - 6.0) * 0.125
    }

    fn reference(context: usize, query_sign: f32) -> f32 {
        if context == 0 { return 0.0; }
        let logits: Vec<f32> = (0..context)
            .map(|i| query_sign * token_key(i))
            .collect();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (weighted, normalizer) = logits.iter().enumerate()
            .fold((0.0f64, 0.0f64), |(weighted, normalizer), (i, &logit)| {
                let weight = (logit - max_logit).exp() as f64;
                (weighted + weight * token_value(i) as f64, normalizer + weight)
            });
        (weighted / normalizer) as f32
    }

    #[test]
    #[ignore = "requires external staged SM61 v2 PTX and an NVIDIA GPU"]
    fn dkr_paged_v2_gpu() -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        const SEQS: usize = 2;
        const HEADS: usize = 2;
        const HD: usize = 64;
        const BLOCK_SIZE: usize = 16;
        const TABLE: usize = 50;
        const BLOCKS: usize = SEQS * TABLE;
        let contexts = [777usize, 513usize];
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let x = if dtype == DType::F32 { 4 } else { 8 };
            let mut queries = vec![0.0f32; SEQS * HEADS * HD];
            for seq in 0..SEQS {
                queries[(seq * HEADS) * HD] = 1.0;
                queries[(seq * HEADS + 1) * HD] = -1.0;
            }
            let mut keys = vec![0.0f32; BLOCKS * HD * BLOCK_SIZE];
            let mut values = vec![0.0f32; BLOCKS * HD * BLOCK_SIZE];
            for seq in 0..SEQS {
                for token in 0..TABLE * BLOCK_SIZE {
                    let page = seq * TABLE + token / BLOCK_SIZE;
                    let slot = token % BLOCK_SIZE;
                    keys[page * HD * BLOCK_SIZE + slot * x] = token_key(token);
                    for dim in 0..HD {
                        values[page * HD * BLOCK_SIZE + dim * BLOCK_SIZE + slot] =
                            token_value(token);
                    }
                }
            }
            let q = Tensor::from_vec(queries, (SEQS, HEADS, HD), &cuda)?.to_dtype(dtype)?;
            let kc = Tensor::from_vec(keys, (BLOCKS, 1, HD / x, BLOCK_SIZE, x), &cuda)?
                .to_dtype(dtype)?;
            let vc = Tensor::from_vec(values, (BLOCKS, 1, HD, BLOCK_SIZE), &cuda)?
                .to_dtype(dtype)?;
            let table = Tensor::from_vec(
                (0..SEQS * TABLE).map(|idx| idx as u32).collect(),
                (SEQS, TABLE), &cuda,
            )?;
            let lens = Tensor::from_vec(
                contexts.map(|n| n as u32).to_vec(), SEQS, &cuda,
            )?;
            let output = super::super::paged_attention::paged_attention(
                &q, None, None, &kc, &vc, &table, &lens,
                None, TABLE * BLOCK_SIZE, 1.0, 1.0, None,
            )?;
            let actual = output.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for seq in 0..SEQS {
                for head in 0..HEADS {
                    let expected = reference(contexts[seq], if head == 0 { 1.0 } else { -1.0 });
                    for dim in 0..HD {
                        let got = actual[(seq * HEADS + head) * HD + dim];
                        let tol = if dtype == DType::BF16 { 0.025 } else { 0.005 };
                        assert!(
                            (got - expected).abs() < tol,
                            "DKR paged v2 dtype={dtype:?} seq={seq} head={head} dim={dim}: \
                             got={got} expected={expected}"
                        );
                    }
                }
            }
            let cached = PLANS.with(|cache| cache.borrow().len());
            let again = super::super::paged_attention::paged_attention(
                &q, None, None, &kc, &vc, &table, &lens,
                None, TABLE * BLOCK_SIZE, 1.0, 1.0, None,
            )?;
            assert_eq!(cached, PLANS.with(|cache| cache.borrow().len()));
            assert_eq!(
                actual,
                again.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}
