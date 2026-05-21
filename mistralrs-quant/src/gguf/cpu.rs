//! CPU implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This dequantizes only the experts selected by the router for this call.

use byteorder::{ByteOrder, LittleEndian};
use candle_core::{
    quantized::{
        ggml_file::qtensor_from_ggml,
        k_quants::{BlockQ4K, BlockQ8K, GgmlType, QK_K},
        GgmlDType, QMatMul, QTensor,
    },
    DType, Device, IndexOp, Result, Tensor,
};
use candle_nn::Linear;
use half::f16;
use rayon::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

type ExpertCacheKey = (usize, usize, u8);
type Q4_1ExpertCacheKey = (usize, usize);
type Q4KExpertCacheKey = (usize, usize);
type Q4KMatmulCacheKey = usize;

const DEFAULT_DEQUANT_EXPERT_CACHE_LIMIT: usize = 0;
const DEFAULT_Q4_1_EXPERT_CACHE_LIMIT: usize = 256;
const DEFAULT_Q4K_EXPERT_CACHE_LIMIT: usize = 1024;

#[derive(Default)]
struct ExpertCache {
    map: HashMap<ExpertCacheKey, Tensor>,
    order: VecDeque<ExpertCacheKey>,
}

static EXPERT_CACHE: OnceLock<Mutex<ExpertCache>> = OnceLock::new();
static Q4_1_EXPERT_CACHE: OnceLock<Mutex<Q4_1ExpertCache>> = OnceLock::new();
static Q4K_EXPERT_CACHE: OnceLock<Mutex<Q4KExpertCache>> = OnceLock::new();
static Q4K_MATMUL_CACHE: OnceLock<Mutex<Q4KMatmulCache>> = OnceLock::new();
static LOG_GGUF_CPU_MOE_FALLBACK: AtomicBool = AtomicBool::new(false);
static LOG_Q4_1_CPU_MOE: AtomicBool = AtomicBool::new(false);
static LOG_Q4K_CPU_MOE: AtomicBool = AtomicBool::new(false);
static LOG_Q4K_FUSED_CPU_MOE: AtomicBool = AtomicBool::new(false);
static LOG_Q4K_CPU_MATMUL: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Q4_1ExpertCache {
    map: HashMap<Q4_1ExpertCacheKey, Arc<Vec<RawQ4_1Block>>>,
    order: VecDeque<Q4_1ExpertCacheKey>,
}

#[derive(Default)]
struct Q4KExpertCache {
    map: HashMap<Q4KExpertCacheKey, Arc<Vec<RawQ4KBlock>>>,
    order: VecDeque<Q4KExpertCacheKey>,
}

#[derive(Default)]
struct Q4KMatmulCache {
    map: HashMap<Q4KMatmulCacheKey, Arc<Vec<RawQ4KBlock>>>,
    order: VecDeque<Q4KMatmulCacheKey>,
}

#[derive(Clone, Copy)]
struct RawQ4_1Block {
    d: f32,
    m: f32,
    qs: [u8; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawQ4KBlock {
    d: f16,
    dmin: f16,
    scales: [u8; 12],
    qs: [u8; 128],
}

struct RawQ8_1Block {
    d: f32,
    s: f32,
    qs: [i8; 32],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawQ8KBlock {
    d: f32,
    qs: [i8; QK_K],
    bsums: [i16; QK_K / 16],
}

struct ExpertTask {
    expert_id: usize,
    q8_idx: usize,
    out_offset: usize,
}

struct Q4KRouteExperts {
    gate: Arc<Vec<RawQ4KBlock>>,
    up: Arc<Vec<RawQ4KBlock>>,
    down: Arc<Vec<RawQ4KBlock>>,
    weight: f32,
}

fn expert_cache_limit() -> usize {
    if let Some(limit) = crate::gguf_cpu_moe_expert_cache_override() {
        return limit;
    }
    std::env::var("MISTRALRS_GGUF_CPU_MOE_EXPERT_CACHE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_DEQUANT_EXPERT_CACHE_LIMIT)
}

fn dtype_key(dtype: DType) -> u8 {
    match dtype {
        DType::U8 => 0,
        DType::U32 => 1,
        DType::I64 => 2,
        DType::BF16 => 3,
        DType::F16 => 4,
        DType::F32 => 5,
        DType::F64 => 6,
        _ => 255,
    }
}

fn cached_expert(
    key: ExpertCacheKey,
    make_expert: impl FnOnce() -> Result<Tensor>,
) -> Result<Tensor> {
    let cache = EXPERT_CACHE.get_or_init(|| Mutex::new(ExpertCache::default()));
    if let Some(expert) = cache.lock().unwrap().map.get(&key).cloned() {
        return Ok(expert);
    }

    let expert = make_expert()?;
    let limit = expert_cache_limit();
    if limit != 0 {
        let mut cache = cache.lock().unwrap();
        if let Some(expert) = cache.map.get(&key).cloned() {
            return Ok(expert);
        }
        cache.map.insert(key, expert.clone());
        cache.order.push_back(key);
        while cache.map.len() > limit {
            if let Some(old_key) = cache.order.pop_front() {
                cache.map.remove(&old_key);
            } else {
                break;
            }
        }
    }
    Ok(expert)
}

fn q4_1_expert_cache_limit() -> usize {
    if let Some(limit) = crate::gguf_cpu_moe_q4_1_expert_cache_override() {
        return limit;
    }
    std::env::var("MISTRALRS_GGUF_CPU_MOE_Q4_1_EXPERT_CACHE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_Q4_1_EXPERT_CACHE_LIMIT)
}

fn cached_q4_1_expert(
    key: Q4_1ExpertCacheKey,
    make_expert: impl FnOnce() -> Result<Vec<RawQ4_1Block>>,
) -> Result<Arc<Vec<RawQ4_1Block>>> {
    let cache = Q4_1_EXPERT_CACHE.get_or_init(|| Mutex::new(Q4_1ExpertCache::default()));
    if let Some(expert) = cache.lock().unwrap().map.get(&key).cloned() {
        return Ok(expert);
    }

    let expert = Arc::new(make_expert()?);
    let limit = q4_1_expert_cache_limit();
    if limit != 0 {
        let mut cache = cache.lock().unwrap();
        if let Some(expert) = cache.map.get(&key).cloned() {
            return Ok(expert);
        }
        cache.map.insert(key, expert.clone());
        cache.order.push_back(key);
        while cache.map.len() > limit {
            if let Some(old_key) = cache.order.pop_front() {
                cache.map.remove(&old_key);
            } else {
                break;
            }
        }
    }
    Ok(expert)
}

fn q4k_expert_cache_limit() -> usize {
    if let Some(limit) = crate::gguf_cpu_moe_q4k_expert_cache_override() {
        return limit;
    }
    std::env::var("MISTRALRS_GGUF_CPU_MOE_Q4K_EXPERT_CACHE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_Q4K_EXPERT_CACHE_LIMIT)
}

fn cached_q4k_expert(
    key: Q4KExpertCacheKey,
    make_expert: impl FnOnce() -> Result<Vec<RawQ4KBlock>>,
) -> Result<Arc<Vec<RawQ4KBlock>>> {
    let cache = Q4K_EXPERT_CACHE.get_or_init(|| Mutex::new(Q4KExpertCache::default()));
    if let Some(expert) = cache.lock().unwrap().map.get(&key).cloned() {
        return Ok(expert);
    }

    let expert = Arc::new(make_expert()?);
    let limit = q4k_expert_cache_limit();
    if limit != 0 {
        let mut cache = cache.lock().unwrap();
        if let Some(expert) = cache.map.get(&key).cloned() {
            return Ok(expert);
        }
        cache.map.insert(key, expert.clone());
        cache.order.push_back(key);
        while cache.map.len() > limit {
            if let Some(old_key) = cache.order.pop_front() {
                cache.map.remove(&old_key);
            } else {
                break;
            }
        }
    }
    Ok(expert)
}

fn q4k_matmul_cache_limit() -> usize {
    if let Some(limit) = crate::gguf_cpu_q4k_matmul_cache_override() {
        return limit;
    }
    std::env::var("MISTRALRS_GGUF_CPU_Q4K_MATMUL_CACHE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

fn q4k_matmul_enabled() -> bool {
    if let Some(enabled) = crate::gguf_cpu_q4k_matmul_override() {
        return enabled;
    }
    std::env::var("MISTRALRS_GGUF_CPU_Q4K_MATMUL")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn q4k_fused_moe_parallel_topk_enabled() -> bool {
    if let Some(enabled) = crate::gguf_cpu_moe_parallel_topk_override() {
        return enabled;
    }
    std::env::var("MISTRALRS_GGUF_CPU_MOE_PARALLEL_TOPK")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn q4k_matmul_max_rows() -> usize {
    if let Some(max_rows) = crate::gguf_cpu_q4k_matmul_max_rows_override() {
        return max_rows;
    }
    std::env::var("MISTRALRS_GGUF_CPU_Q4K_MATMUL_MAX_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

fn cached_q4k_matmul(
    key: Q4KMatmulCacheKey,
    make_blocks: impl FnOnce() -> Result<Vec<RawQ4KBlock>>,
) -> Result<Arc<Vec<RawQ4KBlock>>> {
    let cache = Q4K_MATMUL_CACHE.get_or_init(|| Mutex::new(Q4KMatmulCache::default()));
    if let Some(blocks) = cache.lock().unwrap().map.get(&key).cloned() {
        return Ok(blocks);
    }

    let blocks = Arc::new(make_blocks()?);
    let limit = q4k_matmul_cache_limit();
    if limit != 0 {
        let mut cache = cache.lock().unwrap();
        if let Some(blocks) = cache.map.get(&key).cloned() {
            return Ok(blocks);
        }
        cache.map.insert(key, blocks.clone());
        cache.order.push_back(key);
        while cache.map.len() > limit {
            if let Some(old_key) = cache.order.pop_front() {
                cache.map.remove(&old_key);
            } else {
                break;
            }
        }
    }
    Ok(blocks)
}

fn q4_1_blocks_from_raw(raw: &[u8]) -> Result<Vec<RawQ4_1Block>> {
    let block_bytes = 20;
    if raw.len() % block_bytes != 0 {
        candle_core::bail!(
            "Q4_1 raw byte length {} is not divisible by block size {block_bytes}",
            raw.len()
        );
    }
    let mut blocks = Vec::with_capacity(raw.len() / block_bytes);
    for chunk in raw.chunks_exact(block_bytes) {
        let d = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
        let m = f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32();
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&chunk[4..20]);
        blocks.push(RawQ4_1Block { d, m, qs });
    }
    Ok(blocks)
}

fn q4k_blocks_from_raw(raw: &[u8]) -> Result<Vec<RawQ4KBlock>> {
    let block_bytes = std::mem::size_of::<RawQ4KBlock>();
    if raw.len() % block_bytes != 0 {
        candle_core::bail!(
            "Q4K raw byte length {} is not divisible by block size {block_bytes}",
            raw.len()
        );
    }
    let mut blocks = Vec::with_capacity(raw.len() / block_bytes);
    for chunk in raw.chunks_exact(block_bytes) {
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&chunk[4..16]);
        let mut qs = [0u8; 128];
        qs.copy_from_slice(&chunk[16..144]);
        blocks.push(RawQ4KBlock {
            d: f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])),
            dmin: f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])),
            scales,
            qs,
        });
    }
    Ok(blocks)
}

fn q8_1_from_row(row: &Tensor, in_features: usize) -> Result<Vec<RawQ8_1Block>> {
    let row = row.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let row = row.to_vec1::<f32>()?;
    if row.len() != in_features {
        candle_core::bail!(
            "Q4_1 CPU indexed MoE input row has {} values, expected {in_features}",
            row.len()
        );
    }
    let mut q8 = Vec::with_capacity(in_features / 32);
    for xs in row.chunks_exact(32) {
        let amax = xs.iter().fold(0f32, |acc, &x| acc.max(x.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qs = [0i8; 32];
        let mut sum = 0i32;
        for (q, &x) in qs.iter_mut().zip(xs.iter()) {
            let v = (x * id).round().clamp(-128.0, 127.0) as i8;
            *q = v;
            sum += v as i32;
        }
        q8.push(RawQ8_1Block {
            d,
            s: sum as f32 * d,
            qs,
        });
    }
    Ok(q8)
}

fn q8k_from_row(row: &Tensor, in_features: usize) -> Result<Vec<RawQ8KBlock>> {
    let row = row.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let row = row.to_vec1::<f32>()?;
    q8k_from_slice(&row, in_features)
}

fn q8k_from_slice(row: &[f32], in_features: usize) -> Result<Vec<RawQ8KBlock>> {
    if row.len() != in_features {
        candle_core::bail!(
            "Q4K CPU indexed MoE input row has {} values, expected {in_features}",
            row.len()
        );
    }
    let mut q8 = Vec::with_capacity(in_features / QK_K);
    for xs in row.chunks_exact(QK_K) {
        let mut max = 0f32;
        let mut amax = 0f32;
        for &x in xs.iter() {
            if amax < x.abs() {
                amax = x.abs();
                max = x;
            }
        }
        let mut block = RawQ8KBlock {
            d: 0.0,
            qs: [0; QK_K],
            bsums: [0; QK_K / 16],
        };
        if amax != 0.0 {
            let iscale = -128f32 / max;
            for (j, q) in block.qs.iter_mut().enumerate() {
                let v = (iscale * xs[j]).round();
                *q = v.min(127.0) as i8;
            }
            for j in 0..QK_K / 16 {
                let mut sum = 0i32;
                for ii in 0..16 {
                    sum += block.qs[j * 16 + ii] as i32;
                }
                block.bsums[j] = sum as i16;
            }
            block.d = 1.0 / iscale;
        }
        q8.push(block);
    }
    Ok(q8)
}

fn q4k_cpu_qtensor(qmatmul: &QMatMul) -> Option<&Arc<QTensor>> {
    let QMatMul::QTensor(qtensor) = qmatmul else {
        return None;
    };
    if qtensor.dtype() == GgmlDType::Q4K && qtensor.device().is_cpu() {
        Some(qtensor)
    } else {
        None
    }
}

fn q4k_qtensor_shape(qtensor: &QTensor) -> Result<(usize, usize, usize)> {
    let &[num_experts, out_features, in_features] = qtensor.shape().dims() else {
        candle_core::bail!(
            "GGUF CPU fused MoE expects weights [experts, out, in], got {:?}",
            qtensor.shape().dims()
        );
    };
    if in_features % QK_K != 0 {
        candle_core::bail!(
            "GGUF CPU fused MoE expects in_features {in_features} divisible by {QK_K}"
        );
    }
    Ok((num_experts, out_features, in_features))
}

pub(crate) fn cpu_q4k_matmul(qmatmul: &QMatMul, xs: &Tensor) -> Result<Option<Tensor>> {
    if !q4k_matmul_enabled() {
        return Ok(None);
    }
    let Some(qtensor) = q4k_cpu_qtensor(qmatmul) else {
        return Ok(None);
    };
    if !xs.device().is_cpu() {
        return Ok(None);
    }

    let &[out_features, in_features] = qtensor.shape().dims() else {
        return Ok(None);
    };
    if in_features % QK_K != 0 {
        candle_core::bail!(
            "GGUF CPU Q4K matmul expects in_features {in_features} divisible by {QK_K}"
        );
    }

    let xs_dims = xs.dims();
    if xs_dims.len() < 2 {
        return Ok(None);
    }
    if xs_dims[xs_dims.len() - 1] != in_features {
        return Ok(None);
    }

    let num_rows = xs_dims[..xs_dims.len() - 1].iter().product::<usize>();
    if num_rows > q4k_matmul_max_rows() {
        return Ok(None);
    }

    if !LOG_Q4K_CPU_MATMUL.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using GGUF CPU Q4K/Q8K matmul path (out={out_features}, in={in_features}, max_rows={})",
            q4k_matmul_max_rows()
        );
    }

    let raw = qtensor.data()?;
    let tensor_key = Arc::as_ptr(qtensor) as usize;
    let blocks = cached_q4k_matmul(tensor_key, || q4k_blocks_from_raw(&raw))?;
    let blocks_per_row = in_features / QK_K;
    if blocks.len() != out_features * blocks_per_row {
        candle_core::bail!(
            "GGUF CPU Q4K matmul block count {} does not match shape [{out_features}, {in_features}]",
            blocks.len()
        );
    }

    let xs_cpu = xs
        .contiguous()?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let xs_flat = xs_cpu.flatten_all()?.to_vec1::<f32>()?;
    let mut output = vec![0f32; num_rows * out_features];

    for row_idx in 0..num_rows {
        let row_start = row_idx * in_features;
        let row_end = row_start + in_features;
        let x_q8 = q8k_from_slice(&xs_flat[row_start..row_end], in_features)?;
        let out_base = row_idx * out_features;
        let out_slice = &mut output[out_base..out_base + out_features];

        if out_features >= 1024 {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(out_idx, out)| {
                    let start = out_idx * blocks_per_row;
                    let end = start + blocks_per_row;
                    *out = q4k_q8k_dot(&blocks[start..end], &x_q8, in_features);
                });
        } else {
            for (out_idx, out) in out_slice.iter_mut().enumerate() {
                let start = out_idx * blocks_per_row;
                let end = start + blocks_per_row;
                *out = q4k_q8k_dot(&blocks[start..end], &x_q8, in_features);
            }
        }
    }

    let mut out_dims = xs_dims.to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = out_features;
    Tensor::from_vec(output, out_dims, xs.device()).map(Some)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_fused_moe_q4k_forward<F>(
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    act: F,
) -> Result<Option<Tensor>>
where
    F: Fn(f32) -> f32 + Copy + Send + Sync,
{
    let Some(gate) = q4k_cpu_qtensor(gate) else {
        return Ok(None);
    };
    let Some(up) = q4k_cpu_qtensor(up) else {
        return Ok(None);
    };
    let Some(down) = q4k_cpu_qtensor(down) else {
        return Ok(None);
    };

    let (num_experts, inter, hidden) = q4k_qtensor_shape(gate)?;
    let (up_num_experts, up_inter, up_hidden) = q4k_qtensor_shape(up)?;
    let (down_num_experts, down_hidden, down_inter) = q4k_qtensor_shape(down)?;
    if (up_num_experts, up_inter, up_hidden) != (num_experts, inter, hidden)
        || (down_num_experts, down_hidden, down_inter) != (num_experts, hidden, inter)
    {
        candle_core::bail!(
            "GGUF CPU fused MoE shape mismatch gate={:?} up={:?} down={:?}",
            gate.shape().dims(),
            up.shape().dims(),
            down.shape().dims()
        );
    }

    let (b_size, seq_len, xs_hidden) = xs.dims3()?;
    if xs_hidden != hidden {
        candle_core::bail!("GGUF CPU fused MoE input hidden {xs_hidden} != weight hidden {hidden}");
    }
    let num_tokens = b_size * seq_len;
    let (ids_tokens, topk) = topk_ids.dims2()?;
    if ids_tokens != num_tokens {
        candle_core::bail!(
            "GGUF CPU fused MoE ids tokens {ids_tokens} != input tokens {num_tokens}"
        );
    }
    let weights_dims = topk_weights.dims();
    if weights_dims != [num_tokens, topk] {
        candle_core::bail!(
            "GGUF CPU fused MoE topk weights shape {:?} != [{num_tokens}, {topk}]",
            weights_dims
        );
    }

    let parallel_topk = q4k_fused_moe_parallel_topk_enabled() && topk > 1;
    if !LOG_Q4K_FUSED_CPU_MOE.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using fused GGUF CPU MoE Q4K/Q8K path (experts={num_experts}, topk={topk}, hidden={hidden}, inter={inter}, parallel_topk={parallel_topk})"
        );
    }

    let xs_cpu = xs
        .reshape((num_tokens, hidden))?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let xs_rows = xs_cpu.to_vec2::<f32>()?;
    let ids_vec = topk_ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    let weights_vec = topk_weights
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;

    let gate_raw = gate.data()?;
    let up_raw = up.data()?;
    let down_raw = down.data()?;
    let gate_key = Arc::as_ptr(gate) as usize;
    let up_key = Arc::as_ptr(up) as usize;
    let down_key = Arc::as_ptr(down) as usize;
    let blocks_per_hidden_row = hidden / QK_K;
    let blocks_per_inter_row = inter / QK_K;
    let gate_expert_bytes = inter * blocks_per_hidden_row * std::mem::size_of::<RawQ4KBlock>();
    let down_expert_bytes = hidden * blocks_per_inter_row * std::mem::size_of::<RawQ4KBlock>();
    let mut output = vec![0f32; num_tokens * hidden];

    for token_idx in 0..num_tokens {
        let x_q8 = q8k_from_slice(&xs_rows[token_idx], hidden)?;
        let mut routes = Vec::with_capacity(topk);
        for slot_idx in 0..topk {
            let expert_id = ids_vec[token_idx][slot_idx] as usize;
            if expert_id >= num_experts {
                candle_core::bail!(
                    "GGUF CPU fused MoE expert id {expert_id} out of range {num_experts}"
                );
            }

            let gate_expert = cached_q4k_expert((gate_key, expert_id), || {
                let start = expert_id * gate_expert_bytes;
                let end = start + gate_expert_bytes;
                q4k_blocks_from_raw(&gate_raw[start..end])
            })?;
            let up_expert = cached_q4k_expert((up_key, expert_id), || {
                let start = expert_id * gate_expert_bytes;
                let end = start + gate_expert_bytes;
                q4k_blocks_from_raw(&up_raw[start..end])
            })?;
            let down_expert = cached_q4k_expert((down_key, expert_id), || {
                let start = expert_id * down_expert_bytes;
                let end = start + down_expert_bytes;
                q4k_blocks_from_raw(&down_raw[start..end])
            })?;
            routes.push(Q4KRouteExperts {
                gate: gate_expert,
                up: up_expert,
                down: down_expert,
                weight: weights_vec[token_idx][slot_idx],
            });
        }

        if parallel_topk {
            let mut slot_hidden = vec![0f32; topk * inter];
            let mut slot_outputs = vec![0f32; topk * hidden];
            slot_outputs
                .par_chunks_mut(hidden)
                .zip(slot_hidden.par_chunks_mut(inter))
                .zip(routes.par_iter())
                .try_for_each(|((slot_out, hidden_inter), route)| -> Result<()> {
                    for (row_idx, out) in hidden_inter.iter_mut().enumerate() {
                        let start = row_idx * blocks_per_hidden_row;
                        let end = start + blocks_per_hidden_row;
                        let gate_v = q4k_q8k_dot(&route.gate[start..end], &x_q8, hidden);
                        let up_v = q4k_q8k_dot(&route.up[start..end], &x_q8, hidden);
                        *out = up_v * act(gate_v);
                    }

                    let inter_q8 = q8k_from_slice(&hidden_inter, inter)?;
                    for (row_idx, out) in slot_out.iter_mut().enumerate() {
                        let start = row_idx * blocks_per_inter_row;
                        let end = start + blocks_per_inter_row;
                        *out =
                            route.weight * q4k_q8k_dot(&route.down[start..end], &inter_q8, inter);
                    }

                    Ok(())
                })?;

            let out_base = token_idx * hidden;
            for slot_out in slot_outputs.chunks_exact(hidden) {
                for (row_idx, &value) in slot_out.iter().enumerate() {
                    output[out_base + row_idx] += value;
                }
            }
            continue;
        }

        let mut hidden_inter = vec![0f32; inter];
        for route in routes.iter() {
            for (row_idx, out) in hidden_inter.iter_mut().enumerate() {
                let start = row_idx * blocks_per_hidden_row;
                let end = start + blocks_per_hidden_row;
                let gate_v = q4k_q8k_dot(&route.gate[start..end], &x_q8, hidden);
                let up_v = q4k_q8k_dot(&route.up[start..end], &x_q8, hidden);
                *out = up_v * act(gate_v);
            }

            let inter_q8 = q8k_from_slice(&hidden_inter, inter)?;
            let out_base = token_idx * hidden;
            for row_idx in 0..hidden {
                let start = row_idx * blocks_per_inter_row;
                let end = start + blocks_per_inter_row;
                output[out_base + row_idx] +=
                    route.weight * q4k_q8k_dot(&route.down[start..end], &inter_q8, inter);
            }
        }
    }

    Tensor::from_vec(output, (num_tokens, hidden), xs.device())?
        .to_dtype(xs.dtype())
        .map(Some)
}

fn q4_1_q8_1_dot(xs: &[RawQ4_1Block], ys: &[RawQ8_1Block]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { q4_1_q8_1_dot_avx2(xs, ys) };
        }
    }

    q4_1_q8_1_dot_scalar(xs, ys)
}

fn q4_1_q8_1_dot_scalar(xs: &[RawQ4_1Block], ys: &[RawQ8_1Block]) -> f32 {
    let mut sumf = 0f32;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let mut sumi = 0i32;
        for j in 0..16 {
            let packed = x.qs[j];
            let v0 = (packed & 0x0f) as i32;
            let v1 = (packed >> 4) as i32;
            sumi += v0 * y.qs[j] as i32 + v1 * y.qs[j + 16] as i32;
        }
        sumf += sumi as f32 * x.d * y.d + x.m * y.s;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4_1_q8_1_dot_avx2(xs: &[RawQ4_1Block], ys: &[RawQ8_1Block]) -> f32 {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_maddubs_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm_add_epi32,
        _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_srli_si128,
    };

    let ones = _mm256_set1_epi16(1);
    let mut sumf = 0f32;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let mut q4 = [0u8; 32];
        for j in 0..16 {
            let packed = x.qs[j];
            q4[j] = packed & 0x0f;
            q4[j + 16] = packed >> 4;
        }

        let q4v = _mm256_loadu_si256(q4.as_ptr().cast::<__m256i>());
        let q8v = _mm256_loadu_si256(y.qs.as_ptr().cast::<__m256i>());
        let pair_sums = _mm256_maddubs_epi16(q4v, q8v);
        let i32_sums = _mm256_madd_epi16(pair_sums, ones);
        let i32_sums = _mm256_add_epi32(i32_sums, _mm256_setzero_si256());
        let low = _mm256_extracti128_si256::<0>(i32_sums);
        let high = _mm256_extracti128_si256::<1>(i32_sums);
        let mut sum = _mm_add_epi32(low, high);
        sum = _mm_add_epi32(sum, _mm_shuffle_epi32::<0b1110_1110>(sum));
        sum = _mm_add_epi32(sum, _mm_srli_si128::<4>(sum));
        let sumi = _mm_cvtsi128_si32(sum);

        sumf += sumi as f32 * x.d * y.d + x.m * y.s;
    }
    sumf
}

fn qtensor_indexed_moe_forward_q4_1(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids_vec: &[Vec<u32>],
    num_tokens: usize,
    num_experts_per_tok: usize,
    x_slots: usize,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    raw: &[u8],
) -> Result<Tensor> {
    if !LOG_Q4_1_CPU_MOE.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using GGUF CPU indexed MoE Q4_1/Q8_1 dot path (experts={num_experts}, out={out_features}, in={in_features})"
        );
    }

    let blocks_per_row = in_features / 32;
    let blocks_per_expert = out_features * blocks_per_row;
    let expert_bytes = blocks_per_expert * 20;
    let tensor_key = Arc::as_ptr(qtensor) as usize;
    let mut q8_rows = Vec::with_capacity(num_tokens * x_slots);
    let mut tasks = Vec::with_capacity(num_tokens * num_experts_per_tok);

    for token_idx in 0..num_tokens {
        let token_q8_base = q8_rows.len();
        if x_slots == 1 {
            q8_rows.push(q8_1_from_row(&x.i((token_idx, 0))?, in_features)?);
        } else {
            for slot_idx in 0..num_experts_per_tok {
                q8_rows.push(q8_1_from_row(&x.i((token_idx, slot_idx))?, in_features)?);
            }
        }

        for (slot_idx, &expert_id) in ids_vec[token_idx].iter().enumerate() {
            let expert_id = expert_id as usize;
            if expert_id >= num_experts {
                candle_core::bail!(
                    "GGUF CPU indexed MoE expert id {expert_id} out of range {num_experts}"
                );
            }
            tasks.push(ExpertTask {
                expert_id,
                q8_idx: token_q8_base + if x_slots == 1 { 0 } else { slot_idx },
                out_offset: (token_idx * num_experts_per_tok + slot_idx) * out_features,
            });
        }
    }
    tasks.sort_unstable_by_key(|task| task.expert_id);

    let mut rows = vec![0f32; num_tokens * num_experts_per_tok * out_features];
    let mut task_idx = 0;
    while task_idx < tasks.len() {
        let expert_id = tasks[task_idx].expert_id;
        let expert = cached_q4_1_expert((tensor_key, expert_id), || {
            let start = expert_id * expert_bytes;
            let end = start + expert_bytes;
            q4_1_blocks_from_raw(&raw[start..end])
        })?;
        while task_idx < tasks.len() && tasks[task_idx].expert_id == expert_id {
            let task = &tasks[task_idx];
            let q8 = &q8_rows[task.q8_idx];
            let out_slice = &mut rows[task.out_offset..task.out_offset + out_features];
            for (row_idx, out) in out_slice.iter_mut().enumerate() {
                let start = row_idx * blocks_per_row;
                let end = start + blocks_per_row;
                *out = q4_1_q8_1_dot(&expert[start..end], q8);
            }
            task_idx += 1;
        }
    }

    Tensor::from_vec(
        rows,
        (num_tokens, num_experts_per_tok, out_features),
        x.device(),
    )?
    .to_dtype(x.dtype())
}

fn q4k_q8k_dot(xs: &[RawQ4KBlock], ys: &[RawQ8KBlock], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { q4k_q8k_dot_avx2(xs, ys) };
        }
    }

    let xs = unsafe { std::slice::from_raw_parts(xs.as_ptr().cast::<BlockQ4K>(), xs.len()) };
    let ys = unsafe { std::slice::from_raw_parts(ys.as_ptr().cast::<BlockQ8K>(), ys.len()) };
    <BlockQ4K as GgmlType>::vec_dot(n, xs, ys)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_q8k_dot_avx2(xs: &[RawQ4KBlock], ys: &[RawQ8KBlock]) -> f32 {
    use std::arch::x86_64::{
        __m128i, __m256, __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_castps256_ps128,
        _mm256_castsi128_si256, _mm256_cvtepi32_ps, _mm256_cvtepu8_epi16, _mm256_extractf128_ps,
        _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_insertf128_si256, _mm256_loadu_si256,
        _mm256_madd_epi16, _mm256_maddubs_epi16, _mm256_set1_epi8, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
        _mm_add_ps, _mm_add_ss, _mm_cvtepi32_ps, _mm_cvtss_f32, _mm_fmadd_ps, _mm_hadd_epi16,
        _mm_madd_epi16, _mm_movehdup_ps, _mm_movehl_ps, _mm_set1_ps, _mm_set_epi32, _mm_setzero_ps,
    };

    #[inline(always)]
    unsafe fn hsum_float_8(x: __m256) -> f32 {
        let res = _mm256_extractf128_ps(x, 1);
        let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
        let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        let res = _mm_add_ss(res, _mm_movehdup_ps(res));
        _mm_cvtss_f32(res)
    }

    #[inline(always)]
    unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
        _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
    }

    #[inline(always)]
    unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
        const K_SHUFFLE: [u8; 256] = [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
            1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
            2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4,
            5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
            6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8,
            9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ];
        _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
    }

    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = _mm_setzero_ps();

    for (x, y) in xs.iter().zip(ys.iter()) {
        let d = y.d * x.d.to_f32();
        let dmin = -y.d * x.dmin.to_f32();

        LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        let mut q4 = x.qs.as_ptr();
        let mut q8 = y.qs.as_ptr();

        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));

        let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
        let q8s = _mm_hadd_epi16(
            _mm256_extracti128_si256(q8sums, 0),
            _mm256_extracti128_si256(q8sums, 1),
        );
        let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        let mut sumi = _mm256_setzero_si256();

        for j in 0..QK_K / 64 {
            let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
            let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            let q8l = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let p16l = _mm256_maddubs_epi16(q4l, q8l);
            let p16l = _mm256_madd_epi16(scale_l, p16l);
            sumi = _mm256_add_epi32(sumi, p16l);

            let q8h = _mm256_loadu_si256(q8 as *const __m256i);
            q8 = q8.add(32);
            let p16h = _mm256_maddubs_epi16(q4h, q8h);
            let p16h = _mm256_madd_epi16(scale_h, p16h);
            sumi = _mm256_add_epi32(sumi, p16h);
        }

        let vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
    }

    let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
}

#[allow(clippy::too_many_arguments)]
fn qtensor_indexed_moe_forward_q4k(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids_vec: &[Vec<u32>],
    num_tokens: usize,
    num_experts_per_tok: usize,
    x_slots: usize,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    raw: &[u8],
) -> Result<Tensor> {
    if !LOG_Q4K_CPU_MOE.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using GGUF CPU indexed MoE Q4K/Q8K dot path (experts={num_experts}, out={out_features}, in={in_features})"
        );
    }

    let blocks_per_row = in_features / QK_K;
    let blocks_per_expert = out_features * blocks_per_row;
    let expert_bytes = blocks_per_expert * std::mem::size_of::<RawQ4KBlock>();
    let tensor_key = Arc::as_ptr(qtensor) as usize;
    let mut q8_rows = Vec::with_capacity(num_tokens * x_slots);
    let mut tasks = Vec::with_capacity(num_tokens * num_experts_per_tok);

    for token_idx in 0..num_tokens {
        let token_q8_base = q8_rows.len();
        if x_slots == 1 {
            q8_rows.push(q8k_from_row(&x.i((token_idx, 0))?, in_features)?);
        } else {
            for slot_idx in 0..num_experts_per_tok {
                q8_rows.push(q8k_from_row(&x.i((token_idx, slot_idx))?, in_features)?);
            }
        }

        for (slot_idx, &expert_id) in ids_vec[token_idx].iter().enumerate() {
            let expert_id = expert_id as usize;
            if expert_id >= num_experts {
                candle_core::bail!(
                    "GGUF CPU indexed MoE expert id {expert_id} out of range {num_experts}"
                );
            }
            tasks.push(ExpertTask {
                expert_id,
                q8_idx: token_q8_base + if x_slots == 1 { 0 } else { slot_idx },
                out_offset: (token_idx * num_experts_per_tok + slot_idx) * out_features,
            });
        }
    }
    tasks.sort_unstable_by_key(|task| task.expert_id);

    let mut rows = vec![0f32; num_tokens * num_experts_per_tok * out_features];
    let mut task_idx = 0;
    while task_idx < tasks.len() {
        let expert_id = tasks[task_idx].expert_id;
        let expert = cached_q4k_expert((tensor_key, expert_id), || {
            let start = expert_id * expert_bytes;
            let end = start + expert_bytes;
            q4k_blocks_from_raw(&raw[start..end])
        })?;
        while task_idx < tasks.len() && tasks[task_idx].expert_id == expert_id {
            let task = &tasks[task_idx];
            let q8 = &q8_rows[task.q8_idx];
            let out_slice = &mut rows[task.out_offset..task.out_offset + out_features];
            for (row_idx, out) in out_slice.iter_mut().enumerate() {
                let start = row_idx * blocks_per_row;
                let end = start + blocks_per_row;
                *out = q4k_q8k_dot(&expert[start..end], q8, in_features);
            }
            task_idx += 1;
        }
    }

    Tensor::from_vec(
        rows,
        (num_tokens, num_experts_per_tok, out_features),
        x.device(),
    )?
    .to_dtype(x.dtype())
}

/// Perform indexed MoE forward pass on a QTensor by dequantizing only selected experts.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(
    qtensor: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let shape = qtensor.shape().dims();
    let &[num_experts, out_features, in_features] = shape else {
        candle_core::bail!(
            "GGUF CPU indexed MoE expects weights [experts, out, in], got {:?}",
            shape
        );
    };
    let block_size = qtensor.dtype().block_size();
    if in_features % block_size != 0 {
        candle_core::bail!(
            "GGUF CPU indexed MoE expects in_features {in_features} divisible by block size {block_size}"
        );
    }

    let (num_tokens, num_experts_per_tok, x_slots) = match x.dims() {
        &[tokens, 1, hidden_dim] if hidden_dim == in_features => {
            let (_ids_tokens, topk) = ids.dims2()?;
            (tokens, topk, 1)
        }
        &[tokens, topk, hidden_dim] if hidden_dim == in_features => {
            let (_ids_tokens, ids_topk) = ids.dims2()?;
            if ids_topk != topk {
                candle_core::bail!("GGUF CPU indexed MoE input topk {topk} != ids topk {ids_topk}");
            }
            (tokens, topk, topk)
        }
        dims => {
            candle_core::bail!("GGUF CPU indexed MoE unsupported input shape {:?}", dims);
        }
    };
    let (ids_tokens, ids_topk) = ids.dims2()?;
    if ids_tokens != num_tokens || ids_topk != num_experts_per_tok {
        candle_core::bail!(
            "GGUF CPU indexed MoE ids shape {:?} does not match input ({num_tokens}, {num_experts_per_tok})",
            ids.dims()
        );
    }

    let ids_cpu = ids.to_device(&Device::Cpu)?;
    let ids_vec = ids_cpu.to_vec2::<u32>()?;
    let raw = qtensor.data()?;
    if qtensor.dtype() == GgmlDType::Q4_1 && x.device().is_cpu() {
        return qtensor_indexed_moe_forward_q4_1(
            qtensor,
            x,
            &ids_vec,
            num_tokens,
            num_experts_per_tok,
            x_slots,
            num_experts,
            out_features,
            in_features,
            &raw,
        );
    }
    if qtensor.dtype() == GgmlDType::Q4K && x.device().is_cpu() {
        return qtensor_indexed_moe_forward_q4k(
            qtensor,
            x,
            &ids_vec,
            num_tokens,
            num_experts_per_tok,
            x_slots,
            num_experts,
            out_features,
            in_features,
            &raw,
        );
    }
    if !LOG_GGUF_CPU_MOE_FALLBACK.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using GGUF CPU indexed MoE dequant fallback (dtype={:?}, x_device={:?}, experts={num_experts}, out={out_features}, in={in_features})",
            qtensor.dtype(),
            x.device()
        );
    }
    let expert_bytes = out_features * in_features / block_size * qtensor.dtype().type_size();
    let tensor_key = Arc::as_ptr(qtensor) as usize;
    let mut rows = Vec::with_capacity(num_tokens * num_experts_per_tok);

    for (token_idx, token_ids) in ids_vec.iter().enumerate() {
        for (slot_idx, &expert_id) in token_ids.iter().enumerate() {
            let expert_id = expert_id as usize;
            if expert_id >= num_experts {
                candle_core::bail!(
                    "GGUF CPU indexed MoE expert id {expert_id} out of range {num_experts}"
                );
            }
            let key = (tensor_key, expert_id, dtype_key(x.dtype()));
            let expert = cached_expert(key, || {
                let start = expert_id * expert_bytes;
                let end = start + expert_bytes;
                let expert_qtensor = qtensor_from_ggml(
                    qtensor.dtype(),
                    &raw[start..end],
                    vec![out_features, in_features],
                    x.device(),
                )?;
                expert_qtensor.dequantize(x.device())?.to_dtype(x.dtype())
            })?;
            let x_row = if x_slots == 1 {
                x.i((token_idx, 0))?
            } else {
                x.i((token_idx, slot_idx))?
            };
            rows.push(x_row.unsqueeze(0)?.matmul(&expert.t()?)?.squeeze(0)?);
        }
    }

    Tensor::stack(&rows, 0)?.reshape((num_tokens, num_experts_per_tok, out_features))
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for CPU/Metal GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn cpu_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // For non-quantized tensors, use UnquantLinear directly
            let unquant =
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(t.clone(), None)))?;
            unquant.gather_forward(x, ids)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Module;

    #[test]
    fn cpu_q4k_matmul_matches_candle_qmatmul() -> Result<()> {
        let in_features = QK_K * 2;
        let out_features = 96;
        let rows = 3;
        let weights = (0..out_features * in_features)
            .map(|i| ((i % 97) as f32 - 48.0) / 37.0)
            .collect::<Vec<_>>();
        let input = (0..rows * in_features)
            .map(|i| ((i % 53) as f32 - 26.0) / 29.0)
            .collect::<Vec<_>>();

        let weight = Tensor::from_vec(weights, (out_features, in_features), &Device::Cpu)?;
        let input = Tensor::from_vec(input, (rows, in_features), &Device::Cpu)?;
        let qmatmul = QMatMul::from_arc(Arc::new(QTensor::quantize(&weight, GgmlDType::Q4K)?))?;

        let fast =
            cpu_q4k_matmul(&qmatmul, &input)?.expect("Q4K CPU matmul fast path should match");
        let reference = qmatmul.forward(&input)?;
        let fast = fast.to_vec2::<f32>()?;
        let reference = reference.to_vec2::<f32>()?;

        for (row_fast, row_reference) in fast.iter().zip(reference.iter()) {
            for (&fast, &reference) in row_fast.iter().zip(row_reference.iter()) {
                let diff = (fast - reference).abs();
                assert!(
                    diff < 1e-3,
                    "Q4K CPU matmul mismatch: fast={fast}, reference={reference}, diff={diff}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn cpu_indexed_moe_accepts_supported_gguf_quant_dtypes() -> Result<()> {
        let dtypes = [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q8_1,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ];
        let num_experts = 3;
        let out_features = 8;
        let in_features = QK_K;
        let tokens = 2;
        let topk = 2;
        let weights = (0..num_experts * out_features * in_features)
            .map(|i| ((i % 127) as f32 - 63.0) / 101.0)
            .collect::<Vec<_>>();
        let input = (0..tokens * in_features)
            .map(|i| ((i % 67) as f32 - 33.0) / 79.0)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(vec![0u32, 2, 1, 0], (tokens, topk), &Device::Cpu)?;
        let weight = Tensor::from_vec(
            weights,
            (num_experts, out_features, in_features),
            &Device::Cpu,
        )?;
        let input = Tensor::from_vec(input, (tokens, 1, in_features), &Device::Cpu)?;

        let mut tested = 0;
        for dtype in dtypes {
            let qtensor = match QTensor::quantize(&weight, dtype) {
                Ok(qtensor) => qtensor,
                Err(err) if err.to_string().contains("not supported") => continue,
                Err(err) => return Err(err),
            };
            let qmatmul = match QMatMul::from_arc(Arc::new(qtensor)) {
                Ok(qmatmul) => qmatmul,
                Err(err) if err.to_string().contains("not supported") => continue,
                Err(err) => return Err(err),
            };
            let output = match cpu_indexed_moe_forward(&qmatmul, &input, &ids) {
                Ok(output) => output,
                Err(err) if err.to_string().contains("not supported") => continue,
                Err(err) => return Err(err),
            };

            assert_eq!(output.dims(), &[tokens, topk, out_features], "{dtype:?}");
            tested += 1;
        }
        assert!(tested > 0);

        Ok(())
    }
}
