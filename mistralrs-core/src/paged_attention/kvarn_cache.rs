use candle_core::{DType, Device, IndexOp, Result, Tensor, TensorId};
use half::f16;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};

pub const KVARN_KEY_BITS: usize = 4;
pub const KVARN_VALUE_BITS: usize = 2;
pub const KVARN_GROUP: usize = 128;
pub const KVARN_SINKHORN_ITERS: usize = 16;

const STATUS_RAW: u8 = 0;
const STATUS_QUANTIZED: u8 = 1;
const STATUS_OFFSET: usize = 0;
const PAYLOAD_OFFSET: usize = 1;
const CLIP_STD_MIN: f32 = 1e-3;
const CLIP_STD_MAX: f32 = 1e3;
const LOG_S_MIN: f32 = -0.3;
const LOG_S_MAX: f32 = 10.0;

type BlockHead = (usize, usize);

struct TailStore {
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    rows: HashMap<BlockHead, Vec<f32>>,
}

impl TailStore {
    fn new(block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            block_size,
            num_heads,
            head_dim,
            rows: HashMap::new(),
        }
    }

    fn matches(&self, block_size: usize, num_heads: usize, head_dim: usize) -> bool {
        self.block_size == block_size && self.num_heads == num_heads && self.head_dim == head_dim
    }
}

static TAIL_STORES: OnceLock<Mutex<HashMap<TensorId, TailStore>>> = OnceLock::new();

fn tail_stores() -> &'static Mutex<HashMap<TensorId, TailStore>> {
    TAIL_STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn validate_block_size(block_size: usize) -> Result<()> {
    if block_size != KVARN_GROUP {
        candle_core::bail!(
            "KVarN paged KV cache currently requires block size {KVARN_GROUP}; got {block_size}. \
             Pass `--pa-block-size {KVARN_GROUP}` or omit the block size when using `--pa-cache-type kvarn`."
        );
    }
    Ok(())
}

pub fn key_record_bytes(head_dim: usize, block_size: usize) -> Result<usize> {
    validate_shape(head_dim, block_size)?;
    Ok(PAYLOAD_OFFSET
        + packed_bytes(head_dim, block_size, KVARN_KEY_BITS)
        + key_scale_bytes(head_dim, block_size))
}

pub fn value_record_bytes(head_dim: usize, block_size: usize) -> Result<usize> {
    validate_shape(head_dim, block_size)?;
    Ok(PAYLOAD_OFFSET
        + packed_bytes(block_size, head_dim, KVARN_VALUE_BITS)
        + value_scale_bytes(head_dim, block_size))
}

pub fn is_kvarn_cache(cache: &Tensor) -> bool {
    cache.dtype() == DType::U8 && cache.rank() == 3
}

pub fn cuda_fused_decode_enabled() -> bool {
    std::env::var("MISTRALRS_KVARN_FUSED_DECODE")
        .map(|v| !v.trim().is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

fn cuda_quantizes_partial_blocks(device: &Device) -> bool {
    device.is_cuda()
        && std::env::var("MISTRALRS_KVARN_CUDA_PARTIAL")
            .map(|v| !v.trim().is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
}

pub fn flash_attn_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    {
        return mistralrs_paged_attn::kvarn_flash_attn_decode(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
        );
    }

    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    {
        let _ = (
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
        );
        candle_core::bail!("KVarN fused decode requires the CUDA feature.");
    }
}

pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if !is_kvarn_cache(key_cache) || !is_kvarn_cache(value_cache) {
        candle_core::bail!("KVarN reshape_and_cache expects u8 rank-3 cache tensors.");
    }

    let (num_tokens, num_heads, k_head_dim) = key.dims3()?;
    let (v_tokens, v_heads, v_head_dim) = value.dims3()?;
    if (num_tokens, num_heads) != (v_tokens, v_heads) {
        candle_core::bail!(
            "KVarN cache shape mismatch: key {:?}, value {:?}",
            key.shape(),
            value.shape()
        );
    }

    let (num_blocks, cache_heads, k_record_bytes) = key_cache.dims3()?;
    let (v_num_blocks, v_cache_heads, v_record_bytes) = value_cache.dims3()?;
    if (num_blocks, cache_heads) != (v_num_blocks, v_cache_heads) || cache_heads != num_heads {
        candle_core::bail!(
            "KVarN cache layout mismatch: key_cache {:?}, value_cache {:?}, input heads {num_heads}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    let block_size = KVARN_GROUP;
    if k_record_bytes != key_record_bytes(k_head_dim, block_size)?
        || v_record_bytes != value_record_bytes(v_head_dim, block_size)?
    {
        candle_core::bail!(
            "KVarN record size mismatch: cache records ({k_record_bytes}, {v_record_bytes}), expected ({}, {})",
            key_record_bytes(k_head_dim, block_size)?,
            value_record_bytes(v_head_dim, block_size)?
        );
    }

    let slots = slot_mapping_to_vec(slot_mapping)?;
    if slots.len() != num_tokens {
        candle_core::bail!(
            "KVarN slot mapping length mismatch: got {}, expected {num_tokens}",
            slots.len()
        );
    }

    let key_cpu = key
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let value_cpu = value
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;
    let quantize_partial_blocks = cuda_quantizes_partial_blocks(key_cache.device());

    for token_idx in 0..num_tokens {
        let slot = slots[token_idx];
        if slot < 0 {
            continue;
        }
        let slot = slot as usize;
        let block = slot / block_size;
        let block_offset = slot % block_size;
        if block >= num_blocks {
            candle_core::bail!(
                "KVarN slot {slot} maps to block {block}, but cache has only {num_blocks} blocks."
            );
        }

        for head_idx in 0..num_heads {
            let key_block = store_tail_row(
                key_cache.id(),
                block,
                head_idx,
                block_offset,
                &key_cpu[token_idx][head_idx],
                block_size,
                num_heads,
                k_head_dim,
                quantize_partial_blocks,
            );
            mark_record_raw(key_cache, block, head_idx)?;
            if let Some(raw_block) = key_block {
                let record = quantize_key_block(&raw_block, k_head_dim, block_size)?;
                write_record(key_cache, block, head_idx, &record)?;
            }

            let value_block = store_tail_row(
                value_cache.id(),
                block,
                head_idx,
                block_offset,
                &value_cpu[token_idx][head_idx],
                block_size,
                num_heads,
                v_head_dim,
                quantize_partial_blocks,
            );
            mark_record_raw(value_cache, block, head_idx)?;
            if let Some(raw_block) = value_block {
                let record = quantize_value_block(&raw_block, v_head_dim, block_size)?;
                write_record(value_cache, block, head_idx, &record)?;
            }
        }
    }

    Ok(())
}

pub fn gather_kv_cache(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if !is_kvarn_cache(key_cache) || !is_kvarn_cache(value_cache) {
        candle_core::bail!("KVarN gather_kv_cache expects u8 rank-3 cache tensors.");
    }

    let device = key_cache.device().clone();
    let (num_blocks, num_heads, k_record_bytes) = key_cache.dims3()?;
    let (v_num_blocks, v_num_heads, v_record_bytes) = value_cache.dims3()?;
    if (num_blocks, num_heads) != (v_num_blocks, v_num_heads) {
        candle_core::bail!(
            "KVarN cache layout mismatch: key_cache {:?}, value_cache {:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let block_size = KVARN_GROUP;
    let k_head_dim = key_head_dim_from_record_bytes(k_record_bytes, block_size)?;
    let v_head_dim = value_head_dim_from_record_bytes(v_record_bytes, block_size)?;

    let block_table = block_table_to_vec2(block_table)?;
    let cu_seq_lens = cu_seq_lens_to_vec(cu_seq_lens)?;
    if cu_seq_lens.len() != block_table.len() + 1 {
        candle_core::bail!(
            "KVarN cu_seq_lens length mismatch: got {}, block_table batch {}",
            cu_seq_lens.len(),
            block_table.len()
        );
    }
    let total_tokens = *cu_seq_lens.last().unwrap_or(&0);
    if total_tokens == 0 {
        let k = Tensor::zeros((0, num_heads, k_head_dim), out_dtype, &device)?;
        let v = Tensor::zeros((0, num_heads, v_head_dim), out_dtype, &device)?;
        return Ok((k, v));
    }

    let key_data = key_cache
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let value_data = value_cache
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;

    let mut k_out = Vec::with_capacity(total_tokens * num_heads * k_head_dim);
    let mut v_out = Vec::with_capacity(total_tokens * num_heads * v_head_dim);

    for seq_idx in 0..block_table.len() {
        let seq_start = cu_seq_lens[seq_idx];
        let seq_end = cu_seq_lens[seq_idx + 1];
        let seq_len = seq_end.saturating_sub(seq_start);

        for token_pos in 0..seq_len {
            let table_idx = token_pos / block_size;
            let block = *block_table[seq_idx].get(table_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "KVarN block table too short for sequence {seq_idx}: token {token_pos}, table index {table_idx}"
                ))
            })?;
            if block >= num_blocks {
                candle_core::bail!(
                    "KVarN block table references block {block}, but cache has {num_blocks} blocks."
                );
            }
            let block_offset = token_pos % block_size;

            for head_idx in 0..num_heads {
                let k_base = record_offset(block, head_idx, num_heads, k_record_bytes);
                let v_base = record_offset(block, head_idx, num_heads, v_record_bytes);
                let k_record = &key_data[k_base..k_base + k_record_bytes];
                let v_record = &value_data[v_base..v_base + v_record_bytes];

                if k_record[STATUS_OFFSET] == STATUS_QUANTIZED {
                    k_out.extend_from_slice(&dequant_key_row(
                        k_record,
                        k_head_dim,
                        block_size,
                        block_offset,
                    )?);
                } else {
                    k_out.extend_from_slice(&load_tail_row(
                        key_cache.id(),
                        block,
                        head_idx,
                        block_offset,
                        k_head_dim,
                    ));
                }

                if v_record[STATUS_OFFSET] == STATUS_QUANTIZED {
                    v_out.extend_from_slice(&dequant_value_row(
                        v_record,
                        v_head_dim,
                        block_size,
                        block_offset,
                    )?);
                } else {
                    v_out.extend_from_slice(&load_tail_row(
                        value_cache.id(),
                        block,
                        head_idx,
                        block_offset,
                        v_head_dim,
                    ));
                }
            }
        }
    }

    let k = Tensor::from_vec(k_out, (total_tokens, num_heads, k_head_dim), &Device::Cpu)?;
    let k = if out_dtype == DType::F32 {
        k
    } else {
        k.to_dtype(out_dtype)?
    }
    .to_device(&device)?;
    let v = Tensor::from_vec(v_out, (total_tokens, num_heads, v_head_dim), &Device::Cpu)?;
    let v = if out_dtype == DType::F32 {
        v
    } else {
        v.to_dtype(out_dtype)?
    }
    .to_device(&device)?;
    Ok((k, v))
}

fn validate_shape(head_dim: usize, block_size: usize) -> Result<()> {
    validate_block_size(block_size)?;
    if !head_dim.is_power_of_two() {
        candle_core::bail!(
            "KVarN paged KV cache requires power-of-two head dimensions for the Hadamard rotation, got {head_dim}."
        );
    }
    if head_dim % (8 / KVARN_VALUE_BITS) != 0 {
        candle_core::bail!(
            "KVarN value packing requires head_dim divisible by {}, got {head_dim}.",
            8 / KVARN_VALUE_BITS
        );
    }
    Ok(())
}

fn packed_bytes(rows: usize, cols: usize, bits: usize) -> usize {
    rows * (cols * bits).div_ceil(8)
}

fn key_scale_bytes(head_dim: usize, block_size: usize) -> usize {
    (2 * head_dim + block_size) * 2
}

fn value_scale_bytes(head_dim: usize, block_size: usize) -> usize {
    (head_dim + 2 * block_size) * 2
}

fn key_offsets(head_dim: usize, block_size: usize) -> (usize, usize, usize, usize) {
    let packed = packed_bytes(head_dim, block_size, KVARN_KEY_BITS);
    let s_col = PAYLOAD_OFFSET + packed;
    let zp = s_col + head_dim * 2;
    let s_row = zp + head_dim * 2;
    (PAYLOAD_OFFSET, s_col, zp, s_row)
}

fn value_offsets(head_dim: usize, block_size: usize) -> (usize, usize, usize, usize) {
    let packed = packed_bytes(block_size, head_dim, KVARN_VALUE_BITS);
    let s_col = PAYLOAD_OFFSET + packed;
    let s_row = s_col + head_dim * 2;
    let zp = s_row + block_size * 2;
    (PAYLOAD_OFFSET, s_col, s_row, zp)
}

fn key_head_dim_from_record_bytes(record_bytes: usize, block_size: usize) -> Result<usize> {
    validate_block_size(block_size)?;
    let fixed = PAYLOAD_OFFSET + 2 * block_size;
    let per_dim = block_size * KVARN_KEY_BITS / 8 + 4;
    if record_bytes < fixed || (record_bytes - fixed) % per_dim != 0 {
        candle_core::bail!("KVarN key record byte count {record_bytes} is invalid.");
    }
    let head_dim = (record_bytes - fixed) / per_dim;
    validate_shape(head_dim, block_size)?;
    Ok(head_dim)
}

fn value_head_dim_from_record_bytes(record_bytes: usize, block_size: usize) -> Result<usize> {
    validate_block_size(block_size)?;
    let fixed = PAYLOAD_OFFSET + 4 * block_size;
    let per_dim = block_size * KVARN_VALUE_BITS / 8 + 2;
    if record_bytes < fixed || (record_bytes - fixed) % per_dim != 0 {
        candle_core::bail!("KVarN value record byte count {record_bytes} is invalid.");
    }
    let head_dim = (record_bytes - fixed) / per_dim;
    validate_shape(head_dim, block_size)?;
    Ok(head_dim)
}

fn store_tail_row(
    cache_id: TensorId,
    block: usize,
    head: usize,
    block_offset: usize,
    row: &[f32],
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    quantize_partial_blocks: bool,
) -> Option<Vec<f32>> {
    let mut stores = tail_stores()
        .lock()
        .expect("KVarN tail cache mutex poisoned");
    let store = stores
        .entry(cache_id)
        .or_insert_with(|| TailStore::new(block_size, num_heads, head_dim));
    if !store.matches(block_size, num_heads, head_dim) {
        *store = TailStore::new(block_size, num_heads, head_dim);
    }

    let key = (block, head);
    let entry = store
        .rows
        .entry(key)
        .or_insert_with(|| vec![0.0; block_size * head_dim]);
    if block_offset == 0 {
        entry.fill(0.0);
    }
    let start = block_offset * head_dim;
    entry[start..start + head_dim].copy_from_slice(row);

    if block_offset + 1 == block_size {
        store.rows.remove(&key)
    } else if quantize_partial_blocks {
        Some(entry.clone())
    } else {
        None
    }
}

fn load_tail_row(
    cache_id: TensorId,
    block: usize,
    head: usize,
    block_offset: usize,
    head_dim: usize,
) -> Vec<f32> {
    let stores = tail_stores()
        .lock()
        .expect("KVarN tail cache mutex poisoned");
    let Some(store) = stores.get(&cache_id) else {
        return vec![0.0; head_dim];
    };
    let Some(rows) = store.rows.get(&(block, head)) else {
        return vec![0.0; head_dim];
    };
    let start = block_offset * head_dim;
    rows[start..start + head_dim].to_vec()
}

fn mark_record_raw(cache: &mut Tensor, block: usize, head: usize) -> Result<()> {
    let raw = Tensor::from_vec(vec![STATUS_RAW], (1,), &Device::Cpu)?.to_device(cache.device())?;
    cache.i((block, head))?.slice_set(&raw, 0, STATUS_OFFSET)
}

fn write_record(cache: &mut Tensor, block: usize, head: usize, record: &[u8]) -> Result<()> {
    let record = Tensor::from_vec(record.to_vec(), (record.len(),), &Device::Cpu)?
        .to_device(cache.device())?;
    cache.i((block, head))?.slice_set(&record, 0, 0)
}

fn quantize_key_block(raw: &[f32], head_dim: usize, block_size: usize) -> Result<Vec<u8>> {
    let mut tile = vec![vec![0.0; block_size]; head_dim];
    for token in 0..block_size {
        let mut row = raw[token * head_dim..(token + 1) * head_dim].to_vec();
        hadamard_normalized(&mut row)?;
        for dim in 0..head_dim {
            tile[dim][token] = row[dim];
        }
    }

    let (balanced, s_col, s_row) = variance_normalize(&tile, KVARN_SINKHORN_ITERS);
    let mut record = vec![0u8; key_record_bytes(head_dim, block_size)?];
    record[STATUS_OFFSET] = STATUS_QUANTIZED;
    let (packed_offset, s_col_offset, zp_offset, s_row_offset) = key_offsets(head_dim, block_size);
    let pack = 8 / KVARN_KEY_BITS;
    let row_packed_bytes = block_size / pack;

    for dim in 0..head_dim {
        let (scale, zp) = asymmetric_rtn_row(&balanced[dim], KVARN_KEY_BITS);
        write_f16(&mut record, s_col_offset + dim * 2, s_row[dim] * scale);
        write_f16(&mut record, zp_offset + dim * 2, s_row[dim] * zp);
        for token in 0..block_size {
            let q = quantize_rtn_value(balanced[dim][token], scale, zp, KVARN_KEY_BITS);
            let byte_idx = packed_offset + dim * row_packed_bytes + token / pack;
            record[byte_idx] |= (q as u8) << ((token % pack) * KVARN_KEY_BITS);
        }
    }
    for token in 0..block_size {
        write_f16(&mut record, s_row_offset + token * 2, s_col[token]);
    }
    Ok(record)
}

fn quantize_value_block(raw: &[f32], head_dim: usize, block_size: usize) -> Result<Vec<u8>> {
    let mut tile = vec![vec![0.0; head_dim]; block_size];
    for token in 0..block_size {
        let mut row = raw[token * head_dim..(token + 1) * head_dim].to_vec();
        hadamard_normalized(&mut row)?;
        tile[token] = row;
    }

    let (balanced, s_col, s_row) = variance_normalize(&tile, KVARN_SINKHORN_ITERS);
    let mut record = vec![0u8; value_record_bytes(head_dim, block_size)?];
    record[STATUS_OFFSET] = STATUS_QUANTIZED;
    let (packed_offset, s_col_offset, s_row_offset, zp_offset) =
        value_offsets(head_dim, block_size);
    let pack = 8 / KVARN_VALUE_BITS;
    let row_packed_bytes = head_dim / pack;

    for dim in 0..head_dim {
        write_f16(&mut record, s_col_offset + dim * 2, s_col[dim]);
    }
    for token in 0..block_size {
        let (scale, zp) = asymmetric_rtn_row(&balanced[token], KVARN_VALUE_BITS);
        write_f16(&mut record, s_row_offset + token * 2, s_row[token] * scale);
        write_f16(&mut record, zp_offset + token * 2, s_row[token] * zp);
        for dim in 0..head_dim {
            let q = quantize_rtn_value(balanced[token][dim], scale, zp, KVARN_VALUE_BITS);
            let byte_idx = packed_offset + token * row_packed_bytes + dim / pack;
            record[byte_idx] |= (q as u8) << ((dim % pack) * KVARN_VALUE_BITS);
        }
    }
    Ok(record)
}

fn dequant_key_row(
    record: &[u8],
    head_dim: usize,
    block_size: usize,
    token: usize,
) -> Result<Vec<f32>> {
    let (packed_offset, s_col_offset, zp_offset, s_row_offset) = key_offsets(head_dim, block_size);
    let pack = 8 / KVARN_KEY_BITS;
    let row_packed_bytes = block_size / pack;
    let s_row = read_f16(record, s_row_offset + token * 2);
    let mut row = vec![0.0; head_dim];
    for dim in 0..head_dim {
        let byte_idx = packed_offset + dim * row_packed_bytes + token / pack;
        let q = ((record[byte_idx] >> ((token % pack) * KVARN_KEY_BITS))
            & ((1u8 << KVARN_KEY_BITS) - 1)) as f32;
        let s_col = read_f16(record, s_col_offset + dim * 2);
        let zp = read_f16(record, zp_offset + dim * 2);
        row[dim] = (q * s_col + zp) * s_row;
    }
    hadamard_normalized(&mut row)?;
    Ok(row)
}

fn dequant_value_row(
    record: &[u8],
    head_dim: usize,
    block_size: usize,
    token: usize,
) -> Result<Vec<f32>> {
    let (packed_offset, s_col_offset, s_row_offset, zp_offset) =
        value_offsets(head_dim, block_size);
    let pack = 8 / KVARN_VALUE_BITS;
    let row_packed_bytes = head_dim / pack;
    let s_row = read_f16(record, s_row_offset + token * 2);
    let zp = read_f16(record, zp_offset + token * 2);
    let mut row = vec![0.0; head_dim];
    for dim in 0..head_dim {
        let byte_idx = packed_offset + token * row_packed_bytes + dim / pack;
        let q = ((record[byte_idx] >> ((dim % pack) * KVARN_VALUE_BITS))
            & ((1u8 << KVARN_VALUE_BITS) - 1)) as f32;
        let s_col = read_f16(record, s_col_offset + dim * 2);
        row[dim] = (q * s_row + zp) * s_col;
    }
    hadamard_normalized(&mut row)?;
    Ok(row)
}

fn variance_normalize(tile: &[Vec<f32>], iterations: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
    let rows = tile.len();
    let cols = tile.first().map_or(0, Vec::len);
    let mut log_s_col = vec![0.0f32; cols];
    let mut log_s_row = vec![0.0f32; rows];
    let mut best_s_col = vec![1.0f32; cols];
    let mut best_s_row = vec![1.0f32; rows];
    let mut best_imbalance = f32::INFINITY;

    for _ in 0..iterations.max(1) {
        let cur = apply_log_scales(tile, &log_s_col, &log_s_row);
        let col_std = column_stds(&cur);
        for col in 0..cols {
            let std = col_std[col].clamp(CLIP_STD_MIN, CLIP_STD_MAX);
            log_s_col[col] = (log_s_col[col] + std.ln()).clamp(LOG_S_MIN, LOG_S_MAX);
        }

        let cur = apply_log_scales(tile, &log_s_col, &log_s_row);
        let row_std = row_stds(&cur);
        for row in 0..rows {
            let std = row_std[row].clamp(CLIP_STD_MIN, CLIP_STD_MAX);
            log_s_row[row] = (log_s_row[row] + std.ln()).clamp(LOG_S_MIN, LOG_S_MAX);
        }

        let cur = apply_log_scales(tile, &log_s_col, &log_s_row);
        let imbalance = axis_imbalance(&column_stds(&cur)) + axis_imbalance(&row_stds(&cur));
        if imbalance.is_finite() && imbalance <= best_imbalance {
            best_imbalance = imbalance;
            for col in 0..cols {
                best_s_col[col] = log_s_col[col].exp();
            }
            for row in 0..rows {
                best_s_row[row] = log_s_row[row].exp();
            }
        }
    }

    let balanced = apply_scales(tile, &best_s_col, &best_s_row);
    (balanced, best_s_col, best_s_row)
}

fn apply_log_scales(tile: &[Vec<f32>], log_s_col: &[f32], log_s_row: &[f32]) -> Vec<Vec<f32>> {
    let s_col = log_s_col.iter().map(|v| v.exp()).collect::<Vec<_>>();
    let s_row = log_s_row.iter().map(|v| v.exp()).collect::<Vec<_>>();
    apply_scales(tile, &s_col, &s_row)
}

fn apply_scales(tile: &[Vec<f32>], s_col: &[f32], s_row: &[f32]) -> Vec<Vec<f32>> {
    tile.iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(col, value)| *value / s_col[col].max(1e-12) / s_row[row].max(1e-12))
                .collect()
        })
        .collect()
}

fn column_stds(tile: &[Vec<f32>]) -> Vec<f32> {
    let rows = tile.len();
    let cols = tile.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    (0..cols)
        .map(|col| {
            let mean = tile.iter().map(|row| row[col]).sum::<f32>() / rows as f32;
            let var = tile
                .iter()
                .map(|row| {
                    let diff = row[col] - mean;
                    diff * diff
                })
                .sum::<f32>()
                / rows as f32;
            unbiased_std(var, rows)
        })
        .collect()
}

fn row_stds(tile: &[Vec<f32>]) -> Vec<f32> {
    tile.iter()
        .map(|row| {
            if row.is_empty() {
                return 0.0;
            }
            let mean = row.iter().sum::<f32>() / row.len() as f32;
            let var = row
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f32>()
                / row.len() as f32;
            unbiased_std(var, row.len())
        })
        .collect()
}

fn unbiased_std(var: f32, n: usize) -> f32 {
    if n > 1 {
        (var.max(0.0) * n as f32 / (n - 1) as f32).sqrt()
    } else {
        var.max(0.0).sqrt()
    }
}

fn axis_imbalance(stds: &[f32]) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = 0.0f32;
    for &std in stds {
        let std = std.clamp(CLIP_STD_MIN, CLIP_STD_MAX);
        min = min.min(std);
        max = max.max(std);
    }
    if min.is_finite() && min > 0.0 {
        max / min
    } else {
        f32::INFINITY
    }
}

fn asymmetric_rtn_row(row: &[f32], bits: usize) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &value in row {
        if value.is_finite() {
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (1e-10, 0.0);
    }
    let qmax = ((1u32 << bits) - 1) as f32;
    (((hi - lo) / qmax).max(1e-10), lo)
}

fn quantize_rtn_value(value: f32, scale: f32, zp: f32, bits: usize) -> u32 {
    let qmax = (1u32 << bits) - 1;
    ((value - zp) / scale).round().clamp(0.0, qmax as f32) as u32
}

fn hadamard_normalized(values: &mut [f32]) -> Result<()> {
    if !values.len().is_power_of_two() {
        candle_core::bail!(
            "KVarN Hadamard rotation requires power-of-two head dimensions, got {}.",
            values.len()
        );
    }
    fwht(values);
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= scale;
    }
    Ok(())
}

fn fwht(values: &mut [f32]) {
    debug_assert!(values.len().is_power_of_two());
    let mut h = 1;
    while h < values.len() {
        for i in (0..values.len()).step_by(h * 2) {
            for j in i..i + h {
                let x = values[j];
                let y = values[j + h];
                values[j] = x + y;
                values[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

fn write_f16(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
}

fn read_f16(data: &[u8], offset: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([data[offset], data[offset + 1]])).to_f32()
}

fn record_offset(block: usize, head: usize, num_heads: usize, record_bytes: usize) -> usize {
    (block * num_heads + head) * record_bytes
}

fn slot_mapping_to_vec(slot_mapping: &Tensor) -> Result<Vec<i64>> {
    let slot_mapping = slot_mapping.to_device(&Device::Cpu)?.flatten_all()?;
    match slot_mapping.dtype() {
        DType::I64 => slot_mapping.to_vec1::<i64>(),
        DType::I32 => Ok(slot_mapping
            .to_vec1::<i32>()?
            .into_iter()
            .map(i64::from)
            .collect()),
        DType::U32 => Ok(slot_mapping
            .to_vec1::<u32>()?
            .into_iter()
            .map(|v| v as i64)
            .collect()),
        other => candle_core::bail!("KVarN slot_mapping expects i64/i32/u32, got {other:?}."),
    }
}

fn block_table_to_vec2(block_table: &Tensor) -> Result<Vec<Vec<usize>>> {
    let block_table = block_table.to_device(&Device::Cpu)?;
    match block_table.dtype() {
        DType::I32 => Ok(block_table
            .to_vec2::<i32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v.max(0) as usize).collect())
            .collect()),
        DType::U32 => Ok(block_table
            .to_vec2::<u32>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v as usize).collect())
            .collect()),
        DType::I64 => Ok(block_table
            .to_vec2::<i64>()?
            .into_iter()
            .map(|row| row.into_iter().map(|v| v.max(0) as usize).collect())
            .collect()),
        other => candle_core::bail!("KVarN block_table expects i32/u32/i64, got {other:?}."),
    }
}

fn cu_seq_lens_to_vec(cu_seq_lens: &Tensor) -> Result<Vec<usize>> {
    let cu_seq_lens = cu_seq_lens.to_device(&Device::Cpu)?;
    match cu_seq_lens.dtype() {
        DType::I32 => Ok(cu_seq_lens
            .to_vec1::<i32>()?
            .into_iter()
            .map(|v| v.max(0) as usize)
            .collect()),
        DType::U32 => Ok(cu_seq_lens
            .to_vec1::<u32>()?
            .into_iter()
            .map(|v| v as usize)
            .collect()),
        DType::I64 => Ok(cu_seq_lens
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v.max(0) as usize)
            .collect()),
        other => candle_core::bail!("KVarN cu_seq_lens expects i32/u32/i64, got {other:?}."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_sizes_roundtrip_head_dim() -> Result<()> {
        let key_bytes = key_record_bytes(128, KVARN_GROUP)?;
        let value_bytes = value_record_bytes(128, KVARN_GROUP)?;
        assert_eq!(key_head_dim_from_record_bytes(key_bytes, KVARN_GROUP)?, 128);
        assert_eq!(
            value_head_dim_from_record_bytes(value_bytes, KVARN_GROUP)?,
            128
        );
        Ok(())
    }

    #[test]
    fn gathers_raw_partial_tail() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 16;
        let mut key_cache = Tensor::zeros(
            (1, 1, key_record_bytes(head_dim, KVARN_GROUP)?),
            DType::U8,
            &device,
        )?;
        let mut value_cache = Tensor::zeros(
            (1, 1, value_record_bytes(head_dim, KVARN_GROUP)?),
            DType::U8,
            &device,
        )?;
        let key = Tensor::from_vec(
            (0..3 * head_dim)
                .map(|v| v as f32 / 31.0)
                .collect::<Vec<_>>(),
            (3, 1, head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            (0..3 * head_dim)
                .map(|v| (v as f32 / 17.0).sin())
                .collect::<Vec<_>>(),
            (3, 1, head_dim),
            &device,
        )?;
        let slots = Tensor::from_vec(vec![0i64, 1, 2], (3,), &device)?;
        reshape_and_cache(&key, &value, &mut key_cache, &mut value_cache, &slots)?;

        let block_table = Tensor::from_vec(vec![0i32], (1, 1), &device)?;
        let cu = Tensor::from_vec(vec![0u32, 3], (2,), &device)?;
        let (k, v) = gather_kv_cache(&key_cache, &value_cache, &block_table, &cu, DType::F32)?;
        assert_eq!(k.dims(), &[3, 1, head_dim]);
        assert_eq!(v.dims(), &[3, 1, head_dim]);
        Ok(())
    }

    #[test]
    fn quantized_block_gathers_with_expected_shape() -> Result<()> {
        let device = Device::Cpu;
        let head_dim = 16;
        let mut key_cache = Tensor::zeros(
            (1, 1, key_record_bytes(head_dim, KVARN_GROUP)?),
            DType::U8,
            &device,
        )?;
        let mut value_cache = Tensor::zeros(
            (1, 1, value_record_bytes(head_dim, KVARN_GROUP)?),
            DType::U8,
            &device,
        )?;
        let key = Tensor::from_vec(
            (0..KVARN_GROUP * head_dim)
                .map(|v| ((v as f32) * 0.013).sin())
                .collect::<Vec<_>>(),
            (KVARN_GROUP, 1, head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            (0..KVARN_GROUP * head_dim)
                .map(|v| ((v as f32) * 0.021).cos())
                .collect::<Vec<_>>(),
            (KVARN_GROUP, 1, head_dim),
            &device,
        )?;
        let slots = Tensor::from_vec(
            (0..KVARN_GROUP as i64).collect::<Vec<_>>(),
            (KVARN_GROUP,),
            &device,
        )?;
        reshape_and_cache(&key, &value, &mut key_cache, &mut value_cache, &slots)?;

        let key_record = key_cache.flatten_all()?.to_vec1::<u8>()?;
        assert_eq!(key_record[STATUS_OFFSET], STATUS_QUANTIZED);

        let block_table = Tensor::from_vec(vec![0i32], (1, 1), &device)?;
        let cu = Tensor::from_vec(vec![0u32, KVARN_GROUP as u32], (2,), &device)?;
        let (k, v) = gather_kv_cache(&key_cache, &value_cache, &block_table, &cu, DType::F32)?;
        assert_eq!(k.dims(), &[KVARN_GROUP, 1, head_dim]);
        assert_eq!(v.dims(), &[KVARN_GROUP, 1, head_dim]);
        Ok(())
    }
}
