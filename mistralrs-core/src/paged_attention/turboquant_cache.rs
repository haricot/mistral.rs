use candle_core::{DType, Device, IndexOp, Result, Tensor};

pub const TURBOQUANT_BITS: usize = 4;
const MSE_BITS: usize = TURBOQUANT_BITS - 1;
const NORM_BYTES: usize = std::mem::size_of::<f32>();
const CLIP: f32 = 4.0;

pub fn row_bytes(head_dim: usize) -> Result<usize> {
    if !head_dim.is_power_of_two() {
        candle_core::bail!(
            "TurboQuant paged KV cache requires power-of-two head dimensions for the Hadamard rotation, got {head_dim}."
        );
    }
    Ok(2 * NORM_BYTES + (head_dim * MSE_BITS).div_ceil(8) + head_dim.div_ceil(8))
}

pub fn is_turboquant_cache(cache: &Tensor) -> bool {
    cache.dtype() == DType::U8 && cache.rank() == 4
}

pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if !is_turboquant_cache(key_cache) || !is_turboquant_cache(value_cache) {
        candle_core::bail!("TurboQuant reshape_and_cache expects u8 rank-4 cache tensors.");
    }

    let (num_tokens, num_heads, k_head_dim) = key.dims3()?;
    let (v_tokens, v_heads, v_head_dim) = value.dims3()?;
    if (num_tokens, num_heads) != (v_tokens, v_heads) {
        candle_core::bail!(
            "TurboQuant cache shape mismatch: key {:?}, value {:?}",
            key.shape(),
            value.shape()
        );
    }

    let (num_blocks, cache_heads, block_size, k_row_bytes) = key_cache.dims4()?;
    let (v_num_blocks, v_cache_heads, v_block_size, v_row_bytes) = value_cache.dims4()?;
    if (num_blocks, cache_heads, block_size) != (v_num_blocks, v_cache_heads, v_block_size)
        || cache_heads != num_heads
    {
        candle_core::bail!(
            "TurboQuant cache layout mismatch: key_cache {:?}, value_cache {:?}, input heads {num_heads}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    if k_row_bytes != row_bytes(k_head_dim)? || v_row_bytes != row_bytes(v_head_dim)? {
        candle_core::bail!(
            "TurboQuant row size mismatch: cache rows ({k_row_bytes}, {v_row_bytes}), expected ({}, {})",
            row_bytes(k_head_dim)?,
            row_bytes(v_head_dim)?
        );
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    if key_cache.device().is_cuda() {
        return mistralrs_paged_attn::turboquant_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
        );
    }

    let slots = slot_mapping.to_device(&Device::Cpu)?.to_vec1::<i64>()?;
    if slots.len() != num_tokens {
        candle_core::bail!(
            "TurboQuant slot mapping length mismatch: got {}, expected {num_tokens}",
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
                "TurboQuant slot {slot} maps to block {block}, but cache has only {num_blocks} blocks."
            );
        }

        for head_idx in 0..num_heads {
            let k_row = pack_vector(&key_cpu[token_idx][head_idx])?;
            let v_row = pack_vector(&value_cpu[token_idx][head_idx])?;
            write_row(key_cache, block, head_idx, block_offset, &k_row)?;
            write_row(value_cache, block, head_idx, block_offset, &v_row)?;
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
    if !is_turboquant_cache(key_cache) || !is_turboquant_cache(value_cache) {
        candle_core::bail!("TurboQuant gather_kv_cache expects u8 rank-4 cache tensors.");
    }

    let device = key_cache.device().clone();
    let (num_blocks, num_heads, block_size, k_row_bytes) = key_cache.dims4()?;
    let (v_num_blocks, v_num_heads, v_block_size, v_row_bytes) = value_cache.dims4()?;
    if (num_blocks, num_heads, block_size) != (v_num_blocks, v_num_heads, v_block_size) {
        candle_core::bail!(
            "TurboQuant cache layout mismatch: key_cache {:?}, value_cache {:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let k_head_dim = head_dim_from_row_bytes(k_row_bytes)?;
    let v_head_dim = head_dim_from_row_bytes(v_row_bytes)?;

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    if device.is_cuda() {
        return mistralrs_paged_attn::turboquant_gather_kv_cache(
            key_cache,
            value_cache,
            block_table,
            cu_seq_lens,
            out_dtype,
        );
    }

    let block_table = block_table_to_vec2(block_table)?;
    let cu_seq_lens = cu_seq_lens_to_vec(cu_seq_lens)?;
    if cu_seq_lens.len() != block_table.len() + 1 {
        candle_core::bail!(
            "TurboQuant cu_seq_lens length mismatch: got {}, block_table batch {}",
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
                    "TurboQuant block table too short for sequence {seq_idx}: token {token_pos}, table index {table_idx}"
                ))
            })?;
            if block >= num_blocks {
                candle_core::bail!(
                    "TurboQuant block table references block {block}, but cache has {num_blocks} blocks."
                );
            }
            let block_offset = token_pos % block_size;

            for head_idx in 0..num_heads {
                let k_base = row_offset(
                    block,
                    head_idx,
                    block_offset,
                    num_heads,
                    block_size,
                    k_row_bytes,
                );
                let v_base = row_offset(
                    block,
                    head_idx,
                    block_offset,
                    num_heads,
                    block_size,
                    v_row_bytes,
                );
                k_out.extend_from_slice(&unpack_vector(
                    &key_data[k_base..k_base + k_row_bytes],
                    k_head_dim,
                )?);
                v_out.extend_from_slice(&unpack_vector(
                    &value_data[v_base..v_base + v_row_bytes],
                    v_head_dim,
                )?);
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

fn write_row(
    cache: &mut Tensor,
    block: usize,
    head: usize,
    block_offset: usize,
    row: &[u8],
) -> Result<()> {
    let row =
        Tensor::from_vec(row.to_vec(), (1, row.len()), &Device::Cpu)?.to_device(cache.device())?;
    cache.i((block, head))?.slice_set(&row, 0, block_offset)
}

fn pack_vector(vector: &[f32]) -> Result<Vec<u8>> {
    let dim = vector.len();
    let row_bytes = row_bytes(dim)?;
    let mut row = vec![0u8; row_bytes];
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    row[..NORM_BYTES].copy_from_slice(&norm.to_le_bytes());
    if norm == 0.0 || !norm.is_finite() {
        return Ok(row);
    }

    let mut rotated = vector
        .iter()
        .enumerate()
        .map(|(idx, value)| deterministic_sign(idx) * *value / norm)
        .collect::<Vec<_>>();
    fwht(&mut rotated);

    let mut mse_rotated = Vec::with_capacity(dim);
    let mut bit_offset = 0usize;
    let mse_bytes = (dim * MSE_BITS).div_ceil(8);
    let mse_start = 2 * NORM_BYTES;
    let qjl_start = mse_start + mse_bytes;
    {
        let mse_bits = &mut row[mse_start..qjl_start];
        for value in rotated {
            let idx = quantize_scalar(value, MSE_BITS);
            mse_rotated.push(dequantize_scalar(idx, MSE_BITS));
            push_bits(mse_bits, &mut bit_offset, idx, MSE_BITS);
        }
    }

    fwht(&mut mse_rotated);
    let inv_dim = 1.0 / dim as f32;
    let mse_reconstructed = mse_rotated
        .iter()
        .enumerate()
        .map(|(idx, value)| deterministic_sign(idx) * *value * inv_dim * norm)
        .collect::<Vec<_>>();
    let residual = vector
        .iter()
        .zip(mse_reconstructed.iter())
        .map(|(exact, approx)| exact - approx)
        .collect::<Vec<_>>();
    let residual_norm = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
    row[NORM_BYTES..2 * NORM_BYTES].copy_from_slice(&residual_norm.to_le_bytes());

    if residual_norm != 0.0 && residual_norm.is_finite() {
        let mut qjl_bit_offset = 0usize;
        let qjl_bits = &mut row[qjl_start..];
        let mut projected = residual
            .iter()
            .enumerate()
            .map(|(col, value)| deterministic_sign(col) * *value)
            .collect::<Vec<_>>();
        fwht(&mut projected);
        for projection_row in 0..dim {
            let sign_bit =
                u32::from(deterministic_sign(projection_row) * projected[projection_row] >= 0.0);
            push_bits(qjl_bits, &mut qjl_bit_offset, sign_bit, 1);
        }
    }
    Ok(row)
}

fn unpack_vector(row: &[u8], dim: usize) -> Result<Vec<f32>> {
    if row.len() != row_bytes(dim)? {
        candle_core::bail!(
            "TurboQuant row length mismatch: got {}, expected {}",
            row.len(),
            row_bytes(dim)?
        );
    }

    let norm = f32::from_le_bytes(row[..NORM_BYTES].try_into().unwrap());
    if norm == 0.0 || !norm.is_finite() {
        return Ok(vec![0.0; dim]);
    }
    let residual_norm = f32::from_le_bytes(row[NORM_BYTES..2 * NORM_BYTES].try_into().unwrap());

    let mse_bytes = (dim * MSE_BITS).div_ceil(8);
    let (mse_bits, qjl_bits) = row[2 * NORM_BYTES..].split_at(mse_bytes);
    let mut transformed = Vec::with_capacity(dim);
    let mut reader = BitReader::new(mse_bits, MSE_BITS);
    for _ in 0..dim {
        transformed.push(dequantize_scalar(reader.next()?, MSE_BITS));
    }
    fwht(&mut transformed);
    let inv_dim = 1.0 / dim as f32;
    for (idx, value) in transformed.iter_mut().enumerate() {
        *value = deterministic_sign(idx) * *value * inv_dim * norm;
    }
    if residual_norm != 0.0 && residual_norm.is_finite() {
        let mut qjl_reader = BitReader::new(qjl_bits, 1);
        let mut qjl_signs = (0..dim)
            .map(|row| {
                qjl_reader
                    .next()
                    .map(|bit| if bit == 1 { 1.0 } else { -1.0 } * deterministic_sign(row))
            })
            .collect::<Result<Vec<_>>>()?;
        fwht(&mut qjl_signs);
        let scale = (std::f32::consts::FRAC_PI_2).sqrt() / dim as f32;
        for col in 0..dim {
            transformed[col] += residual_norm * scale * deterministic_sign(col) * qjl_signs[col];
        }
    }
    Ok(transformed)
}

fn head_dim_from_row_bytes(row_len: usize) -> Result<usize> {
    if row_len < NORM_BYTES {
        candle_core::bail!("TurboQuant row has fewer than {NORM_BYTES} bytes.");
    }
    let packed_bytes = row_len - 2 * NORM_BYTES;
    let dim = (packed_bytes * 8) / TURBOQUANT_BITS;
    if row_bytes(dim)? != row_len {
        candle_core::bail!(
            "TurboQuant row byte count {row_len} is not valid for {TURBOQUANT_BITS}-bit packing."
        );
    }
    Ok(dim)
}

fn row_offset(
    block: usize,
    head: usize,
    block_offset: usize,
    num_heads: usize,
    block_size: usize,
    row_bytes: usize,
) -> usize {
    ((block * num_heads + head) * block_size + block_offset) * row_bytes
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
        other => candle_core::bail!("TurboQuant block_table expects i32/u32, got {other:?}."),
    }
}

pub fn cu_seq_lens_to_vec(cu_seq_lens: &Tensor) -> Result<Vec<usize>> {
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
        other => candle_core::bail!("TurboQuant cu_seq_lens expects i32/u32, got {other:?}."),
    }
}

fn quantize_scalar(value: f32, bits: usize) -> u32 {
    let levels = (1u32 << bits) - 1;
    let value = value.clamp(-CLIP, CLIP);
    (((value + CLIP) / (2.0 * CLIP)) * levels as f32).round() as u32
}

fn dequantize_scalar(index: u32, bits: usize) -> f32 {
    let levels = (1u32 << bits) - 1;
    -CLIP + (index as f32 / levels as f32) * (2.0 * CLIP)
}

fn push_bits(data: &mut [u8], bit_offset: &mut usize, value: u32, bits: usize) {
    for bit in 0..bits {
        if (value >> bit) & 1 == 1 {
            let pos = *bit_offset + bit;
            data[pos / 8] |= 1 << (pos % 8);
        }
    }
    *bit_offset += bits;
}

struct BitReader<'a> {
    data: &'a [u8],
    bits: usize,
    bit_offset: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], bits: usize) -> Self {
        Self {
            data,
            bits,
            bit_offset: 0,
        }
    }

    fn next(&mut self) -> Result<u32> {
        let mut value = 0u32;
        for bit in 0..self.bits {
            let pos = self.bit_offset + bit;
            let byte = self.data.get(pos / 8).ok_or_else(|| {
                candle_core::Error::Msg("TurboQuant packed row ended unexpectedly.".to_string())
            })?;
            value |= (((byte >> (pos % 8)) & 1) as u32) << bit;
        }
        self.bit_offset += self.bits;
        Ok(value)
    }
}

fn deterministic_sign(index: usize) -> f32 {
    let mut x = index as u32;
    x = x.wrapping_add(0x9E3779B9);
    x = (x ^ (x >> 15)).wrapping_mul(0x85EBCA6B);
    x = (x ^ (x >> 13)).wrapping_mul(0xC2B2AE35);
    if ((x ^ (x >> 16)) & 1) == 0 {
        1.0
    } else {
        -1.0
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip_has_expected_shape_and_reasonable_error() -> Result<()> {
        let vector = (0..64)
            .map(|i| ((i as f32) * 0.17).sin())
            .collect::<Vec<_>>();
        let packed = pack_vector(&vector)?;
        assert_eq!(packed.len(), row_bytes(64)?);
        let unpacked = unpack_vector(&packed, 64)?;
        let mse = vector
            .iter()
            .zip(unpacked.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vector.len() as f32;
        assert!(mse < 0.08, "mse={mse}");
        Ok(())
    }

    #[test]
    fn writes_and_gathers_cache_rows() -> Result<()> {
        let device = Device::Cpu;
        let mut k_cache = Tensor::zeros((2, 2, 4, row_bytes(8)?), DType::U8, &device)?;
        let mut v_cache = Tensor::zeros((2, 2, 4, row_bytes(8)?), DType::U8, &device)?;
        let k = Tensor::randn(0f32, 1f32, (3, 2, 8), &device)?;
        let v = Tensor::randn(0f32, 1f32, (3, 2, 8), &device)?;
        let slots = Tensor::new(&[0i64, 1, 5], &device)?;

        reshape_and_cache(&k, &v, &mut k_cache, &mut v_cache, &slots)?;

        let block_table = Tensor::new(&[[0u32, 1u32]], &device)?;
        let cu = Tensor::new(&[0u32, 6u32], &device)?;
        let (kg, vg) = gather_kv_cache(&k_cache, &v_cache, &block_table, &cu, DType::F32)?;
        assert_eq!(kg.dims(), &[6, 2, 8]);
        assert_eq!(vg.dims(), &[6, 2, 8]);

        let (kg, vg) = gather_kv_cache(&k_cache, &v_cache, &block_table, &cu, DType::F16)?;
        assert_eq!(kg.dtype(), DType::F16);
        assert_eq!(vg.dtype(), DType::F16);
        assert_eq!(kg.dims(), &[6, 2, 8]);
        assert_eq!(vg.dims(), &[6, 2, 8]);
        Ok(())
    }

    #[test]
    fn gathers_empty_cache_without_dtype_conversion_kernel() -> Result<()> {
        let device = Device::Cpu;
        let k_cache = Tensor::zeros((2, 2, 4, row_bytes(8)?), DType::U8, &device)?;
        let v_cache = Tensor::zeros((2, 2, 4, row_bytes(8)?), DType::U8, &device)?;
        let block_table = Tensor::new(&[[0u32]], &device)?;
        let cu = Tensor::new(&[0u32, 0u32], &device)?;

        let (kg, vg) = gather_kv_cache(&k_cache, &v_cache, &block_table, &cu, DType::F16)?;
        assert_eq!(kg.dtype(), DType::F16);
        assert_eq!(vg.dtype(), DType::F16);
        assert_eq!(kg.dims(), &[0, 2, 8]);
        assert_eq!(vg.dims(), &[0, 2, 8]);
        Ok(())
    }
}
