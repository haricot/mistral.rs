use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::CudaDType;
use candle_core::cuda_backend::cudarc::driver::DeviceRepr;
use candle_core::{DType, Result, Storage, Tensor};
use std::ffi::c_int;

const NORM_BYTES: usize = std::mem::size_of::<f32>();
const MSE_BITS: usize = 3;

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!(
            "legacy_flash_attn_decode_turboquant only supports f16/bf16/f32 ({other:?})"
        ),
    }
}

fn head_dim_from_row_bytes(row_len: usize) -> Result<usize> {
    if row_len < 2 * NORM_BYTES {
        candle_core::bail!("TurboQuant row has fewer than {} bytes.", 2 * NORM_BYTES);
    }

    let packed_bytes = row_len - 2 * NORM_BYTES;
    let max_dim = packed_bytes * 8;
    let mut dim = 1usize;
    while dim <= max_dim {
        let expected = 2 * NORM_BYTES + (dim * MSE_BITS).div_ceil(8) + dim.div_ceil(8);
        if expected == row_len {
            return Ok(dim);
        }
        dim *= 2;
    }

    candle_core::bail!("TurboQuant row byte count {row_len} is not valid.")
}

/// Runtime gate used by the integration call-site.
///
/// The user command already sets `ALLOW_LEGACY=bf16,fp8`; this function treats any
/// non-empty value as approval to route the decode attention through the legacy
/// TurboQuant-direct kernel.
pub fn legacy_flash_attn_turboquant_allowed() -> bool {
    std::env::var_os("ALLOW_LEGACY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

fn legacy_flash_attn_decode_turboquant_t<T: CudaDType + DeviceRepr>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant expects u8 TurboQuant cache tensors, got {:?} and {:?}",
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    if !matches!(block_tables.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant expects i32/u32 block_tables, got {:?}",
            block_tables.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant expects i32/u32 cu_seq_lens, got {:?}",
            cu_seq_lens.dtype()
        );
    }

    let query = query.contiguous()?;
    let block_tables = block_tables.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;

    let q_rank = query.dims().len();
    let (num_seqs, num_heads, q_len, head_dim) = match q_rank {
        3 => {
            let (s, h, d) = query.dims3()?;
            (s, h, 1usize, d)
        }
        4 => query.dims4()?,
        other => candle_core::bail!(
            "legacy_flash_attn_decode_turboquant expects query rank 3 or 4, got rank {other} shape {:?}",
            query.shape()
        ),
    };
    if q_len != 1 {
        candle_core::bail!("legacy_flash_attn_decode_turboquant expects q_len == 1, got {q_len}");
    }

    let (num_blocks, num_kv_heads, block_size, k_row_bytes) = key_cache.dims4()?;
    let (v_num_blocks, v_num_kv_heads, v_block_size, v_row_bytes) = value_cache.dims4()?;
    if (v_num_blocks, v_num_kv_heads, v_block_size) != (num_blocks, num_kv_heads, block_size) {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant cache layout mismatch: key={:?}, value={:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let k_head_dim = head_dim_from_row_bytes(k_row_bytes)?;
    let v_head_dim = head_dim_from_row_bytes(v_row_bytes)?;
    if k_head_dim != head_dim || v_head_dim != head_dim {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant head dim mismatch: q={head_dim}, k={k_head_dim}, v={v_head_dim}"
        );
    }
    if num_heads % num_kv_heads != 0 {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
    }

    let (bt_num_seqs, block_table_stride) = block_tables.dims2()?;
    if bt_num_seqs != num_seqs {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant block_tables seq mismatch: got {bt_num_seqs}, expected {num_seqs}"
        );
    }
    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    if cu_seq_lens_len != num_seqs + 1 {
        candle_core::bail!(
            "legacy_flash_attn_decode_turboquant cu_seq_lens length mismatch: got {cu_seq_lens_len}, expected {}",
            num_seqs + 1
        );
    }

    let out = if q_rank == 4 {
        Tensor::zeros((num_seqs, num_heads, 1usize, head_dim), query.dtype(), query.device())?
    } else {
        Tensor::zeros((num_seqs, num_heads, head_dim), query.dtype(), query.device())?
    };

    let dtype_code = dtype_code(query.dtype())?;

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: query must be CUDA"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: key_cache must be CUDA"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: value_cache must be CUDA"),
    };
    let binding = out.clone();
    let (o_s, o_l) = binding.storage_and_layout();
    let o_s = match &*o_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: output must be CUDA"),
    };

    let (q_ptr, _q_guard) = slice_ptr(q_s.as_cuda_slice::<T>()?, q_l.start_offset());
    let (kc_ptr, _kc_guard) = slice_ptr(kc_s.as_cuda_slice::<u8>()?, kc_l.start_offset());
    let (vc_ptr, _vc_guard) = slice_ptr(vc_s.as_cuda_slice::<u8>()?, vc_l.start_offset());
    let (out_ptr, _out_guard) = slice_ptr(o_s.as_cuda_slice::<T>()?, o_l.start_offset());

    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: block_tables must be CUDA"),
    };
    let (bt_ptr, _bt_guard) = if block_tables.dtype() == DType::I32 {
        let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<i32>()?, bt_l.start_offset());
        (ptr as *const i32, guard)
    } else {
        let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());
        (ptr as *const i32, guard)
    };

    let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
    let cu_s = match &*cu_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("legacy_flash_attn_decode_turboquant: cu_seq_lens must be CUDA"),
    };
    let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
        let (ptr, guard) = slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset());
        (ptr as *const i32, guard)
    } else {
        let (ptr, guard) = slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset());
        (ptr as *const i32, guard)
    };

    let dev = q_s.device();
    unsafe {
        ffi::legacy_flash_attn_decode_turboquant(
            q_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            bt_ptr,
            cu_ptr,
            out_ptr as *mut core::ffi::c_void,
            num_seqs as c_int,
            block_size as c_int,
            block_table_stride as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            k_row_bytes as c_int,
            v_row_bytes as c_int,
            softmax_scale,
            window_size as c_int,
            dev.cuda_stream().cu_stream(),
            dtype_code,
        );
    }

    Ok(out)
}

/// Decode-only online-softmax attention that reads TurboQuant paged K/V directly.
///
/// Shapes accepted:
/// - `query`: `[num_seqs, num_heads, head_dim]` or `[num_seqs, num_heads, 1, head_dim]`
/// - `key_cache`: `[num_blocks, num_kv_heads, block_size, k_row_bytes]` U8 TurboQuant
/// - `value_cache`: `[num_blocks, num_kv_heads, block_size, v_row_bytes]` U8 TurboQuant
/// - `block_tables`: `[num_seqs, max_num_blocks_per_seq]` I32/U32
/// - `cu_seq_lens`: `[num_seqs + 1]` I32/U32 prefix sums
///
/// This is intentionally opt-in and intended for the `ALLOW_LEGACY` decode path.
#[allow(clippy::too_many_arguments)]
pub fn legacy_flash_attn_decode_turboquant(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => legacy_flash_attn_decode_turboquant_t::<half::f16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            window_size,
        ),
        DType::BF16 => legacy_flash_attn_decode_turboquant_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            window_size,
        ),
        DType::F32 => legacy_flash_attn_decode_turboquant_t::<f32>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            window_size,
        ),
        dt => candle_core::bail!("legacy_flash_attn_decode_turboquant unsupported dtype {dt:?}"),
    }
}
