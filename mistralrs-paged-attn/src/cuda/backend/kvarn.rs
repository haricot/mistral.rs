use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::DeviceRepr;
use candle_core::cuda_backend::CudaDType;
use candle_core::{DType, Result, Storage, Tensor};
use std::ffi::c_int;

const KVARN_GROUP: usize = 128;
const KVARN_KEY_BITS: usize = 4;
const KVARN_VALUE_BITS: usize = 2;

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => {
            candle_core::bail!("kvarn_flash_attn_decode only supports f16/bf16/f32 ({other:?})")
        }
    }
}

fn k_head_dim_from_record_bytes(record_bytes: usize) -> Result<usize> {
    let fixed = 1 + 2 * KVARN_GROUP;
    let per_dim = KVARN_GROUP * KVARN_KEY_BITS / 8 + 4;
    if record_bytes < fixed || (record_bytes - fixed) % per_dim != 0 {
        candle_core::bail!("KVarN key record byte count {record_bytes} is invalid.");
    }
    Ok((record_bytes - fixed) / per_dim)
}

fn v_head_dim_from_record_bytes(record_bytes: usize) -> Result<usize> {
    let fixed = 1 + 4 * KVARN_GROUP;
    let per_dim = KVARN_GROUP * KVARN_VALUE_BITS / 8 + 2;
    if record_bytes < fixed || (record_bytes - fixed) % per_dim != 0 {
        candle_core::bail!("KVarN value record byte count {record_bytes} is invalid.");
    }
    Ok((record_bytes - fixed) / per_dim)
}

fn use_cc61_kernel() -> bool {
    !crate::cuda::USE_FLASHINFER
        || std::env::var("MISTRALRS_KVARN_DECODE_CC61")
            .map(|v| !v.trim().is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
}

fn kvarn_flash_attn_decode_t<T: CudaDType + DeviceRepr>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        candle_core::bail!(
            "kvarn_flash_attn_decode expects u8 KVarN cache tensors, got {:?} and {:?}",
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    if !matches!(block_tables.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "kvarn_flash_attn_decode expects i32/u32 block_tables, got {:?}",
            block_tables.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "kvarn_flash_attn_decode expects i32/u32 cu_seq_lens, got {:?}",
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
            "kvarn_flash_attn_decode expects query rank 3 or 4, got rank {other} shape {:?}",
            query.shape()
        ),
    };
    if q_len != 1 {
        candle_core::bail!("kvarn_flash_attn_decode expects q_len == 1, got {q_len}");
    }

    let (num_blocks, num_kv_heads, k_record_bytes) = key_cache.dims3()?;
    let (v_num_blocks, v_num_kv_heads, v_record_bytes) = value_cache.dims3()?;
    if (num_blocks, num_kv_heads) != (v_num_blocks, v_num_kv_heads) {
        candle_core::bail!(
            "kvarn_flash_attn_decode cache layout mismatch: key={:?}, value={:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let k_head_dim = k_head_dim_from_record_bytes(k_record_bytes)?;
    let v_head_dim = v_head_dim_from_record_bytes(v_record_bytes)?;
    if k_head_dim != head_dim || v_head_dim != head_dim {
        candle_core::bail!(
            "kvarn_flash_attn_decode head dim mismatch: q={head_dim}, k={k_head_dim}, v={v_head_dim}"
        );
    }
    if !matches!(head_dim, 32 | 64 | 128 | 256 | 512) {
        candle_core::bail!("kvarn_flash_attn_decode unsupported head_dim={head_dim}");
    }
    if num_heads % num_kv_heads != 0 {
        candle_core::bail!(
            "kvarn_flash_attn_decode: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
    }

    let (bt_num_seqs, block_table_stride) = block_tables.dims2()?;
    if bt_num_seqs != num_seqs {
        candle_core::bail!(
            "kvarn_flash_attn_decode block_tables seq mismatch: got {bt_num_seqs}, expected {num_seqs}"
        );
    }
    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    if cu_seq_lens_len != num_seqs + 1 {
        candle_core::bail!(
            "kvarn_flash_attn_decode cu_seq_lens length mismatch: got {cu_seq_lens_len}, expected {}",
            num_seqs + 1
        );
    }

    let out = if q_rank == 4 {
        Tensor::zeros(
            (num_seqs, num_heads, 1usize, head_dim),
            query.dtype(),
            query.device(),
        )?
    } else {
        Tensor::zeros(
            (num_seqs, num_heads, head_dim),
            query.dtype(),
            query.device(),
        )?
    };

    let dtype_code = dtype_code(query.dtype())?;

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_flash_attn_decode: query must be CUDA"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_flash_attn_decode: key_cache must be CUDA"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_flash_attn_decode: value_cache must be CUDA"),
    };
    let binding = out.clone();
    let (o_s, o_l) = binding.storage_and_layout();
    let o_s = match &*o_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_flash_attn_decode: output must be CUDA"),
    };

    let (q_ptr, _q_guard) = slice_ptr(q_s.as_cuda_slice::<T>()?, q_l.start_offset());
    let (kc_ptr, _kc_guard) = slice_ptr(kc_s.as_cuda_slice::<u8>()?, kc_l.start_offset());
    let (vc_ptr, _vc_guard) = slice_ptr(vc_s.as_cuda_slice::<u8>()?, vc_l.start_offset());
    let (out_ptr, _out_guard) = slice_ptr(o_s.as_cuda_slice::<T>()?, o_l.start_offset());

    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_flash_attn_decode: block_tables must be CUDA"),
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
        _ => candle_core::bail!("kvarn_flash_attn_decode: cu_seq_lens must be CUDA"),
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
        if use_cc61_kernel() {
            ffi::kvarn_flash_attn_decode_cc61(
                q_ptr as *const core::ffi::c_void,
                kc_ptr as *const core::ffi::c_void,
                vc_ptr as *const core::ffi::c_void,
                bt_ptr,
                cu_ptr,
                out_ptr as *mut core::ffi::c_void,
                num_seqs as c_int,
                block_table_stride as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                k_record_bytes as c_int,
                v_record_bytes as c_int,
                softmax_scale,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        } else {
            ffi::kvarn_flash_attn_decode(
                q_ptr as *const core::ffi::c_void,
                kc_ptr as *const core::ffi::c_void,
                vc_ptr as *const core::ffi::c_void,
                bt_ptr,
                cu_ptr,
                out_ptr as *mut core::ffi::c_void,
                num_seqs as c_int,
                block_table_stride as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                k_record_bytes as c_int,
                v_record_bytes as c_int,
                softmax_scale,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        }
    }

    Ok(out)
}

pub fn kvarn_flash_attn_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => kvarn_flash_attn_decode_t::<half::f16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
        ),
        DType::BF16 => kvarn_flash_attn_decode_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
        ),
        DType::F32 => kvarn_flash_attn_decode_t::<f32>(
            query,
            key_cache,
            value_cache,
            block_tables,
            cu_seq_lens,
            softmax_scale,
        ),
        dt => candle_core::bail!("kvarn_flash_attn_decode unsupported dtype {dt:?}"),
    }
}
