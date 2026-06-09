use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice};
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
    if let Ok(v) = std::env::var("MISTRALRS_KVARN_DECODE_CC61") {
        return !v.trim().is_empty() && v != "0" && !v.eq_ignore_ascii_case("false");
    }
    !crate::cuda::USE_FLASHINFER
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvarnDecodeKernel {
    Decode,
    Mtp,
}

fn kvarn_flash_attn_decode_t<T: CudaDType + DeviceRepr>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    tail_pool: Option<(&Tensor, &Tensor, &Tensor, &Tensor)>,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
    kernel: KvarnDecodeKernel,
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
    if kernel == KvarnDecodeKernel::Mtp {
        if q_rank != 3 {
            candle_core::bail!(
                "kvarn_flash_attn_decode_mtp expects flattened rank-3 query, got rank {q_rank}"
            );
        }
        if !(2..=8).contains(&num_seqs) {
            candle_core::bail!(
                "kvarn_flash_attn_decode_mtp expects 2..=8 query rows, got {num_seqs}"
            );
        }
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
    let stream = dev.cuda_stream().cu_stream();
    if let Some((key_tail, value_tail, key_tail_slots, value_tail_slots)) = tail_pool {
        if key_tail.dtype() != query.dtype() || value_tail.dtype() != query.dtype() {
            candle_core::bail!(
                "kvarn_flash_attn_decode tail pool dtype mismatch: query={:?}, key_tail={:?}, value_tail={:?}",
                query.dtype(),
                key_tail.dtype(),
                value_tail.dtype()
            );
        }
        let (tail_slots, tail_heads, tail_block, tail_dim) = key_tail.dims4()?;
        let (v_tail_slots, v_tail_heads, v_tail_block, v_tail_dim) = value_tail.dims4()?;
        if (tail_slots, tail_heads, tail_block, tail_dim)
            != (v_tail_slots, v_tail_heads, v_tail_block, v_tail_dim)
            || tail_heads != num_kv_heads
            || tail_block != KVARN_GROUP
            || tail_dim != head_dim
        {
            candle_core::bail!(
                "kvarn_flash_attn_decode tail pool shape mismatch: key_tail={:?}, value_tail={:?}, num_kv_heads={num_kv_heads}, head_dim={head_dim}",
                key_tail.shape(),
                value_tail.shape()
            );
        }
        if key_tail_slots.dtype() != DType::I32 || value_tail_slots.dtype() != DType::I32 {
            candle_core::bail!("kvarn_flash_attn_decode tail slot maps must be i32 tensors");
        }
        if key_tail_slots.dims1()? != num_blocks || value_tail_slots.dims1()? != num_blocks {
            candle_core::bail!(
                "kvarn_flash_attn_decode tail slot map length mismatch: got {} and {}, expected {num_blocks}",
                key_tail_slots.dims1()?,
                value_tail_slots.dims1()?
            );
        }

        let key_tail = key_tail.contiguous()?;
        let value_tail = value_tail.contiguous()?;
        let key_tail_slots = key_tail_slots.contiguous()?;
        let value_tail_slots = value_tail_slots.contiguous()?;
        let (kt_s, kt_l) = key_tail.storage_and_layout();
        let kt_s = match &*kt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kvarn_flash_attn_decode: key_tail must be CUDA"),
        };
        let (vt_s, vt_l) = value_tail.storage_and_layout();
        let vt_s = match &*vt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kvarn_flash_attn_decode: value_tail must be CUDA"),
        };
        let (kts_s, kts_l) = key_tail_slots.storage_and_layout();
        let kts_s = match &*kts_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kvarn_flash_attn_decode: key_tail_slots must be CUDA"),
        };
        let (vts_s, vts_l) = value_tail_slots.storage_and_layout();
        let vts_s = match &*vts_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("kvarn_flash_attn_decode: value_tail_slots must be CUDA"),
        };
        let (kt_ptr, _kt_guard) = slice_ptr(kt_s.as_cuda_slice::<T>()?, kt_l.start_offset());
        let (vt_ptr, _vt_guard) = slice_ptr(vt_s.as_cuda_slice::<T>()?, vt_l.start_offset());
        let (kts_ptr, _kts_guard) = slice_ptr(kts_s.as_cuda_slice::<i32>()?, kts_l.start_offset());
        let (vts_ptr, _vts_guard) = slice_ptr(vts_s.as_cuda_slice::<i32>()?, vts_l.start_offset());

        unsafe {
            if kernel == KvarnDecodeKernel::Mtp {
                let mtp_kernel = if use_cc61_kernel() {
                    ffi::kvarn_flash_attn_decode_mtp_cc61
                } else {
                    ffi::kvarn_flash_attn_decode_mtp
                };
                mtp_kernel(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    kt_ptr as *const core::ffi::c_void,
                    vt_ptr as *const core::ffi::c_void,
                    kts_ptr as *const i32,
                    vts_ptr as *const i32,
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
                    stream,
                    dtype_code,
                );
            } else if use_cc61_kernel() {
                ffi::kvarn_flash_attn_decode_cc61(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    kt_ptr as *const core::ffi::c_void,
                    vt_ptr as *const core::ffi::c_void,
                    kts_ptr as *const i32,
                    vts_ptr as *const i32,
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
                    stream,
                    dtype_code,
                );
            } else {
                ffi::kvarn_flash_attn_decode(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    kt_ptr as *const core::ffi::c_void,
                    vt_ptr as *const core::ffi::c_void,
                    kts_ptr as *const i32,
                    vts_ptr as *const i32,
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
                    stream,
                    dtype_code,
                );
            }
        }
    } else {
        unsafe {
            if kernel == KvarnDecodeKernel::Mtp {
                let mtp_kernel = if use_cc61_kernel() {
                    ffi::kvarn_flash_attn_decode_mtp_cc61
                } else {
                    ffi::kvarn_flash_attn_decode_mtp
                };
                mtp_kernel(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
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
                    stream,
                    dtype_code,
                );
            } else if use_cc61_kernel() {
                ffi::kvarn_flash_attn_decode_cc61(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
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
                    stream,
                    dtype_code,
                );
            } else {
                ffi::kvarn_flash_attn_decode(
                    q_ptr as *const core::ffi::c_void,
                    kc_ptr as *const core::ffi::c_void,
                    vc_ptr as *const core::ffi::c_void,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
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
                    stream,
                    dtype_code,
                );
            }
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
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        DType::BF16 => kvarn_flash_attn_decode_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        DType::F32 => kvarn_flash_attn_decode_t::<f32>(
            query,
            key_cache,
            value_cache,
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        dt => candle_core::bail!("kvarn_flash_attn_decode unsupported dtype {dt:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kvarn_flash_attn_decode_with_tail(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    key_tail_pool: &Tensor,
    value_tail_pool: &Tensor,
    key_tail_slots: &Tensor,
    value_tail_slots: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => kvarn_flash_attn_decode_t::<half::f16>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        DType::BF16 => kvarn_flash_attn_decode_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        DType::F32 => kvarn_flash_attn_decode_t::<f32>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Decode,
        ),
        dt => candle_core::bail!("kvarn_flash_attn_decode unsupported dtype {dt:?}"),
    }
}

pub fn kvarn_flash_attn_decode_mtp(
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
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        DType::BF16 => kvarn_flash_attn_decode_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        DType::F32 => kvarn_flash_attn_decode_t::<f32>(
            query,
            key_cache,
            value_cache,
            None,
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        dt => candle_core::bail!("kvarn_flash_attn_decode_mtp unsupported dtype {dt:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn kvarn_flash_attn_decode_mtp_with_tail(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    key_tail_pool: &Tensor,
    value_tail_pool: &Tensor,
    key_tail_slots: &Tensor,
    value_tail_slots: &Tensor,
    block_tables: &Tensor,
    cu_seq_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => kvarn_flash_attn_decode_t::<half::f16>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        DType::BF16 => kvarn_flash_attn_decode_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        DType::F32 => kvarn_flash_attn_decode_t::<f32>(
            query,
            key_cache,
            value_cache,
            Some((
                key_tail_pool,
                value_tail_pool,
                key_tail_slots,
                value_tail_slots,
            )),
            block_tables,
            cu_seq_lens,
            softmax_scale,
            KvarnDecodeKernel::Mtp,
        ),
        dt => candle_core::bail!("kvarn_flash_attn_decode_mtp unsupported dtype {dt:?}"),
    }
}

pub fn kvarn_store_tail(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    key_tail_pool: &Tensor,
    value_tail_pool: &Tensor,
    slot_mapping: &Tensor,
    key_tail_slots: &Tensor,
    value_tail_slots: &Tensor,
) -> Result<()> {
    match key.dtype() {
        DType::F16 => kvarn_store_tail_t::<half::f16>(
            key,
            value,
            key_cache,
            value_cache,
            key_tail_pool,
            value_tail_pool,
            slot_mapping,
            key_tail_slots,
            value_tail_slots,
        ),
        DType::BF16 => kvarn_store_tail_t::<half::bf16>(
            key,
            value,
            key_cache,
            value_cache,
            key_tail_pool,
            value_tail_pool,
            slot_mapping,
            key_tail_slots,
            value_tail_slots,
        ),
        DType::F32 => kvarn_store_tail_t::<f32>(
            key,
            value,
            key_cache,
            value_cache,
            key_tail_pool,
            value_tail_pool,
            slot_mapping,
            key_tail_slots,
            value_tail_slots,
        ),
        dt => candle_core::bail!("kvarn_store_tail unsupported dtype {dt:?}"),
    }
}

fn kvarn_store_tail_t<T: CudaDType + DeviceRepr>(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    key_tail_pool: &Tensor,
    value_tail_pool: &Tensor,
    slot_mapping: &Tensor,
    key_tail_slots: &Tensor,
    value_tail_slots: &Tensor,
) -> Result<()> {
    if value.dtype() != key.dtype()
        || key_tail_pool.dtype() != key.dtype()
        || value_tail_pool.dtype() != key.dtype()
    {
        candle_core::bail!(
            "kvarn_store_tail expects matching key/value/tail dtypes, got key={:?}, value={:?}, key_tail={:?}, value_tail={:?}",
            key.dtype(),
            value.dtype(),
            key_tail_pool.dtype(),
            value_tail_pool.dtype()
        );
    }
    if slot_mapping.dtype() != DType::I64
        || key_tail_slots.dtype() != DType::I32
        || value_tail_slots.dtype() != DType::I32
    {
        candle_core::bail!(
            "kvarn_store_tail expects i64 slot_mapping and i32 tail slot tensors, got {:?}, {:?}, {:?}",
            slot_mapping.dtype(),
            key_tail_slots.dtype(),
            value_tail_slots.dtype()
        );
    }

    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let slot_mapping = slot_mapping.contiguous()?;
    let key_tail_slots = key_tail_slots.contiguous()?;
    let value_tail_slots = value_tail_slots.contiguous()?;

    let (num_tokens, num_heads, head_dim) = key.dims3()?;
    let (v_tokens, v_heads, v_head_dim) = value.dims3()?;
    if (num_tokens, num_heads, head_dim) != (v_tokens, v_heads, v_head_dim) {
        candle_core::bail!(
            "kvarn_store_tail key/value shape mismatch: key={:?}, value={:?}",
            key.shape(),
            value.shape()
        );
    }
    let (num_tail_slots, tail_heads, tail_block_size, tail_head_dim) = key_tail_pool.dims4()?;
    let (v_tail_slots, v_tail_heads, v_tail_block_size, v_tail_head_dim) =
        value_tail_pool.dims4()?;
    if (num_tail_slots, tail_heads, tail_block_size, tail_head_dim)
        != (
            v_tail_slots,
            v_tail_heads,
            v_tail_block_size,
            v_tail_head_dim,
        )
        || tail_heads != num_heads
        || tail_head_dim != head_dim
        || tail_block_size != KVARN_GROUP
    {
        candle_core::bail!(
            "kvarn_store_tail tail pool shape mismatch: key_tail={:?}, value_tail={:?}, input heads={num_heads}, head_dim={head_dim}",
            key_tail_pool.shape(),
            value_tail_pool.shape()
        );
    }
    if slot_mapping.dims1()? != num_tokens
        || key_tail_slots.dims1()? != num_tokens
        || value_tail_slots.dims1()? != num_tokens
    {
        candle_core::bail!(
            "kvarn_store_tail metadata length mismatch: slot_mapping={}, key_tail_slots={}, value_tail_slots={}, expected {num_tokens}",
            slot_mapping.dims1()?,
            key_tail_slots.dims1()?,
            value_tail_slots.dims1()?
        );
    }

    let dtype_code = dtype_code(key.dtype())?;
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        candle_core::bail!(
            "kvarn_store_tail expects u8 KVarN cache tensors, got {:?} and {:?}",
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    let (num_blocks, cache_heads, key_record_bytes) = key_cache.dims3()?;
    let (value_blocks, value_heads, value_record_bytes) = value_cache.dims3()?;
    if (num_blocks, cache_heads) != (value_blocks, value_heads) || cache_heads != num_heads {
        candle_core::bail!(
            "kvarn_store_tail cache shape mismatch: key_cache={:?}, value_cache={:?}, num_heads={num_heads}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let (k_s, k_l) = key.storage_and_layout();
    let k_s = match &*k_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: key must be CUDA"),
    };
    let (v_s, v_l) = value.storage_and_layout();
    let v_s = match &*v_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: value must be CUDA"),
    };
    let (kt_s, kt_l) = key_tail_pool.storage_and_layout();
    let kt_s = match &*kt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: key_tail_pool must be CUDA"),
    };
    let (vt_s, vt_l) = value_tail_pool.storage_and_layout();
    let vt_s = match &*vt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: value_tail_pool must be CUDA"),
    };
    let (sm_s, sm_l) = slot_mapping.storage_and_layout();
    let sm_s = match &*sm_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: slot_mapping must be CUDA"),
    };
    let (kts_s, kts_l) = key_tail_slots.storage_and_layout();
    let kts_s = match &*kts_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: key_tail_slots must be CUDA"),
    };
    let (vts_s, vts_l) = value_tail_slots.storage_and_layout();
    let vts_s = match &*vts_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: value_tail_slots must be CUDA"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: key_cache must be CUDA"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("kvarn_store_tail: value_cache must be CUDA"),
    };

    let k_slice = k_s.as_cuda_slice::<T>()?;
    let v_slice = v_s.as_cuda_slice::<T>()?;
    let kc_slice = kc_s.as_cuda_slice::<u8>()?;
    let vc_slice = vc_s.as_cuda_slice::<u8>()?;
    let kt_slice = kt_s.as_cuda_slice::<T>()?;
    let vt_slice = vt_s.as_cuda_slice::<T>()?;
    let sm_slice = sm_s.as_cuda_slice::<i64>()?;
    let kts_slice = kts_s.as_cuda_slice::<i32>()?;
    let vts_slice = vts_s.as_cuda_slice::<i32>()?;

    let k_slice = k_slice.slice(k_l.start_offset()..);
    let v_slice = v_slice.slice(v_l.start_offset()..);
    let sm_slice = sm_slice.slice(sm_l.start_offset()..);
    let kts_slice = kts_slice.slice(kts_l.start_offset()..);
    let vts_slice = vts_slice.slice(vts_l.start_offset()..);
    let (k_ptr, _k_guard) = k_slice.device_ptr(k_slice.stream());
    let (v_ptr, _v_guard) = v_slice.device_ptr(v_slice.stream());
    let (sm_ptr, _sm_guard) = sm_slice.device_ptr(sm_slice.stream());
    let (kts_ptr, _kts_guard) = kts_slice.device_ptr(kts_slice.stream());
    let (vts_ptr, _vts_guard) = vts_slice.device_ptr(vts_slice.stream());
    let (kc_ptr, _kc_guard) = slice_ptr(kc_slice, kc_l.start_offset());
    let (vc_ptr, _vc_guard) = slice_ptr(vc_slice, vc_l.start_offset());
    let (kt_ptr, _kt_guard) = slice_ptr(kt_slice, kt_l.start_offset());
    let (vt_ptr, _vt_guard) = slice_ptr(vt_slice, vt_l.start_offset());

    let key_stride = k_l.stride()[0] as c_int;
    let value_stride = v_l.stride()[0] as c_int;
    let dev = k_s.device();
    unsafe {
        ffi::kvarn_store_tail(
            k_ptr as *const core::ffi::c_void,
            v_ptr as *const core::ffi::c_void,
            kc_ptr as *mut core::ffi::c_void,
            vc_ptr as *mut core::ffi::c_void,
            kt_ptr as *mut core::ffi::c_void,
            vt_ptr as *mut core::ffi::c_void,
            sm_ptr as *const core::ffi::c_long,
            kts_ptr as *const i32,
            vts_ptr as *const i32,
            num_tokens as c_int,
            num_heads as c_int,
            head_dim as c_int,
            KVARN_GROUP as c_int,
            num_blocks as c_int,
            key_record_bytes as c_int,
            value_record_bytes as c_int,
            num_tail_slots as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            dtype_code,
        );
    }

    Ok(())
}
