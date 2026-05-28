use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::turboquant_gather_kv_cache as ffi_turboquant_gather_kv_cache;
use crate::cuda::ffi::turboquant_reshape_and_cache as ffi_turboquant_reshape_and_cache;
use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice};
use candle_core::cuda_backend::CudaDType;
use candle_core::{DType, IndexOp, Result, Storage, Tensor};
use std::ffi::c_int;

const NORM_BYTES: usize = std::mem::size_of::<f32>();
const MSE_BITS: usize = 3;

pub fn turboquant_reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    match key.dtype() {
        DType::F16 => {
            update_turboquant_cache::<half::f16>(key, value, key_cache, value_cache, slot_mapping)
        }
        DType::BF16 => {
            update_turboquant_cache::<half::bf16>(key, value, key_cache, value_cache, slot_mapping)
        }
        DType::F32 => {
            update_turboquant_cache::<f32>(key, value, key_cache, value_cache, slot_mapping)
        }
        dt => candle_core::bail!(
            "turboquant_reshape_and_cache is only supported for f32, f16 and bf16 ({dt:?})"
        ),
    }
}

fn update_turboquant_cache<T: CudaDType + DeviceRepr>(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    if value.dtype() != key.dtype() {
        candle_core::bail!(
            "turboquant_reshape_and_cache expects matching key/value dtypes, got {:?} and {:?}",
            key.dtype(),
            value.dtype()
        );
    }
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        candle_core::bail!(
            "turboquant_reshape_and_cache expects u8 cache tensors, got {:?} and {:?}",
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    if slot_mapping.dtype() != DType::I64 {
        candle_core::bail!(
            "turboquant_reshape_and_cache expects i64 slot_mapping (got {:?})",
            slot_mapping.dtype()
        );
    }

    let key = key.contiguous()?;
    let value = value.contiguous()?;
    let slot_mapping = slot_mapping.contiguous()?;

    let (num_tokens, num_heads, k_head_dim) = key.dims3()?;
    let (v_tokens, v_heads, v_head_dim) = value.dims3()?;
    if (num_tokens, num_heads) != (v_tokens, v_heads) {
        candle_core::bail!(
            "turboquant_reshape_and_cache input shape mismatch: key {:?}, value {:?}",
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
            "turboquant_reshape_and_cache cache layout mismatch: key_cache {:?}, value_cache {:?}, input heads {num_heads}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    if k_row_bytes != row_bytes(k_head_dim) || v_row_bytes != row_bytes(v_head_dim) {
        candle_core::bail!(
            "turboquant_reshape_and_cache row size mismatch: cache rows ({k_row_bytes}, {v_row_bytes}), expected ({}, {})",
            row_bytes(k_head_dim),
            row_bytes(v_head_dim)
        );
    }
    if slot_mapping.dims1()? != num_tokens {
        candle_core::bail!(
            "turboquant_reshape_and_cache slot mapping length mismatch: got {}, expected {num_tokens}",
            slot_mapping.dims1()?
        );
    }

    let dtype_code: u32 = match key.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        _ => unreachable!(),
    };

    {
        let (k_s, k_l) = key.storage_and_layout();
        let k_s = match &*k_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("key must be a cuda tensor"),
        };
        let (v_s, v_l) = value.storage_and_layout();
        let v_s = match &*v_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("value must be a cuda tensor"),
        };
        let (kc_s, kc_l) = key_cache.storage_and_layout();
        let kc_s = match &*kc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };
        let (vc_s, vc_l) = value_cache.storage_and_layout();
        let vc_s = match &*vc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };
        let (s_s, s_l) = slot_mapping.storage_and_layout();
        let s_s = match &*s_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("slot_mapping must be a cuda tensor"),
        };

        let k = k_s.as_cuda_slice::<T>()?;
        let v = v_s.as_cuda_slice::<T>()?;
        let key_cache = kc_s.as_cuda_slice::<u8>()?;
        let value_cache = vc_s.as_cuda_slice::<u8>()?;
        let slot_mapping = s_s.as_cuda_slice::<i64>()?;

        let k = k.slice(k_l.start_offset()..);
        let v = v.slice(v_l.start_offset()..);
        let (key_cache_ptr, _kc_guard) = slice_ptr(key_cache, kc_l.start_offset());
        let (value_cache_ptr, _vc_guard) = slice_ptr(value_cache, vc_l.start_offset());
        let slot_mapping = slot_mapping.slice(s_l.start_offset()..);

        let (k_ptr, _k_guard) = k.device_ptr(k.stream());
        let (v_ptr, _v_guard) = v.device_ptr(v.stream());
        let (slot_ptr, _slot_guard) = slot_mapping.device_ptr(slot_mapping.stream());

        let key_stride = k_l.stride()[0] as c_int;
        let value_stride = v_l.stride()[0] as c_int;
        let dev = k_s.device();

        unsafe {
            ffi_turboquant_reshape_and_cache(
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                key_cache_ptr as *const core::ffi::c_void,
                value_cache_ptr as *const core::ffi::c_void,
                slot_ptr as *const core::ffi::c_long,
                num_tokens as c_int,
                num_heads as c_int,
                k_head_dim as c_int,
                v_head_dim as c_int,
                block_size as c_int,
                num_blocks as c_int,
                k_row_bytes as c_int,
                v_row_bytes as c_int,
                key_stride,
                value_stride,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        }
    }

    Ok(())
}

pub fn turboquant_gather_kv_cache(
    key_cache: &Tensor,   // [num_blocks, kv_heads, block_size, k_row_bytes]
    value_cache: &Tensor, // [num_blocks, kv_heads, block_size, v_row_bytes]
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    if key_cache.dtype() != DType::U8 || value_cache.dtype() != DType::U8 {
        candle_core::bail!(
            "turboquant_gather_kv_cache expects u8 cache tensors, got {:?} and {:?}",
            key_cache.dtype(),
            value_cache.dtype()
        );
    }

    let (num_blocks, num_kv_heads, block_size, k_row_bytes) = key_cache.dims4()?;
    let (v_num_blocks, v_num_kv_heads, v_block_size, v_row_bytes) = value_cache.dims4()?;
    if (num_blocks, num_kv_heads, block_size) != (v_num_blocks, v_num_kv_heads, v_block_size) {
        candle_core::bail!(
            "turboquant_gather_kv_cache cache layout mismatch: key_cache {:?}, value_cache {:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }

    let k_head_dim = head_dim_from_row_bytes(k_row_bytes)?;
    let v_head_dim = head_dim_from_row_bytes(v_row_bytes)?;

    let out_dtype_code: u32 = match out_dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        other => candle_core::bail!(
            "turboquant_gather_kv_cache only supports f16, bf16, f32 output (got {other:?})"
        ),
    };

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;
    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "turboquant_gather_kv_cache expects i32/u32 block_table (got {:?})",
            block_table.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "turboquant_gather_kv_cache expects i32/u32 cu_seq_lens (got {:?})",
            cu_seq_lens.dtype()
        );
    }

    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    if cu_seq_lens_len == 0 {
        candle_core::bail!("turboquant_gather_kv_cache expects non-empty cu_seq_lens");
    }
    let num_seqs = cu_seq_lens_len.saturating_sub(1);
    let num_tokens = if cu_seq_lens.dtype() == DType::I32 {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<i32>()? as usize
    } else {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<u32>()? as usize
    };

    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, k_head_dim), out_dtype, key_cache.device())?;
        let v_out = Tensor::zeros((0, num_kv_heads, v_head_dim), out_dtype, key_cache.device())?;
        return Ok((k_out, v_out));
    }

    let k_out = Tensor::zeros(
        (num_tokens, num_kv_heads, k_head_dim),
        out_dtype,
        key_cache.device(),
    )?;
    let v_out = Tensor::zeros(
        (num_tokens, num_kv_heads, v_head_dim),
        out_dtype,
        key_cache.device(),
    )?;

    {
        let (kc_s, kc_l) = key_cache.storage_and_layout();
        let kc_s = match &*kc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("key_cache must be a cuda tensor"),
        };
        let (vc_s, vc_l) = value_cache.storage_and_layout();
        let vc_s = match &*vc_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("value_cache must be a cuda tensor"),
        };
        let (kc_ptr, _kc_guard) = slice_ptr(kc_s.as_cuda_slice::<u8>()?, kc_l.start_offset());
        let (vc_ptr, _vc_guard) = slice_ptr(vc_s.as_cuda_slice::<u8>()?, vc_l.start_offset());

        let (ko_s, ko_l) = k_out.storage_and_layout();
        let ko_s = match &*ko_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("k_out must be a cuda tensor"),
        };
        let (vo_s, vo_l) = v_out.storage_and_layout();
        let vo_s = match &*vo_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("v_out must be a cuda tensor"),
        };
        let (ko_ptr, _ko_guard) = match out_dtype {
            DType::F16 => slice_ptr(ko_s.as_cuda_slice::<half::f16>()?, ko_l.start_offset()),
            DType::BF16 => slice_ptr(ko_s.as_cuda_slice::<half::bf16>()?, ko_l.start_offset()),
            DType::F32 => slice_ptr(ko_s.as_cuda_slice::<f32>()?, ko_l.start_offset()),
            _ => unreachable!(),
        };
        let (vo_ptr, _vo_guard) = match out_dtype {
            DType::F16 => slice_ptr(vo_s.as_cuda_slice::<half::f16>()?, vo_l.start_offset()),
            DType::BF16 => slice_ptr(vo_s.as_cuda_slice::<half::bf16>()?, vo_l.start_offset()),
            DType::F32 => slice_ptr(vo_s.as_cuda_slice::<f32>()?, vo_l.start_offset()),
            _ => unreachable!(),
        };

        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt_s = match &*bt_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("block_table must be a cuda tensor"),
        };
        let (bt_ptr, _bt_guard) = if block_table.dtype() == DType::I32 {
            let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<i32>()?, bt_l.start_offset());
            (ptr as *const i32, guard)
        } else {
            let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());
            (ptr as *const i32, guard)
        };

        let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
        let cu_s = match &*cu_s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a cuda tensor"),
        };
        let (cu_ptr, _cu_guard) = if cu_seq_lens.dtype() == DType::I32 {
            let (ptr, guard) = slice_ptr(cu_s.as_cuda_slice::<i32>()?, cu_l.start_offset());
            (ptr as *const i32, guard)
        } else {
            let (ptr, guard) = slice_ptr(cu_s.as_cuda_slice::<u32>()?, cu_l.start_offset());
            (ptr as *const i32, guard)
        };

        let (_, block_table_stride) = bt_l.shape().dims2()?;
        let dev = kc_s.device();

        unsafe {
            ffi_turboquant_gather_kv_cache(
                kc_ptr as *const core::ffi::c_void,
                vc_ptr as *const core::ffi::c_void,
                ko_ptr as *const core::ffi::c_void,
                vo_ptr as *const core::ffi::c_void,
                bt_ptr,
                cu_ptr,
                num_tokens as i32,
                num_seqs as i32,
                block_size as i32,
                block_table_stride as i32,
                num_kv_heads as i32,
                k_head_dim as i32,
                v_head_dim as i32,
                k_row_bytes as i32,
                v_row_bytes as i32,
                dev.cuda_stream().cu_stream(),
                out_dtype_code,
            );
        }
    }

    Ok((k_out, v_out))
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

    candle_core::bail!("TurboQuant row byte count {row_len} is not valid for 4-bit packing.")
}

fn row_bytes(head_dim: usize) -> usize {
    2 * NORM_BYTES + (head_dim * MSE_BITS).div_ceil(8) + head_dim.div_ceil(8)
}
