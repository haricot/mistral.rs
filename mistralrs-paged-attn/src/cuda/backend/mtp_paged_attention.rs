use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle_core::backend::BackendStorage;
use candle_core::{DType, Result, Storage, Tensor};
use float8::F8E4M3;
use std::ffi::{c_int, c_void};

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("mtp_paged_attention only supports f16/bf16/f32 ({other:?})"),
    }
}

fn cache_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        DType::F8E4M3 => Ok(3),
        other => candle_core::bail!(
            "mtp_paged_attention only supports f16/bf16/f32/f8e4m3 cache ({other:?})"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn mtp_paged_attention_t<
    T: candle_core::cuda_backend::CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    query: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    batch_size: usize,
    q_len: usize,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    if q_len == 0 {
        candle_core::bail!("mtp_paged_attention expects q_len > 0");
    }
    if q_len > 8 {
        candle_core::bail!("mtp_paged_attention supports q_len <= 8, got {q_len}");
    }

    let query = query.contiguous()?;
    let block_tables = block_tables.contiguous()?;
    let context_lens = context_lens.contiguous()?;

    let dtype = query.dtype();
    let dtype_code = dtype_code(dtype)?;
    let cache_dtype = key_cache.dtype();
    let cache_dtype_code = cache_dtype_code(cache_dtype)?;
    if value_cache.dtype() != cache_dtype {
        candle_core::bail!(
            "mtp_paged_attention expects matching cache dtypes, got {:?} and {:?}",
            cache_dtype,
            value_cache.dtype()
        );
    }
    if cache_dtype == DType::F8E4M3 && !crate::cuda::USE_FP8 {
        candle_core::bail!("FP8 is not supported on this system");
    }
    if cache_dtype != DType::F8E4M3 && cache_dtype != dtype {
        candle_core::bail!(
            "mtp_paged_attention non-FP8 cache must match query dtype, got query={dtype:?}, cache={cache_dtype:?}"
        );
    }

    if !matches!(block_tables.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "mtp_paged_attention expects i32/u32 block_tables, got {:?}",
            block_tables.dtype()
        );
    }
    if !matches!(context_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "mtp_paged_attention expects i32/u32 context_lens, got {:?}",
            context_lens.dtype()
        );
    }

    let (num_rows, num_heads, head_dim) = query.dims3()?;
    if num_rows != batch_size * q_len {
        candle_core::bail!(
            "mtp_paged_attention query rows mismatch: got {num_rows}, expected {}",
            batch_size * q_len
        );
    }
    let (num_blocks, num_kv_heads, head_dim_over_x, block_size, x) = key_cache.dims5()?;
    if head_dim_over_x * x != head_dim {
        candle_core::bail!(
            "mtp_paged_attention key cache head dim mismatch: cache {:?}, query {:?}",
            key_cache.shape(),
            query.shape()
        );
    }
    if value_cache.dims4()? != (num_blocks, num_kv_heads, head_dim, block_size) {
        candle_core::bail!(
            "mtp_paged_attention value cache shape mismatch: key={:?}, value={:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    let (bt_rows, max_num_blocks_per_seq) = block_tables.dims2()?;
    if bt_rows != num_rows {
        candle_core::bail!(
            "mtp_paged_attention block table rows mismatch: got {bt_rows}, expected {num_rows}"
        );
    }
    if context_lens.dims1()? != num_rows {
        candle_core::bail!(
            "mtp_paged_attention context_lens mismatch: got {}, expected {num_rows}",
            context_lens.dims1()?
        );
    }
    if num_heads % num_kv_heads != 0 {
        candle_core::bail!(
            "mtp_paged_attention num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
    }

    let effective_max_context_len = max_context_len.min(max_num_blocks_per_seq * block_size);
    let out = Tensor::zeros((num_rows, num_heads, head_dim), dtype, query.device())?;

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention query must be CUDA"),
    };
    let (q_ptr, _q_guard) = slice_ptr(q_s.as_cuda_slice::<T>()?, q_l.start_offset());

    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention key_cache must be CUDA"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention value_cache must be CUDA"),
    };

    let (kc_ptr, _kc_guard) = if cache_dtype == DType::F8E4M3 {
        slice_ptr(kc_s.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset())
    } else {
        slice_ptr(kc_s.as_cuda_slice::<T>()?, kc_l.start_offset())
    };
    let (vc_ptr, _vc_guard) = if cache_dtype == DType::F8E4M3 {
        slice_ptr(vc_s.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset())
    } else {
        slice_ptr(vc_s.as_cuda_slice::<T>()?, vc_l.start_offset())
    };

    let out_binding = out.clone();
    let (o_s, o_l) = out_binding.storage_and_layout();
    let o_s = match &*o_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention output must be CUDA"),
    };
    let (out_ptr, _out_guard) = slice_ptr(o_s.as_cuda_slice::<T>()?, o_l.start_offset());

    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention block_tables must be CUDA"),
    };
    let (bt_ptr, _bt_guard) = if block_tables.dtype() == DType::I32 {
        let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<i32>()?, bt_l.start_offset());
        (ptr as *const i32, guard)
    } else {
        let (ptr, guard) = slice_ptr(bt_s.as_cuda_slice::<u32>()?, bt_l.start_offset());
        (ptr as *const i32, guard)
    };

    let (cl_s, cl_l) = context_lens.storage_and_layout();
    let cl_s = match &*cl_s {
        Storage::Cuda(s) => s,
        _ => candle_core::bail!("mtp_paged_attention context_lens must be CUDA"),
    };
    let (cl_ptr, _cl_guard) = if context_lens.dtype() == DType::I32 {
        let (ptr, guard) = slice_ptr(cl_s.as_cuda_slice::<i32>()?, cl_l.start_offset());
        (ptr as *const i32, guard)
    } else {
        let (ptr, guard) = slice_ptr(cl_s.as_cuda_slice::<u32>()?, cl_l.start_offset());
        (ptr as *const i32, guard)
    };

    let _ks_storage = k_scale.map(|ks| ks.storage_and_layout());
    let (k_scale_ptr, _ks_guard) = if let Some((ref s, l)) = _ks_storage {
        let s = match &**s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("mtp_paged_attention k_scale must be CUDA"),
        };
        let (ptr, guard) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
        (ptr as *const f32, Some(guard))
    } else if cache_dtype == DType::F8E4M3 {
        candle_core::bail!("mtp_paged_attention FP8 cache requires k_scale")
    } else {
        (std::ptr::null(), None)
    };

    let _vs_storage = v_scale.map(|vs| vs.storage_and_layout());
    let (v_scale_ptr, _vs_guard) = if let Some((ref s, l)) = _vs_storage {
        let s = match &**s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("mtp_paged_attention v_scale must be CUDA"),
        };
        let (ptr, guard) = slice_ptr(s.as_cuda_slice::<f32>()?, l.start_offset());
        (ptr as *const f32, Some(guard))
    } else if cache_dtype == DType::F8E4M3 {
        candle_core::bail!("mtp_paged_attention FP8 cache requires v_scale")
    } else {
        (std::ptr::null(), None)
    };

    let k_stride = kc_l.stride();
    let v_stride = vc_l.stride();
    let dev = q_s.device();

    unsafe {
        ffi::mtp_paged_attention(
            q_ptr as *const c_void,
            kc_ptr as *const c_void,
            vc_ptr as *const c_void,
            k_scale_ptr,
            v_scale_ptr,
            bt_ptr,
            cl_ptr,
            out_ptr as *mut c_void,
            batch_size as c_int,
            q_len as c_int,
            effective_max_context_len as c_int,
            block_size as c_int,
            max_num_blocks_per_seq as c_int,
            num_heads as c_int,
            num_kv_heads as c_int,
            head_dim as c_int,
            x as c_int,
            k_stride[0] as c_int,
            k_stride[1] as c_int,
            v_stride[0] as c_int,
            v_stride[1] as c_int,
            softmax_scale,
            softcapping,
            dev.cuda_stream().cu_stream(),
            dtype_code,
            cache_dtype_code,
        );
    }

    Ok(out)
}

/// Multi-token MTP verification over paged KV cache.
///
/// `query` is `[batch * q_len, num_heads, head_dim]`; `block_tables` and
/// `context_lens` must also have one row per verification token so the kernel
/// can enforce each query token's causal boundary.
#[allow(clippy::too_many_arguments)]
pub fn mtp_paged_attention(
    query: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    batch_size: usize,
    q_len: usize,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => mtp_paged_attention_t::<half::f16>(
            query,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            batch_size,
            q_len,
            max_context_len,
            softmax_scale,
            softcapping,
        ),
        DType::BF16 => mtp_paged_attention_t::<half::bf16>(
            query,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            batch_size,
            q_len,
            max_context_len,
            softmax_scale,
            softcapping,
        ),
        DType::F32 => mtp_paged_attention_t::<f32>(
            query,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            batch_size,
            q_len,
            max_context_len,
            softmax_scale,
            softcapping,
        ),
        other => candle_core::bail!("mtp_paged_attention unsupported dtype {other:?}"),
    }
}
