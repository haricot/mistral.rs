use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle::backend::BackendStorage;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceSlice};
use std::ffi::c_int;

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle::bail!("legacy_flash_attn only supports f16/bf16/f32 ({other:?})"),
    }
}

struct LegacyFlashAttnDecodeDense {
    key: Tensor,
    value: Tensor,
    softmax_scale: f32,
    window_size: usize,
}

impl LegacyFlashAttnDecodeDense {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        let dtype_code = dtype_code(dtype)?;
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let (batch_size, num_heads, q_len, head_dim) = q_l.shape().dims4()?;
        if q_len != 1 {
            candle::bail!("legacy_flash_attn_decode_dense expects q_len == 1, got {q_len}");
        }

        let (k_s, k_l) = self.key.storage_and_layout();
        let k_cuda = match &*k_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("legacy_flash_attn_decode_dense: key must be CUDA"),
        };
        let (k_batch, num_kv_heads, kv_len, k_head_dim) = k_l.shape().dims4()?;
        if k_batch != batch_size || k_head_dim != head_dim {
            candle::bail!(
                "legacy_flash_attn_decode_dense: key shape mismatch, q={:?}, k={:?}",
                q_l.shape(),
                k_l.shape()
            );
        }

        let (v_s, v_l) = self.value.storage_and_layout();
        let v_cuda = match &*v_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("legacy_flash_attn_decode_dense: value must be CUDA"),
        };
        let (v_batch, v_kv_heads, v_len, v_head_dim) = v_l.shape().dims4()?;
        if (v_batch, v_kv_heads, v_len, v_head_dim) != (batch_size, num_kv_heads, kv_len, head_dim)
        {
            candle::bail!(
                "legacy_flash_attn_decode_dense: value shape mismatch, k={:?}, v={:?}",
                k_l.shape(),
                v_l.shape()
            );
        }
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "legacy_flash_attn_decode_dense: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            );
        }

        let q_slice = q.as_cuda_slice::<T>()?;
        let q_view = q_slice.slice(q_l.start_offset()..);
        let (q_ptr, _q_guard) = q_view.device_ptr(q_view.stream());

        let k_slice = k_cuda.as_cuda_slice::<T>()?;
        let k_view = k_slice.slice(k_l.start_offset()..);
        let (k_ptr, _k_guard) = k_view.device_ptr(k_view.stream());

        let v_slice = v_cuda.as_cuda_slice::<T>()?;
        let v_view = v_slice.slice(v_l.start_offset()..);
        let (v_ptr, _v_guard) = v_view.device_ptr(v_view.stream());

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;
        let (out_ptr, out_guard) = out.device_ptr(out.stream());

        unsafe {
            ffi::legacy_flash_attn_decode_dense(
                q_ptr as *const std::ffi::c_void,
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                batch_size as c_int,
                kv_len as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                self.softmax_scale,
                self.window_size as c_int,
                dev.cuda_stream().cu_stream(),
                dtype_code,
            );
        }
        drop(out_guard);
        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out_storage, out_shape))
    }
}

impl candle::CustomOp1 for LegacyFlashAttnDecodeDense {
    fn name(&self) -> &'static str {
        "legacy-flash-attn-decode-dense"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no CPU support for legacy_flash_attn_decode_dense")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F16 => self.cuda_fwd_t::<half::f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<half::bf16>(q, q_l),
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            dt => candle::bail!("legacy_flash_attn_decode_dense unsupported dtype {dt:?}"),
        }
    }
}

/// Decode-only legacy streaming attention over dense K/V.
///
/// Shapes:
/// - q: `[batch, num_heads, 1, head_dim]`
/// - k: `[batch, num_kv_heads, kv_len, head_dim]`
/// - v: `[batch, num_kv_heads, kv_len, head_dim]`
/// - returns `[batch, num_heads, 1, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn legacy_flash_attn_decode_dense(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let op = LegacyFlashAttnDecodeDense {
        key: k,
        value: v,
        softmax_scale,
        window_size,
    };
    q.apply_op1(op)
}

fn legacy_flash_attn_decode_paged_t<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let query = query.contiguous()?;
    let block_tables = block_tables.contiguous()?;
    let context_lens = context_lens.contiguous()?;

    if key_cache.dtype() != query.dtype() || value_cache.dtype() != query.dtype() {
        candle::bail!(
            "legacy_flash_attn_decode_paged expects q/k/v same dtype, got q={:?}, k={:?}, v={:?}",
            query.dtype(),
            key_cache.dtype(),
            value_cache.dtype()
        );
    }
    if !matches!(block_tables.dtype(), DType::I32 | DType::U32) {
        candle::bail!("legacy_flash_attn_decode_paged expects i32/u32 block_tables");
    }
    if !matches!(context_lens.dtype(), DType::I32 | DType::U32) {
        candle::bail!("legacy_flash_attn_decode_paged expects i32/u32 context_lens");
    }

    let (num_seqs, num_heads, head_dim) = query.dims3()?;
    let (num_blocks, num_kv_heads, head_dim_over_x, block_size, x) = key_cache.dims5()?;
    if head_dim_over_x * x != head_dim {
        candle::bail!(
            "legacy_flash_attn_decode_paged key cache head dim mismatch: head_dim_over_x={head_dim_over_x}, x={x}, query head_dim={head_dim}"
        );
    }
    let (v_num_blocks, v_num_kv_heads, v_head_dim, v_block_size) = value_cache.dims4()?;
    if (v_num_blocks, v_num_kv_heads, v_head_dim, v_block_size)
        != (num_blocks, num_kv_heads, head_dim, block_size)
    {
        candle::bail!(
            "legacy_flash_attn_decode_paged value cache layout mismatch: key={:?}, value={:?}",
            key_cache.shape(),
            value_cache.shape()
        );
    }
    let (bt_num_seqs, max_num_blocks_per_seq) = block_tables.dims2()?;
    if bt_num_seqs != num_seqs {
        candle::bail!(
            "legacy_flash_attn_decode_paged block_tables seq mismatch: got {bt_num_seqs}, expected {num_seqs}"
        );
    }
    if context_lens.dims1()? != num_seqs {
        candle::bail!(
            "legacy_flash_attn_decode_paged context_lens mismatch: got {}, expected {num_seqs}",
            context_lens.dims1()?
        );
    }
    if num_heads % num_kv_heads != 0 {
        candle::bail!(
            "legacy_flash_attn_decode_paged: num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        );
    }

    let out = Tensor::zeros(
        (num_seqs, num_heads, head_dim),
        query.dtype(),
        query.device(),
    )?;
    let dtype_code = dtype_code(query.dtype())?;

    let (q_s, q_l) = query.storage_and_layout();
    let q_s = match &*q_s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("legacy_flash_attn_decode_paged: query must be CUDA"),
    };
    let (kc_s, kc_l) = key_cache.storage_and_layout();
    let kc_s = match &*kc_s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("legacy_flash_attn_decode_paged: key_cache must be CUDA"),
    };
    let (vc_s, vc_l) = value_cache.storage_and_layout();
    let vc_s = match &*vc_s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("legacy_flash_attn_decode_paged: value_cache must be CUDA"),
    };
    let binding = out.clone();
    let (o_s, o_l) = binding.storage_and_layout();
    let o_s = match &*o_s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("legacy_flash_attn_decode_paged: output must be CUDA"),
    };

    let (q_ptr, _q_guard) = slice_ptr(q_s.as_cuda_slice::<T>()?, q_l.start_offset());
    let (kc_ptr, _kc_guard) = slice_ptr(kc_s.as_cuda_slice::<T>()?, kc_l.start_offset());
    let (vc_ptr, _vc_guard) = slice_ptr(vc_s.as_cuda_slice::<T>()?, vc_l.start_offset());
    let (out_ptr, _out_guard) = slice_ptr(o_s.as_cuda_slice::<T>()?, o_l.start_offset());

    let (bt_s, bt_l) = block_tables.storage_and_layout();
    let bt_s = match &*bt_s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("legacy_flash_attn_decode_paged: block_tables must be CUDA"),
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
        _ => candle::bail!("legacy_flash_attn_decode_paged: context_lens must be CUDA"),
    };
    let (cl_ptr, _cl_guard) = if context_lens.dtype() == DType::I32 {
        let (ptr, guard) = slice_ptr(cl_s.as_cuda_slice::<i32>()?, cl_l.start_offset());
        (ptr as *const i32, guard)
    } else {
        let (ptr, guard) = slice_ptr(cl_s.as_cuda_slice::<u32>()?, cl_l.start_offset());
        (ptr as *const i32, guard)
    };

    let k_stride = kc_l.stride();
    let v_stride = vc_l.stride();
    let max_context_len = max_num_blocks_per_seq * block_size;
    let dev = q_s.device();

    unsafe {
        ffi::legacy_flash_attn_decode_paged(
            q_ptr as *const std::ffi::c_void,
            kc_ptr as *const std::ffi::c_void,
            vc_ptr as *const std::ffi::c_void,
            bt_ptr,
            cl_ptr,
            out_ptr as *mut std::ffi::c_void,
            num_seqs as c_int,
            max_context_len as c_int,
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
            window_size as c_int,
            dev.cuda_stream().cu_stream(),
            dtype_code,
        );
    }

    Ok(out)
}

/// Decode-only legacy streaming attention over paged K/V cache.
///
/// Shapes:
/// - query: `[num_seqs, num_heads, head_dim]`
/// - key_cache: `[num_blocks, num_kv_heads, head_dim / x, block_size, x]`
/// - value_cache: `[num_blocks, num_kv_heads, head_dim, block_size]`
/// - block_tables: `[num_seqs, max_num_blocks_per_seq]`
/// - context_lens: `[num_seqs]`
/// - returns `[num_seqs, num_heads, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn legacy_flash_attn_decode_paged(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    match query.dtype() {
        DType::F16 => legacy_flash_attn_decode_paged_t::<half::f16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            softmax_scale,
            window_size,
        ),
        DType::BF16 => legacy_flash_attn_decode_paged_t::<half::bf16>(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            softmax_scale,
            window_size,
        ),
        DType::F32 => legacy_flash_attn_decode_paged_t::<f32>(
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            softmax_scale,
            window_size,
        ),
        dt => candle::bail!("legacy_flash_attn_decode_paged unsupported dtype {dt:?}"),
    }
}
