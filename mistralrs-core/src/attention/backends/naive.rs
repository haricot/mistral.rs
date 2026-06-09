#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::MatMul;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::attention::{chunked_attention, SdpaParams};

static LOG_LEGACY_REDUCED_PRECISION_ATTENTION_F32: AtomicBool = AtomicBool::new(false);
static NAIVE_SDPA_TOTAL: AtomicUsize = AtomicUsize::new(0);
static NAIVE_SDPA_DECODE: AtomicUsize = AtomicUsize::new(0);
static NAIVE_SDPA_PREFILL: AtomicUsize = AtomicUsize::new(0);

fn naive_sdpa_trace_enabled() -> bool {
    std::env::var("MISTRALRS_NAIVE_SDPA_TRACE")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
}

/// Not *really* sure why this is necessary but it is.
pub(crate) fn maybe_synchronize(device: &Device) -> Result<()> {
    // If less that 4 GB available, synchronize
    #[cfg(target_pointer_width = "64")]
    const FOUR_GIB: usize = 4 * 1024 * 1024 * 1024;
    #[cfg(not(target_pointer_width = "64"))]
    const FOUR_GIB: usize = usize::MAX;
    if MemoryUsage.query(device)?.available() < FOUR_GIB {
        device.synchronize()?;
    }
    Ok(())
}

fn to_dtype_avoiding_legacy_cuda_cast(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        return Ok(tensor.clone());
    }
    if crate::utils::is_legacy_cuda_device(tensor.device()) {
        tensor
            .to_device(&Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(tensor.device())
    } else {
        tensor.to_dtype(dtype)
    }
}

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    let k_dims = k.dims();
    let v_dims = v.dims();

    let k_seq_len = match k.rank() {
        4 => k_dims[2],
        3 => k_dims[0],
        _ => candle_core::bail!(
            "naive_sdpa expected rank-3 or rank-4 K tensor, got {:?}",
            k.shape()
        ),
    };

    let v_seq_len = match v.rank() {
        4 => v_dims[2],
        3 => v_dims[0],
        _ => candle_core::bail!(
            "naive_sdpa expected rank-3 or rank-4 V tensor, got {:?}",
            v.shape()
        ),
    };

    if k_seq_len == 0 || v_seq_len == 0 {
        return Tensor::zeros(q.shape(), q.dtype(), q.device());
    }

    let (batch_size, num_heads, query_len, _) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let value_head_dim = v.dim(3)?;

    let total = NAIVE_SDPA_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if query_len == 1 {
        NAIVE_SDPA_DECODE.fetch_add(1, Ordering::Relaxed);
    } else if query_len <= 64 {
        NAIVE_SDPA_PREFILL.fetch_add(1, Ordering::Relaxed);
    }

    if naive_sdpa_trace_enabled() {
        eprintln!(
            "[naive_sdpa_nonempty] q={:?} k={:?} v={:?} mask_custom={} dtype={:?} device={:?}",
            q.shape(),
            k.shape(),
            v.shape(),
            mask.is_some(),
            q.dtype(),
            q.device(),
        );

        if total % 100 == 0 {
            eprintln!(
                "[naive_sdpa_stats] total={} decode_q1={} prefill_smallq={}",
                total,
                NAIVE_SDPA_DECODE.load(Ordering::Relaxed),
                NAIVE_SDPA_PREFILL.load(Ordering::Relaxed),
            );
        }
    }

    if query_len == 0 || kv_len == 0 {
        return Tensor::zeros(
            (batch_size, num_heads, query_len, value_head_dim),
            q.dtype(),
            q.device(),
        );
    }

    let original_dtype = q.dtype();
    let use_legacy_f32_path = matches!(original_dtype, DType::BF16 | DType::F16)
        && crate::utils::is_legacy_cuda_device(q.device());
    if use_legacy_f32_path
        && !LOG_LEGACY_REDUCED_PRECISION_ATTENTION_F32.swap(true, Ordering::Relaxed)
    {
        tracing::warn!(
            "Keeping naive attention matmuls in F32 on legacy CUDA for reduced-precision stability."
        );
    }

    let q_f32 = use_legacy_f32_path
        .then(|| to_dtype_avoiding_legacy_cuda_cast(q, DType::F32))
        .transpose()?;
    let k_f32 = use_legacy_f32_path
        .then(|| to_dtype_avoiding_legacy_cuda_cast(k, DType::F32))
        .transpose()?;
    let v_f32 = use_legacy_f32_path
        .then(|| to_dtype_avoiding_legacy_cuda_cast(v, DType::F32))
        .transpose()?;
    let mask_f32 = if use_legacy_f32_path {
        mask.map(|mask| to_dtype_avoiding_legacy_cuda_cast(mask, DType::F32))
            .transpose()?
    } else {
        None
    };
    let q = q_f32.as_ref().unwrap_or(q);
    let k = k_f32.as_ref().unwrap_or(k);
    let v = v_f32.as_ref().unwrap_or(v);
    let mask = if use_legacy_f32_path {
        mask_f32.as_ref()
    } else {
        mask
    };

    // Use chunked attention with a closure that captures the necessary parameters
    let out = chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
        let q_chunk = q_chunk.contiguous()?;
        let kt = k.t()?.contiguous()?;
        let mut att = MatMul.matmul_affine_mul(&q_chunk, &kt, sdpa_params.softmax_scale.into())?;

        if let Some(softcap) = sdpa_params.softcap {
            att = (att / softcap as f64)?;
            att = att.tanh()?;
            att = (att * softcap as f64)?;
        }

        if let Some(mask) = mask_chunk {
            att = att.broadcast_add(mask)?;
        }

        // Compute softmax in F32 for precision (BF16 exp() loses information).
        let att_dtype = att.dtype();
        if att_dtype == candle_core::DType::BF16 || att_dtype == candle_core::DType::F16 {
            att = to_dtype_avoiding_legacy_cuda_cast(&att, DType::F32)?;
        }
        att = candle_nn::ops::softmax_last_dim(&att)?;
        if !use_legacy_f32_path && att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att.contiguous()?, &v.contiguous()?)
    })?;

    if use_legacy_f32_path {
        to_dtype_avoiding_legacy_cuda_cast(&out, original_dtype)
    } else {
        Ok(out)
    }
}
