#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::MemoryUsage;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::MatMul;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::attention::{chunked_attention, SdpaParams};

static LOG_LEGACY_REDUCED_PRECISION_ATTENTION_F32: AtomicBool = AtomicBool::new(false);

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

/// Computes softmax(QK^T*sqrt(d_k))V
pub(crate) fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    // Use chunked attention with a closure that captures the necessary parameters
    chunked_attention(q, k, v, mask, |q_chunk, k, v, mask_chunk| {
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
            att = att.to_dtype(candle_core::DType::F32)?;
        }
        att = candle_nn::ops::softmax_last_dim(&att)?;
        if att.dtype() != att_dtype {
            att = att.to_dtype(att_dtype)?;
        }
        MatMul.matmul(&att.contiguous()?, &v.contiguous()?)
    })





    //     let reduced_precision = att_dtype == DType::BF16 || att_dtype == DType::F16;
    //     if reduced_precision {
    //         att = att.to_dtype(DType::F32)?;
    //     }
    //     att = candle_nn::ops::softmax_last_dim(&att)?;
    //     let keep_attention_f32 =
    //         reduced_precision && crate::utils::is_legacy_cuda_device(att.device());
    //     if keep_attention_f32 {
    //         if !LOG_LEGACY_REDUCED_PRECISION_ATTENTION_F32.swap(true, Ordering::Relaxed) {
    //             tracing::warn!(
    //                 "Keeping naive attention probabilities and values in F32 on legacy CUDA for reduced-precision stability."
    //             );
    //         }
    //         MatMul
    //             .matmul(&att.contiguous()?, &v.to_dtype(DType::F32)?.contiguous()?)?
    //             .to_dtype(att_dtype)
    //     } else {
    //         if att.dtype() != att_dtype {
    //             att = att.to_dtype(att_dtype)?;
    //         }
    //         MatMul.matmul(&att.contiguous()?, &v.contiguous()?)
    //     }
    // })
}
