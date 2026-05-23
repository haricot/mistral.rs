mod cpu_dequant;
#[cfg(feature = "cuda")]
mod ffi;
#[cfg(not(feature = "cuda"))]
mod gptq_cpu;
#[cfg(feature = "cuda")]
mod gptq_cuda;
#[cfg(feature = "cuda")]
mod marlin_backend;
#[cfg(feature = "cuda")]
mod marlin_ffi;

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

use crate::{
    has_missing_required_tensors, make_dummy_or_error, QuantMethod, QuantMethodConfig,
    QuantizedConfig, ShardedVarBuilder,
};

#[cfg(not(feature = "cuda"))]
pub use gptq_cpu::{gptq_linear, GptqLayer};
#[cfg(feature = "cuda")]
pub use gptq_cuda::{gptq_linear, GptqLayer};

fn pack_factor(bits: usize) -> Result<usize> {
    match bits {
        2 | 4 | 8 => Ok(32 / bits),
        other => candle_core::bail!("GPTQ/AWQ MoE loading does not support {other} bits"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gptq_moe_linear(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    bias: bool,
    experts_vb: ShardedVarBuilder,
    projection: &str,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        candle_core::bail!("GPTQ/AWQ MoE experts do not support bias.");
    }

    let QuantizedConfig::GptqAwq {
        bits,
        group_size,
        checkpoint_format,
        is_awq,
    } = config
    else {
        candle_core::bail!("Unexpected quantization config.")
    };
    if checkpoint_format
        .as_ref()
        .is_some_and(|fmt| fmt == "marlin")
    {
        candle_core::bail!("Marlin GPTQ/AWQ MoE checkpoints cannot run CPU-MoE yet.");
    }

    let pack = pack_factor(*bits)?;
    let is_awq = *is_awq;
    let mut required = vec!["qweight", "qzeros", "scales"];
    if !is_awq {
        required.push("g_idx");
    }

    let expert0_vb = experts_vb.pp("0").pp(projection);
    if has_missing_required_tensors(&expert0_vb, &required) {
        return make_dummy_or_error("gptq_awq_moe_linear", &expert0_vb, &required);
    }

    let qw_shape = if is_awq {
        (in_dim, out_dim / pack)
    } else {
        (in_dim / pack, out_dim)
    };
    let scale_and_zero_size = in_dim / group_size;
    let qzeros_shape = (scale_and_zero_size, out_dim / pack);
    let scales_shape = (scale_and_zero_size, out_dim);

    let mut qweights = Vec::with_capacity(num_experts);
    let mut qzeros = Vec::with_capacity(num_experts);
    let mut scales = Vec::with_capacity(num_experts);
    let mut g_idx = Vec::with_capacity(num_experts);

    for expert in 0..num_experts {
        let expert_vb = experts_vb.pp(expert).pp(projection);
        qweights.push(expert_vb.get_with_hints_dtype(
            qw_shape,
            "qweight",
            Default::default(),
            DType::I32,
        )?);
        qzeros.push(expert_vb.get_with_hints_dtype(
            qzeros_shape,
            "qzeros",
            Default::default(),
            DType::I32,
        )?);
        scales.push(expert_vb.get_with_hints_dtype(
            scales_shape,
            "scales",
            Default::default(),
            DType::F16,
        )?);
        if !is_awq {
            g_idx.push(expert_vb.get_with_hints_dtype(
                (in_dim,),
                "g_idx",
                Default::default(),
                DType::I32,
            )?);
        }
    }

    let config = QuantMethodConfig::GptqAwq {
        bits: *bits as i32,
        use_exllama: false,
        q_weight: Tensor::stack(&qweights, 0)?,
        qzeros: Some(Tensor::stack(&qzeros, 0)?),
        scales: Tensor::stack(&scales, 0)?,
        g_idx: if is_awq {
            None
        } else {
            Some(Tensor::stack(&g_idx, 0)?)
        },
        bias: None,
        workspace: None,
        is_marlin: false,
        is_awq,
    };
    Ok(Arc::new(GptqLayer::new(config)?))
}
