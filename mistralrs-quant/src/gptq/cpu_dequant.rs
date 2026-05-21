use candle_core::{DType, Device, Result, Shape, Tensor};

use crate::{gather_forward_dequantized, QuantMethod, QuantMethodConfig, UnquantLinear};

pub(super) struct GptqCpuParams<'a> {
    pub q_weight: &'a Tensor,
    pub qzeros: Option<&'a Tensor>,
    pub scales: &'a Tensor,
    pub g_idx: Option<&'a Tensor>,
    pub bias: Option<&'a Tensor>,
    pub bits: i32,
    pub is_marlin: bool,
    pub is_awq: bool,
    pub name: &'static str,
}

fn pack_factor(bits: usize) -> Result<usize> {
    match bits {
        2 | 4 | 8 => Ok(32 / bits),
        other => candle_core::bail!("GPTQ/AWQ CPU dequantization does not support {other} bits"),
    }
}

fn signed_to_u32(value: i32) -> u32 {
    value as u32
}

fn unpack_bits(packed: i32, offset: usize, bits: usize) -> f32 {
    ((signed_to_u32(packed) >> offset) & ((1u32 << bits) - 1)) as f32
}

fn group_for(k: usize, groups: usize, in_features: usize, g_idx: Option<&[i32]>) -> Result<usize> {
    let group = if let Some(g_idx) = g_idx {
        g_idx[k] as usize
    } else {
        k * groups / in_features
    };
    if group >= groups {
        candle_core::bail!("GPTQ/AWQ group index {group} out of range {groups}");
    }
    Ok(group)
}

struct CpuGptqWeights {
    q_weight: Vec<i32>,
    qzeros: Vec<i32>,
    scales: Vec<f32>,
    g_idx: Option<Vec<i32>>,
    experts: usize,
    in_features: usize,
    out_features: usize,
    groups: usize,
    bits: usize,
    is_awq: bool,
}

impl CpuGptqWeights {
    fn new(params: &GptqCpuParams<'_>) -> Result<Self> {
        if params.is_marlin {
            candle_core::bail!(
                "GPTQ/AWQ CPU dequantization does not support Marlin-packed weights"
            );
        }
        let Some(qzeros) = params.qzeros else {
            candle_core::bail!("GPTQ/AWQ CPU dequantization requires qzeros");
        };
        let bits = params.bits as usize;
        let pack = pack_factor(bits)?;
        let q_dims = params.q_weight.dims();
        let s_dims = params.scales.dims();
        let z_dims = qzeros.dims();
        let (experts, in_features, out_features, groups) = match (q_dims, s_dims, z_dims) {
            ([q_rows, q_cols], [groups, out], [z_groups, z_cols]) => {
                let in_features = if params.is_awq {
                    *q_rows
                } else {
                    q_rows * pack
                };
                let out_features = if params.is_awq {
                    q_cols * pack
                } else {
                    *q_cols
                };
                if *groups != *z_groups || *out != out_features {
                    candle_core::bail!(
                        "GPTQ/AWQ CPU shape mismatch q={q_dims:?} scales={s_dims:?} qzeros={z_dims:?}"
                    );
                }
                if *z_cols != out_features.div_ceil(pack) {
                    candle_core::bail!(
                        "GPTQ/AWQ CPU qzeros shape {z_dims:?} does not match out_features {out_features}"
                    );
                }
                (1, in_features, out_features, *groups)
            }
            (
                [experts, q_rows, q_cols],
                [s_experts, groups, out],
                [z_experts, z_groups, z_cols],
            ) => {
                let in_features = if params.is_awq {
                    *q_rows
                } else {
                    q_rows * pack
                };
                let out_features = if params.is_awq {
                    q_cols * pack
                } else {
                    *q_cols
                };
                if experts != s_experts
                    || experts != z_experts
                    || groups != z_groups
                    || *out != out_features
                {
                    candle_core::bail!(
                        "GPTQ/AWQ CPU MoE shape mismatch q={q_dims:?} scales={s_dims:?} qzeros={z_dims:?}"
                    );
                }
                if *z_cols != out_features.div_ceil(pack) {
                    candle_core::bail!(
                        "GPTQ/AWQ CPU MoE qzeros shape {z_dims:?} does not match out_features {out_features}"
                    );
                }
                (*experts, in_features, out_features, *groups)
            }
            _ => {
                candle_core::bail!(
                    "GPTQ/AWQ CPU dequantization expects 2D or 3D tensors, got q={q_dims:?} scales={s_dims:?} qzeros={z_dims:?}"
                )
            }
        };
        let q_weight = params
            .q_weight
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<i32>()?;
        let qzeros = qzeros
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<i32>()?;
        let scales = params
            .scales
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let g_idx = params
            .g_idx
            .map(|g_idx| {
                g_idx
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<i32>()
            })
            .transpose()?;
        Ok(Self {
            q_weight,
            qzeros,
            scales,
            g_idx,
            experts,
            in_features,
            out_features,
            groups,
            bits,
            is_awq: params.is_awq,
        })
    }

    fn qvalue(&self, expert: usize, k: usize, out: usize) -> f32 {
        let pack = 32 / self.bits;
        if self.is_awq {
            let q_cols = self.out_features / pack;
            let base = expert * self.in_features * q_cols;
            return unpack_bits(
                self.q_weight[base + k * q_cols + out / pack],
                (out % pack) * self.bits,
                self.bits,
            );
        }
        let q_rows = self.in_features / pack;
        let base = expert * q_rows * self.out_features;
        unpack_bits(
            self.q_weight[base + (k / pack) * self.out_features + out],
            (k % pack) * self.bits,
            self.bits,
        )
    }

    fn zero(&self, expert: usize, group: usize, out: usize) -> f32 {
        let pack = 32 / self.bits;
        let z_cols = self.out_features.div_ceil(pack);
        let base = expert * self.groups * z_cols + group * z_cols;
        let zero = unpack_bits(
            self.qzeros[base + out / pack],
            (out % pack) * self.bits,
            self.bits,
        );
        if self.is_awq {
            zero
        } else {
            zero + 1.0
        }
    }

    fn scale(&self, expert: usize, group: usize, out: usize) -> f32 {
        self.scales[expert * self.groups * self.out_features + group * self.out_features + out]
    }

    fn group(&self, expert: usize, k: usize) -> Result<usize> {
        let g_idx = self.g_idx.as_deref().map(|g_idx| {
            if self.experts == 1 {
                &g_idx[..self.in_features]
            } else {
                let start = expert * self.in_features;
                &g_idx[start..start + self.in_features]
            }
        });
        group_for(k, self.groups, self.in_features, g_idx)
    }

    fn dequant_value(&self, expert: usize, k: usize, out: usize) -> Result<f32> {
        let group = self.group(expert, k)?;
        Ok(
            (self.qvalue(expert, k, out) - self.zero(expert, group, out))
                * self.scale(expert, group, out),
        )
    }

    fn dequantize(&self) -> Result<Tensor> {
        let mut weights = vec![0f32; self.experts * self.out_features * self.in_features];
        for expert in 0..self.experts {
            for out in 0..self.out_features {
                for k in 0..self.in_features {
                    let dst =
                        expert * self.out_features * self.in_features + out * self.in_features + k;
                    weights[dst] = self.dequant_value(expert, k, out)?;
                }
            }
        }
        let shape = if self.experts == 1 {
            vec![self.out_features, self.in_features]
        } else {
            vec![self.experts, self.out_features, self.in_features]
        };
        Tensor::from_vec(weights, shape.as_slice(), &Device::Cpu)?.to_dtype(DType::F16)
    }

    fn matmul_route(&self, expert: usize, x: &[f32], output: &mut [f32]) -> Result<()> {
        for (out_idx, out) in output.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (k, &xv) in x.iter().enumerate() {
                acc += xv * self.dequant_value(expert, k, out_idx)?;
            }
            *out = acc;
        }
        Ok(())
    }
}

pub(super) fn dequantize_w(params: &GptqCpuParams<'_>) -> Result<Tensor> {
    CpuGptqWeights::new(params)?.dequantize()
}

pub(super) fn forward(params: &GptqCpuParams<'_>, x: &Tensor) -> Result<Tensor> {
    let weight = dequantize_w(params)?
        .to_device(x.device())?
        .to_dtype(x.dtype())?;
    let bias = params
        .bias
        .map(|bias| bias.to_device(x.device()))
        .transpose()?;
    let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
        weight, bias,
    )))?;
    unquant.forward(x)
}

pub(super) fn gather_forward(
    params: &GptqCpuParams<'_>,
    x: &Tensor,
    indices: &Tensor,
) -> Result<Tensor> {
    let weights = CpuGptqWeights::new(params)?;
    if weights.experts == 1 {
        return gather_forward_dequantized(
            params.name,
            weights.dequantize()?,
            params.bias.cloned(),
            x,
            indices,
        );
    }
    let original_device = x.device().clone();
    let original_dtype = x.dtype();
    let (num_tokens, topk, x_slots) = match x.dims() {
        &[tokens, 1, hidden] if hidden == weights.in_features => {
            let (_, topk) = indices.dims2()?;
            (tokens, topk, 1)
        }
        &[tokens, topk, hidden] if hidden == weights.in_features => {
            let (_, ids_topk) = indices.dims2()?;
            if ids_topk != topk {
                candle_core::bail!("GPTQ/AWQ CPU MoE input topk {topk} != ids topk {ids_topk}");
            }
            (tokens, topk, topk)
        }
        dims => candle_core::bail!("GPTQ/AWQ CPU MoE unsupported input shape {dims:?}"),
    };
    let ids = indices.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
    let x = x
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten(0, 1)?
        .to_vec2::<f32>()?;
    let mut out = vec![0f32; num_tokens * topk * weights.out_features];
    for token in 0..num_tokens {
        for slot in 0..topk {
            let expert = ids[token][slot] as usize;
            if expert >= weights.experts {
                candle_core::bail!(
                    "GPTQ/AWQ CPU MoE expert id {expert} out of range {}",
                    weights.experts
                );
            }
            let x_index = token * x_slots + if x_slots == 1 { 0 } else { slot };
            let out_start = (token * topk + slot) * weights.out_features;
            weights.matmul_route(
                expert,
                &x[x_index],
                &mut out[out_start..out_start + weights.out_features],
            )?;
        }
    }
    let mut result = Tensor::from_vec(
        out,
        Shape::from_dims(&[num_tokens, topk, weights.out_features]),
        &Device::Cpu,
    )?;
    if let Some(bias) = params.bias {
        result = result.broadcast_add(&bias.to_device(&Device::Cpu)?.to_dtype(DType::F32)?)?;
    }
    result.to_dtype(original_dtype)?.to_device(&original_device)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack4(values: &[u32]) -> i32 {
        values
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, value)| acc | (value << (i * 4))) as i32
    }

    fn params<'a>(
        q_weight: &'a Tensor,
        qzeros: &'a Tensor,
        scales: &'a Tensor,
        g_idx: Option<&'a Tensor>,
        is_awq: bool,
    ) -> GptqCpuParams<'a> {
        GptqCpuParams {
            q_weight,
            qzeros: Some(qzeros),
            scales,
            g_idx,
            bias: None,
            bits: 4,
            is_marlin: false,
            is_awq,
            name: "gptq",
        }
    }

    #[test]
    fn dequantizes_gptq_weight() -> Result<()> {
        let q_weight = Tensor::from_vec(
            vec![
                pack4(&[1, 2, 3, 4, 5, 6, 7, 8]),
                pack4(&[2, 3, 4, 5, 6, 7, 8, 9]),
            ],
            (1, 2),
            &Device::Cpu,
        )?;
        let qzeros = Tensor::from_vec(vec![0i32], (1, 1), &Device::Cpu)?;
        let scales =
            Tensor::from_vec(vec![1f32, 1.0], (1, 2), &Device::Cpu)?.to_dtype(DType::F16)?;
        let g_idx = Tensor::from_vec(vec![0i32; 8], (8,), &Device::Cpu)?;

        let weight = dequantize_w(&params(&q_weight, &qzeros, &scales, Some(&g_idx), false))?
            .to_dtype(DType::F32)?;
        let values = weight.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(weight.dims(), &[2, 8]);
        assert_eq!(&values[..8], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        assert_eq!(&values[8..], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        Ok(())
    }

    #[test]
    fn gather_forward_dequantizes_selected_awq_experts() -> Result<()> {
        let mut q_data = Vec::new();
        for expert in 0..2 {
            for k in 0..8 {
                let values = (0..8)
                    .map(|out| (k + out + expert) as u32)
                    .collect::<Vec<_>>();
                q_data.push(pack4(&values));
            }
        }
        let q_weight = Tensor::from_vec(q_data, (2, 8, 1), &Device::Cpu)?;
        let qzeros = Tensor::from_vec(
            vec![pack4(&[1; 8]), pack4(&[1; 8])],
            (2, 1, 1),
            &Device::Cpu,
        )?;
        let scales =
            Tensor::from_vec(vec![1f32; 16], (2, 1, 8), &Device::Cpu)?.to_dtype(DType::F16)?;
        let input = Tensor::from_vec(vec![1f32; 16], (2, 1, 8), &Device::Cpu)?;
        let indices = Tensor::from_vec(vec![0u32, 1, 1, 0], (2, 2), &Device::Cpu)?;

        let output = gather_forward(
            &params(&q_weight, &qzeros, &scales, None, true),
            &input,
            &indices,
        )?
        .to_dtype(DType::F32)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;

        assert_eq!(output.dims(), &[2, 2, 8]);
        assert_eq!(&values[..8], &[20., 28., 36., 44., 52., 60., 68., 76.]);
        assert_eq!(&values[8..16], &[28., 36., 44., 52., 60., 68., 76., 84.]);
        assert_eq!(&values[16..24], &[28., 36., 44., 52., 60., 68., 76., 84.]);
        assert_eq!(&values[24..], &[20., 28., 36., 44., 52., 60., 68., 76.]);
        Ok(())
    }
}
