use std::fs;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::UqffTensor;
use serde_json::json;

use super::{
    dflash::{CtxAppend, DFlashConfig, DFlashDraftModel, DFlashLoadTarget},
    MtpConfig,
};

const HIDDEN: usize = 16;
const VOCAB: usize = 32;
const FP4: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];

pub(crate) fn checkpoint(packed: bool) -> Result<tempfile::TempDir> {
    let dir = tempfile::tempdir().map_err(candle_core::Error::msg)?;
    let mut config = json!({
        "architectures": ["DFlashDraftModel"], "hidden_size": HIDDEN,
        "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8, "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB, "num_target_layers": 4, "block_size": 4,
        "rope_theta": 1_000_000., "has_own_lm_head": true,
        "layer_types": ["sliding_attention", "sliding_attention"], "sliding_window": 1024,
        "dflash_config": {"mask_token_id": 4, "target_layer_ids": [0, 2],
            "causal": false, "use_swa": true, "swa_window_size": 1024}
    });
    if packed {
        config["quantization_config"] = json!({"quant_method": "modelopt_fp4", "quant_algo": "W4A16_NVFP4",
            "group_size": 16, "exclude_modules": ["*norm", "*layernorm"]});
    }
    fs::write(dir.path().join("config.json"), config.to_string())
        .map_err(candle_core::Error::msg)?;
    let mut tensors = Vec::new();
    let mut weight = |name: String, rows: usize, cols: usize, norm: bool| -> Result<()> {
        if norm {
            let t = Tensor::ones((rows,), DType::F32, &Device::Cpu)?;
            tensors.push(UqffTensor::from_tensor(name, &t)?);
        } else {
            let bytes = (0..rows * cols / 2)
                .map(|i| (i % 16) as u8 | ((((i + 5) % 16) as u8) << 4))
                .collect::<Vec<_>>();
            if packed {
                tensors.push(UqffTensor::from_tensor(
                    &name,
                    &Tensor::from_vec(bytes, (rows, cols / 2), &Device::Cpu)?,
                )?);
                let prefix = name.strip_suffix("weight").unwrap();
                tensors.push(UqffTensor::from_tensor(
                    format!("{prefix}weight_scale"),
                    &Tensor::ones((rows, cols / 16), DType::F32, &Device::Cpu)?
                        .to_dtype(DType::F8E4M3)?,
                )?);
                tensors.push(UqffTensor::from_tensor(
                    format!("{prefix}weight_scale_2"),
                    &Tensor::new(0.125f32, &Device::Cpu)?,
                )?);
            } else {
                let values = bytes
                    .into_iter()
                    .flat_map(|b| [b & 15, b >> 4])
                    .map(|b| {
                        let value = FP4[(b & 7) as usize] * 0.125;
                        if b & 8 == 0 {
                            value
                        } else {
                            -value
                        }
                    })
                    .collect::<Vec<_>>();
                tensors.push(UqffTensor::from_tensor(
                    name,
                    &Tensor::from_vec(values, (rows, cols), &Device::Cpu)?,
                )?);
            }
        }
        Ok(())
    };
    for layer in 0..2 {
        let p = format!("layers.{layer}");
        for (name, rows, cols) in [
            ("self_attn.q_proj", 16, 16),
            ("self_attn.k_proj", 8, 16),
            ("self_attn.v_proj", 8, 16),
            ("self_attn.o_proj", 16, 16),
            ("mlp.gate_proj", 32, 16),
            ("mlp.up_proj", 32, 16),
            ("mlp.down_proj", 16, 32),
        ] {
            weight(format!("{p}.{name}.weight"), rows, cols, false)?;
        }
        for (name, size) in [
            ("self_attn.q_norm", 8),
            ("self_attn.k_norm", 8),
            ("input_layernorm", 16),
            ("post_attention_layernorm", 16),
        ] {
            weight(format!("{p}.{name}.weight"), size, 1, true)?;
        }
    }
    weight("fc.weight".into(), 16, 32, false)?;
    weight("lm_head.weight".into(), VOCAB, HIDDEN, false)?;
    weight("norm.weight".into(), 16, 1, true)?;
    weight("hidden_norm.weight".into(), 16, 1, true)?;
    safetensors::serialize_to_file(
        tensors.iter().map(|t| (t.name(), t)),
        None,
        &dir.path().join("model.safetensors"),
    )
    .map_err(candle_core::Error::msg)?;
    Ok(dir)
}

fn load(dir: &tempfile::TempDir) -> Result<DFlashDraftModel> {
    DFlashDraftModel::load(
        &MtpConfig::new(dir.path().to_string_lossy(), Some(3)),
        DFlashLoadTarget {
            num_layers: 4,
            hidden_size: HIDDEN,
            yarn_rope_config: None,
            device: &Device::Cpu,
            dtype: DType::F32,
        },
        true,
    )
}

#[test]
fn bolt_native_nvfp4_matches_independent_e2m1_expansion() -> Result<()> {
    let packed = checkpoint(true)?;
    let dense = checkpoint(false)?;
    let a = load(&packed)?;
    let b = load(&dense)?;
    let h_a = a
        .own_lm_head()
        .unwrap()
        .dequantize_w()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let h_b = b
        .own_lm_head()
        .unwrap()
        .dequantize_w()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(h_a, h_b);
    let taps = vec![
        Tensor::arange(0f32, 48., &Device::Cpu)?.reshape((1, 3, HIDDEN))?,
        Tensor::arange(1f32, 49., &Device::Cpu)?.reshape((1, 3, HIDDEN))?,
    ];
    let entries = [CtxAppend {
        seq_id: 7,
        rows: 3,
        start_pos: 0,
    }];
    for model in [&a, &b] {
        model.append_ctx_batch(&taps, vec![0, 1, 2], &entries)?;
    }
    let noise = (Tensor::arange(0f32, 64., &Device::Cpu)?.reshape((1, 4, HIDDEN))? * 0.01)?;
    let x = a.draft_hidden_batch(&[7], &noise, &[3])?;
    let y = b.draft_hidden_batch(&[7], &noise, &[3])?;
    let error = (x - y)?
        .abs()?
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .fold(0f32, f32::max);
    assert!(
        error < 1e-6,
        "native NVFP4 draft differs from its dense reference: {error}"
    );
    a.release_seqs(&[7]);
    assert_eq!(a.ctx_next_pos(7), None);
    Ok(())
}

#[test]
fn bolt_native_reserves_expanded_runtime_weights() -> Result<()> {
    let dir = checkpoint(true)?;
    let config = MtpConfig::new(dir.path().to_string_lossy(), Some(3));
    let bytes = config.external_weight_size_in_bytes(DType::F16)?;
    const WEIGHT_ELEMENTS: usize = 5760;
    assert_eq!(bytes, WEIGHT_ELEMENTS * DType::F16.size_in_bytes());
    let cfg: DFlashConfig =
        serde_json::from_str(&fs::read_to_string(dir.path().join("config.json")).unwrap()).unwrap();
    assert_eq!(cfg.target_layer_ids()?, vec![0, 2]);
    assert_eq!(cfg.block_size(), 4);
    Ok(())
}
