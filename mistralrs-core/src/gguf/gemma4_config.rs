use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::Value;
use serde_json::json;

use crate::vision_models::gemma4::config::Gemma4Config;

const PROPORTIONAL_ROTARY_FACTOR: f64 = 0.25;

pub(crate) fn prepare_gemma4_text_config(
    external: Option<&str>,
    metadata: &HashMap<String, Value>,
    tensor_names: &[String],
) -> Result<String> {
    ensure!(
        matches!(metadata.get("general.architecture"), Some(Value::String(s)) if s == "gemma4"),
        "Gemma 4 text loading requires gemma4 GGUF metadata"
    );
    let vocab = match metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(tokens)) if !tokens.is_empty() => tokens.len(),
        _ => bail!("Gemma 4 GGUF has no tokenizer vocabulary"),
    };
    let layers = integer(metadata, "block_count")?;
    let hidden = integer(metadata, "embedding_length")?;
    let global_head = integer(metadata, "attention.key_length")?;
    ensure!(
        global_head == integer(metadata, "attention.value_length")?,
        "Gemma 4 K/V head dimensions differ"
    );
    let local_head = integer(metadata, "attention.key_length_swa")?;
    ensure!(
        local_head == integer(metadata, "attention.value_length_swa")?,
        "Gemma 4 sliding K/V head dimensions differ"
    );
    let shared = optional_integer(metadata, "attention.shared_kv_layers")?.unwrap_or(0);
    ensure!(shared < layers, "Gemma 4 shared KV layer count is invalid");
    let ple = optional_integer(metadata, "embedding_length_per_layer_input")?.unwrap_or(0);
    ensure!(
        ple == 0
            || tensor_names
                .iter()
                .any(|name| name == "per_layer_token_embd.weight"),
        "Gemma 4 GGUF is missing its per-layer token embeddings"
    );
    let pattern = match metadata.get("gemma4.attention.sliding_window_pattern") {
        Some(Value::Array(values)) if values.len() == layers => values
            .iter()
            .map(|value| match value {
                Value::Bool(true) => Ok("sliding_attention"),
                Value::Bool(false) => Ok("full_attention"),
                _ => bail!("Gemma 4 sliding attention pattern must contain booleans"),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => bail!("Gemma 4 GGUF needs a sliding attention pattern for every layer"),
    };
    let tie = !tensor_names.iter().any(|name| name == "output.weight");
    let mut config = if let Some(external) = external {
        serde_json::from_str::<serde_json::Value>(external)?
    } else {
        ensure!(
            optional_integer(metadata, "expert_count")?.unwrap_or(0) == 0,
            "Gemma 4 MoE text GGUF requires its original model configuration"
        );
        json!({"text_config": {
            "hidden_size": hidden,
            "intermediate_size": integer(metadata, "feed_forward_length")?,
            "num_hidden_layers": layers,
            "num_attention_heads": integer(metadata, "attention.head_count")?,
            "num_key_value_heads": integer(metadata, "attention.head_count_kv")?,
            "head_dim": local_head,
            "global_head_dim": global_head,
            "num_kv_shared_layers": shared,
            "hidden_size_per_layer_input": (ple > 0).then_some(ple),
            "vocab_size_per_layer_input": (ple > 0).then_some(vocab),
            "rms_norm_eps": real(metadata, "attention.layer_norm_rms_epsilon")?,
            "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": pattern,
            "sliding_window": integer(metadata, "attention.sliding_window")?,
            "max_position_embeddings": integer(metadata, "context_length")?,
            "rope_theta": real(metadata, "rope.freq_base")?,
            "rope_parameters": {
                "full_attention": {"rope_type": "proportional",
                    "rope_theta": real(metadata, "rope.freq_base")?,
                    "partial_rotary_factor": PROPORTIONAL_ROTARY_FACTOR},
                "sliding_attention": {"rope_type": "default", "rope_theta": real(metadata, "rope.freq_base_swa")?}
            },
            "final_logit_softcapping": metadata.get("gemma4.final_logit_softcapping").map(number).transpose()?,
            "use_bidirectional_attention": null,
            "use_double_wide_mlp": false
        }})
    };
    let object = config
        .as_object_mut()
        .context("Gemma 4 configuration must be an object")?;
    if !object.contains_key("text_config") {
        let text = object.clone();
        object.clear();
        object.insert("text_config".into(), text.into());
    }
    object.insert(
        "architectures".into(),
        json!(["Gemma4ForConditionalGeneration"]),
    );
    object.insert("vision_config".into(), serde_json::Value::Null);
    object.insert("audio_config".into(), serde_json::Value::Null);
    object.remove("quantization_config");
    let text = object
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
        .context("Gemma 4 text_config must be an object")?;
    text.remove("quantization_config");
    text.insert("vocab_size".into(), json!(vocab));
    text.insert("tie_word_embeddings".into(), json!(tie));
    let encoded = serde_json::to_string(&config)?;
    let parsed: Gemma4Config = serde_json::from_str(&encoded)?;
    let c = &parsed.text_config;
    ensure!(
        c.hidden_size == hidden
            && c.num_hidden_layers == layers
            && c.num_attention_heads == integer(metadata, "attention.head_count")?
            && c.num_key_value_heads == integer(metadata, "attention.head_count_kv")?
            && c.intermediate_size == integer(metadata, "feed_forward_length")?
            && c.head_dim == local_head
            && c.global_head_dim == global_head
            && c.num_kv_shared_layers == shared
            && c.hidden_size_per_layer_input.unwrap_or(0) == ple
            && c.layer_types == pattern,
        "Gemma 4 external configuration disagrees with its GGUF architecture"
    );
    ensure!(
        c.max_position_embeddings <= integer(metadata, "context_length")?,
        "Gemma 4 requested context exceeds the GGUF context limit"
    );
    Ok(encoded)
}

fn optional_integer(metadata: &HashMap<String, Value>, suffix: &str) -> Result<Option<usize>> {
    metadata
        .get(&format!("gemma4.{suffix}"))
        .map(|v| match v {
            Value::U32(v) => Ok(*v as usize),
            Value::U64(v) => Ok(usize::try_from(*v)?),
            Value::I32(v) => Ok(usize::try_from(*v)?),
            _ => bail!("Gemma 4 metadata {suffix} must be an integer"),
        })
        .transpose()
}

fn integer(metadata: &HashMap<String, Value>, suffix: &str) -> Result<usize> {
    let value =
        optional_integer(metadata, suffix)?.with_context(|| format!("Missing gemma4.{suffix}"))?;
    ensure!(value > 0, "gemma4.{suffix} must be positive");
    Ok(value)
}

fn number(value: &Value) -> Result<f64> {
    let value = match value {
        Value::F32(v) => f64::from(*v),
        Value::F64(v) => *v,
        _ => bail!("Gemma 4 metadata must be floating point"),
    };
    ensure!(value.is_finite(), "Gemma 4 metadata must be finite");
    Ok(value)
}

fn real(metadata: &HashMap<String, Value>, suffix: &str) -> Result<f64> {
    let value = number(
        metadata
            .get(&format!("gemma4.{suffix}"))
            .with_context(|| format!("Missing gemma4.{suffix}"))?,
    )?;
    ensure!(value > 0.0, "gemma4.{suffix} must be positive");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bolt_official_gguf_and_hf_configs_describe_the_same_text_architecture() -> Result<()> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../speculative/fixtures/bolt/target-gguf-metadata.json"
        ))?;
        let mut metadata = HashMap::new();
        for (key, value) in fixture["metadata"].as_object().unwrap() {
            let value = match value {
                serde_json::Value::String(s) => Value::String(s.clone()),
                serde_json::Value::Number(n) if n.is_u64() => Value::U64(n.as_u64().unwrap()),
                serde_json::Value::Number(n) => Value::F64(n.as_f64().unwrap()),
                serde_json::Value::Array(a) => Value::Array(
                    a.iter()
                        .map(|v| Value::Bool(v.as_bool().unwrap()))
                        .collect(),
                ),
                _ => continue,
            };
            metadata.insert(key.clone(), value);
        }
        let vocab = fixture["vocabulary_size"].as_u64().unwrap() as usize;
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); vocab]),
        );
        let names = fixture["tensors"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let hf = include_str!("../speculative/fixtures/bolt/target-hf-text-config.json");
        let a: Gemma4Config =
            serde_json::from_str(&prepare_gemma4_text_config(None, &metadata, &names)?)?;
        let b: Gemma4Config =
            serde_json::from_str(&prepare_gemma4_text_config(Some(hf), &metadata, &names)?)?;
        assert_eq!(a.text_config.hidden_size, b.text_config.hidden_size);
        assert_eq!(a.text_config.vocab_size, 262144);
        assert_eq!(
            a.text_config.num_kv_shared_layers,
            b.text_config.num_kv_shared_layers
        );
        assert_eq!(a.text_config.layer_types, b.text_config.layer_types);
        assert_eq!(
            a.text_config.partial_rotary_factor(),
            b.text_config.partial_rotary_factor()
        );
        assert_eq!(
            a.text_config.rope_local_base_freq(),
            b.text_config.rope_local_base_freq()
        );
        Ok(())
    }

    fn metadata() -> HashMap<String, Value> {
        let mut m = HashMap::from([
            (
                "general.architecture".into(),
                Value::String("gemma4".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                Value::Array(vec![Value::String(String::new()); 64]),
            ),
        ]);
        for (key, value) in [
            ("block_count", 42),
            ("context_length", 65536),
            ("embedding_length", 2560),
            ("feed_forward_length", 10240),
            ("attention.head_count", 8),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 512),
            ("attention.value_length", 512),
            ("attention.key_length_swa", 256),
            ("attention.value_length_swa", 256),
            ("attention.sliding_window", 512),
            ("attention.shared_kv_layers", 18),
            ("embedding_length_per_layer_input", 256),
        ] {
            m.insert(format!("gemma4.{key}"), Value::U32(value));
        }
        for (key, value) in [
            ("rope.freq_base", 1_000_000.),
            ("rope.freq_base_swa", 10_000.),
            ("attention.layer_norm_rms_epsilon", 1e-6),
            ("final_logit_softcapping", 30.),
        ] {
            m.insert(format!("gemma4.{key}"), Value::F64(value));
        }
        m.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            Value::Array((0..42).map(|i| Value::Bool(i % 6 != 5)).collect()),
        );
        m
    }

    #[test]
    fn bolt_gguf_preserves_ple_shared_kv_and_proportional_rope() -> Result<()> {
        let c =
            prepare_gemma4_text_config(None, &metadata(), &["per_layer_token_embd.weight".into()])?;
        let c: Gemma4Config = serde_json::from_str(&c)?;
        assert!(c.vision_config.is_none() && c.audio_config.is_none());
        assert_eq!(c.text_config.num_kv_shared_layers, 18);
        assert_eq!(c.text_config.hidden_size_per_layer_input, Some(256));
        assert_eq!(c.text_config.partial_rotary_factor(), 0.25);
        assert_eq!(c.text_config.head_dim, 256);
        assert_eq!(c.text_config.global_head_dim, 512);
        assert_eq!(c.text_config.max_position_embeddings, 65536);
        Ok(())
    }

    #[test]
    fn bolt_gguf_rejects_missing_ple_and_truncated_layer_pattern() {
        let mut m = metadata();
        assert!(prepare_gemma4_text_config(None, &m, &[]).is_err());
        m.insert(
            "gemma4.attention.sliding_window_pattern".into(),
            Value::Array(vec![Value::Bool(true)]),
        );
        assert!(
            prepare_gemma4_text_config(None, &m, &["per_layer_token_embd.weight".into()]).is_err()
        );
    }

    #[test]
    fn bolt_external_config_cannot_override_the_gguf_architecture() -> Result<()> {
        let m = metadata();
        let names = ["per_layer_token_embd.weight".into()];
        let mut c: serde_json::Value =
            serde_json::from_str(&prepare_gemma4_text_config(None, &m, &names)?)?;
        c["text_config"]["hidden_size"] = json!(2048);
        assert!(prepare_gemma4_text_config(Some(&c.to_string()), &m, &names).is_err());
        Ok(())
    }
}
