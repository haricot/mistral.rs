//! Strict inventory of Voxtral4 GGUF weights exported by CrispASR.
//! B0: the mapping is a prerequisite; it is not a working audio inference loader.
use std::collections::HashSet;

use anyhow::{bail, ensure, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding};

pub(crate) fn is_voxtral4_architecture(name: &str) -> bool {
    name.eq_ignore_ascii_case("voxtral4") || name.eq_ignore_ascii_case("voxtral4b")
}

fn required_count(archive: &GgufArchive, namespace: &str, component: &str) -> Result<usize> {
    let key = format!("{namespace}.{component}.n_layers");
    let count = match archive.metadata_value(&key) {
        Some(Value::U32(n)) => usize::try_from(*n)?,
        Some(Value::U64(n)) => usize::try_from(*n)?,
        _ => bail!("Voxtral GGUF requires unsigned integer metadata {key}"),
    };
    ensure!(
        (1..=128).contains(&count),
        "Voxtral GGUF {key} must be between 1 and 128, got {count}"
    );
    Ok(count)
}

pub(crate) fn build_voxtral4_bindings(archive: &GgufArchive) -> Result<GgufBindingMap> {
    let arch = match archive.metadata_value("general.architecture") {
        Some(Value::String(name)) => name.as_str(),
        _ => bail!("Voxtral GGUF requires string general.architecture"),
    };
    ensure!(
        is_voxtral4_architecture(arch),
        "unexpected architecture {arch}"
    );
    // Preserve the independently versioned architecture and metadata namespace.
    let ns = if archive.metadata_value("voxtral4b.llm.n_layers").is_some() {
        "voxtral4b"
    } else if archive.metadata_value("voxtral4.llm.n_layers").is_some() {
        "voxtral4"
    } else {
        bail!("Voxtral GGUF missing voxtral4b/voxtral4 LLM layer-count metadata");
    };
    let audio = required_count(archive, ns, "audio")?;
    let text = required_count(archive, ns, "llm")?;
    let names = archive
        .tensors()
        .keys()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    build_from_inventory(&names, audio, text)
}

fn build_from_inventory(
    names: &HashSet<&str>,
    audio_layers: usize,
    text_layers: usize,
) -> Result<GgufBindingMap> {
    ensure!(
        (1..=128).contains(&audio_layers) && (1..=128).contains(&text_layers),
        "invalid Voxtral encoder/decoder layer count"
    );
    let root = "mm_streams_embeddings.embedding_module";
    let enc = format!("{root}.whisper_encoder");
    let mut bindings = GgufBindingMap::new();
    let mut insert = |native: String, source: String| -> Result<()> {
        ensure!(
            names.contains(source.as_str()),
            "Voxtral GGUF missing tensor {source} for native weight {native}"
        );
        ensure!(
            bindings
                .insert(native.clone(), GgufTensorBinding::tensor(source))
                .is_none(),
            "duplicate native Voxtral binding {native}"
        );
        Ok(())
    };
    for (native, gguf) in [
        (format!("{root}.tok_embeddings.weight"), "token_embd.weight"),
        ("norm.weight".to_string(), "output_norm.weight"),
        (
            format!("{enc}.conv_layers.0.conv.weight"),
            "audio.conv.1.weight",
        ),
        (
            format!("{enc}.conv_layers.0.conv.bias"),
            "audio.conv.1.bias",
        ),
        (
            format!("{enc}.conv_layers.1.conv.weight"),
            "audio.conv.2.weight",
        ),
        (
            format!("{enc}.conv_layers.1.conv.bias"),
            "audio.conv.2.bias",
        ),
        (
            format!("{enc}.transformer.norm.weight"),
            "audio.ln_post.weight",
        ),
        (
            format!("{root}.audio_language_projection.0.weight"),
            "proj1.weight",
        ),
        (
            format!("{root}.audio_language_projection.2.weight"),
            "proj2.weight",
        ),
    ] {
        insert(native, gguf.to_string())?;
    }

    for i in 0..audio_layers {
        let native = format!("{enc}.transformer.layers.{i}");
        let gguf = format!("audio.blk.{i}");
        for (dst, src) in [
            ("attention_norm.weight", "attn_norm.weight"),
            ("ffn_norm.weight", "ffn_norm.weight"),
            ("attention.wq.weight", "attn_q.weight"),
            ("attention.wq.bias", "attn_q.bias"),
            ("attention.wk.weight", "attn_k.weight"),
            ("attention.wv.weight", "attn_v.weight"),
            ("attention.wv.bias", "attn_v.bias"),
            ("attention.wo.weight", "attn_out.weight"),
            ("attention.wo.bias", "attn_out.bias"),
            ("feed_forward.w1.weight", "ffn_gate.weight"),
            ("feed_forward.w2.weight", "ffn_down.weight"),
            ("feed_forward.w2.bias", "ffn_down.bias"),
            ("feed_forward.w3.weight", "ffn_up.weight"),
        ] {
            insert(format!("{native}.{dst}"), format!("{gguf}.{src}"))?;
        }
    }

    for i in 0..text_layers {
        let native = format!("layers.{i}");
        let gguf = format!("blk.{i}");
        for (dst, src) in [
            ("attention_norm.weight", "attn_norm.weight"),
            ("ffn_norm.weight", "ffn_norm.weight"),
            ("attention.wq.weight", "attn_q.weight"),
            ("attention.wk.weight", "attn_k.weight"),
            ("attention.wv.weight", "attn_v.weight"),
            ("attention.wo.weight", "attn_output.weight"),
            ("feed_forward.w1.weight", "ffn_gate.weight"),
            ("feed_forward.w2.weight", "ffn_down.weight"),
            ("feed_forward.w3.weight", "ffn_up.weight"),
            ("ada_rms_norm_t_cond.0.weight", "ada_norm_down.weight"),
            ("ada_rms_norm_t_cond.2.weight", "ada_norm_up.weight"),
        ] {
            insert(format!("{native}.{dst}"), format!("{gguf}.{src}"))?;
        }
    }
    Ok(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_aliases() {
        for a in ["voxtral4", "voxtral4b", "VOXTRAL4B"] {
            assert!(is_voxtral4_architecture(a));
        }
        for a in ["voxtral", "mistral", "llama"] {
            assert!(!is_voxtral4_architecture(a));
        }
    }

    #[test]
    fn incomplete_audio_inventory_is_not_silently_accepted() {
        let names = HashSet::from(["token_embd.weight", "output_norm.weight"]);
        let error = build_from_inventory(&names, 32, 26)
            .unwrap_err()
            .to_string();
        assert!(error.contains("audio.conv.1.weight"), "{error}");
    }

    #[test]
    fn invalid_layer_counts_rejected() {
        let names = HashSet::new();
        assert!(build_from_inventory(&names, 0, 26).is_err());
        assert!(build_from_inventory(&names, 32, 0).is_err());
        assert!(build_from_inventory(&names, 129, 26).is_err());
    }
}
