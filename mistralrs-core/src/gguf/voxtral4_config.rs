//! B3-A: strict native VoxtralLoader params.json <> GGUF topology contract.
//!
//! Reuses the existing VoxtralModel, VoxtralProcessor and GgufWeightSource.
//! B3-A does not qualify GGUF Q4_K/encoder numerical or audio parity.
use anyhow::{bail, ensure, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::GgufArchive;

use crate::vision_models::voxtral::config::VoxtralConfig;

fn integer(archive: &GgufArchive, key: &str) -> Result<usize> {
    match archive.metadata_value(key) {
        Some(Value::U32(n)) => Ok(usize::try_from(*n)?),
        Some(Value::U64(n)) => Ok(usize::try_from(*n)?),
        _ => bail!("Voxtral4 GGUF missing unsigned metadata {key}"),
    }
}

fn real(archive: &GgufArchive, key: &str) -> Result<f64> {
    match archive.metadata_value(key) {
        Some(Value::F32(n)) => Ok(f64::from(*n)),
        Some(Value::F64(n)) => Ok(*n),
        _ => bail!("Voxtral4 GGUF missing float metadata {key}"),
    }
}

fn require_dimension(actual: usize, expected: usize, key: &str) -> Result<()> {
    ensure!(
        actual == expected,
        "VoxtralLoader params.json {key}={actual} disagrees with GGUF {key}={expected}"
    );
    Ok(())
}

fn require_real(actual: f64, expected: f64, key: &str) -> Result<()> {
    ensure!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= 1e-4_f64.max(expected.abs() * 1e-6),
        "VoxtralLoader params.json {key}={actual} disagrees with GGUF {key}={expected}"
    );
    Ok(())
}

pub(crate) fn validate_native_voxtral4_config(
    archive: &GgufArchive,
    native_config_json: &str,
) -> Result<()> {
    let cfg: VoxtralConfig = serde_json::from_str(native_config_json)?;
    let ns = match archive.metadata_value("general.architecture") {
        Some(Value::String(arch)) if arch == "voxtral4b" => "voxtral4b",
        Some(Value::String(arch)) if arch == "voxtral4" => "voxtral4",
        _ => bail!("invalid Voxtral4 GGUF architecture/config source"),
    };
    let dec = format!("{ns}.llm");
    let audio = format!("{ns}.audio");
    let enc = &cfg.multimodal.whisper_model_args.encoder_args;
    let signal = &enc.audio_encoding_args;
    let ds = &cfg.multimodal.whisper_model_args.downsample_args;
    for (native, key) in [
        (cfg.dim, "d_model"),
        (cfg.n_layers, "n_layers"),
        (cfg.head_dim, "head_dim"),
        (cfg.hidden_dim, "ff_dim"),
        (cfg.n_heads, "n_heads"),
        (cfg.n_kv_heads, "n_kv_heads"),
        (cfg.vocab_size, "vocab_size"),
    ] {
        require_dimension(native, integer(archive, &format!("{dec}.{key}"))?, key)?;
    }
    for (native, key) in [
        (enc.dim, "d_model"),
        (enc.n_layers, "n_layers"),
        (enc.head_dim, "head_dim"),
        (enc.hidden_dim, "ff_dim"),
        (enc.n_heads, "n_heads"),
    ] {
        require_dimension(native, integer(archive, &format!("{audio}.{key}"))?, key)?;
    }
    require_dimension(
        cfg.model_max_length,
        integer(archive, &format!("{dec}.max_pos"))?,
        "llm.max_pos",
    )?;
    require_dimension(
        cfg.sliding_window.unwrap_or(0),
        integer(archive, &format!("{dec}.sliding_window"))?,
        "llm.sliding_window",
    )?;
    require_dimension(
        enc.sliding_window.unwrap_or(0),
        integer(archive, &format!("{audio}.sliding_window"))?,
        "audio.sliding_window",
    )?;
    require_dimension(
        signal.num_mel_bins,
        integer(archive, &format!("{ns}.n_mels"))?,
        "n_mels",
    )?;
    require_dimension(
        signal.sampling_rate as usize,
        integer(archive, &format!("{ns}.sample_rate"))?,
        "sample_rate",
    )?;
    require_dimension(
        signal.window_size,
        integer(archive, &format!("{ns}.win_length"))?,
        "win_length",
    )?;
    require_dimension(
        signal.hop_length,
        integer(archive, &format!("{ns}.hop_length"))?,
        "hop_length",
    )?;
    require_dimension(
        ds.downsample_factor,
        integer(archive, &format!("{ns}.proj.frame_stack"))?,
        "proj.frame_stack",
    )?;
    require_dimension(
        enc.dim * ds.downsample_factor,
        integer(archive, &format!("{ns}.proj.in_dim"))?,
        "proj.in_dim",
    )?;
    require_dimension(
        cfg.dim,
        integer(archive, &format!("{ns}.proj.out_dim"))?,
        "proj.out_dim",
    )?;
    require_dimension(
        cfg.ada_rms_norm_t_cond_dim,
        integer(archive, &format!("{dec}.ada_norm_dim"))?,
        "llm.ada_norm_dim",
    )?;
    require_real(cfg.rope_theta, real(archive, &format!("{dec}.rope_theta"))?, "llm.rope_theta")?;
    require_real(enc.rope_theta, real(archive, &format!("{audio}.rope_theta"))?, "audio.rope_theta")?;
    require_real(cfg.norm_eps, real(archive, &format!("{dec}.rms_norm_eps"))?, "llm.rms_norm_eps")?;
    ensure!(cfg.tied_embeddings, "Voxtral4 GGUF requires tied embeddings");
    ensure!(
        cfg.ada_rms_norm_t_cond,
        "Voxtral4 GGUF decoder requires adaptive RMS normalization"
    );
    ensure!(
        enc.use_biases,
        "Voxtral4 GGUF encoder requires biased query/value/output projections"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_dimension_conflict_is_rejected() {
        let err = require_dimension(3072, 1280, "audio.d_model").unwrap_err().to_string();
        assert!(err.contains("audio.d_model"));
    }

    #[test]
    fn float_tolerance_is_explicit_not_unbounded() {
        assert!(require_real(1e6, 1e6, "rope_theta").is_ok());
        assert!(require_real(1e-5, 9.999_999_747e-6, "norm_eps").is_ok());
        assert!(require_real(1.0, 0.5, "rope_theta").is_err());
        assert!(require_real(f64::NAN, 1.0, "rope_theta").is_err());
    }
}
