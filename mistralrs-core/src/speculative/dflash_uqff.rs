//! Schema and preflight for independent DFlash UQFF exports.
//!
//! This module deliberately does not claim to have exported a draft checkpoint.
//! A DFlash draft is not a normal target model, and its compatibility metadata
//! must survive quantization before the runtime can read a draft UQFF.

use candle_core::{Error, Result};
use serde::{Deserialize, Serialize};
use crate::IsqType;
use super::dflash::DFlashConfig;

pub const DFLASH_UQFF_PLAN_SCHEMA: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftUqffPlan {
    pub schema_version: u32,
    pub source_architecture: String,
    pub quantization: String,
    pub layers: usize,
    pub hidden_size: usize,
    pub vocabulary_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub owns_lm_head: bool,
    pub source_is_nvfp4: bool,
    pub note: String,
}

/// Validate a draft checkpoint's architecture and token-tap compatibility
/// before invoking any generic UQFF model loader or CUDA allocation.
pub fn plan_draft_uqff(cfg: &DFlashConfig, target: IsqType) -> Result<DraftUqffPlan> {
    let architecture = cfg.architectures.iter()
        .find(|architecture| architecture.contains("DFlash"))
        .cloned()
        .ok_or_else(|| Error::msg("draft UQFF requires DFlash/DFlash2 architecture"))?;
    if cfg.num_hidden_layers == 0 || cfg.hidden_size == 0 || cfg.vocab_size == 0 {
        candle_core::bail!("draft UQFF config must specify nonzero geometry");
    }
    let taps = cfg.target_layer_ids()?;
    if taps.is_empty() || taps.iter().enumerate().any(|(i, id)| taps[..i].contains(id)) {
        candle_core::bail!("draft UQFF target taps must be nonempty and unique");
    }
    // Supported intent for the first exporter iteration. This is only a
    // preflight plan, NOT evidence that quantization or UQFF loading works.
    if target != IsqType::Q8_0 {
        candle_core::bail!("draft UQFF preflight currently accepts Q8_0 only");
    }
    Ok(DraftUqffPlan {
        schema_version: DFLASH_UQFF_PLAN_SCHEMA,
        source_architecture: architecture,
        quantization: target.to_string(),
        layers: cfg.num_hidden_layers,
        hidden_size: cfg.hidden_size,
        vocabulary_size: cfg.vocab_size,
        target_layer_ids: taps,
        owns_lm_head: cfg.has_own_lm_head,
        source_is_nvfp4: cfg.quantization_config.is_some(),
        note: "preflight-only; export, verification and reload are not yet implemented".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bolt_draft_preflight_preserves_taps_and_head_contract() -> Result<()> {
        let cfg: DFlashConfig = serde_json::from_str(
            include_str!("fixtures/bolt/native-draft-config.json")
        ).map_err(Error::msg)?;
        let plan = plan_draft_uqff(&cfg, IsqType::Q8_0)?;
        assert_eq!(plan.layers, cfg.num_hidden_layers);
        assert_eq!(plan.target_layer_ids, vec![5, 11, 17, 23, 35]);
        assert!(plan.owns_lm_head);
        assert!(plan.source_is_nvfp4);
        Ok(())
    }

    #[test]
    fn non_dflash_model_is_not_a_draft() {
        let mut cfg: DFlashConfig = serde_json::from_str(
            include_str!("fixtures/bolt/native-draft-config.json")
        ).unwrap();
        cfg.architectures = vec!["Qwen3ForCausalLM".into()];
        assert!(plan_draft_uqff(&cfg, IsqType::Q8_0).is_err());
    }
}
