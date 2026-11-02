use std::{collections::BTreeMap, path::Path};

use candle_core::{Result, Tensor};
use serde::Deserialize;

const PROFILE_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DSparkVerifyProfile {
    schema_version: u32,
    draft_tokens: usize,
    batch_step_ms: BTreeMap<usize, Vec<f64>>,
}

impl DSparkVerifyProfile {
    pub(crate) fn load(path: &Path, max_drafts: usize) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(candle_core::Error::msg)?;
        let profile: Self = serde_json::from_str(&text).map_err(candle_core::Error::msg)?;
        profile.validate(max_drafts)?;
        Ok(profile)
    }

    fn validate(&self, max_drafts: usize) -> Result<()> {
        if self.schema_version != PROFILE_VERSION
            || self.draft_tokens != max_drafts
            || max_drafts == 0
            || self.batch_step_ms.is_empty()
        {
            candle_core::bail!("DSpark verify profile requires schema_version 1 and batch_step_ms");
        }
        for (batch, costs) in &self.batch_step_ms {
            if *batch == 0
                || costs.len() != max_drafts
                || costs.iter().any(|cost| !cost.is_finite() || *cost <= 0.0)
            {
                candle_core::bail!(
                    "DSpark verify profile batch {batch} requires exactly {max_drafts} positive finite step costs"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn select_prefix(&self, confidence: &Tensor) -> Result<usize> {
        let conditional = confidence
            .to_dtype(candle_core::DType::F32)?
            .to_vec2::<f32>()?;
        self.select_rows(&conditional)
    }

    fn select_rows(&self, conditional: &[Vec<f32>]) -> Result<usize> {
        let max_drafts = conditional.first().map_or(0, Vec::len);
        if max_drafts == 0
            || conditional.iter().any(|row| {
                row.len() != max_drafts
                    || row
                        .iter()
                        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
            })
        {
            candle_core::bail!(
                "DSpark confidence must contain equal nonempty rows of probabilities"
            );
        }
        let batch = conditional.len();
        let Some(costs) = self.batch_step_ms.get(&batch) else {
            tracing::debug!(
                batch,
                max_drafts,
                reason = "unprofiled_batch",
                "DSpark verify decision"
            );
            return Ok(max_drafts);
        };
        if costs.len() != max_drafts {
            candle_core::bail!("DSpark verify profile does not cover {max_drafts} drafts");
        }
        let mut survival = vec![1.0f64; batch];
        let mut expected_tokens = batch as f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut selected = 1;
        for position in 0..max_drafts {
            for (alive, row) in survival.iter_mut().zip(conditional) {
                *alive *= f64::from(row[position]);
                expected_tokens += *alive;
            }
            let score = expected_tokens / costs[position];
            if score > best_score {
                best_score = score;
                selected = position + 1;
            }
        }
        tracing::debug!(
            batch,
            max_drafts,
            selected,
            best_score,
            "DSpark verify decision"
        );
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::DSparkVerifyProfile;

    fn cost_profile(costs: &[f64]) -> DSparkVerifyProfile {
        DSparkVerifyProfile {
            schema_version: 1,
            draft_tokens: costs.len(),
            batch_step_ms: [(1, costs.to_vec())].into(),
        }
    }

    #[test]
    fn dspark_confidence_and_measured_cost_change_the_verify_prefix() -> candle_core::Result<()> {
        let profile = cost_profile(&[1.0, 1.05, 2.0]);
        assert_eq!(profile.select_rows(&[vec![0.9, 0.9, 0.9]])?, 2);
        assert_eq!(profile.select_rows(&[vec![0.9, 0.01, 0.01]])?, 1);
        assert_eq!(cost_profile(&[1.0; 3]).select_rows(&[vec![0.9; 3]])?, 3);
        Ok(())
    }

    #[test]
    fn dspark_expected_progress_uses_conditional_prefix_survival() -> candle_core::Result<()> {
        assert_eq!(cost_profile(&[1.0, 1.2]).select_rows(&[vec![0.5, 0.5]])?, 1);
        Ok(())
    }

    #[test]
    fn dspark_dense_batches_share_one_width_and_unknown_sizes_keep_full_width(
    ) -> candle_core::Result<()> {
        let profile = DSparkVerifyProfile {
            schema_version: 1,
            draft_tokens: 3,
            batch_step_ms: [(2, vec![1.0, 1.2, 3.0])].into(),
        };
        assert_eq!(profile.select_rows(&[vec![0.9; 3], vec![0.9; 3]])?, 2);
        assert_eq!(profile.select_rows(&[vec![0.01; 3]])?, 3);
        Ok(())
    }

    #[test]
    fn dspark_verify_profile_rejects_incomplete_or_invalid_costs_and_scores() {
        assert!(cost_profile(&[1.0]).validate(2).is_err());
        assert!(cost_profile(&[0.0]).validate(1).is_err());
        assert!(cost_profile(&[f64::NAN]).validate(1).is_err());
        assert!(cost_profile(&[1.0]).select_rows(&[vec![f32::NAN]]).is_err());
        assert!(cost_profile(&[1.0]).select_rows(&[vec![1.1]]).is_err());
        assert!(cost_profile(&[1.0, 1.0])
            .select_rows(&[vec![0.5], vec![0.5, 0.5]])
            .is_err());
    }
}
