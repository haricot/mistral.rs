use candle_core::{Result, Tensor};

use crate::speculative::{
    dflash::{CtxAppend, DFlashDraftModel, DFlashLoadTarget, DFlashProposalBatch},
    MtpConfig, SpeculativeBatchPlan, SpeculativePrefillCtx, SpeculativeProposal,
    SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeTapRouting,
};

use super::{config::Gemma4TextConfig, text::TextModel};

pub(super) struct Gemma4DflashRuntime {
    pub drafter: DFlashDraftModel,
    pub depth: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::text_models_inputs_processor::FlashParams;
    use crate::pipeline::{ModelForwardContext, MultimodalModel, NormalLoadingMetadata};
    use crate::vision_models::gemma4::{config::Gemma4Config, Gemma4Model, Gemma4SpecificArgs};
    use candle_core::{DType, Device};
    use mistralrs_quant::ShardedSafeTensors;
    use serde_json::json;
    use std::{collections::HashMap, sync::Arc};

    fn target() -> Result<Gemma4Model> {
        let cfg: Gemma4Config = serde_json::from_value(json!({"text_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
            "global_head_dim": 8, "vocab_size": 32, "sliding_window": 8,
            "max_position_embeddings": 32, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 4, "vocab_size_per_layer_input": 32
        }})).map_err(candle_core::Error::msg)?;
        let mut tensors = HashMap::new();
        let mut add = |name: String, rows: usize, cols: Option<usize>| -> Result<()> {
            let t = match cols {
                Some(cols) => {
                    let values = (0..rows * cols)
                        .map(|i| ((i % 29) as f32 - 14.) * 0.01)
                        .collect::<Vec<_>>();
                    Tensor::from_vec(values, (rows, cols), &Device::Cpu)?
                }
                None => Tensor::ones((rows,), DType::F32, &Device::Cpu)?,
            };
            tensors.insert(format!("model.language_model.{name}"), t);
            Ok(())
        };
        add("embed_tokens.weight".into(), 32, Some(16))?;
        add("embed_tokens_per_layer.weight".into(), 32, Some(16))?;
        add("per_layer_model_projection.weight".into(), 16, Some(16))?;
        add("per_layer_projection_norm.weight".into(), 4, None)?;
        add("norm.weight".into(), 16, None)?;
        for i in 0..4 {
            for (name, rows, cols) in [
                ("self_attn.q_proj", 16, 16),
                ("self_attn.k_proj", 8, 16),
                ("self_attn.v_proj", 8, 16),
                ("self_attn.o_proj", 16, 16),
                ("mlp.gate_proj", 32, 16),
                ("mlp.up_proj", 32, 16),
                ("mlp.down_proj", 16, 32),
                ("per_layer_input_gate", 4, 16),
                ("per_layer_projection", 16, 4),
            ] {
                add(format!("layers.{i}.{name}.weight"), rows, Some(cols))?;
            }
            for (name, size) in [
                ("self_attn.q_norm", 8),
                ("self_attn.k_norm", 8),
                ("input_layernorm", 16),
                ("post_attention_layernorm", 16),
                ("pre_feedforward_layernorm", 16),
                ("post_feedforward_layernorm", 16),
                ("post_per_layer_input_norm", 16),
            ] {
                add(format!("layers.{i}.{name}.weight"), size, None)?;
            }
        }
        let vb = ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu);
        Gemma4Model::new(
            &cfg,
            vb,
            true,
            NormalLoadingMetadata {
                mapper: Box::new(crate::device_map::DummyDeviceMapper {
                    nm_device: Device::Cpu,
                }),
                loading_isq: false,
                real_device: Device::Cpu,
                multi_progress: Arc::new(crate::utils::progress::new_multi_progress()),
                matformer_slicing_config: None,
                rope_pairing: None,
            },
            AttentionImplementation::Eager,
        )
    }

    fn forward(model: &Gemma4Model, tokens: &[u32], offset: usize) -> Result<Tensor> {
        let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu)?;
        let offsets = [offset];
        let selection = [(tokens.len() - 1, 1)];
        let positions = [offset + tokens.len()];
        let flash = FlashParams::empty(true);
        let mut ctx = ModelForwardContext::new(&offsets, &selection, &positions, None, &flash);
        MultimodalModel::forward(
            model,
            &ids,
            None,
            Box::<Gemma4SpecificArgs>::default(),
            &mut ctx,
        )
    }

    #[test]
    fn bolt_layer_taps_preserve_logits_order_and_all_shared_kv_rows() -> Result<()> {
        let baseline = target()?;
        let tapped = target()?;
        tapped.language_model.set_dflash_tap_layers(vec![2, 0])?;
        let a = forward(&baseline, &[1, 2, 3, 4], 0)?;
        let b = forward(&tapped, &[1, 2, 3, 4], 0)?;
        let max_error = (a - b)?
            .abs()?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .fold(0f32, f32::max);
        assert!(
            max_error < 1e-4,
            "capturing DFlash layers changed target logits: {max_error}"
        );
        let full = tapped.language_model.dflash_full_taps();
        let selected = tapped.language_model.dflash_spec_taps();
        assert_eq!(full.len(), 2);
        assert_eq!(full[0].dims(), &[1, 4, 16]);
        assert_eq!(selected[0].dims(), &[1, 1, 16]);
        let single = target()?;
        single.language_model.set_dflash_tap_layers(vec![0])?;
        forward(&single, &[1, 2, 3, 4], 0)?;
        assert_eq!(
            full[1].flatten_all()?.to_vec1::<f32>()?,
            single.language_model.dflash_full_taps()[0]
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        forward(&tapped, &[5], 4)?;
        assert_eq!(
            tapped.language_model.dflash_full_taps()[0].dims(),
            &[1, 1, 16]
        );
        tapped.language_model.set_dflash_tap_layers(Vec::new())?;
        assert!(tapped.language_model.dflash_full_taps().is_empty());
        assert!(tapped
            .language_model
            .set_dflash_tap_layers(vec![4])
            .is_err());
        assert!(tapped
            .language_model
            .set_dflash_tap_layers(vec![0, 0])
            .is_err());
        Ok(())
    }

    #[test]
    fn bolt_verified_context_excludes_rejected_rows_and_keeps_sequences_separate() -> Result<()> {
        let dir = crate::speculative::dflash_bolt_tests::checkpoint(true)?;
        let model = target()?;
        let cfg = MtpConfig::new(dir.path().to_string_lossy(), Some(3));
        let actual =
            Gemma4DflashRuntime::load(&cfg, &model.cfg.text_config, &model.language_model)?;
        let expected =
            Gemma4DflashRuntime::load(&cfg, &model.cfg.text_config, &model.language_model)?;
        let taps = vec![
            (Tensor::arange(0f32, 128., &Device::Cpu)?.reshape((2, 4, 16))? * 0.01)?,
            (Tensor::arange(128f32, 256., &Device::Cpu)?.reshape((2, 4, 16))? * 0.01)?,
        ];
        let prefix = [
            CtxAppend {
                seq_id: 7,
                rows: 3,
                start_pos: 0,
            },
            CtxAppend {
                seq_id: 9,
                rows: 2,
                start_pos: 0,
            },
        ];
        for runtime in [&actual, &expected] {
            runtime
                .drafter
                .append_ctx_batch(&taps, vec![0, 1, 2, 4, 5], &prefix)?;
        }
        // Verification rows are permuted; rows after each accepted prefix were rejected.
        assert!(actual.append_verified(&taps, &[9, 7], &[3, 6], &[(0, 1), (1, 3)])?);
        expected.drafter.append_ctx_batch(
            &taps,
            vec![0, 4, 5, 6],
            &[
                CtxAppend {
                    seq_id: 9,
                    rows: 1,
                    start_pos: 2,
                },
                CtxAppend {
                    seq_id: 7,
                    rows: 3,
                    start_pos: 3,
                },
            ],
        )?;
        // Replaying the same notification must not append the prefix twice.
        assert!(actual.append_verified(&taps, &[9, 7], &[3, 6], &[(0, 1), (1, 3)])?);
        assert_eq!(actual.drafter.ctx_next_pos(7), Some(6));
        assert_eq!(actual.drafter.ctx_next_pos(9), Some(3));
        let noise = Tensor::ones((2, 4, 16), DType::F32, &Device::Cpu)?;
        let a = actual
            .drafter
            .draft_hidden_batch(&[9, 7], &noise, &[3, 6])?;
        let b = expected
            .drafter
            .draft_hidden_batch(&[9, 7], &noise, &[3, 6])?;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        actual.drafter.release_seqs(&[9]);
        assert_eq!(actual.drafter.ctx_next_pos(9), None);
        assert_eq!(actual.drafter.ctx_next_pos(7), Some(6));
        assert!(!actual.append_verified(&taps, &[7], &[], &[(0, 1)])?);
        assert!(actual
            .append_verified(&taps, &[7], &[5], &[(0, 1)])
            .is_err());
        Ok(())
    }
}

impl Gemma4DflashRuntime {
    pub fn load(config: &MtpConfig, target: &Gemma4TextConfig, text: &TextModel) -> Result<Self> {
        let cfg = crate::speculative::dflash::peek_config(config)?
            .ok_or_else(|| candle_core::Error::msg("Expected a DFlash checkpoint"))?;
        if cfg.vocab_size != target.vocab_size {
            candle_core::bail!("DFlash vocabulary does not match Gemma 4");
        }
        if cfg
            .architectures
            .iter()
            .any(|name| name.contains("DFlash2"))
        {
            candle_core::bail!(
                "Gemma 4 supports the Bolt DFlash 1 drafter; DFlash 2 is not qualified"
            );
        }
        let drafter = DFlashDraftModel::load(
            config,
            DFlashLoadTarget {
                num_layers: target.num_hidden_layers,
                hidden_size: target.hidden_size,
                yarn_rope_config: None,
                device: text.device(),
                dtype: text.dtype(),
            },
            false,
        )?;
        let max = drafter
            .block_size()
            .checked_sub(1)
            .ok_or_else(|| candle_core::Error::msg("DFlash block must contain an anchor"))?;
        let depth = config
            .n_predict
            .unwrap_or(max.min(crate::speculative::dflash::DEFAULT_MAX_DRAFTS));
        if depth == 0 || depth > max {
            candle_core::bail!("DFlash draft depth {depth} must be between 1 and {max}");
        }
        Ok(Self { drafter, depth })
    }

    pub fn plan(&self) -> SpeculativeBatchPlan {
        SpeculativeBatchPlan::new(self.depth)
    }

    pub fn prefill(&self, text: &TextModel, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        let taps = text.dflash_full_taps();
        if taps.is_empty() {
            return Ok(());
        }
        let (batch, rows, _) = taps[0].dims3()?;
        let routing = SpeculativeTapRouting::new(
            ctx.capture_layout(),
            batch,
            rows,
            ctx.batch_indices,
            ctx.chunk_ranges,
        )?;
        self.drafter.activate_seqs(ctx.seq_ids);
        let entries = ctx
            .seq_ids
            .iter()
            .zip(ctx.chunk_ranges)
            .zip(routing.spans())
            .map(|((&seq_id, &(start_pos, _)), span)| CtxAppend {
                seq_id,
                start_pos,
                rows: span.rows(),
            })
            .collect::<Vec<_>>();
        self.drafter
            .append_ctx_batch(&taps, routing.flat_row_indices()?, &entries)
    }

    pub fn propose(
        &self,
        text: &TextModel,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let depth = ctx.proposal_len;
        if depth == 0
            || ctx.seq_ids.is_empty()
            || ctx.seq_ids.len() != ctx.sampled_tokens.len()
            || self.drafter.has_dormant_seq(ctx.seq_ids)
        {
            return Ok(None);
        }
        if depth > self.depth {
            candle_core::bail!("DFlash proposal exceeds its reserved depth");
        }
        let taps = text.dflash_spec_taps();
        if taps.is_empty() {
            return Ok(None);
        }
        if !self.append_verified(&taps, ctx.seq_ids, ctx.base_lens, ctx.target_rows)? {
            return Ok(None);
        }
        if !self.drafter.contexts_ready_for_draft(ctx.seq_ids) {
            return Ok(None);
        }
        let block = depth + 1;
        let ids = ctx
            .sampled_tokens
            .iter()
            .flat_map(|&anchor| {
                std::iter::once(anchor)
                    .chain(std::iter::repeat_n(self.drafter.mask_token_id(), depth))
            })
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (ctx.seq_ids.len(), block), text.device())?;
        let noise = (text.embed_tokens(&ids)? * self.drafter.input_embedding_scale())?;
        let hidden = self
            .drafter
            .draft_hidden_batch(ctx.seq_ids, &noise, ctx.base_lens)?;
        let head = self.drafter.own_lm_head().unwrap_or_else(|| text.lm_head());
        match self
            .drafter
            .finish_proposals(&hidden, ctx.sampled_tokens, None, head)?
        {
            DFlashProposalBatch::Tokens(tokens) => Ok(Some(SpeculativeProposalBatch::new(
                tokens.into_iter().map(SpeculativeProposal::new).collect(),
            ))),
            #[cfg(feature = "cuda")]
            _ => candle_core::bail!("Gemma 4 DFlash expected greedy token proposals"),
        }
    }

    fn append_verified(
        &self,
        taps: &[Tensor],
        seq_ids: &[usize],
        base_lens: &[usize],
        target_rows: &[(usize, usize)],
    ) -> Result<bool> {
        if taps.is_empty() || seq_ids.len() != base_lens.len() || seq_ids.len() != target_rows.len()
        {
            return Ok(false);
        }
        let (batch, rows, _) = taps[0].dims3()?;
        let mut entries = Vec::new();
        let mut indices = Vec::new();
        for ((&seq_id, &base), &(batch_idx, count)) in
            seq_ids.iter().zip(base_lens).zip(target_rows)
        {
            let next = self
                .drafter
                .ctx_next_pos(seq_id)
                .unwrap_or(base.saturating_sub(count));
            let Some(needed) = base.checked_sub(next) else {
                candle_core::bail!("DFlash context advanced past the verified Gemma 4 prefix");
            };
            if needed == 0 {
                continue;
            }
            if batch_idx >= batch || needed > count || count > rows {
                return Ok(false);
            }
            let offset = batch_idx
                .checked_mul(rows)
                .and_then(|n| n.checked_add(count - needed))
                .ok_or_else(|| candle_core::Error::msg("DFlash context row overflow"))?;
            for row in offset..offset + needed {
                indices.push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
            }
            entries.push(CtxAppend {
                seq_id,
                rows: needed,
                start_pos: next,
            });
        }
        self.drafter.append_ctx_batch(&taps, indices, &entries)?;
        Ok(true)
    }
}
