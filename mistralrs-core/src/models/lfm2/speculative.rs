use std::sync::{atomic::Ordering, Arc};

use candle_core::{Result, Tensor};

use crate::speculative::{
    dflash::{CtxAppend, DFlashDraftModel, DFlashLoadTarget, DFlashProposalBatch},
    MtpRuntimeConfig, SpeculativeAttachInfo, SpeculativeBatchPlan, SpeculativeCommitRow,
    SpeculativeConfig, SpeculativeGraphPlan, SpeculativePrefillCtx, SpeculativePrefixReplay,
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
    SpeculativeProposePreparation, SpeculativeProposePrepareCtx, SpeculativeTapRouting,
    SpeculativeTargetMixin,
};

use super::{LayerType, Model};

fn dspark_speculative_batch(batch: DFlashProposalBatch) -> Result<SpeculativeProposalBatch> {
    let proposals = match batch {
        DFlashProposalBatch::Tokens(tokens) => {
            tokens.into_iter().map(SpeculativeProposal::new).collect()
        }
        #[cfg(feature = "cuda")]
        DFlashProposalBatch::DeviceTokens(tokens) => {
            let batch = tokens.dim(0)?;
            (0..batch)
                .map(|row| SpeculativeProposal::from_device(tokens.get(row)?))
                .collect::<Result<Vec<_>>>()?
        }
        #[cfg(feature = "cuda")]
        DFlashProposalBatch::DeviceSparse { .. } => {
            candle_core::bail!("DSpark does not produce sparse DFlash2 proposals")
        }
    };
    Ok(SpeculativeProposalBatch::new(proposals))
}

impl Model {
    fn mtp_n_predict(&self) -> usize {
        self.mtp_n_predict.load(Ordering::Relaxed)
    }

    fn set_speculative_capture(&self, store: bool, tap_layers: Vec<usize>) {
        self.store_spec_hidden.store(store, Ordering::Relaxed);
        *self
            .dflash_tap_layers
            .lock()
            .expect("LFM2 DSpark taps poisoned") = tap_layers;
        if !store {
            *self
                .last_spec_capture
                .lock()
                .expect("LFM2 DSpark capture poisoned") = None;
            *self
                .last_full_capture
                .lock()
                .expect("LFM2 full DSpark capture poisoned") = None;
        }
    }

    fn supports_shortconv_checkpoints(&self) -> bool {
        self.layer_types
            .iter()
            .any(|layer| *layer == LayerType::Conv)
    }

    fn attach_dspark(
        &mut self,
        config: crate::speculative::MtpConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let mut drafter = DFlashDraftModel::load(
            &config,
            DFlashLoadTarget {
                num_layers: self.layers.len(),
                hidden_size: self.cfg.hidden_size,
                yarn_rope_config: None,
                device: &self.device,
                dtype: self.dtype,
            },
            false,
        )?;
        if !drafter.is_dspark() {
            candle_core::bail!(
                "LFM2 external speculation requires an Lfm2DSparkDraftModel checkpoint"
            );
        }
        let block = drafter.block_size();
        let n_predict = match config.n_predict {
            Some(0) => candle_core::bail!("DSpark n_predict must be at least 1"),
            Some(n) if n + 1 > block => candle_core::bail!(
                "requested {n} draft tokens but the DSpark block size is {block} (max {})",
                block - 1
            ),
            Some(n) => n,
            None => block - 1,
        };
        if self.supports_shortconv_checkpoints()
            && self
                .cache
                .hybrid()
                .configure_checkpoint_lanes(n_predict + 1)?
        {
            self.device.synchronize()?;
        }
        let sequence_capacity = self.cache.hybrid().recurrent_capacity();
        drafter.enable_windowed_kv(sequence_capacity, runtime.prefix_cache_capacity())?;
        self.set_speculative_capture(true, drafter.target_layer_ids.clone());
        self.mtp_n_predict.store(n_predict, Ordering::Relaxed);
        if let Some(ty) = config.draft_lm_head_isq {
            let head = self.lm_head.clone().apply_isq(
                Some(ty),
                self.device.clone(),
                &std::sync::atomic::AtomicUsize::new(0),
                None,
                mistralrs_quant::QuantizeOntoGuard::new(),
            )?;
            *self
                .draft_lm_head
                .lock()
                .expect("LFM2 draft lm_head poisoned") = Some(head);
        } else {
            *self
                .draft_lm_head
                .lock()
                .expect("LFM2 draft lm_head poisoned") = None;
        }
        let name = format!(
            "DSpark `{}` (block {block}, depth {n_predict}, Markov head, taps {:?})",
            config.model.as_deref().unwrap_or("dspark"),
            drafter.target_layer_ids
        );
        *self.dflash.lock().expect("LFM2 DSpark poisoned") = Some(Arc::new(drafter));
        Ok(Some(SpeculativeAttachInfo::mtp(name, n_predict)))
    }

    fn dspark_noise_embedding(
        &self,
        drafter: &DFlashDraftModel,
        anchors: &[u32],
        n_predict: usize,
    ) -> Result<Tensor> {
        let block = n_predict + 1;
        let mut ids = Vec::with_capacity(anchors.len() * block);
        for anchor in anchors {
            ids.push(*anchor);
            ids.extend(std::iter::repeat_n(drafter.mask_token_id(), n_predict));
        }
        let ids = Tensor::from_vec(ids, (anchors.len(), block), &self.device)?;
        let mut embedding = self.embed_tokens.embedding_forward(&ids, self.dtype)?;
        if (drafter.input_embedding_scale() - 1.0).abs() > f64::EPSILON {
            embedding = (embedding * drafter.input_embedding_scale())?;
        }
        Ok(embedding)
    }

    fn dspark_propose(
        &self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        let drafter = self
            .dflash
            .lock()
            .expect("LFM2 DSpark poisoned")
            .clone()
            .ok_or_else(|| candle_core::Error::msg("DSpark propose without a drafter"))?;
        let batch = ctx.seq_ids.len();
        let n_predict = ctx.proposal_len;
        if batch == 0 || n_predict == 0 {
            return Ok(None);
        }
        if ctx.sequences.iter().any(|sequence| {
            crate::speculative::verifier::stochastic_verification_allowed_for_sequence(sequence)
        }) {
            return Ok(None);
        }
        if n_predict > self.mtp_n_predict() {
            candle_core::bail!(
                "DSpark proposal length {n_predict} exceeds configured maximum {}",
                self.mtp_n_predict()
            );
        }
        if drafter.has_dormant_seq(ctx.seq_ids) {
            return Ok(None);
        }
        let capture = self
            .last_spec_capture
            .lock()
            .expect("LFM2 DSpark capture poisoned")
            .clone();
        let Some(capture) = capture else {
            return Ok(None);
        };
        if capture.taps.len() != drafter.target_layer_ids.len() {
            return Ok(None);
        }

        let source_rows = capture.taps[0].dim(1)?;
        let mut appends = Vec::with_capacity(batch);
        let mut flat_row_indices = Vec::new();
        for (index, seq_id) in ctx.seq_ids.iter().enumerate() {
            let (batch_idx, count) = ctx.target_rows[index];
            let base_len = ctx.base_lens[index];
            let needed = match drafter.ctx_next_pos(*seq_id) {
                Some(next) if next <= base_len => base_len - next,
                _ => count.min(base_len),
            };
            if needed == 0 {
                continue;
            }
            if needed > count || count > source_rows {
                return Ok(None);
            }
            let start_row = count - needed;
            let flat_start = batch_idx
                .checked_mul(source_rows)
                .and_then(|row| row.checked_add(start_row))
                .ok_or_else(|| candle_core::Error::msg("DSpark tap row index overflow"))?;
            for row in flat_start..flat_start + needed {
                flat_row_indices.push(u32::try_from(row).map_err(candle_core::Error::wrap)?);
            }
            appends.push(CtxAppend {
                seq_id: *seq_id,
                rows: needed,
                start_pos: base_len - needed,
            });
        }
        drafter.append_ctx_batch(&capture.taps, flat_row_indices, &appends)?;
        if !drafter.contexts_ready_for_draft(ctx.seq_ids) {
            return Ok(None);
        }
        let noise = self.dspark_noise_embedding(&drafter, ctx.sampled_tokens, n_predict)?;
        let hidden = drafter.draft_hidden_batch(ctx.seq_ids, &noise, ctx.base_lens)?;
        let draft_head = self
            .draft_lm_head
            .lock()
            .expect("LFM2 draft lm_head poisoned");
        let lm_head = draft_head.as_ref().unwrap_or(&self.lm_head);
        dspark_speculative_batch(drafter.finish_proposals(
            &hidden,
            ctx.sampled_tokens,
            None,
            lm_head,
        )?)
        .map(Some)
    }

    fn dspark_prefill(&self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        let drafter = self
            .dflash
            .lock()
            .expect("LFM2 DSpark poisoned")
            .clone()
            .ok_or_else(|| candle_core::Error::msg("DSpark prefill without a drafter"))?;
        let capture = self
            .last_full_capture
            .lock()
            .expect("LFM2 full DSpark capture poisoned")
            .clone();
        let Some(capture) = capture else {
            return Ok(());
        };
        if capture.taps.len() != drafter.target_layer_ids.len() {
            return Ok(());
        }
        let (capture_batch, capture_rows, _) = capture.taps[0].dims3()?;
        let routing = SpeculativeTapRouting::new(
            ctx.capture_layout(),
            capture_batch,
            capture_rows,
            ctx.batch_indices,
            ctx.chunk_ranges,
        )?;
        drafter.activate_seqs(ctx.seq_ids);
        let appends = ctx
            .seq_ids
            .iter()
            .zip(ctx.chunk_ranges)
            .zip(routing.spans())
            .map(|((seq_id, &(start, _)), span)| CtxAppend {
                seq_id: *seq_id,
                rows: span.rows(),
                start_pos: start,
            })
            .collect::<Vec<_>>();
        drafter.append_ctx_batch(&capture.taps, routing.flat_row_indices()?, &appends)
    }
}

impl SpeculativeTargetMixin for Model {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        self.attach_speculative_with_runtime(config, MtpRuntimeConfig::default())
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: SpeculativeConfig,
        runtime: MtpRuntimeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        let SpeculativeConfig::Mtp(config) = config else {
            self.mtp_n_predict.store(0, Ordering::Relaxed);
            self.set_speculative_capture(false, Vec::new());
            *self.dflash.lock().expect("LFM2 DSpark poisoned") = None;
            return Ok(None);
        };
        if config.is_builtin() {
            candle_core::bail!(
                "LFM2 has no built-in MTP head; pass the DSpark checkpoint with `--mtp-model`"
            );
        }
        self.attach_dspark(config, runtime)
    }

    fn has_speculative_proposer(&self) -> bool {
        self.mtp_n_predict() > 0
    }

    fn supports_recurrent_speculative_checkpoints(&self) -> bool {
        self.supports_shortconv_checkpoints()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.dflash.lock().expect("LFM2 DSpark poisoned").is_some()
    }

    fn supports_speculative_packed_prefill(&self) -> bool {
        self.dflash.lock().expect("LFM2 DSpark poisoned").is_some()
    }

    fn speculative_prefix_replay(&self) -> SpeculativePrefixReplay {
        self.dflash
            .lock()
            .expect("LFM2 DSpark poisoned")
            .as_ref()
            .map_or(SpeculativePrefixReplay::NotRequired, |drafter| {
                drafter.prefix_replay()
            })
    }

    fn speculative_plan(&self, _batch_size: usize) -> Option<SpeculativeBatchPlan> {
        let n_predict = self.mtp_n_predict();
        (n_predict > 0).then(|| SpeculativeBatchPlan::new(n_predict).without_target_hiddens())
    }

    fn speculative_graph_plans(&self) -> Vec<SpeculativeGraphPlan> {
        self.speculative_plan(1)
            .map(|plan| SpeculativeGraphPlan::new(plan.proposal_len, None))
            .into_iter()
            .collect()
    }

    fn speculative_bypass(&mut self, seq_ids: &[usize]) {
        if let Some(drafter) = self.dflash.lock().expect("LFM2 DSpark poisoned").as_ref() {
            drafter.mark_seqs_dormant(seq_ids);
        }
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) {
        if let Some(drafter) = self.dflash.lock().expect("LFM2 DSpark poisoned").as_ref() {
            drafter.release_seqs(seq_ids);
        }
    }

    fn speculative_propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        self.dspark_propose(ctx)
    }

    fn speculative_prepare_propose(
        &mut self,
        _ctx: SpeculativeProposePrepareCtx<'_>,
    ) -> Result<Option<Box<dyn SpeculativeProposePreparation>>> {
        Ok(None)
    }

    fn speculative_prefill(&mut self, ctx: SpeculativePrefillCtx<'_>) -> Result<()> {
        if self.mtp_n_predict() == 0 {
            return Ok(());
        }
        self.dspark_prefill(ctx)
    }

    fn speculative_commit(&mut self, rows: &[SpeculativeCommitRow]) -> Result<()> {
        let checkpoint_rows = rows
            .iter()
            .map(|row| (row.batch_idx, row.keep_rows))
            .collect::<Vec<_>>();
        if !self
            .cache
            .hybrid()
            .commit_speculative_rows(&checkpoint_rows)?
        {
            candle_core::bail!("LFM2 DSpark verification has no ShortConv checkpoints")
        }
        Ok(())
    }
}
