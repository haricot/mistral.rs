//! Unified MoE experts layer supporting multiple backends and weight formats.
//!
//! `MoEExperts` does NOT carry the gate (router); the caller computes routing. Three orthogonal
//! axes, each in one place: [`config`] (which backend to run), [`checkpoint`] (what's on disk ->
//! canonical ENK), and [`backends`] (materialize + forward). [`forward`] holds the per-call shapes.

mod backends;
#[cfg(feature = "cuda")]
pub(crate) use mistralrs_quant::routed_moe::GROUPED_PREFILL_MIN_TOKENS;
mod checkpoint;
pub(crate) use checkpoint::{expert_stack_available, rebuild_expert_projection};
mod config;
mod cpu_moe_baseline;
mod cpu_moe_resident;
mod forward;
#[cfg(test)]
mod isq_tests;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{
    LoraExpertExecution, LoraExpertProjectionNames, LoraExpertSiteHandle, LoraExpertSiteSpec,
    LoraSiteKey, QuantizedConfig, Shard, ShardedVarBuilder, SumAllReduce,
};
use std::sync::Arc;

use crate::layers::Activation;
use cpu_moe_baseline::CpuMoeBaseline;
use cpu_moe_resident::CpuMoeResident;

pub use config::{prelog_moe_backend, ExpertProj, ExpertProjNames, MoEExpertsConfig};

#[cfg(feature = "cutile")]
use backends::CutileExpertsWeights;
#[cfg(feature = "cutile")]
use backends::CutileFp8ExpertsWeights;
#[cfg(feature = "cuda")]
use backends::CutlassExpertsWeights;
use backends::{
    experts_are_prequantized, FastExpertsWeights, FusedExpertsWeights, StackedExpertWeights,
};
use checkpoint::ExpertCheckpoint;
use config::{BackendChoice, MoEExpertsBackend};
use forward::{MoEForward, MoEForwardConfig, MoEForwardShape};

/// MoE experts layer without the gate; the caller computes routing weights + topk indices.
pub struct MoEExperts {
    backend: MoEExpertsBackendImpl,
    resident: Option<CpuMoeResident>,
    lora_site: Option<Arc<LoraExpertSiteHandle>>,
    act: Activation,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    num_experts: usize,
    num_experts_per_tok: usize,
    all_reduce: SumAllReduce,
    world_size: usize,
}

enum MoEExpertsBackendImpl {
    Fused(FusedExpertsWeights),
    #[cfg(feature = "cutile")]
    Cutile(CutileExpertsWeights),
    #[cfg(feature = "cutile")]
    CutileFp8(CutileFp8ExpertsWeights),
    #[cfg(feature = "cuda")]
    Cutlass(CutlassExpertsWeights),
    Fast(FastExpertsWeights),
}

impl MoEExpertsBackendImpl {
    fn forward(&self, forward: &MoEForward, config: MoEForwardConfig) -> Result<Tensor> {
        match self {
            Self::Fused(w) => w.forward_impl(forward, config),
            #[cfg(feature = "cutile")]
            Self::Cutile(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile")),
            #[cfg(feature = "cutile")]
            Self::CutileFp8(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutile fp8")),
            #[cfg(feature = "cuda")]
            Self::Cutlass(w) => w
                .forward_impl(forward, config)
                .map_err(|err| err.context("moe experts cutlass")),
            Self::Fast(w) => w.forward_impl(forward, config),
        }
    }
}

// The gather-based forward requires `gather_forward` support from the quantized layer.
fn check_isq_gather_support() -> Result<()> {
    let Some(params) = mistralrs_quant::get_immediate_isq() else {
        return Ok(());
    };
    if let Some(ty) = params.ty {
        if !ty.supports_stacked_gather() {
            candle_core::bail!("ISQ type {ty} is not supported for MoE experts.");
        }
    }
    Ok(())
}

impl MoEExperts {
    /// Create MoEExperts for a standard model: experts live under `vb.pp("experts")`.
    pub fn new(
        cfg: &MoEExpertsConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let policy = *CpuMoeBaseline::current();
        policy.validate(comm.world_size())?;
        let experts_vb = vb
            .pp("experts")
            .set_device(policy.expert_device(&layer_device));
        if let Some(fast) = FastExpertsWeights::from_weight_source(cfg, &experts_vb, comm)? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, fast.sharded)?;
            return Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                lora_site,
                cfg,
                comm,
                act,
                &layer_device,
            );
        }
        #[cfg(feature = "cutile")]
        if let Some(fp8) = CutileFp8ExpertsWeights::try_from_checkpoint(
            cfg,
            &experts_vb,
            comm,
            quantization_config,
            act,
        )? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, false)?;
            return Self::from_backend(
                MoEExpertsBackendImpl::CutileFp8(fp8),
                lora_site,
                cfg,
                comm,
                act,
                &layer_device,
            );
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            experts_vb.device().clone(),
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        // only the stacked backends need a readable checkpoint; Fast detects with a dummy fallback
        let backend_impl = match MoEExpertsBackend::resolve(&choice)? {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ExpertCheckpoint::new(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(quantization_config, &experts_vb) {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_prequantized(
                        cfg,
                        vb.set_device(policy.expert_device(&layer_device)),
                        quantization_config,
                    )?)
                } else {
                    MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                        cfg,
                        experts_vb.clone(),
                        comm,
                    )?)
                }
            }
        };
        let lora_sharded = match &backend_impl {
            MoEExpertsBackendImpl::Fast(weights) => weights.sharded,
            _ => comm.world_size() > 1,
        };
        let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, lora_sharded)?;

        Self::from_backend(backend_impl, lora_site, cfg, comm, act, &layer_device)
    }

    /// Create MoEExperts from a VB already at the experts level (gemma4's flat `moe.*`, no
    /// `experts.*` sublevel).
    pub fn new_direct(
        cfg: &MoEExpertsConfig,
        experts_vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Result<Self> {
        let policy = *CpuMoeBaseline::current();
        policy.validate(comm.world_size())?;
        let layer_device = experts_vb.device().clone();
        let expert_device = policy.expert_device(experts_vb.device());
        let experts_vb = experts_vb.set_device(expert_device);
        if let Some(fast) = FastExpertsWeights::from_weight_source(cfg, &experts_vb, comm)? {
            let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, fast.sharded)?;
            return Self::from_backend(
                MoEExpertsBackendImpl::Fast(fast),
                lora_site,
                cfg,
                comm,
                act,
                &layer_device,
            );
        }
        check_isq_gather_support()?;
        let choice = BackendChoice::new(
            experts_vb.device().clone(),
            experts_vb.dtype(),
            loading_isq,
            quantization_config,
            act,
        );
        // only the stacked backends need a readable checkpoint; Fast detects with a dummy fallback
        let backend_impl = match MoEExpertsBackend::resolve(&choice)? {
            MoEExpertsBackend::Fused => MoEExpertsBackendImpl::Fused(FusedExpertsWeights {
                w: StackedExpertWeights::from_checkpoint(&ExpertCheckpoint::new(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)?,
            }),
            #[cfg(feature = "cutile")]
            MoEExpertsBackend::Cutile => {
                MoEExpertsBackendImpl::Cutile(CutileExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            #[cfg(feature = "cuda")]
            MoEExpertsBackend::Cutlass => {
                MoEExpertsBackendImpl::Cutlass(CutlassExpertsWeights::from_checkpoint(
                    &ExpertCheckpoint::new(cfg, experts_vb.clone(), comm)?,
                )?)
            }
            MoEExpertsBackend::Fast => {
                if experts_are_prequantized(quantization_config, &experts_vb) {
                    candle_core::bail!(
                        "Pre-quantized experts are not supported for flat expert trees."
                    );
                }
                MoEExpertsBackendImpl::Fast(FastExpertsWeights::load_unquantized(
                    cfg,
                    experts_vb.clone(),
                    comm,
                )?)
            }
        };
        let lora_sharded = match &backend_impl {
            MoEExpertsBackendImpl::Fast(weights) => weights.sharded,
            _ => comm.world_size() > 1,
        };
        let lora_site = Self::register_lora_site(cfg, &experts_vb, comm, lora_sharded)?;

        Self::from_backend(backend_impl, lora_site, cfg, comm, act, &layer_device)
    }

    fn from_backend(
        backend: MoEExpertsBackendImpl,
        lora_site: Option<Arc<LoraExpertSiteHandle>>,
        cfg: &MoEExpertsConfig,
        comm: &Arc<mistralrs_quant::Comm>,
        act: Activation,
        original_device: &Device,
    ) -> Result<Self> {
        let slots = CpuMoeResident::configured_slots()?;
        if slots != 0 && !CpuMoeBaseline::current().enabled() {
            candle_core::bail!("CPU-MoE resident slots require MISTRALRS_CPU_MOE_BASELINE=1");
        }
        let resident = match &backend {
            MoEExpertsBackendImpl::Fast(fast) if slots > 0 => {
                CpuMoeResident::try_new(fast, MoEForwardConfig {
                    num_experts: cfg.num_experts,
                    num_experts_per_tok: cfg.num_experts_per_tok,
                    act,
                }, original_device, slots)?
            }
            _ => None,
        };
        Ok(Self {
            backend,
            resident,
            lora_site,
            act,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            all_reduce: SumAllReduce::new(comm),
            world_size: comm.world_size(),
        })
    }

    fn register_lora_site(
        cfg: &MoEExpertsConfig,
        experts_vb: &ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        sharded: bool,
    ) -> Result<Option<Arc<LoraExpertSiteHandle>>> {
        let Some(registry) = experts_vb.lora_registry() else {
            return Ok(None);
        };
        let names = cfg.expert_proj_names;
        let (gate_up_output_shard, down_input_shard) = if sharded {
            (
                Shard::Simple {
                    dim: 0,
                    rank: comm.rank(),
                    world_size: comm.world_size(),
                },
                Shard::Simple {
                    dim: 1,
                    rank: comm.rank(),
                    world_size: comm.world_size(),
                },
            )
        } else {
            (Shard::default(), Shard::default())
        };
        let spec = LoraExpertSiteSpec::new(
            cfg.num_experts,
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            LoraExpertProjectionNames::new(names.gate, names.up, names.down),
            gate_up_output_shard,
            down_input_shard,
        )?;
        registry
            .register_expert(
                LoraSiteKey::new(experts_vb.prefix()),
                spec,
                experts_vb.dtype(),
                experts_vb.device().clone(),
            )
            .map(Some)
    }
    /// Forward pass through experts
    ///
    /// # Arguments
    /// * `xs` - Input tensor of shape [batch, seq_len, hidden_dim]
    /// * `topk_weights` - Top-k routing weights of shape [num_tokens, num_experts_per_tok]
    /// * `topk_ids` - Top-k expert indices of shape [num_tokens, num_experts_per_tok]
    ///
    /// # Returns
    /// Output tensor of shape [batch, seq_len, hidden_dim]
    pub fn forward(&self, xs: &Tensor, topk_weights: Tensor, topk_ids: &Tensor) -> Result<Tensor> {
        let policy = *CpuMoeBaseline::current();
        let lora = self
            .lora_site
            .as_ref()
            .map(LoraExpertExecution::current)
            .transpose()?
            .flatten();
        if policy.enabled() && lora.is_some() {
            candle_core::bail!("CPU MoE baseline does not support active expert LoRA");
        }
        if let Some(resident) = &self.resident {
            let (batch, seq_len, hidden) = xs.dims3()?;
            if seq_len == 1 && xs.device().is_cuda() {
                let shape = MoEForwardShape::new(batch, seq_len, hidden);
                let xs_flat = xs.reshape(shape.flat())?;
                let forward = MoEForward {
                    xs_flat: &xs_flat,
                    topk_weights: &topk_weights,
                    topk_ids,
                    original_dtype: xs.dtype(),
                    shape,
                    lora: None,
                };
                return resident.forward(&forward)?.reshape(shape.output());
            }
        }
        let (xs_exec, weights_exec, ids_exec) = policy.prepare(xs, topk_weights, topk_ids)?;
        let (batch_size, seq_len, hidden_dim) = xs_exec.dims3()?;
        let shape = MoEForwardShape::new(batch_size, seq_len, hidden_dim);
        let xs_flat = xs_exec.reshape(shape.flat())?;
        let forward = MoEForward {
            xs_flat: &xs_flat,
            topk_weights: &weights_exec,
            topk_ids: &ids_exec,
            original_dtype: xs.dtype(),
            shape,
            lora,
        };

        let config = self.forward_config();
        let mut ys = self
            .backend
            .forward(&forward, config)
            .map_err(|err| err.context("moe experts forward"))?;

        // Sharded experts produce partial sums; replicated ones are already complete.
        let sharded = match &self.backend {
            MoEExpertsBackendImpl::Fast(w) => w.sharded,
            _ => true,
        };
        if self.world_size > 1 && sharded {
            ys = self.all_reduce.sum_all_reduce(&ys)?;
        }

        policy.finish(ys.reshape(forward.shape.output())?, xs.device())
    }

    fn forward_config(&self) -> MoEForwardConfig {
        MoEForwardConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            act: self.act,
        }
    }
}
