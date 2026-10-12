//! Haricot EmbeddingGemma 2 bridge to the native Mistral.rs embedding pipeline.
//! The encoder remains implemented once, in haricot/candle:extended.
//!
//! Initial contract: text-only, F32, one sequence, 768-dimensional pooled
//! normalized result, one device, no ISQ/UQFF or tensor parallelism.
use candle_core::{DType, Device, Result, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use candle_transformers::models::embedding_gemma2::{
    Config as EmbeddingGemma2Config, Model as CandleEmbeddingGemma2,
};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    pipeline::{
        text_models_inputs_processor::FlashParams, EmbeddingModel, IsqModel, NormalLoadingMetadata,
    },
};

/// Allows Candle's native model to request tensors by their existing HF
/// names without materializing or duplicating the whole safetensors store.
/// The source continues to be owned by Mistral.rs's verified weight loader.
#[derive(Clone)]
struct ShardedCandleBackend {
    source: ShardedVarBuilder,
}

impl SimpleBackend for ShardedCandleBackend {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _init: Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let value = self.source.get(shape, name)?;
        let value = if value.dtype() == dtype {
            value
        } else {
            value.to_dtype(dtype)?
        };
        value.to_device(device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        let value = self.source.get_unchecked_dtype(name, dtype)?;
        value.to_device(device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.source.contains_tensor(name)
    }
}

/// Native Mistral.rs adapter that supplies a pooled and L2-normalized
/// embedding. Consequently, do not add a sentence-transformers Pooling or
/// Normalize module on top of this model.
pub struct EmbeddingGemma2 {
    model: CandleEmbeddingGemma2,
    device: Device,
}

impl EmbeddingGemma2 {
    pub fn new(
        config: &EmbeddingGemma2Config,
        vb: ShardedVarBuilder,
        loading: NormalLoadingMetadata,
    ) -> Result<Self> {
        if vb.dtype() != DType::F32 {
            candle_core::bail!(
                "EmbeddingGemma 2 native integration currently requires F32; got {:?}",
                vb.dtype()
            )
        }
        if loading.loading_isq {
            candle_core::bail!("EmbeddingGemma 2 does not support ISQ/UQFF loading yet")
        }
        if config.text_config.embedding_dim < 768 {
            candle_core::bail!("EmbeddingGemma 2 requires a model with >=768 embedding dimensions")
        }
        let device = loading.real_device.clone();
        // The first native release intentionally does not shard layers across
        // devices. No silent CPU / GPU cross-device dispatch is permitted.
        for layer in 0..config.text_config.num_hidden_layers {
            if let Some(mapped) = loading.mapper.device_for(layer, false) {
                if mapped.location() != device.location() {
                    candle_core::bail!(
                        "EmbeddingGemma 2 native integration requires one device; layer {layer} is mapped elsewhere"
                    )
                }
            }
        }
        let backend = ShardedCandleBackend { source: vb.root() };
        let candle_vb = VarBuilder::from_backend(Box::new(backend), DType::F32, device.clone());
        let model = CandleEmbeddingGemma2::load(config, candle_vb)?;
        Ok(Self { model, device })
    }
}

impl IsqModel for EmbeddingGemma2 {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        // No quantization support; do not serialize incomplete residuals.
        Vec::new()
    }
}

impl EmbeddingModel for EmbeddingGemma2 {
    fn forward(&self, input_ids: &Tensor, _flash_params: &FlashParams) -> Result<Tensor> {
        // Candle rejects batch != 1 and performs Matryoshka truncation,
        // mean pooling and L2 normalization in precisely that order.
        self.model.embed(input_ids, 768)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for EmbeddingGemma2 {}
