//! Text-only EmbeddingGemma 2 adapter, independently testable before Mistral.rs
//! pipeline integration. Uses the implementation in pinned haricot Candle.
//! This is not interchangeable with the older EmbeddingGemma architecture.
use std::{
    collections::BTreeSet,
    path::{Component, Path, PathBuf},
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::embedding_gemma2::{Config, Model};
use serde::Serialize;
use tokenizers::Tokenizer;

pub const MAX_TOKENS: usize = 8192;
pub const SUPPORTED_DIMENSIONS: [usize; 4] = [128, 256, 512, 768];

/// Metadata required to prevent mixing this embedding space with older
/// MiniLM or EmbeddingGemma vectors in persistent memory indexes.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResult {
    pub architecture: &'static str,
    pub dtype: &'static str,
    pub dimensions: usize,
    pub token_count: usize,
    pub vector: Vec<f32>,
}

pub struct TextEmbedder {
    tokenizer: Tokenizer,
    model: Model,
    device: Device,
}

impl TextEmbedder {
    /// Loads the native Candle EmbeddingGemma 2 text tower from local
    /// Hugging Face files. F32 is deliberately forced for Pascal SM61.
    ///
    /// This preliminary standalone does not yet support arbitrary batch
    /// sizes, quantization, automatic HF downloads, or image/audio towers.
    pub fn load_local(root: impl AsRef<Path>, device: Device) -> Result<Self> {
        let root = root.as_ref();
        let config_bytes = std::fs::read(root.join("config.json"))
            .with_context(|| format!("missing config.json in {}", root.display()))?;
        let config: Config = serde_json::from_slice(&config_bytes)
            .context("EmbeddingGemma 2 text config invalid")?;
        let tokenizer = Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| anyhow!("unable to load tokenizer.json: {e}"))?;
        let paths = safetensors_shards(root)?;
        // SAFETY: Candle owns the memory-mapped weights via its tensor backend.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, DType::F32, &device)?
        };
        let model = Model::load(&config, vb)?;
        Ok(Self { tokenizer, model, device })
    }

    pub fn embed(&self, text: &str, dimensions: usize) -> Result<EmbeddingResult> {
        ensure!(
            SUPPORTED_DIMENSIONS.contains(&dimensions),
            "unsupported Matryoshka dimension {dimensions}; expected 128,256,512,768"
        );
        // Task instructions (Query:/Passage:) belong to the caller's
        // embedding policy; do not silently prepend or modify them.
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;
        let ids = encoding.get_ids();
        ensure!(!ids.is_empty() && ids.len() <= MAX_TOKENS, "token count out of bounds");
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let mut rows = self.model.embed(&input, dimensions)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let vector = rows.pop().ok_or_else(|| anyhow!("encoder returned no embeddings"))?;
        ensure!(vector.len() == dimensions, "wrong embedding width");
        ensure!(vector.iter().all(|v| v.is_finite()), "non-finite embeddings");
        Ok(EmbeddingResult {
            architecture: "EmbeddingGemma2Model",
            dtype: "f32",
            dimensions,
            token_count: ids.len(),
            vector,
        })
    }
}

/// Only local safetensors, no remote IO. Shards are discovered through HF's
/// weight_map; filename traversal outside the selected model dir is rejected.
pub fn safetensors_shards(root: &Path) -> Result<Vec<PathBuf>> {
    let index = root.join("model.safetensors.index.json");
    if !index.is_file() {
        let one = root.join("model.safetensors");
        ensure!(one.is_file(), "missing safetensors checkpoint in {}", root.display());
        return Ok(vec![one]);
    }
    let body: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
    let weights = body.get("weight_map").and_then(|v| v.as_object())
        .context("missing safetensors weight_map")?;
    ensure!(!weights.is_empty(), "empty safetensors weight_map");
    let mut shards = BTreeSet::new();
    for value in weights.values() {
        let filename = value.as_str().context("invalid weight_map filename")?;
        if filename.is_empty() || !Path::new(filename).components().all(|c| matches!(c, Component::Normal(_))) {
            bail!("invalid or escaping safetensors shard name: {filename}");
        }
        let path = root.join(filename);
        ensure!(path.is_file(), "missing safetensors shard {}", path.display());
        shards.insert(path);
    }
    Ok(shards.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_weights() {
        let dir = tempfile::tempdir().unwrap();
        assert!(safetensors_shards(dir.path()).is_err());
    }

    #[test]
    fn rejects_empty_and_traversing_indices() {
        let dir = tempfile::tempdir().unwrap();
        let index = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index, r#"{"weight_map":{}}"#).unwrap();
        assert!(safetensors_shards(dir.path()).is_err());
        std::fs::write(&index, r#"{"weight_map":{"x":"../secret.safetensors"}}"#).unwrap();
        assert!(safetensors_shards(dir.path()).is_err());
    }

    #[test]
    fn accepts_unique_local_shards_in_stable_order() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["shard-b.safetensors", "shard-a.safetensors"] {
            std::fs::write(dir.path().join(name), b"fixture").unwrap();
        }
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            r#"{"weight_map":{"b":"shard-b.safetensors","a":"shard-a.safetensors","c":"shard-a.safetensors"}}"#,
        ).unwrap();
        let names: Vec<_> = safetensors_shards(dir.path()).unwrap()
            .into_iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["shard-a.safetensors", "shard-b.safetensors"]);
    }
}
