//! Model-backed G3V smoke test. NOT a numerical parity or SM61 proof.
//!
//! Example usage:
//! MEXT_G3V_MODEL=/path/to/google-embeddinggemma-2 \
//!   cargo run --release --locked -p mistralrs --example embedding_gemma2_native_smoke
//!
//! Remote HF model use requires MEXT_G3V_REV=<full model commit>.
use anyhow::{bail, Context, Result};
use mistralrs::{core::EmbeddingLoaderType, EmbeddingModelBuilder, EmbeddingRequest, ModelDType};

const SENTENCE: &str = "Passage: Haricot Memory Dynamics conserve les preuves par document.";
const EXPECTED_DIMENSIONS: usize = 768;
const MAX_L2_ERROR: f64 = 0.01;

#[tokio::main]
async fn main() -> Result<()> {
    let model_id = std::env::var("MEXT_G3V_MODEL")
        .context("Set MEXT_G3V_MODEL to a local google/embeddinggemma-2 checkpoint")?;
    let revision = std::env::var("MEXT_G3V_REV").ok();
    if !std::path::Path::new(&model_id).exists()
        && revision
            .as_ref()
            .is_none_or(|rev| rev.trim().len() != 40 || !rev.bytes().all(|c| c.is_ascii_hexdigit()))
    {
        bail!("Remote models require MEXT_G3V_REV=<full 40-character immutable HF commit>");
    }

    let mut builder = EmbeddingModelBuilder::new(&model_id)
        .with_loader_type(EmbeddingLoaderType::EmbeddingGemma2)
        .with_dtype(ModelDType::F32)
        .with_force_cpu()
        .with_max_num_seqs(1)
        .with_logging();
    if let Some(rev) = revision {
        builder = builder.with_hf_revision(rev);
    }
    let model = builder
        .build()
        .await
        .context("native EmbeddingGemma 2 load failed")?;

    let vectors = model
        .generate_embeddings(EmbeddingRequest::builder().add_prompt(SENTENCE))
        .await
        .context("native EmbeddingGemma 2 inference failed")?;
    if vectors.len() != 1 {
        bail!("expected exactly one embedding; got {}", vectors.len());
    }
    let embedding = &vectors[0];
    if embedding.len() != EXPECTED_DIMENSIONS {
        bail!("expected 768-dimensional output; got {}", embedding.len());
    }
    if !embedding.iter().all(|x| x.is_finite()) {
        bail!("native embedding contains non-finite values");
    }
    let norm = embedding
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum::<f64>()
        .sqrt();
    if (norm - 1.0).abs() > MAX_L2_ERROR {
        bail!("native output L2 norm {norm:.9} differs from 1 beyond tolerance");
    }
    println!(
        "G3V_NATIVE_SMOKE=PASS architecture=EmbeddingGemma2Model          dtype=f32 dims={} inputs=1 l2={norm:.9}",
        embedding.len()
    );
    println!("G3V_MODEL_ID={model_id}");
    println!(
        "G3V_NUMERIC_PARITY=UNVERIFIED G3V_HTTP_ENDPOINT=UNVERIFIED          G3V_SM61_PHYSICAL=UNVERIFIED"
    );
    Ok(())
}
