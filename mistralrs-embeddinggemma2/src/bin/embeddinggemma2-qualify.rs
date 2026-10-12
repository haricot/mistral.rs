//! Compares one locally produced embedding JSON to an independently supplied
//! reference. This does not authenticate provenance or run any GPU hardware.
use anyhow::{Context, Result};
use clap::Parser;
use mistralrs_embeddinggemma2::parity::{compare, ObservedEmbedding, ReferenceEmbedding};
use std::{fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Qualify exact token IDs and numerical EmbeddingGemma 2 vector parity")]
struct Args {
    #[arg(long)]
    observed: PathBuf,
    #[arg(long)]
    reference: PathBuf,
    #[arg(long, default_value_t = 0.999)]
    min_cosine: f64,
    #[arg(long, default_value_t = 0.01)]
    max_abs: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let observed: ObservedEmbedding = serde_json::from_slice(
        &fs::read(&args.observed)
            .with_context(|| format!("reading {}", args.observed.display()))?,
    )
    .context("invalid observed embedding JSON (must include token_ids)")?;
    let reference: ReferenceEmbedding = serde_json::from_slice(
        &fs::read(&args.reference)
            .with_context(|| format!("reading {}", args.reference.display()))?,
    )
    .context("invalid reference embedding JSON")?;
    let report = compare(&observed, &reference, args.min_cosine, args.max_abs)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
