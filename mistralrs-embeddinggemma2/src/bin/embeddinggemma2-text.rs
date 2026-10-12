//! Local-only CLI for EmbeddingGemma 2 embedding smoke tests.
use std::path::PathBuf;

use anyhow::{ensure, Result};
use candle_core::Device;
use clap::Parser;
use mistralrs_embeddinggemma2::TextEmbedder;

#[derive(Debug, Parser)]
#[command(about = "Native EmbeddingGemma 2 text smoke test (local Candle checkpoint)")]
struct Args {
    #[arg(long)]
    local_dir: PathBuf,
    #[arg(long)]
    text: String,
    #[arg(long, default_value_t = 256)]
    dims: usize,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    cuda: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.cuda {
        ensure!(cfg!(feature = "cuda"), "rebuild with --features cuda");
    }
    let device = if args.cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let model = TextEmbedder::load_local(&args.local_dir, device)?;
    let result = model.embed(&args.text, args.dims)?;
    if args.json {
        println!("{}", serde_json::to_string(&result)?);
    } else {
        let l2 = result.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "architecture={} dtype={} dims={} tokens={} l2={l2:.8}",
            result.architecture, result.dtype, result.dimensions, result.token_count
        );
        println!(
            "first_16={:?}",
            &result.vector[..result.vector.len().min(16)]
        );
    }
    Ok(())
}
