//! Paged DKR graph proof and S1B streaming timings for the SM61 qualification.
mod graph_metrics;
mod probe;

use anyhow::{ensure, Context, Result};
use graph_metrics::{GraphMetrics, Snapshot};
use mistralrs::{
    DType, Device, DeviceMapMetadata, GgufModelBuilder, MemoryGoal, Model,
    PagedAttentionMetaBuilder, PrefixCacheCapacity, RequestBuilder, Response, TextMessageRole,
};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

fn request(text: &str, raw: bool) -> RequestBuilder {
    RequestBuilder::new()
        .set_deterministic_sampler()
        // The engine validates max_tokens even for raw-logit requests.
        .set_sampler_max_len(if raw { 1 } else { 24 })
        .set_sampler_ignore_eos(!raw)
        .return_logprobs(!raw)
        .set_sampler_topn_logprobs(if raw { 0 } else { 5 })
        .add_message(TextMessageRole::User, text)
}

async fn raw(model: &Model, text: &str) -> Result<Value> {
    let (chunks, tokens) = model.send_raw_chat_request(request(text, true)).await?;
    ensure!(
        !chunks.is_empty() && !tokens.is_empty(),
        "missing raw logits or prompt tokens"
    );
    let mut vocab = None;
    let mut rows = 0usize;
    let mut last = None;
    for chunk in chunks {
        let dims = chunk.dims();
        let v = *dims.last().context("scalar logits")?;
        ensure!(v > 1 && chunk.elem_count() % v == 0, "invalid logits shape");
        ensure!(
            vocab.is_none() || vocab == Some(v),
            "vocabulary changed between chunks"
        );
        vocab = Some(v);
        let count = chunk.elem_count() / v;
        ensure!(count > 0, "empty logits chunk");
        rows += count;
        let last_row = chunk
            .reshape((count, v))?
            .narrow(0, count - 1, 1)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        ensure!(
            last_row.len() == v && last_row.iter().all(|x| x.is_finite()),
            "invalid last vocabulary row"
        );
        last = Some(last_row);
    }
    Ok(json!({"prompt_ids":tokens,"vocab":vocab.unwrap(),"raw_rows":rows,"logits":last.unwrap()}))
}
async fn generation(model: &Model, text: &str) -> Result<Value> {
    let start = Instant::now();
    let response = model.send_chat_request(request(text, false)).await?;
    let wall = start.elapsed().as_secs_f64();
    ensure!(
        response.choices.len() == 1,
        "unexpected generation choice count"
    );
    let choice = &response.choices[0];
    let probs = choice
        .logprobs
        .as_ref()
        .and_then(|p| p.content.as_ref())
        .context("missing generated logprobs")?;
    ensure!(
        response.usage.completion_tokens == 24 && probs.len() == 24,
        "incomplete 24-token generation"
    );
    ensure!(
        probs.iter().all(|p| p.logprob.is_finite()),
        "nonfinite generation logprob"
    );
    // Thinking-enabled templates may place the entire 24-token budget in
    // reasoning_content. Preserve both channels; do not require final prose.
    let content = &choice.message.content;
    let reasoning = &choice.message.reasoning_content;
    ensure!(
        content.is_some() || reasoning.is_some(),
        "both response channels missing"
    );
    // Raw generated token spellings and byte sequences, not retokenized text.
    let tokens = probs
        .iter()
        .map(|p| json!({"token":p.token,"bytes":p.bytes,"logprob":p.logprob}))
        .collect::<Vec<_>>();
    Ok(
        json!({"content":content,"reasoning_content":reasoning,"finish_reason":choice.finish_reason,"tokens":tokens,
        "usage":response.usage,"wall_seconds":wall}),
    )
}

async fn streaming(
    model: &Model,
    text: &str,
    max_tokens: usize,
    graphs: &GraphMetrics,
) -> Result<Value> {
    let req = RequestBuilder::new()
        .set_deterministic_sampler()
        .set_sampler_max_len(max_tokens)
        .set_sampler_ignore_eos(true)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(5)
        .add_message(TextMessageRole::User, text);
    let before = graphs.snapshot();
    let start = Instant::now();
    let mut stream = model.stream_chat_request(req).await?;
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut timestamps_ms = Vec::with_capacity(max_tokens);
    let mut first_chunk_ms = None;
    let mut first_content_ms = None;
    let mut first_reasoning_ms = None;
    let mut finish = None;
    let mut usage = None;
    while let Some(response) = stream.next().await {
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if first_chunk_ms.is_none() {
            first_chunk_ms = Some(elapsed_ms);
        }
        match response {
            Response::Chunk(chunk) => {
                if let Some(u) = chunk.usage {
                    ensure!(usage.is_none(), "duplicate streaming usage");
                    usage = Some(u);
                }
                for choice in chunk.choices {
                    ensure!(choice.index == 0, "unexpected streaming choice");
                    if choice.delta.content.as_ref().is_some_and(|x| !x.is_empty())
                        && first_content_ms.is_none()
                    {
                        first_content_ms = Some(elapsed_ms);
                    }
                    if choice
                        .delta
                        .reasoning_content
                        .as_ref()
                        .is_some_and(|x| !x.is_empty())
                        && first_reasoning_ms.is_none()
                    {
                        first_reasoning_ms = Some(elapsed_ms);
                    }
                    if let Some(p) = choice.logprobs {
                        ensure!(p.logprob.is_finite(), "nonfinite generation logprob");
                        tokens.push(json!({"token":p.token,"bytes":p.bytes,"logprob":p.logprob}));
                        timestamps_ms.push(elapsed_ms);
                    }
                    if let Some(reason) = choice.finish_reason {
                        ensure!(finish.is_none(), "duplicate streaming finish");
                        finish = Some(reason);
                    }
                }
            }
            Response::InternalError(e) | Response::ValidationError(e) => {
                anyhow::bail!("stream error: {e}")
            }
            Response::ModelError(e, _) => anyhow::bail!("stream model error: {e}"),
            _ => anyhow::bail!("unexpected non-chunk streaming response"),
        }
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    ensure!(
        tokens.len() == max_tokens,
        "incomplete token timestamp inventory: expected {max_tokens}, got {}",
        tokens.len()
    );
    ensure!(finish.is_some(), "missing finish reason");
    if let Some(ref u) = usage {
        ensure!(
            u.completion_tokens == max_tokens,
            "stream usage count mismatch"
        );
    }
    let token_intervals_ms = timestamps_ms
        .windows(2)
        .map(|w| w[1] - w[0])
        .collect::<Vec<_>>();
    ensure!(
        token_intervals_ms
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0),
        "nonmonotonic token receipt"
    );
    Ok(json!({"tokens":tokens,"token_receipt_ms":timestamps_ms,
        "inter_token_receipt_ms":token_intervals_ms,"ttft_logprob_ms":timestamps_ms[0],
        "first_chunk_ms":first_chunk_ms,"first_content_ms":first_content_ms,
        "first_reasoning_ms":first_reasoning_ms,"finish_reason":finish,"usage":usage,
        "wall_ms":wall_ms,"token_count":max_tokens,"graph":graphs.since(&before)}))
}

#[tokio::main]
async fn main() -> Result<()> {
    let _evidence = mistralrs_kernel_runtime::RuntimeEvidenceGuard;
    let dest = std::env::var("DKR_S684_OUTPUT")?;
    if std::env::args().nth(1).as_deref() == Some("--probe") {
        std::fs::write(dest, serde_json::to_vec(&probe::run()?)?)?;
        return Ok(());
    }
    let graphs = GraphMetrics::default();
    metrics::set_global_recorder(graphs.clone())
        .map_err(|_| anyhow::anyhow!("graph metrics recorder already installed"))?;
    let graph_enabled = match std::env::var("MISTRALRS_CUDA_GRAPHS")?.as_str() {
        "0" => false,
        "1" => true,
        flag => anyhow::bail!("invalid CUDA graph mode: {flag}"),
    };
    let mode = std::env::var("DKR_S684_MODE")?;
    ensure!(mode == "legacy" || mode == "external", "invalid A/B mode");
    ensure!(
        std::env::var_os("CANDLE_DKR_S31_DIR").is_some() == (mode == "external"),
        "native provider mode mismatch"
    );
    // A Hugging Face snapshot path is a named symlink to a content blob.
    // Preserve that filename and parent for the local GGUF loader.
    let gguf = std::path::PathBuf::from(std::env::var("DKR_S684_GGUF")?);
    ensure!(
        gguf.is_absolute() && gguf.is_file(),
        "absolute local GGUF file missing"
    );
    ensure!(
        gguf.extension().and_then(|s| s.to_str()) == Some("gguf"),
        "expected named GGUF file"
    );
    let parent = gguf
        .parent()
        .context("GGUF parent")?
        .to_str()
        .context("non-UTF8 GGUF path")?;
    let filename = gguf
        .file_name()
        .context("GGUF filename")?
        .to_str()
        .context("non-UTF8 filename")?;
    let paged = std::env::var("DKR_S684_PAGED").unwrap_or_else(|_| "0".into());
    ensure!(paged == "0" || paged == "1", "invalid Paged Attention mode");
    let device = Device::new_cuda_with_stream(0)?;
    let start = Instant::now();
    let mut builder = GgufModelBuilder::new(parent, vec![filename])
        .with_device(device)
        .with_device_mapping(DeviceMapMetadata::dummy())
        .with_max_num_seqs(1)
        .with_max_model_len(1024)
        .with_prefix_cache_n(PrefixCacheCapacity::DISABLED)
        .with_logging();
    if let Ok(template) = std::env::var("DKR_S684_CHAT_TEMPLATE") {
        ensure!(Path::new(&template).is_file(), "chat template missing");
        builder = builder.with_chat_template(template);
    }
    builder = if paged == "1" {
        builder.with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_memory_goal(MemoryGoal::KvTokens(1024))
                .build()?,
        )
    } else {
        builder.without_paged_attn()
    };
    let model = builder.build().await?;
    let load_seconds = start.elapsed().as_secs_f64();
    let graph_at_load = graphs.since(&Snapshot::default());
    let prompts=[
        ("short","Write one sentence about the Moon.".to_string()),
        ("medium","Explain how to verify a matrix multiplication. Use short concrete steps. ".repeat(5)),
        ("long",format!("{} Summarize the key technical ideas in two sentences.",
            "A CUDA kernel operates on blocks of threads. Quantized weights reduce memory traffic. A stream preserves the order of submitted work. ".repeat(14))),
    ];
    let mut results = serde_json::Map::new();
    for (key, prompt) in prompts {
        eprintln!("DKR_S684_CASE_BEGIN {key} raw=1");
        let a = raw(&model, &prompt)
            .await
            .with_context(|| format!("{key} raw 1"))?;
        eprintln!("DKR_S684_CASE_BEGIN {key} raw=2");
        let b = raw(&model, &prompt)
            .await
            .with_context(|| format!("{key} raw 2"))?;
        eprintln!("DKR_S684_CASE_BEGIN {key} generation=1");
        let first = generation(&model, &prompt)
            .await
            .with_context(|| format!("{key} generation 1"))?;
        eprintln!("DKR_S684_CASE_BEGIN {key} generation=2");
        let second = generation(&model, &prompt)
            .await
            .with_context(|| format!("{key} generation 2"))?;
        eprintln!("DKR_S684_CASE_PASS {key}");
        eprintln!("DKR_S1B_STREAM_BEGIN {key} tokens=24");
        let check = streaming(&model, &prompt, 24, &graphs)
            .await
            .with_context(|| format!("{key} streaming parity"))?;
        ensure!(
            check["tokens"] == second["tokens"],
            "24-token streaming/nonstreaming mismatch: {key}"
        );
        eprintln!("DKR_S1B_STREAM_BEGIN {key} tokens=128 repetition=1");
        let long_first = streaming(&model, &prompt, 128, &graphs)
            .await
            .with_context(|| format!("{key} long stream 1"))?;
        eprintln!("DKR_S1B_STREAM_BEGIN {key} tokens=128 repetition=2");
        let long_second = streaming(&model, &prompt, 128, &graphs)
            .await
            .with_context(|| format!("{key} long stream 2"))?;
        ensure!(
            long_first["tokens"] == long_second["tokens"],
            "long streaming repeat mismatch: {key}"
        );
        results.insert(
            key.into(),
            json!({"raw":[a,b],"generation":[first,second],
            "streaming_24":check,"streaming_128":[long_first,long_second]}),
        );
    }
    std::fs::write(
        dest,
        serde_json::to_vec(&json!({"schema":"DKR-SM61-GRAPH-STREAMING-V1",
        "mode":mode,"paged_requested":paged=="1","model":gguf.to_string_lossy(),
        "load_seconds":load_seconds,"cases":results,"graphs_requested":graph_enabled,
        "graph_at_load":graph_at_load,"graph_total":graphs.since(&Snapshot::default())}))?,
    )?;
    Ok(())
}
