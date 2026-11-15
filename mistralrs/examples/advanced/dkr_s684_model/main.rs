//! S6.8.4 candidate: model-level A/B at fixed Mistral typed DKR configuration.
//! The only A/B runtime switch is Candle's external native package root.
use anyhow::{ensure, Context, Result};
use mistralrs::{DType, Device, DeviceMapMetadata, GgufModelBuilder, Model,
    PrefixCacheCapacity, RequestBuilder, TextMessageRole, PagedAttentionMetaBuilder, MemoryGoal};
use serde_json::{json, Value};
use std::{path::Path, time::Instant};

fn request(text:&str, raw:bool)->RequestBuilder {
    RequestBuilder::new().set_deterministic_sampler()
        .set_sampler_max_len(if raw { 0 } else { 24 })
        .set_sampler_ignore_eos(!raw)
        .return_logprobs(!raw)
        .set_sampler_topn_logprobs(if raw {0}else{5})
        .add_message(TextMessageRole::User,text)
}

async fn raw(model:&Model,text:&str)->Result<Value>{
    let (chunks,tokens)=model.send_raw_chat_request(request(text,true)).await?;
    ensure!(!chunks.is_empty() && !tokens.is_empty(),"missing raw logits or prompt tokens");
    let mut vocab=None;
    let mut rows=0usize;
    let mut last=None;
    for chunk in chunks {
        let dims=chunk.dims();
        let v=*dims.last().context("scalar logits")?;
        ensure!(v>1 && chunk.elem_count()%v==0,"invalid logits shape");
        ensure!(vocab.is_none()||vocab==Some(v),"vocabulary changed between chunks");
        vocab=Some(v);
        let count=chunk.elem_count()/v;
        ensure!(count>0,"empty logits chunk");
        rows+=count;
        let last_row=chunk.reshape((count,v))?.narrow(0,count-1,1)?
            .to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        ensure!(last_row.len()==v && last_row.iter().all(|x|x.is_finite()),"invalid last vocabulary row");
        last=Some(last_row);
    }
    Ok(json!({"prompt_ids":tokens,"vocab":vocab.unwrap(),"raw_rows":rows,"logits":last.unwrap()}))
}
async fn generation(model:&Model,text:&str)->Result<Value>{
    let start=Instant::now();
    let response=model.send_chat_request(request(text,false)).await?;
    let wall=start.elapsed().as_secs_f64();
    ensure!(response.choices.len()==1,"unexpected generation choice count");
    let choice=&response.choices[0];
    let probs=choice.logprobs.as_ref().and_then(|p|p.content.as_ref()).context("missing generated logprobs")?;
    ensure!(response.usage.completion_tokens==24 && probs.len()==24,"incomplete 24-token generation");
    ensure!(probs.iter().all(|p|p.logprob.is_finite()),"nonfinite generation logprob");
    let content=choice.message.content.as_ref().context("missing generated text")?;
    // Raw generated token spellings and byte sequences, not retokenized text.
    let tokens=probs.iter().map(|p|json!({"token":p.token,"bytes":p.bytes,"logprob":p.logprob})).collect::<Vec<_>>();
    Ok(json!({"content":content,"finish_reason":choice.finish_reason,"tokens":tokens,
        "usage":response.usage,"wall_seconds":wall}))
}

#[tokio::main]
async fn main()->Result<()>{
    let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
    let mode=std::env::var("DKR_S684_MODE")?;
    ensure!(mode=="legacy"||mode=="external","invalid A/B mode");
    ensure!(std::env::var_os("CANDLE_DKR_S31_DIR").is_some()==(mode=="external"),"native provider mode mismatch");
    // A Hugging Face snapshot path is a named symlink to a content blob.
    // Preserve that filename and parent for the local GGUF loader.
    let gguf=std::path::PathBuf::from(std::env::var("DKR_S684_GGUF")?);
    ensure!(gguf.is_absolute() && gguf.is_file(),"absolute local GGUF file missing");
    ensure!(gguf.extension().and_then(|s|s.to_str())==Some("gguf"),"expected named GGUF file");
    let dest=std::env::var("DKR_S684_OUTPUT")?;
    let parent=gguf.parent().context("GGUF parent")?.to_str().context("non-UTF8 GGUF path")?;
    let filename=gguf.file_name().context("GGUF filename")?.to_str().context("non-UTF8 filename")?;
    let paged=std::env::var("DKR_S684_PAGED").unwrap_or_else(|_|"0".into());
    ensure!(paged=="0"||paged=="1","invalid Paged Attention mode");
    let device=Device::new_cuda(0)?;
    let start=Instant::now();
    let mut builder=GgufModelBuilder::new(parent,vec![filename])
        .with_device(device)
        .with_device_mapping(DeviceMapMetadata::dummy())
        .with_max_num_seqs(1)
        .with_max_model_len(1024)
        .with_prefix_cache_n(PrefixCacheCapacity::DISABLED)
        .with_logging();
    if let Ok(template)=std::env::var("DKR_S684_CHAT_TEMPLATE"){
        ensure!(Path::new(&template).is_file(),"chat template missing");
        builder=builder.with_chat_template(template);
    }
    builder=if paged=="1"{
        builder.with_paged_attn(PagedAttentionMetaBuilder::default()
            .with_block_size(32).with_memory_goal(MemoryGoal::KvTokens(1024)).build()?)
    }else{builder.without_paged_attn()};
    let model=builder.build().await?;
    let load_seconds=start.elapsed().as_secs_f64();
    let prompts=[
        ("short","Write one sentence about the Moon.".to_string()),
        ("medium","Explain how to verify a matrix multiplication. Use short concrete steps. ".repeat(5)),
        ("long",format!("{} Summarize the key technical ideas in two sentences.",
            "A CUDA kernel operates on blocks of threads. Quantized weights reduce memory traffic. A stream preserves the order of submitted work. ".repeat(14))),
    ];
    let mut results=serde_json::Map::new();
    for (key,prompt) in prompts {
        let a=raw(&model,&prompt).await?;
        let b=raw(&model,&prompt).await?;
        let first=generation(&model,&prompt).await?;
        let second=generation(&model,&prompt).await?;
        results.insert(key.into(),json!({"raw":[a,b],"generation":[first,second]}));
    }
    std::fs::write(dest,serde_json::to_vec(&json!({"schema":"DKR-S684-MODEL-V1",
        "mode":mode,"paged_requested":paged=="1","model":gguf.to_string_lossy(),
        "load_seconds":load_seconds,"cases":results}))?)?;
    Ok(())
}
