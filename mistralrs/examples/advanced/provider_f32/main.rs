use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use mistralrs::{
    DeviceMapMetadata, ModelDType, PrefixCacheCapacity, RequestBuilder, TextMessageRole,
    TextModelBuilder, TokenSource,
};
use mistralrs_kernel_runtime::{compare, digest, execution_evidence, oracle};
use mistralrs_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};
use std::{path::PathBuf, time::Duration};

const PROMPT: &str = "hello world mechanism";
const GENERATION_TOKENS: usize = 4;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

fn device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Ok(Device::new_cuda(0)?),
        _ => anyhow::bail!("device must be cpu or cuda"),
    }
}

fn operator(device: &Device) -> Result<serde_json::Value> {
    let (n, k) = (17, 33);
    let w: Vec<_> = (0..n * k)
        .map(|i| ((i * 7 % 23) as f32 - 11.) / 32.)
        .collect();
    let b: Vec<_> = (0..n).map(|i| (i as f32 - 8.) / 16.).collect();
    let weights = Tensor::from_vec(w.clone(), (n, k), device)?;
    let bias = Tensor::from_vec(b.clone(), n, device)?;
    let mut cases = Vec::new();
    for with_bias in [false, true] {
        let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
            weights.clone(),
            with_bias.then(|| bias.clone()),
        )))?;
        if execution_evidence()["profile_id"].is_string() {
            ensure!(
                layer.unquant_forward_weight().is_none() && layer.unquant_weight_bias().is_none(),
                "raw-weight fast paths can bypass the external provider"
            );
        }
        for shape in [vec![1, k], vec![3, k], vec![5, k], vec![2, 3, k]] {
            let m = shape.iter().product::<usize>() / k;
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 11 % 31) as f32 - 15.) / 16.)
                .collect();
            let mut expected = oracle(&x, &w, [m, n, k]);
            if with_bias {
                for (index, value) in expected.iter_mut().enumerate() {
                    *value += b[index % n];
                }
            }
            let input = Tensor::from_vec(x, shape.clone(), device)?;
            let output = layer.forward(&input)?;
            let mut expected_shape = shape;
            *expected_shape.last_mut().unwrap() = n;
            ensure!(
                output.dims() == expected_shape,
                "UnquantLinear changed output shape"
            );
            let actual = output.flatten_all()?.to_vec1::<f32>()?;
            cases.push(serde_json::json!({"shape":expected_shape, "bias":with_bias,
                "max_abs_error":compare(&actual,&expected)?}));
        }
        let source = Tensor::arange(0f32, (6 * k) as f32, device)?.reshape((2, 3, k))? / 256.;
        let source = source?;
        for input in [source.transpose(0, 1)?, source.narrow(0, 1, 1)?] {
            let m = input.elem_count() / k;
            let x = input.flatten_all()?.to_vec1::<f32>()?;
            let mut expected = oracle(&x, &w, [m, n, k]);
            if with_bias {
                for (index, value) in expected.iter_mut().enumerate() {
                    *value += b[index % n];
                }
            }
            let output = layer.forward(&input)?;
            compare(&output.flatten_all()?.to_vec1::<f32>()?, &expected)?;
            cases.push(
                serde_json::json!({"layout":"strided_or_offset", "rows":m, "bias":with_bias}),
            );
        }
    }
    Ok(serde_json::json!({"status":"PASS", "cases":cases, "provider":execution_evidence()}))
}

fn request() -> RequestBuilder {
    RequestBuilder::new()
        .add_message(TextMessageRole::User, PROMPT)
        .set_deterministic_sampler()
        .set_sampler_max_len(GENERATION_TOKENS)
        .return_logprobs(true)
        .set_sampler_topn_logprobs(5)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(args.len() >= 3, "usage: provider_f32_inference operator|model cpu|cuda OUTPUT [MODEL_DIR] [remove-control-plane]");
    let device = device(&args[1])?;
    #[cfg(feature = "cuda")]
    candle_core::cuda_backend::set_gemm_reduced_precision_f32(false);
    let report = if args[0] == "profile" {
        mistralrs_kernel_runtime::profile::LoadedProfile::read(std::path::Path::new(
            args.get(3).context("missing profile")?,
        ))?
        .evidence()
    } else if args[0] == "operator" {
        operator(&device)?
    } else {
        ensure!(args[0] == "model", "unknown command");
        let path = args.get(3).context("missing local model path")?;
        let model = TextModelBuilder::new(path)
            .with_device(device.clone())
            .with_dtype(ModelDType::F32)
            .with_token_source(TokenSource::None)
            .without_paged_attn()
            .with_device_mapping(DeviceMapMetadata::dummy())
            .with_max_num_seqs(1)
            .with_prefix_cache_n(PrefixCacheCapacity::Fixed(0))
            .build()
            .await?;
        let provider_after_load = execution_evidence();
        let removed = args.get(4).is_some_and(|s| s == "remove-control-plane");
        if removed {
            let profile_path = PathBuf::from(std::env::var(
                mistralrs_kernel_runtime::profile::PROFILE_ENV,
            )?);
            let loaded = mistralrs_kernel_runtime::profile::LoadedProfile::read(&profile_path)?;
            let parent = profile_path.parent().context("profile parent")?;
            let mut files = std::collections::HashSet::new();
            for d in loaded.decisions {
                let manifest = parent.join(&d.decision.manifest);
                files.insert(manifest.parent().unwrap().join(&d.module.manifest.artifact));
                files.insert(manifest);
            }
            files.insert(profile_path);
            for path in files {
                std::fs::remove_file(path)?;
            }
        }
        let (logits, prompt_tokens) =
            tokio::time::timeout(REQUEST_TIMEOUT, model.send_raw_chat_request(request())).await??;
        let logits: Vec<Vec<f32>> = logits
            .into_iter()
            .map(|t| t.flatten_all()?.to_vec1::<f32>())
            .collect::<candle_core::Result<_>>()?;
        let response =
            tokio::time::timeout(REQUEST_TIMEOUT, model.send_chat_request(request())).await??;
        ensure!(
            response.usage.completion_tokens == GENERATION_TOKENS,
            "decode did not complete four tokens"
        );
        serde_json::json!({"status":"PASS", "model":"local deterministic one-layer Llama",
            "device":format!("{:?}",device.location()), "dtype":"f32", "logits":logits,
            "prompt_tokens":prompt_tokens, "response":response,
            "provider_after_load":provider_after_load, "provider":execution_evidence(),
            "control_plane_files_removed_before_requests":removed})
    };
    let binary_hash = digest(&std::fs::read(std::env::current_exe()?)?);
    let mut report = report;
    report["binary_sha256"] = binary_hash.into();
    std::fs::write(&args[2], serde_json::to_vec_pretty(&report)?)?;
    println!("PROVIDER_F32_{}=PASS", args[0].to_uppercase());
    Ok(())
}
