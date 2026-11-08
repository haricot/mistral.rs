use anyhow::{bail, Result};
#[cfg(feature = "cuda")]
use anyhow::{ensure, Context};
#[cfg(feature = "cuda")]
use legacy_provider_f32_poc::digest;
use legacy_provider_f32_poc::VerifiedModule;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, manifest] if command == "validate" => {
            let module = VerifiedModule::read(Path::new(manifest))?;
            println!(
                "{}",
                serde_json::json!({"package":"PASS", "cuda_initialized":false,
                "implementation_id":module.manifest.implementation_id,
                "artifact_sha256":module.manifest.sha256, "entry":module.manifest.entry})
            );
            Ok(())
        }
        [command, a, b] if command == "swap" => swap(Path::new(a), Path::new(b)),
        _ => bail!("usage: legacy-provider-f32-poc validate MANIFEST | swap A.json B.json"),
    }
}

#[cfg(not(feature = "cuda"))]
fn swap(_: &Path, _: &Path) -> Result<()> {
    bail!("swap requires compilation with the cuda feature")
}

#[cfg(feature = "cuda")]
fn swap(a: &Path, b: &Path) -> Result<()> {
    use candle_core::{DType, Device, Tensor};
    use legacy_provider_f32_poc::{
        compare,
        cuda::{compute_capability, PreparedLinear},
        oracle, ATOL, RTOL,
    };
    let executable = std::env::current_exe()?;
    let binary_hash = digest(&std::fs::read(&executable)?);
    let a_manifest = VerifiedModule::read(a)?;
    let b_manifest = VerifiedModule::read(b)?;
    ensure!(
        a_manifest.manifest.sha256 != b_manifest.manifest.sha256,
        "A and B must have distinct content"
    );
    let device = Device::new_cuda(0)?;
    candle_core::cuda_backend::set_gemm_reduced_precision_f32(false);
    let cc = compute_capability(&device)?;
    let mut inputs = Vec::new();
    for m in [1, 4, 5] {
        for (n, k) in [(17, 33), (32, 64), (31, 65)] {
            let x: Vec<_> = (0..m * k)
                .map(|i| ((i * 7 % 23) as f32 - 11.) / 16.)
                .collect();
            let w: Vec<_> = (0..n * k)
                .map(|i| ((i * 11 % 29) as f32 - 14.) / 32.)
                .collect();
            let reference = oracle(&x, &w, [m, n, k]);
            let xt = Tensor::from_vec(x, (m, k), &device)?;
            let wt = Tensor::from_vec(w, (n, k), &device)?;
            let candle = xt.matmul(&wt.t()?)?.flatten_all()?.to_vec1::<f32>()?;
            compare(&candle, &reference).context("Candle GPU reference")?;
            inputs.push(([m, n, k], xt, wt, reference, candle));
        }
    }
    let mut passes = Vec::new();
    let mut first_outputs: Vec<Vec<f32>> = Vec::new();
    let mut rejections = 0;
    for (step, path) in [a, b, a].into_iter().enumerate() {
        let prepared = PreparedLinear::load(VerifiedModule::read(path)?, &device)?;
        if step == 0 {
            let (_, x, w, _, _) = &inputs[0];
            let noncontiguous = x.t()?;
            let offset = Tensor::zeros((2, 33), DType::F32, &device)?.narrow(0, 1, 1)?;
            let half = x.to_dtype(DType::F16)?;
            let wrong_k = Tensor::zeros((17, 32), DType::F32, &device)?;
            let other_device = Device::new_cuda(0)?;
            let other_w = w.to_device(&other_device)?;
            for result in [
                prepared.run(&noncontiguous, w),
                prepared.run(&offset, w),
                prepared.run(&half, w),
                prepared.run(x, &wrong_k),
                prepared.run(x, &other_w),
                prepared.run(&x.to_device(&Device::Cpu)?, &w.to_device(&Device::Cpu)?),
            ] {
                ensure!(result.is_err(), "invalid input unexpectedly admitted");
                rejections += 1;
            }
        }
        let mut cases = Vec::new();
        for (index, (shape, x, w, reference, candle)) in inputs.iter().enumerate() {
            let actual = prepared.run(x, w)?.flatten_all()?.to_vec1::<f32>()?;
            let cpu_error = compare(&actual, reference)?;
            let candle_error = compare(&actual, candle)?;
            if step == 0 {
                first_outputs.push(actual.clone());
            }
            if step == 2 {
                ensure!(actual == first_outputs[index], "A replay differs after B");
            }
            cases.push(
                serde_json::json!({"shape":shape, "cpu_max_abs_error":cpu_error,
                "candle_max_abs_error":candle_error, "status":"PASS"}),
            );
        }
        passes.push(serde_json::json!({"pass":step+1, "implementation_id":prepared.manifest.implementation_id,
            "entry_launched":prepared.manifest.entry, "artifact_sha256":prepared.manifest.sha256,
            "module_cache_key":prepared.manifest.module_key(), "cases":cases}));
    }
    ensure!(
        digest(&std::fs::read(&executable)?) == binary_hash,
        "executable changed during swap"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "status":"PASS", "abi":legacy_provider_f32_poc::ABI,
            "scope":"standalone F32 mechanism; no model or performance qualification",
            "binary_sha256":binary_hash, "compute_capability":cc,
            "sequence":"A-B-A", "same_process":true, "same_candle_device_and_stream":true,
            "fallback_used":false, "atol":ATOL, "rtol":RTOL,
            "invalid_input_rejections":rejections, "passes":passes
        }))?
    );
    Ok(())
}
