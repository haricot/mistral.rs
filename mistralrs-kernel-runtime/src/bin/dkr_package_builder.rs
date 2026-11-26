//! Build all external DKR PTX modules with cudaforge for one target GPU.
//!
//! Unlike the historical Python builders, NVCC compilation is owned by the
//! pinned cudaforge crate. Existing Python emitters only package and fingerprint
//! already compiled PTX; this preserves the versioned typed manifests.

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};
use sha2::{Digest, Sha256};
use mistralrs_kernel_runtime::universal::VerifiedKernel;

struct Package {
    name: &'static str,
    source: &'static str,
    fast_math: bool,
}

const PACKAGES: &[Package] = &[
    Package { name: "add-rms-norm", source: "add_rms_norm.cu", fast_math: true },
    Package { name: "causal-mask", source: "causal_mask.cu", fast_math: true },
    Package { name: "gated-add", source: "gated_add.cu", fast_math: true },
    Package { name: "indexed-copy", source: "indexed_copy.cu", fast_math: false },
    Package { name: "input-pad", source: "input_pad.cu", fast_math: false },
    Package { name: "logits", source: "logits.cu", fast_math: false },
    Package { name: "moe", source: "moe.cu", fast_math: false },
    Package { name: "reduce-norm", source: "reduce_norm.cu", fast_math: true },
    Package { name: "rms-variants", source: "rms_variants.cu", fast_math: true },
    Package { name: "ssm", source: "ssm.cu", fast_math: true },
    Package { name: "gdn-recurrence", source: "gdn_recurrence.cu", fast_math: true },
    Package { name: "gdn-preparation", source: "gdn_preparation.cu", fast_math: true },
    Package { name: "gdn-decode", source: "gdn_decode.cu", fast_math: true },
    Package { name: "gdn-transaction", source: "gdn_transaction.cu", fast_math: true },
    Package { name: "dynamic-conv", source: "dynamic_conv.cu", fast_math: false },
    Package { name: "rounded-rope", source: "rounded_rope.cu", fast_math: true },
    Package { name: "dense-gemv", source: "dense_gemv.cu", fast_math: false },
    Package { name: "kv-scales", source: "kv_scales.cu", fast_math: false },
    Package { name: "fp8-cast", source: "fp8_cast.cu", fast_math: false },
    Package { name: "kv-gather", source: "kv_gather.cu", fast_math: false },
    Package { name: "dense-gemm", source: "dense_gemm.cu", fast_math: false },
    Package { name: "fp4-dequant", source: "fp4_dequant.cu", fast_math: false },
    Package { name: "quant-gemv", source: "quant_gemv.cu", fast_math: false },
    Package { name: "paged-decode", source: "paged_decode.cu", fast_math: false },
    Package { name: "nvfp4-gemv", source: "nvfp4_gemv.cu", fast_math: false },
    Package { name: "nvfp4-gemm", source: "nvfp4_gemm.cu", fast_math: false },
    Package { name: "quant-gemm", source: "quant_gemm.cu", fast_math: false },
    Package { name: "paged-fp8", source: "paged_fp8.cu", fast_math: false },
    Package { name: "paged-v2", source: "paged_v2.cu", fast_math: false },
    Package { name: "paged-fp8-v2", source: "paged_fp8_v2.cu", fast_math: false },
    Package { name: "paged-options", source: "paged_options.cu", fast_math: false },
    Package { name: "prefill-sinks", source: "prefill_sinks.cu", fast_math: false },
    Package { name: "mla-cache-io", source: "mla_cache_io.cu", fast_math: false },
    Package { name: "mla-decode", source: "mla_decode.cu", fast_math: true },
    Package { name: "flashinfer-merge", source: "flashinfer_merge.cu", fast_math: true },
    Package { name: "flashinfer-decode", source: "flashinfer_decode.cu", fast_math: true },
    Package { name: "flashinfer-hnd-io", source: "flashinfer_hnd_io.cu", fast_math: false },
    Package { name: "kv-cache-ops", source: "kv_cache_ops.cu", fast_math: false },
    Package { name: "kv-fp8-io", source: "kv_fp8_io.cu", fast_math: false },
    Package { name: "routed-quant", source: "routed_quant.cu", fast_math: false },
    Package { name: "gguf-mmvq", source: "gguf_mmvq.cu", fast_math: false },
    Package { name: "gguf-mmq", source: "gguf_mmq.cu", fast_math: false },
    Package { name: "gguf-moe-compact", source: "gguf_moe_compact.cu", fast_math: false },
    Package { name: "gguf-glu-encode", source: "gguf_glu_encode.cu", fast_math: false },
    Package { name: "gguf-gated-mix", source: "gguf_gated_mix.cu", fast_math: false },
    Package { name: "gguf-mmvq-fused", source: "gguf_mmvq_fused.cu", fast_math: false },
    Package { name: "gguf-moe-glu-encode", source: "gguf_moe_glu_encode.cu", fast_math: false },
    Package { name: "gguf-moe-decode", source: "gguf_moe_decode.cu", fast_math: false },
    Package { name: "gguf-residual", source: "gguf_residual.cu", fast_math: false },
    Package { name: "gguf-grouped-row", source: "gguf_grouped_row.cu", fast_math: false },
    Package { name: "attention-prep", source: "attention_prep.cu", fast_math: false },
];

fn parse_args() -> Result<(usize, PathBuf, String, bool, bool, Option<String>), Box<dyn std::error::Error>> {
    let mut cc: Option<usize> = None;
    let mut output = PathBuf::from("kernel-packages/build/cuda");
    let mut python = String::from("python3");
    let mut cubin = false;
    let mut pair_existing = false;
    let mut family: Option<String> = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--cubin" {
            cubin = true;
            continue;
        }
        if flag == "--pair-existing" {
            cubin = true;
            pair_existing = true;
            continue;
        }
        let value = args.next().ok_or_else(|| format!("missing argument for {flag}"))?;
        match flag.as_str() {
            "--cc" => cc = Some(value.parse()?),
            "--output" => output = PathBuf::from(value),
            "--python" => python = value,
            "--family" => family = Some(value),
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let cc = cc.ok_or("usage: --cc 61 [--output DIR] [--python python3]")?;
    if !(50..=120).contains(&cc) {
        return Err(format!("unsupported compute capability {cc}").into());
    }
    if family.as_ref().is_some_and(|name| !PACKAGES.iter().any(|p| p.name == name.as_str())) {
        return Err("unknown DKR package family".into());
    }
    if pair_existing && family.is_some() {
        return Err("--family cannot be combined with --pair-existing; use an isolated output root".into());
    }
    Ok((cc, output, python, cubin, pair_existing, family))
}

fn compile_ptx(
    source: &Path,
    out_dir: &Path,
    cc: usize,
    fast_math: bool,
    mode: Option<&str>,
    header_dependencies: &[PathBuf],
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let mut builder = cudaforge::KernelBuilder::new()
        .source_files([source])
        .watch(header_dependencies)
        .out_dir(out_dir)
        .compute_cap(cc)
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--expt-relaxed-constexpr")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__");
    if fast_math {
        builder = builder.arg("--use_fast_math");
    }
    if !header_dependencies.is_empty() {
        builder = builder.arg("-DALLOW_LEGACY_BF16");
    }
    if source.file_stem().is_some_and(|s| s == "mla_decode" || s == "flashinfer_merge" || s == "flashinfer_decode") {
        builder = builder.arg("--expt-extended-lambda");
    }
    if let Some(mode) = mode {
        builder = builder.arg(mode);
    }
    builder.build_ptx()?;
    let name = source.file_stem().ok_or("CUDA source missing stem")?;
    let ptx = out_dir.join(name).with_extension("ptx");
    if !ptx.is_file() {
        return Err(format!("cudaforge did not generate {}", ptx.display()).into());
    }
    // Prevent mislabeling a PTX module for another architecture.
    let source_text = fs::read_to_string(&ptx)?;
    if !source_text.contains(&format!(".target sm_{cc}")) {
        return Err(format!("PTX target does not match sm_{cc}: {}", ptx.display()).into());
    }
    Ok(ptx)
}

fn package(
    python: &str,
    script: &Path,
    cc: usize,
    output: &Path,
    precompiled: &Path,
    moe: bool,
    root: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new(python);
    command
        .arg(script)
        .args(["--cc", &cc.to_string()])
        .arg("--output").arg(output)
        .arg(if moe { "--ptx-dir" } else { "--ptx-file" })
        .arg(precompiled)
        .current_dir(root);
    let status = command.status()?;
    if !status.success() {
        return Err(format!("DKR manifest packaging failed: {}", script.display()).into());
    }
    Ok(())
}

// Assemble the exact PTX produced by cudaforge. ptxas is the CUDA toolchain's
// PTX-to-SASS stage: no second nvcc C++ compilation or Python device build.
fn stage_cubins(root: &Path, cc: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let ptxas = env::var("PTXAS").unwrap_or_else(|_| "ptxas".into());
    let version = Command::new(&ptxas).arg("--version").output()?;
    if !version.status.success() {
        return Err(format!("ptxas --version failed: {}", version.status).into());
    }
    let mut paths = Vec::<PathBuf>::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file() && entry.file_name() == "manifest.json" {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err("no typed manifests found for CUBIN pairing".into());
    }
    let image_cache = root.join("_cudaforge_cache/cubin");
    fs::create_dir_all(&image_cache)?;
    let mut compiled = HashMap::<String, PathBuf>::new();
    for manifest_path in paths {
        // Validate every entry before assembling any physical image in this package.
        let verified_profiles = VerifiedKernel::read_package(&manifest_path)?;
        let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        let is_shard = manifest["schema"].as_str() == Some(mistralrs_kernel_runtime::universal::SHARD_SCHEMA);
        let mut destinations = HashMap::<PathBuf, (String, String)>::new();
        for (index, verified) in verified_profiles.iter().enumerate() {
        if verified.manifest.target_cc as usize != cc {
            return Err(format!("wrong target in {}", manifest_path.display()).into());
        }
        let mut cache_key = Sha256::new();
        cache_key.update(verified.ptx.as_bytes());
        cache_key.update(&version.stdout);
        cache_key.update(&version.stderr);
        cache_key.update(cc.to_le_bytes());
        let identity = format!("{:x}", cache_key.finalize());
        let binary = if let Some(previous) = compiled.get(&identity) {
            previous.clone()
        } else {
            let binary = image_cache.join(format!("{identity}.cubin"));
            if !binary.is_file() {
                let source = manifest_path.parent().ok_or("manifest missing directory")?
                    .join(&verified.manifest.artifact);
                let partial = image_cache.join(format!("{identity}.tmp.cubin"));
                let status = Command::new(&ptxas)
                    .arg(format!("-arch=sm_{cc}"))
                    .arg("-o").arg(&partial).arg(&source)
                    .status()?;
                if !status.success() {
                    let _ = fs::remove_file(&partial);
                    return Err(format!("ptxas failed for {}: {status}", source.display()).into());
                }
                fs::rename(&partial, &binary)?;
            }
            compiled.insert(identity.clone(), binary.clone());
            binary
        };
        let destination = verified.manifest.artifact.with_extension("cubin");
        let dest_path = manifest_path.parent().ok_or("manifest missing directory")?
            .join(&destination);
        let sha = if let Some((previous_identity, sha)) = destinations.get(&dest_path) {
            if previous_identity != &identity {
                return Err(format!("CUBIN destination collision: {}", dest_path.display()).into());
            }
            sha.clone()
        } else {
            fs::copy(&binary, &dest_path)?;
            let sha = format!("{:x}", Sha256::digest(&fs::read(&dest_path)?));
            destinations.insert(dest_path, (identity, sha.clone()));
            sha
        };
        let row = if is_shard { &mut manifest["profiles"][index] } else { &mut manifest };
        row["cubin_artifact"] = destination.to_string_lossy().into_owned().into();
        row["cubin_sha256"] = sha.into();
        }
        let encoded = serde_json::to_vec_pretty(&manifest)?;
        if encoded.len() as u64 > mistralrs_kernel_runtime::universal::MAX_MANIFEST_BYTES {
            return Err("paired manifest budget exceeded".into());
        }
        fs::write(&manifest_path, encoded)?;
        // Validate all paired entries, not only the first one.
        let validated = VerifiedKernel::read_package(&manifest_path)?;
        if validated.iter().any(|profile| profile.cubin.is_none()) {
            return Err("CUBIN pairing did not validate".into());
        }
    }
    Ok(compiled.len())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (cc, requested_output, python, cubin, pair_existing, family) = parse_args()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().ok_or("no repository root")?;
    let output = if requested_output.is_absolute() {
        requested_output
    } else {
        root.join(requested_output)
    };
    let sources = root.join("kernel-packages/cuda");
    let vllm = root.join("mistralrs-paged-attn/src/cuda");
    let paged_headers = [
        sources.join("paged_vllm_compat.cuh"),
        vllm.join("pagedattention.cuh"),
        vllm.join("cuda_compat.h"),
        vllm.join("attention"),
        vllm.join("quantization/fp8/nvidia"),
    ];
    // Track every vendored FlashInfer header explicitly for cache invalidation.
    let mut flashinfer_headers = Vec::new();
    let mut dirs = vec![vllm.join("flashinfer")];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() { dirs.push(entry.path()); }
            else if entry.file_type()?.is_file() { flashinfer_headers.push(entry.path()); }
        }
    }
    flashinfer_headers.sort();
    let work = output.join(format!("sm{cc}/_cudaforge_cache"));
    if !pair_existing {
    for pkg in PACKAGES.iter().filter(|p| family.as_ref().is_none_or(|name| p.name == name.as_str())) {
        if matches!(pkg.name, "gguf-mmq" | "gguf-moe-compact" | "gguf-glu-encode" | "gguf-gated-mix" | "gguf-moe-glu-encode" | "gguf-moe-decode" | "gguf-residual" | "gguf-grouped-row") && cc != 61 {
            if family.is_some() { return Err("GGUF MMQ typed package is qualified for source target SM61 only".into()); }
            println!("DKR_GGUF_MMQ_SKIPPED target_cc={cc} reason=sm61_source_only");
            continue;
        }
        let dir = sources.join(pkg.name);
        let source = dir.join(pkg.source);
        if !source.is_file() {
            return Err(format!("missing DKR CUDA source: {}", source.display()).into());
        }
        let script = dir.join("build.py");
        if pkg.name == "flashinfer-decode" {
            // Python emits only the bounded source plan and stamps verified PTX.
            // Cudaforge owns every device compilation, cached per image identity.
            let planned = Command::new(&python).arg(&script)
                .args(["--cc", &cc.to_string(), "--list-images"])
                .current_dir(root).output()?;
            if !planned.status.success() {
                return Err(format!("FlashInfer image plan failed: {}", String::from_utf8_lossy(&planned.stderr)).into());
            }
            let ids = String::from_utf8(planned.stdout)?;
            let mut seen = std::collections::BTreeSet::new();
            for line in ids.lines() {
                let id: u32 = line.parse()?;
                if id == 0 || id >= (1 << 14) || !seen.insert(id) {
                    return Err("invalid or duplicate FlashInfer image identity".into());
                }
                let define = format!("-DDKR_FLASHINFER_IMAGE={id}");
                let mut watched = flashinfer_headers.clone();
                watched.push(dir.join("safe_decode.cuh"));
                let compiled = compile_ptx(&source, &work.join(pkg.name).join(format!("safe2-image-{id:05}")),
                    cc, pkg.fast_math, Some(&define), &watched)?;
                let status = Command::new(&python).arg(&script)
                    .args(["--cc", &cc.to_string(), "--image", &id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled)
                    .current_dir(root).status()?;
                if !status.success() {
                    return Err(format!("FlashInfer shard packaging failed for {id}: {status}").into());
                }
            }
            if seen.is_empty() { return Err("empty FlashInfer image plan".into()); }
         } else if pkg.name == "gguf-gated-mix" {
            let kernels=root.join("mistralrs-quant/kernels");
            let watched=["gguf/gated_mix.cu","gguf/vecdot.cuh","gemv/row.cuh"].map(|n|kernels.join(n));
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image","1"])
                .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
            if !status.success(){return Err(format!("GGUF gated mix packaging failed: {status}").into());}
        } else if matches!(pkg.name,"gguf-glu-encode" | "gguf-moe-glu-encode") {
            let kernels = root.join("mistralrs-quant/kernels/mmq_gguf");
            let watched = ["mmq_common.cuh","mmq_gguf.cuh","mmq_vecdotq.cuh","mmq_mma.cuh","mmq_quantize.cu"].map(|n|kernels.join(n));
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image","1"])
                .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
            if !status.success(){return Err(format!("GGUF GLU packaging failed: {status}").into());}
        } else if matches!(pkg.name, "gguf-mmq" | "gguf-moe-compact") {
            let kernels = root.join("mistralrs-quant/kernels/mmq_gguf");
            let watched = ["mmq_common.cuh","mmq_gguf.cuh","mmq_vecdotq.cuh",
                "mmq_mma.cuh","mmq_quantize.cu"].map(|name| kernels.join(name));
            for id in 1..=11 {
                let define = if pkg.name == "gguf-mmq" { format!("-DDKR_GGUF_MMQ_IMAGE={id}") } else { format!("-DDKR_GGUF_MOE_IMAGE={id}") };
                let compiled = compile_ptx(&source, &work.join(pkg.name).join(format!("image-{id:02}")),
                    cc, pkg.fast_math, Some(&define), &watched)?;
                let status = Command::new(&python).arg(&script)
                    .args(["--cc", &cc.to_string(), "--image", &id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled)
                    .current_dir(root).status()?;
                if !status.success() { return Err(format!("GGUF MMQ shard packaging failed for {id}: {status}").into()); }
            }
           } else if pkg.name=="attention-prep" {
            let watched=[root.join("mistralrs-core/src/cuda/attention_prep.cu"),
                root.join("mistralrs-quant/kernels/rotary/rotary.cu"),
                root.join("mistralrs-quant/kernels/rotary/cuda_compat.h"),
                root.join("mistralrs-core/src/cuda/gdn_norm_device.cuh")];
            for id in 1..=3 {
                let define=format!("-DDKR_AP_DTYPE={id}");
                let compiled=compile_ptx(&source,&work.join(pkg.name).join(format!("dtype-{id}")),cc,pkg.fast_math,Some(&define),&watched)?;
                let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image",&id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
                if !status.success(){return Err(format!("attention prep packaging failed: {status}").into());}
            }
        } else if pkg.name=="gguf-grouped-row" {
            let watched=[root.join("mistralrs-quant/kernels/moe_grouped/moe_grouped.cu")];
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image","1"])
                .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
            if !status.success(){return Err(format!("GGUF grouped row packaging failed: {status}").into());}
        } else if matches!(pkg.name,"gguf-moe-decode" | "gguf-residual") {
            let kernels=root.join("mistralrs-quant/kernels");
            let watched=["indexed_moe/indexed_moe.cu","indexed_moe/moe_decode_prefetch.cu","gguf/vecdot.cuh","mmvq_gguf/mmvq_gguf.cu","mmvq_gguf/prefetch_row.cuh","gemv/row.cuh"].map(|n|kernels.join(n));
            let residual=pkg.name=="gguf-residual";
            for id in 1..=if residual{3}else{2} {
                let define=if residual{format!("-DDKR_GGUF_RESIDUAL_IMAGE={id}")}else{format!("-DDKR_GGUF_MOE_DECODE_IMAGE={id}")};
                let compiled=compile_ptx(&source,&work.join(pkg.name).join(format!("image-{id:02}")),cc,pkg.fast_math,Some(&define),&watched)?;
                let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image",&id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
                if !status.success(){return Err(format!("GGUF MoE decode shard {id}: {status}").into());}
            }
        } else if matches!(pkg.name,"gguf-mmvq" | "gguf-mmvq-fused") {
            let kernels = root.join("mistralrs-quant/kernels");
            let watched = [
                kernels.join("mmvq_gguf/mmvq_gguf.cu"),
                kernels.join("mmvq_gguf/prefetch_row.cuh"),
                kernels.join("gguf/vecdot.cuh"),
                kernels.join("gemv/row.cuh"),
            ];
            let fused=pkg.name=="gguf-mmvq-fused";
            let mut watched=watched.to_vec();
            if fused {watched.push(kernels.join("indexed_moe/indexed_moe.cu"));}
            for id in 1..=if fused {10}else{11} {
                let define = if fused {format!("-DDKR_GGUF_MMVQ_FUSED_IMAGE={id}")}else{format!("-DDKR_GGUF_MMVQ_IMAGE={id}")};
                let compiled = compile_ptx(&source, &work.join(pkg.name).join(format!("image-{id:02}")),
                    cc, pkg.fast_math, Some(&define), &watched)?;
                let status = Command::new(&python).arg(&script)
                    .args(["--cc", &cc.to_string(), "--image", &id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled)
                    .current_dir(root).status()?;
                if !status.success() {
                    return Err(format!("GGUF MMVQ shard packaging failed for {id}: {status}").into());
                }
            }
        } else if pkg.name == "gdn-transaction" {
            let watched=[root.join("mistralrs-core/src/cuda/gdn_transition_device.cuh")];
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            package(&python,&script,cc,&output,&compiled,false,root)?;
        } else if pkg.name == "gdn-decode" {
            let watched=[root.join("mistralrs-core/src/cuda/gdn_decode_device.cuh"),
                         root.join("mistralrs-core/src/cuda/gdn_recurrence_conv_device.cuh")];
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            package(&python,&script,cc,&output,&compiled,false,root)?;
        } else if pkg.name == "gdn-preparation" {
            let watched=[root.join("mistralrs-core/src/cuda/gdn_preparation_device.cuh"),
                         root.join("mistralrs-core/src/cuda/gdn_norm_device.cuh")];
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            package(&python,&script,cc,&output,&compiled,false,root)?;
        } else if pkg.name == "gdn-recurrence" {
            let watched=[root.join("mistralrs-core/src/cuda/gdn_recurrence_conv_device.cuh")];
            for id in 1..=3 {
                let define=format!("-DDKR_GDN_DTYPE={id}");
                let compiled=compile_ptx(&source,&work.join(pkg.name).join(format!("dtype-{id}")),cc,pkg.fast_math,Some(&define),&watched)?;
                let status=Command::new(&python).arg(&script).args(["--cc",&cc.to_string(),"--image",&id.to_string()])
                    .arg("--output").arg(&output).arg("--ptx-file").arg(&compiled).current_dir(root).status()?;
                if !status.success(){return Err(format!("GDN recurrence packaging failed: {status}").into());}
            }
        } else if pkg.name == "dynamic-conv" {
            let watched=[root.join("mistralrs-core/src/cuda/dynamic_conv_device.cuh")];
            let compiled=compile_ptx(&source,&work.join(pkg.name),cc,pkg.fast_math,None,&watched)?;
            package(&python,&script,cc,&output,&compiled,false,root)?;
        } else if pkg.name == "moe" {
            let moe_dir = work.join("moe");
            for (variant, define) in [
                ("f16", "-UDKR_BF16_MODE"),
                ("bf16", "-DDKR_BF16_MODE=1"),
            ] {
                compile_ptx(
                    &source, &moe_dir.join(variant), cc, pkg.fast_math, Some(define), &[],
                )?;
            }
            package(&python, &script, cc, &output, &moe_dir, true, root)?;
        } else {
            let watched = if matches!(pkg.name, "mla-decode" | "flashinfer-merge") {
                flashinfer_headers.as_slice()
            } else if cc == 61 && matches!(pkg.name, "paged-decode" | "paged-v2") {
                paged_headers.as_slice()
            } else {
                &[]
            };
            let compiled = compile_ptx(
                &source, &work.join(pkg.name), cc, pkg.fast_math, None, watched,
            )?;
            package(&python, &script, cc, &output, &compiled, false, root)?;
        }
    }
    }
    if cubin {
        let images = stage_cubins(&output.join(format!("sm{cc}")), cc)?;
        println!("DKR_CUDAFORGE_CUBIN_PAIRED=STAGED target_cc={cc} images={images}");
    }
    println!("DKR_CUDAFORGE_PACKAGES=STAGED target_cc={cc} families={}", if pair_existing { 0 } else if family.is_some() { 1 } else { PACKAGES.len() });
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("DKR cudaforge package build failed: {error}");
        std::process::exit(1);
    }
}
