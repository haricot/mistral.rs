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
    Package { name: "kv-cache-ops", source: "kv_cache_ops.cu", fast_math: false },
    Package { name: "kv-fp8-io", source: "kv_fp8_io.cu", fast_math: false },
    Package { name: "routed-quant", source: "routed_quant.cu", fast_math: false },
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
        // Reads and validates the PTX source, including the entry parameter ABI.
        let verified = VerifiedKernel::read(&manifest_path)?;
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
            compiled.insert(identity, binary.clone());
            binary
        };
        let destination = verified.manifest.artifact.with_extension("cubin");
        let dest_path = manifest_path.parent().ok_or("manifest missing directory")?
            .join(&destination);
        fs::copy(&binary, &dest_path)?;
        let bytes = fs::read(&dest_path)?;
        let sha = format!("{:x}", Sha256::digest(&bytes));
        let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
        manifest["cubin_artifact"] = destination.to_string_lossy().into_owned().into();
        manifest["cubin_sha256"] = sha.into();
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
        // Do not publish an image that fails the same checks as runtime discovery.
        let validated = VerifiedKernel::read(&manifest_path)?;
        if validated.cubin.is_none() {
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
    let work = output.join(format!("sm{cc}/_cudaforge_cache"));
    if !pair_existing {
    for pkg in PACKAGES.iter().filter(|p| family.as_ref().is_none_or(|name| p.name == name.as_str())) {
        let dir = sources.join(pkg.name);
        let source = dir.join(pkg.source);
        if !source.is_file() {
            return Err(format!("missing DKR CUDA source: {}", source.display()).into());
        }
        let script = dir.join("build.py");
        if pkg.name == "moe" {
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
            let watched = if cc == 61 && matches!(pkg.name, "paged-decode" | "paged-v2") {
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
