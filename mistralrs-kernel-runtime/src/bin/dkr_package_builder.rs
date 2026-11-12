//! Build all external DKR PTX modules with cudaforge for one target GPU.
//!
//! Unlike the historical Python builders, NVCC compilation is owned by the
//! pinned cudaforge crate. Existing Python emitters only package and fingerprint
//! already compiled PTX; this preserves the versioned typed manifests.

use std::{env, fs, path::{Path, PathBuf}, process::Command};

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
    Package { name: "quant-gemm", source: "quant_gemm.cu", fast_math: false },
];

fn parse_args() -> Result<(usize, PathBuf, String), Box<dyn std::error::Error>> {
    let mut cc: Option<usize> = None;
    let mut output = PathBuf::from("kernel-packages/build/cuda");
    let mut python = String::from("python3");
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("missing argument for {flag}"))?;
        match flag.as_str() {
            "--cc" => cc = Some(value.parse()?),
            "--output" => output = PathBuf::from(value),
            "--python" => python = value,
            other => return Err(format!("unknown option {other}").into()),
        }
    }
    let cc = cc.ok_or("usage: --cc 61 [--output DIR] [--python python3]")?;
    if !(50..=120).contains(&cc) {
        return Err(format!("unsupported compute capability {cc}").into());
    }
    Ok((cc, output, python))
}

fn compile_ptx(
    source: &Path,
    out_dir: &Path,
    cc: usize,
    fast_math: bool,
    mode: Option<&str>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::create_dir_all(out_dir)?;
    let mut builder = cudaforge::KernelBuilder::new()
        .source_files([source])
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

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (cc, requested_output, python) = parse_args()?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().ok_or("no repository root")?;
    let output = if requested_output.is_absolute() {
        requested_output
    } else {
        root.join(requested_output)
    };
    let sources = root.join("kernel-packages/cuda");
    let work = output.join(format!("sm{cc}/_cudaforge_cache"));
    for pkg in PACKAGES {
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
                    &source, &moe_dir.join(variant), cc, pkg.fast_math, Some(define),
                )?;
            }
            package(&python, &script, cc, &output, &moe_dir, true, root)?;
        } else {
            let compiled = compile_ptx(
                &source, &work.join(pkg.name), cc, pkg.fast_math, None,
            )?;
            package(&python, &script, cc, &output, &compiled, false, root)?;
        }
    }
    println!("DKR_CUDAFORGE_PACKAGES=STAGED target_cc={cc} families={}", PACKAGES.len());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("DKR cudaforge package build failed: {error}");
        std::process::exit(1);
    }
}
