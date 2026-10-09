#!/usr/bin/env python3
"""Strict semantic port of 30f21d8f (quant BF16) onto the 2026 upstream.

The upstream AFQ kernels already implement BF16 conversion and float
accumulation on SM61; preserve those kernels instead of resurrecting old
preprocessor branches. Port only the missing build-time opt-in and GEMV /
cuBLASLt dispatch avoidance. Abort on unexpected shapes or conflicts.
"""
from pathlib import Path
import subprocess

EXPECTED = "30f21d8f79f2209f8a6fa74edb57a350e9ad017c"
PATHS = {
    "mistralrs-quant/build.rs",
    "mistralrs-quant/kernels/afq/afq.cu",
    "mistralrs-quant/kernels/afq/afq_gemm.cu",
    "mistralrs-quant/kernels/afq/afq_utils.cuh",
    "mistralrs-quant/src/unquantized/mod.rs",
}

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def require(flag: bool, msg: str) -> None:
    if not flag:
        raise RuntimeError(msg)

def replace_one(data: str, anchor: str, value: str, label: str) -> str:
    hits = data.count(anchor)
    require(hits == 1, f"{label}: expected 1 exact occurrence, got {hits}")
    return data.replace(anchor, value, 1)

require(git("rev-parse", "REBASE_HEAD") == EXPECTED, "Unexpected patch SHA")
unmerged = set(git("diff", "--name-only", "--diff-filter=U").splitlines())
require(unmerged == PATHS, f"Unexpected conflicting files: {sorted(unmerged)}")
for path in sorted(PATHS):
    Path(path).write_text(subprocess.check_output(["git", "show", f":2:{path}"], text=True))

# Confirm the current upstream already integrates each former AFQ BF16 fix
# without SM80-only guards. Never accept a backend that silently drops BF16.
afq = Path("mistralrs-quant/kernels/afq/afq.cu").read_text()
qmm = Path("mistralrs-quant/kernels/afq/afq_gemm.cu").read_text()
utils = Path("mistralrs-quant/kernels/afq/afq_utils.cuh").read_text()
require("#if __CUDA_ARCH__ >= 800" not in afq, "AFQ BF16 is SM80 gated")
require("#if __CUDA_ARCH__ >= 800" not in qmm, "AFQ GEMM BF16 is SM80 gated")
require(afq.count("__bfloat162float") >= 2 and afq.count("__float2bfloat16") >= 2,
        "AFQ lacks portable BF16 arithmetic")
require(qmm.count("__bfloat162float") >= 5 and qmm.count("__float2bfloat16") >= 2,
        "AFQ GEMM lacks portable BF16 arithmetic")
require("dequant_value<__nv_bfloat16>" in utils and
        "__CUDA_ARCH__ >= 800" in utils and
        "fmaf((float)q, __bfloat162float(scale), __bfloat162float(bias))" in utils,
        "Portable AFQ BF16 dequantization missing")

# Add build-time flag to the MODERN quant builder rather than replacing its
# header hash, CUTLASS, deepgemm and CUDA architecture selection.
path = Path("mistralrs-quant/build.rs")
build = path.read_text()
build = replace_one(build,
    '    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");\n',
    '    println!("cargo::rustc-check-cfg=cfg(has_mxfp4_wmma_kernels)");\n'
    '    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");\n',
    "declare BF16 build cfg")
build = replace_one(build,
    '        println!("cargo:rerun-if-changed=build.rs");\n',
    '        println!("cargo:rerun-if-changed=build.rs");\n'
    '        println!("cargo:rerun-if-env-changed=ALLOW_LEGACY");\n',
    "rerun on ALLOW_LEGACY")
anchor = '        let compute_cap = builder.get_compute_cap().unwrap_or(80);\n'
flag = '''        let allow_legacy = std::env::var("ALLOW_LEGACY").unwrap_or_default();
        let allow_legacy_bf16 = allow_legacy == "all"
            || allow_legacy
                .split(',')
                .map(str::trim)
                .any(|feature| feature == "bf16");
        if allow_legacy_bf16 {
            builder = builder.arg("-DALLOW_LEGACY_BF16");
            println!("cargo:rustc-cfg=allow_legacy_bf16");
        }

'''
build = replace_one(build, anchor, flag + anchor, "modern quant builder capability")
path.write_text(build)

# The modern UnquantLinear code optimizes matmul on CUDA, but it must not
# send Pascal BF16 inputs to specialized GEMV/cuBLASLt paths that are not
# guaranteed on SM61. Candle owns the BF16 fallback.
path = Path("mistralrs-quant/src/unquantized/mod.rs")
src = path.read_text()
marker = "impl UnquantLinear {\n"
helper = '''// Legacy BF16 routes through Candle's supported matmul path on Pascal,
// avoiding the specialized GEMV/cuBLASLt kernels. Opt-in only.
fn use_legacy_bf16_fallback(input: &Tensor) -> bool {
    cfg!(feature = "cuda")
        && cfg!(allow_legacy_bf16)
        && input.device().is_cuda()
        && input.dtype() == DType::BF16
}

'''
src = replace_one(src, marker, helper + marker, "legacy BF16 helper")
src = replace_one(src,
    "        maybe_init_cublas_lt_wrapper(a.device().clone());\n",
    "        maybe_init_cublas_lt_wrapper(a.device().clone());\n"
    "        let legacy_bf16 = use_legacy_bf16_fallback(a);\n",
    "unquantized dispatch selection")
src = replace_one(src,
    "        if crate::gemv::should_use_gemv(a, &self.w) {\n",
    "        if !legacy_bf16 && crate::gemv::should_use_gemv(a, &self.w) {\n",
    "specialized GEMV bypass")
cublas = "if supports_cublaslt_batch_matmul(a, &w) {"
require(src.count(cublas) == 2,
        f"Unexpected cuBLASLt decision count: {src.count(cublas)}")
src = src.replace(cublas, "if !legacy_bf16 && supports_cublaslt_batch_matmul(a, &w) {")
path.write_text(src)

# Add a physical test for THE dispatch semantics, not just tensor dtype
# conversions. New helper is local, so test it at the source module.
with path.open("a") as handle:
    handle.write('''
#[cfg(all(test, feature = "cuda"))]
mod legacy_bf16_pascal_proof {
    use super::*;

    #[test]
    fn legacy_bf16_unquantized_linear_avoids_unsupported_dispatch() -> Result<()> {
        if !cfg!(allow_legacy_bf16) {
            return Ok(());
        }
        let device = Device::new_cuda(0)?;
        let a = Tensor::new(&[[1.0_f32, 2.0]], &device)?.to_dtype(DType::BF16)?;
        assert!(use_legacy_bf16_fallback(&a));
        let weight = Tensor::new(&[[1.0_f32, 1.0], [2.0, 0.0]], &device)?
            .to_dtype(DType::BF16)?;
        let layer = <UnquantLinear as QuantMethod>::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None))
        )?;
        let result = layer.forward_raw(&a)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        assert!((result[0][0] - 3.0).abs() < 0.1);
        assert!((result[0][1] - 2.0).abs() < 0.1);
        Ok(())
    }
}
''')

for path in sorted(PATHS):
    subprocess.run(["git", "add", path], check=True)
subprocess.run(["git", "diff", "--cached", "--check"], check=True)
print("PORT_QUANT_R3_READY: upstream AFQ portable; Pascal BF16 dispatch + test")
