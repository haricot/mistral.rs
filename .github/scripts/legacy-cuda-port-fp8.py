#!/usr/bin/env python3
"""Port only the opt-in FP8 flags from the exact historical commit bd7e8d43.

Keep all upstream 2026 headers, CUTLASS/DeepGEMM guards and quant/PagedAttn
builders. FP8 conversions on Pascal are deliberately gated by ALLOW_LEGACY;
hosted CUDA compilation and physical SM61 tests must pass before promotion.
"""
import subprocess
from pathlib import Path

EXPECTED = "bd7e8d43bc680f8e3a5d4d291a076dffe1e8d3b3"
PATHS = {
    "mistralrs-paged-attn/build.rs",
    "mistralrs-paged-attn/src/cuda/quantization/fp8/nvidia/quant_utils.cuh",
    "mistralrs-quant/build.rs",
}
def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()
def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)
def edit_one(src, needle, replacement, label):
    require(src.count(needle) == 1,
            f"{label}: expected one exact anchor, got {src.count(needle)}")
    return src.replace(needle, replacement, 1)

def enable_pascal_fp8_sources(src):
    """Edit only the FP8 filters inside the modern excluded_files block."""
    start = '        let mut excluded_files = if cc_over_80 {\n'
    require(src.count(start) == 1,
            f"FP8 source inclusion: expected one excluded_files assignment, got {src.count(start)}")
    begin = src.index(start)
    end_marker = '        };\n'
    end = src.find(end_marker, begin)
    require(end != -1, "FP8 source inclusion: excluded_files block has no terminator")
    end += len(end_marker)

    block = src[begin:end]
    lines = block.splitlines(keepends=True)
    # FP8 MMA requires tensor cores (SM80+); never enable it for Pascal.
    # Only the scalar/vector FP8 source families are conditionally enabled.
    filters = ('"*_fp8.cu",', '"*_fp8_gemm.cu",')
    for pattern in filters:
        hits = [i for i, line in enumerate(lines) if pattern in line]
        require(len(hits) == 1,
                f"FP8 source inclusion: expected one {pattern} filter, got {len(hits)}")
        del lines[hits[0]]

    guard = '''        if !cc_over_80 && !allow_legacy_fp8 {
            excluded_files.extend(["*_fp8.cu", "*_fp8_gemm.cu"]);
        }
'''
    require(guard not in src, "FP8 source inclusion: conditional filter guard already present")
    return src[:begin] + ''.join(lines) + guard + src[end:]

require(git("rev-parse", "REBASE_HEAD") == EXPECTED, "Wrong FP8 commit")
conflicts = set(git("diff", "--name-only", "--diff-filter=U").splitlines())
require(bool(conflicts) and conflicts <= PATHS,
        f"Unexpected conflict set: {sorted(conflicts)}")
# Use the last committed rebase state for ALL files: it includes modern
# upstream plus earlier ports. This also resets a cleanly auto-merged portion
# of the historical patch before semantic replay, avoiding duplicate flags.
for path in sorted(PATHS):
    Path(path).write_text(subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True))

qpath = Path("mistralrs-quant/build.rs")
quant = qpath.read_text()
# R3 owns BF16 forwarding and must already have run.
assert '"cargo::rustc-check-cfg=cfg(allow_legacy_bf16)"' in quant
quant = edit_one(quant,
    '    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");\n',
    '    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");\n'
    '    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");\n',
    "FP8 cfg declaration")
optin = '''        if allow_legacy_bf16 {
            builder = builder.arg("-DALLOW_LEGACY_BF16");
            println!("cargo:rustc-cfg=allow_legacy_bf16");
        }

'''
newoptin = '''        let allow_legacy_fp8 = allow_legacy == "all"
            || allow_legacy
                .split(',')
                .map(str::trim)
                .any(|feature| feature == "fp8");
        if allow_legacy_bf16 {
            builder = builder.arg("-DALLOW_LEGACY_BF16");
            println!("cargo:rustc-cfg=allow_legacy_bf16");
        }
        if allow_legacy_fp8 {
            builder = builder.arg("-DALLOW_LEGACY_FP8");
            println!("cargo:rustc-cfg=allow_legacy_fp8");
        }

'''
quant = edit_one(quant, optin, newoptin, "FP8 environment switch")
quant = edit_one(quant,
    '        let cc_over_80 = compute_cap >= 80;\n',
    '        let cc_over_80 = compute_cap >= 80;\n'
    '        let enable_legacy_fp8 = cc_over_80 || allow_legacy_fp8;\n',
    "SM61 FP8 feature gate")
fast = '''        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
'''
portable = '''        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
            println!("cargo:rustc-cfg=has_mxfp4_wmma_kernels");
        }
        if enable_legacy_fp8 {
            println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");
            println!("cargo:rustc-cfg=has_scalar_fp8_kernels");
            println!("cargo:rustc-cfg=has_vector_fp8_kernels");
        }
'''
quant = edit_one(quant, fast, portable, "keep WMMA disabled on Pascal")
quant = enable_pascal_fp8_sources(quant)
qpath.write_text(quant)

path = Path("mistralrs-paged-attn/build.rs")
paged = path.read_text()
paged = edit_one(paged,
    '    println!("cargo:rerun-if-changed=build.rs");\n',
    '    println!("cargo:rerun-if-changed=build.rs");\n'
    '    println!("cargo:rerun-if-env-changed=ALLOW_LEGACY");\n',
    "paged FP8 rerun hook")
old = '''    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    // Enable FP8 if compute capability >= 8.0 (Ampere and newer)
    let using_fp8 = if compute_cap >= 80 {
        builder = builder.arg("-DENABLE_FP8");
        true
    } else {
        false
    };
'''
new = '''    let compute_cap = builder.get_compute_cap().unwrap_or(80);
    let allow_legacy = std::env::var("ALLOW_LEGACY").unwrap_or_default();
    let allow_legacy_fp8 = allow_legacy == "all"
        || allow_legacy.split(',').map(str::trim).any(|x| x == "fp8");

    // Native FP8 is SM80+. Pascal uses opt-in software conversion.
    let using_fp8 = if compute_cap >= 80 || allow_legacy_fp8 {
        builder = builder.arg("-DENABLE_FP8");
        if allow_legacy_fp8 {
            builder = builder.arg("-DALLOW_LEGACY_FP8");
        }
        true
    } else {
        false
    };
'''
paged = edit_one(paged, old, new, "PagedAttn legacy FP8 gate")
path.write_text(paged)

path = Path("mistralrs-paged-attn/src/cuda/quantization/fp8/nvidia/quant_utils.cuh")
fp8 = path.read_text()
original = '''    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning'''
fixed = '''  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;'''
fp8 = edit_one(fp8, original, fixed, "BF16 to FP8 software conversion")
path.write_text(fp8)

for path in sorted(PATHS):
    subprocess.run(["git", "add", path], check=True)
subprocess.run(["git", "diff", "--cached", "--check"], check=True)
print("PORT_FP8_R7_READY: semantic source exclusions + SM61 software FP8")
