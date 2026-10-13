#!/usr/bin/env python3
"""Split Pascal FP8 software kernels from SM80+ tensor-core FP8 MMA.

Modern upstream has_blockwise_fp8_kernels gates both portable FP8 and
TensorCoreGemv/MMA FFI. Historical Pascal opt-in enables software kernels
with that cfg while excluding *_fp8_mma.cu, leaving undefined native symbols.

Older asd_runner avoided this issue by not having the MMA module.
Retain modern upstream MMA on SM80+ (where the .cu source is compiled) with a
separate cfg; keep portable blockwise FP8 enabled on SM61 when opted in.
"""
from pathlib import Path


def edit_once(src: str, old: str, new: str, context: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"FP8 MMA split {context}: expected one anchor, got {n}")
    return src.replace(old, new, 1)


build_path = Path("mistralrs-quant/build.rs")
build = build_path.read_text()
module_path = Path("mistralrs-quant/src/blockwise_fp8/mod.rs")
module = module_path.read_text()
ffi_path = Path("mistralrs-quant/src/blockwise_fp8/ffi.rs")
ffi = ffi_path.read_text()

# The historical SM61 candidate may already have the native/portable MMA
# split. Accept it only after validating every known component together.
present = {
    "build cfg declaration": build.count('println!("cargo::rustc-check-cfg=cfg(has_fp8_mma_kernels)");'),
    "native SM80 cfg": build.count('println!("cargo:rustc-cfg=has_fp8_mma_kernels");'),
    "MMA consumer guards": module.count("has_fp8_mma_kernels"),
    "MMA native FFI guard": ffi.count("#[cfg(has_fp8_mma_kernels)]"),
}
if any(present.values()):
    if not (
        present["build cfg declaration"] == 1
        and present["native SM80 cfg"] == 1
        and present["MMA consumer guards"] == 20
        and module.count("has_blockwise_fp8_kernels") == 0
        and present["MMA native FFI guard"] == 1
        and ffi.count("#[cfg(has_blockwise_fp8_kernels)]") == 1
        and 'HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels)' in ffi
        and 'println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");' in build
        and '"*_fp8_mma.cu",' in build
    ):
        raise RuntimeError(
            "FP8 MMA pre-existing split is partial or duplicated; "
            f"markers={present}; refusing replay"
        )
    print("PORT_FP8_MMA_GUARD_ALREADY_VALIDATED: native MMA SM80+, Pascal software FP8 retained")
    raise SystemExit(0)

build = edit_once(
    build,
    '    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");\n',
    '    println!("cargo::rustc-check-cfg=cfg(has_blockwise_fp8_kernels)");\n'
    '    println!("cargo::rustc-check-cfg=cfg(has_fp8_mma_kernels)");\n',
    "cfg declaration",
)
build = edit_once(
    build,
    '''        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
''',
    '''        if cc_over_80 {
            println!("cargo:rustc-cfg=has_marlin_kernels");
            // Never advertise native MMA when *_fp8_mma.cu is excluded (SM61).
            // Keep the original SM80+ compilation capability unchanged.
            println!("cargo:rustc-cfg=has_fp8_mma_kernels");
            // WMMA tensor core MXFP4 kernel (FP16/BF16 WMMA requires SM >= 80)
''',
    "native MMA cap gate",
)
# The prior historical FP8 port explicitly excludes this CUDA source on SM61.
assert '"*_fp8_mma.cu",' in build, "Native MMA source exclusion no longer present"
assert 'println!("cargo:rustc-cfg=has_blockwise_fp8_kernels");' in build, (
    "Portable FP8 cfg must be preserved"
)
build_path.write_text(build)

module_path = Path("mistralrs-quant/src/blockwise_fp8/mod.rs")
module = module_path.read_text()
assert 'pub(crate) mod mma;' in module, "Modern MMA module missing"
assert 'mma::weight_supported(' in module and 'mma::gemv(' in module, (
    "Unexpected modern FP8 MMA consumer layout"
)
n = module.count("has_blockwise_fp8_kernels")
assert n == 20, f"Expected 20 native MMA-only guards in module; got {n}"
module = module.replace("has_blockwise_fp8_kernels", "has_fp8_mma_kernels")
assert module.count("has_fp8_mma_kernels") == 20
assert "has_blockwise_fp8_kernels" not in module
module_path.write_text(module)

ffi_path = Path("mistralrs-quant/src/blockwise_fp8/ffi.rs")
ffi = ffi_path.read_text()
ffi = edit_once(
    ffi,
    '''#[cfg(has_blockwise_fp8_kernels)]
extern "C" {
    pub(crate) fn mistralrs_fp8_mma_error_string''',
    '''#[cfg(has_fp8_mma_kernels)]
extern "C" {
    pub(crate) fn mistralrs_fp8_mma_error_string''',
    "native MMA FFI",
)
assert ffi.count("#[cfg(has_blockwise_fp8_kernels)]") == 1, (
    "Portable fused RMSNorm FP8 FFI must remain enabled"
)
assert 'HAVE_BLOCKWISE_DEQUANT_KERNELS: bool = cfg!(has_blockwise_fp8_kernels)' in ffi
ffi_path.write_text(ffi)

print("PORT_FP8_MMA_GUARD_R1_READY: native MMA SM80+ only; SM61 software FP8 retained")
