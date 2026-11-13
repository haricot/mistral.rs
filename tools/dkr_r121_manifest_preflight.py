#!/usr/bin/env python3
"""CPU-only R1.21 manifest contract preflight.

This stages sample manifests against dummy PTX in a temporary directory:
it tests Python packaging, SHA integrity, entry identities, argument counts
and compiler-flag separation. It does NOT certify PTX/kernel correctness and
never runs Cargo, NVCC, or a GPU.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = {
    "quant-gemv": {
        "profiles": 6,
        "ops": {
            "quant.fp8.blockwise.gemv.b128.v1": (10, 6),
            "quant.mxfp4.gemv.plain.b128.v1": (7, 3),
            "quant.mxfp4.gemv.bias.b128.v1": (8, 3),
        },
    },
    "paged-decode": {
        "profiles": 3,
        "ops": {"attention.paged.decode.online.v1": (15, 9)},
    },
    "kv-cache-ops": {
        "profiles": 4,
        "ops": {
            "kv_cache.copy_blocks.bytes.b256.v1": (7, 4),
            "kv_cache.reshape_store.vllm.b256.v1": (11, 6),
        },
    },
    "paged-v2": {
        "profiles": 6,
        "ops": {
            "attention.paged.v2.partition.p256.v1": (16, 9),
            "attention.paged.v2.merge.p256.v1": (7, 4),
        },
    },
    "paged-fp8": {
        "profiles": 3,
        "ops": {"attention.paged.fp8_decode.online.v1": (17, 9)},
    },
    "quant-gemm": {
        "profiles": 6,
        "ops": {
            "quant.fp8.blockwise.gemm.t16.v1": (10, 6),
            "quant.mxfp4.gemm.plain.t16.v1": (7, 3),
            "quant.mxfp4.gemm.bias.t16.v1": (8, 3),
        },
    },
    "nvfp4-gemv": {
        "profiles": 4,
        "ops": {
            "quant.nvfp4.gemv.plain.b128.v1": (8, 3),
            "quant.nvfp4.gemv.bias.b128.v1": (9, 3),
        },
    },
}


def validate(pkg, options, root):
    script = ROOT / "kernel-packages" / "cuda" / pkg / "build.py"
    cu = script.with_name({
        "quant-gemv": "quant_gemv.cu",
        "paged-decode": "paged_decode.cu",
        "nvfp4-gemv": "nvfp4_gemv.cu",
        "quant-gemm": "quant_gemm.cu",
        "paged-fp8": "paged_fp8.cu",
        "paged-v2": "paged_v2.cu",
        "kv-cache-ops": "kv_cache_ops.cu",
    }[pkg])
    if not (script.is_file() and cu.is_file()):
        raise AssertionError(f"{pkg}: source or packager missing")
    ptx = root / "fixture.ptx"
    ptx.write_text(
        ".version 7.0\n.target sm_61\n.address_size 64\n"
        "// FAKE. Not a compiled CUDA kernel.\n"
    )
    out = root / "out"
    invocation = [
        sys.executable, str(script), "--cc", "61",
        "--output", str(out), "--ptx-file", str(ptx),
    ]
    subprocess.run(invocation, check=True, capture_output=True, text=True)
    manifests = sorted((out / "sm61" / pkg).rglob("manifest.json"))
    if len(manifests) != options["profiles"]:
        raise AssertionError(
            f"{pkg}: expected {options['profiles']} profiles, found {len(manifests)}"
        )
    counts = {}
    for path in manifests:
        data = json.loads(path.read_text())
        if data["schema"] != "DKR-KERNEL-MODULE-V1" or data["abi"] != "dkr.typed.v1":
            raise AssertionError(f"{path}: invalid schema or ABI")
        if data["target_cc"] != 61 or data["block"] not in ([128, 1, 1], [16, 16, 1], [256, 1, 1]):
            raise AssertionError(f"{path}: wrong SM61 geometry")
        op = data["operation"]
        if op not in options["ops"]:
            raise AssertionError(f"{path}: unexpected op {op}")
        expected_total, expected_scalars = options["ops"][op]
        args = data["arguments"]
        if len(args) != expected_total:
            raise AssertionError(f"{path}: {len(args)} args, expected {expected_total}")
        scalar_count = sum(arg["kind"] == "scalar" for arg in args)
        if scalar_count != expected_scalars:
            raise AssertionError(f"{path}: unexpected scalar count {scalar_count}")
        if data["sha256"] != hashlib.sha256(
            (path.parent / data["artifact"]).read_bytes()
        ).hexdigest():
            raise AssertionError(f"{path}: artifact integrity failure")
        if data["entry"] not in (
            {
                "quant-gemv": [
                    f"dkr_fp8_block_gemv_{dt}"
                    for dt in ("f16", "bf16")
                ] + [
                    f"dkr_mxfp4_gemv_{dt}_{mode}"
                    for dt in ("f16", "bf16") for mode in ("plain", "bias")
                ],
                "kv-cache-ops": [
                    "dkr_kv_copy_blocks_bytes",
                    *[f"dkr_kv_reshape_store_{dt}"
                      for dt in ("f16", "bf16", "f32")],
                ],
                "paged-v2": [
                    f"dkr_paged_v2_{stage}_{dt}"
                    for stage in ("partition", "merge")
                    for dt in ("f16", "bf16", "f32")
                ],
                "paged-fp8": [
                    f"dkr_paged_fp8_decode_{dt}"
                    for dt in ("f16", "bf16", "f32")
                ],
                "paged-decode": [
                    f"dkr_paged_decode_{dt}"
                    for dt in ("f16", "bf16", "f32")
                ],
                "quant-gemm": [
                    f"dkr_fp8_block_gemm_{dt}"
                    for dt in ("f16", "bf16")
                ] + [
                    f"dkr_mxfp4_gemm_{dt}_{mode}"
                    for dt in ("f16", "bf16") for mode in ("plain", "bias")
                ],
                "nvfp4-gemv": [
                    f"dkr_nvfp4_gemv_{dt}_{mode}"
                    for dt in ("f16", "bf16") for mode in ("plain", "bias")
                ],
            }[pkg]
        ):
            raise AssertionError(f"{path}: unexpected symbol identity")
        counts[op] = counts.get(op, 0) + 1
    subprocess.run(invocation, check=True, capture_output=True, text=True)
    print(f"DKR_R121_MANIFEST_PREFLIGHT={pkg}:PASS profiles={len(manifests)} "
          f"operations={len(counts)}")
    return len(manifests)


def main():
    total = 0
    with tempfile.TemporaryDirectory(prefix="dkr-r121-manifest-") as temp:
        for pkg, options in PACKAGES.items():
            root = Path(temp) / pkg
            root.mkdir()
            total += validate(pkg, options, root)
    builder = (ROOT / "mistralrs-kernel-runtime" / "src" / "bin" /
               "dkr_package_builder.rs").read_text()
    if not all(f'name: "{pkg}"' in builder for pkg in PACKAGES):
        raise AssertionError("a new source family is missing from cudaforge builder")
    # Avoid environment mutation and NVCC-wide options in source preflight.
    build_files = ["mistralrs-quant", "mistralrs-core", "mistralrs-paged-attn"]
    for name in build_files:
        source = (ROOT / name / "build.rs").read_text()
        for macro in ("DKR_QUANT_GEMV", "DKR_QUANT_GEMM", "DKR_PAGED_FP8", "DKR_PAGED_V2", "DKR_KV_CACHE_OPS", "DKR_PAGED_DECODE", "DKR_NVFP4_GEMV"):
            if f'-DMISTRALRS_{macro}' in source:
                raise AssertionError(f"global CUDA flag regression in {name}: {macro}")
    print(f"DKR_R121_MANIFEST_PREFLIGHT=PASS profiles={total} no_cuda_compilation")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"DKR_R121_MANIFEST_PREFLIGHT=FAIL {exc}", file=sys.stderr)
        sys.exit(1)
