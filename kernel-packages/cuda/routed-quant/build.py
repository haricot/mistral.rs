#!/usr/bin/env python3
"""DKR-R1.21/S8: typed routed quantized GEMV/GEMM packages.

The Rust cudaforge builder compiles PTX. This script only packages an
already-compiled PTX and stamps its SHA/ABI; no Python NVCC compilation
occurs when --ptx-file is supplied.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("routed_quant.cu")
CODECS = ("fp8", "mxfp8", "mxfp4", "nvfp4")
DTYPES = ("f16", "bf16", "f32")
STAGES = ("gemv", "gemm")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def tensor(name, dtype, rank, access, representation=None):
    value = {"kind": "tensor", "name": name, "dtype": dtype,
             "rank": rank, "access": access}
    if representation:
        value["representation"] = representation
    return value


def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}


def params(codec, dtype):
    weight_dtype = "fp8_e4m3" if codec in ("fp8", "mxfp8") else "u8"
    weight_repr = {
        "fp8": "fp8_e4m3_blockwise_weight.v1",
        "mxfp8": "mxfp8_e4m3_group32.v1",
        "mxfp4": "mxfp4_e2m1_group32.v1",
        "nvfp4": "nvfp4_e2m1_group16.v1",
    }[codec]
    scale_dtype = {
        "fp8": "f32", "mxfp8": "u8", "mxfp4": "u8",
        "nvfp4": "fp8_e4m3",
    }[codec]
    scale_repr = {
        "fp8": "fp8_blockwise_f32.v1",
        "mxfp8": "mxfp8_e8m0_group32.v1",
        "mxfp4": "mxfp4_e8m0_group32.v1",
        "nvfp4": "nvfp4_e4m3_group16.v1",
    }[codec]
    result = [
        tensor("input", dtype, 3, "read"),
        tensor("weight", weight_dtype, 3, "read", weight_repr),
        tensor("scales", scale_dtype, 3, "read", scale_repr),
    ]
    if codec == "nvfp4":
        result.append(tensor("global_scale", "f32", 2, "read",
                             "nvfp4_f32_row_global.v1"))
    result += [
        tensor("expert_ids", "u32", 2, "read"),
        tensor("output", dtype, 3, "write"),
    ]
    result += [scalar(name) for name in (
        "tokens", "topk", "n", "k", "experts", "input_routes"
    )]
    if codec == "fp8":
        result += [scalar(name) for name in (
            "block_y", "block_x", "scale_cols"
        )]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("kernel-packages/build/cuda"))
    p.add_argument("--ptx-file", type=Path)
    p.add_argument("--nvcc", default="nvcc")
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported CUDA compute capability")
    root = a.output / f"sm{a.cc}" / "routed-quant"
    original = root / "gemv" / "fp8" / "f16" / "routed_quant.ptx"
    original.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}",
        f"-code=compute_{a.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(original), str(SOURCE),
    ]
    version = (b"cudaforge:" + digest(a.ptx_file.read_bytes()).encode()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    fingerprint = digest(SOURCE.read_bytes() + version
                         + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def target(stage, codec, dtype):
        return root / stage / codec / dtype / original.name

    if (stamp.is_file() and stamp.read_text().strip() == fingerprint
            and all(target(stage, codec, dtype).is_file()
                    and target(stage, codec, dtype).with_name(
                        "manifest.json").is_file()
                    and json.loads(target(stage, codec, dtype).with_name(
                        "manifest.json").read_text())["sha256"]
                    == digest(target(stage, codec, dtype).read_bytes())
                    for stage in STAGES for codec in CODECS for dtype in DTYPES)):
        print(f"DKR_ROUTED_QUANT_REUSED=PASS cc={a.cc}")
        return

    if a.ptx_file:
        shutil.copyfile(a.ptx_file, original)
    else:
        subprocess.run(command, check=True)
    sha = digest(original.read_bytes())
    for stage in STAGES:
        for codec in CODECS:
            for dtype in DTYPES:
                file = target(stage, codec, dtype)
                file.parent.mkdir(parents=True, exist_ok=True)
                if file != original:
                    shutil.copyfile(original, file)
                geometry = [128, 1, 1] if stage == "gemv" else [16, 16, 1]
                manifest = {
                    "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
                    "operation": f"moe.routed.{codec}.{stage}.plain.v1",
                    "implementation_id":
                        f"cuda.routed_quant.{codec}.{stage}.{dtype}.v1",
                    "artifact": file.name,
                    "sha256": sha,
                    "entry": f"dkr_routed_{stage}_{codec}_{dtype}",
                    "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
                    "block": geometry, "shared_mem_bytes": 0,
                    "arguments": params(codec, dtype),
                }
                file.with_name("manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n"
                )
    stamp.write_text(fingerprint + "\n")
    print(
        f"DKR_ROUTED_QUANT_PACKAGES=STAGED profiles="
        f"{len(STAGES)*len(CODECS)*len(DTYPES)} "
        f"operations={len(STAGES)*len(CODECS)} cc={a.cc}"
    )


if __name__ == "__main__":
    main()
