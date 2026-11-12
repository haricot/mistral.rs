#!/usr/bin/env python3
"""Package DKR R1.21 FP8-blockwise and MXFP4 fused decode GEMV PTX.

The CUDA PTX is produced by the Rust cudaforge builder when --ptx-file
is provided. Python owns only manifests and integrity fingerprints.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("quant_gemv.cu")
BLOCK = 128
DTYPES = ("f16", "bf16")
PROFILES = (
    [("fp8", dtype, "plain") for dtype in DTYPES]
    + [("mxfp4", dtype, mode) for dtype in DTYPES for mode in ("plain", "bias")]
)

def checksum(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access, representation=None):
    result = {"kind": "tensor", "name": name, "dtype": dtype,
              "rank": rank, "access": access}
    if representation is not None:
        result["representation"] = representation
    return result

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def entry(fmt, dtype, mode):
    return (f"dkr_fp8_block_gemv_{dtype}" if fmt == "fp8"
            else f"dkr_mxfp4_gemv_{dtype}_{mode}")

def operation(fmt, mode):
    return ("quant.fp8.blockwise.gemv.b128.v1" if fmt == "fp8"
            else f"quant.mxfp4.gemv.{mode}.b128.v1")

def parameters(fmt, dtype, mode):
    args = [tensor("input", dtype, 2, "read")]
    if fmt == "fp8":
        args.extend([
            tensor("weight", "fp8_e4m3", 2, "read",
                   "fp8_e4m3_blockwise_weight.v1"),
            tensor("scales", "f32", 2, "read",
                   "fp8_blockwise_f32.v1"),
        ])
    else:
        args.extend([
            tensor("weight", "u8", 2, "read",
                   "mxfp4_e2m1_group32.v1"),
            tensor("scales", "u8", 2, "read",
                   "mxfp4_e8m0_group32.v1"),
        ])
        if mode == "bias":
            args.append(tensor("bias", dtype, 1, "read"))
    args.append(tensor("output", dtype, 2, "write"))
    args.extend(scalar(name) for name in ("m", "n", "k"))
    if fmt == "fp8":
        args.extend(scalar(name) for name in (
            "scale_stride", "block_y", "block_x"))
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True, type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--ptx-file", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported target")
    root = args.output / f"sm{args.cc}" / "quant-gemv"
    base = root / "fp8" / "f16" / "plain" / "quant_gemv.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(base), str(SOURCE),
    ]
    version = (b"cudaforge:" + checksum(args.ptx_file.read_bytes()).encode()
               if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = checksum(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def artifact(fmt, dtype, mode):
        return root / fmt / dtype / mode / base.name

    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        artifact(fmt, dtype, mode).is_file()
        and artifact(fmt, dtype, mode).with_name("manifest.json").is_file()
        and json.loads(artifact(fmt, dtype, mode)
                       .with_name("manifest.json").read_text())["sha256"]
            == checksum(artifact(fmt, dtype, mode).read_bytes())
        for fmt, dtype, mode in PROFILES
    ):
        print(f"DKR_QUANT_GEMV_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file:
        shutil.copyfile(args.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    sha = checksum(base.read_bytes())
    for fmt, dtype, mode in PROFILES:
        path = artifact(fmt, dtype, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != base:
            shutil.copyfile(base, path)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": operation(fmt, mode),
            "implementation_id": f"cuda.quant_gemv.{fmt}.{dtype}.{mode}.b128.v1",
            "artifact": path.name,
            "sha256": sha,
            "entry": entry(fmt, dtype, mode),
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [BLOCK, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": parameters(fmt, dtype, mode),
        }
        path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_QUANT_GEMV_PACKAGES=STAGED profiles={len(PROFILES)} cc={args.cc}")

if __name__ == "__main__":
    main()
