#!/usr/bin/env python3
"""DKR-R1.21/S4: stage fused FP8-blockwise and MXFP4 prefill GEMM profiles."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("quant_gemm.cu")
DTYPES = ("f16", "bf16")
PROFILES = ([("fp8", dt, "plain") for dt in DTYPES]
            + [("mxfp4", dt, mode) for dt in DTYPES
               for mode in ("plain", "bias")])

def sha(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access, representation=None):
    result = {"kind": "tensor", "name": name, "dtype": dtype, "rank": rank,
              "access": access}
    if representation:
        result["representation"] = representation
    return result

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True, type=int)
    p.add_argument("--output", type=Path,
                   default=Path("kernel-packages/build/cuda"))
    p.add_argument("--ptx-file", type=Path)
    p.add_argument("--nvcc", default="nvcc")
    args = p.parse_args()
    if not 50 <= args.cc <= 120:
        p.error("unsupported target CC")
    root = args.output / f"sm{args.cc}" / "quant-gemm"
    base = root / "fp8" / "f16" / "plain" / "quant_gemm.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
               f"-code=compute_{args.cc}", "-std=c++17", "-O3",
               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
               "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
               "-o", str(base), str(SOURCE)]
    version = (b"cudaforge:" + sha(args.ptx_file.read_bytes()).encode()
               if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    fingerprint = sha(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def artifact(fmt, dt, mode):
        return root / fmt / dt / mode / base.name

    if stamp.is_file() and stamp.read_text().strip() == fingerprint and all(
        artifact(fmt, dt, mode).is_file()
        and artifact(fmt, dt, mode).with_name("manifest.json").is_file()
        and json.loads(artifact(fmt, dt, mode).with_name("manifest.json").read_text())["sha256"]
            == sha(artifact(fmt, dt, mode).read_bytes())
        for fmt, dt, mode in PROFILES
    ):
        print(f"DKR_QUANT_GEMM_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file:
        shutil.copyfile(args.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    digest = sha(base.read_bytes())
    for fmt, dt, mode in PROFILES:
        file = artifact(fmt, dt, mode)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != base:
            shutil.copyfile(base, file)
        params = [tensor("input", dt, 2, "read")]
        if fmt == "fp8":
            params += [
                tensor("weight", "fp8_e4m3", 2, "read",
                       "fp8_e4m3_blockwise_weight.v1"),
                tensor("scales", "f32", 2, "read",
                       "fp8_blockwise_f32.v1"),
            ]
            operation = "quant.fp8.blockwise.gemm.t16.v1"
            entry = f"dkr_fp8_block_gemm_{dt}"
        else:
            params += [
                tensor("weight", "u8", 2, "read",
                       "mxfp4_e2m1_group32.v1"),
                tensor("scales", "u8", 2, "read",
                       "mxfp4_e8m0_group32.v1"),
            ]
            if mode == "bias":
                params.append(tensor("bias", dt, 1, "read"))
            operation = f"quant.mxfp4.gemm.{mode}.t16.v1"
            entry = f"dkr_mxfp4_gemm_{dt}_{mode}"
        params += [tensor("output", dt, 2, "write")]
        params += [scalar(n) for n in ("m", "n", "k")]
        if fmt == "fp8":
            params += [scalar(n) for n in ("scale_stride", "block_y", "block_x")]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": operation,
            "implementation_id": f"cuda.quant_gemm.{fmt}.{dt}.{mode}.t16.v1",
            "artifact": file.name, "sha256": digest, "entry": entry,
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [16, 16, 1], "shared_mem_bytes": 0,
            "arguments": params,
        }
        file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(fingerprint + "\n")
    print(f"DKR_QUANT_GEMM_PACKAGES=STAGED profiles={len(PROFILES)} cc={args.cc}")

if __name__ == "__main__":
    main()
