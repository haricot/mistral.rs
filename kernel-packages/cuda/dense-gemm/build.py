#!/usr/bin/env python3
"""Manifest-only packager for DKR dense GEMM, separate plain/bias ABIs."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("dense_gemm.cu")
DTYPES = ("f16", "bf16", "f32")
MODES = ("plain", "bias")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access):
    return {"kind": "tensor", "name": name, "dtype": dtype,
            "rank": rank, "access": access}

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True, type=int)
    p.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--ptx-file", type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target")
    root = a.output / f"sm{a.cc}" / "dense-gemm"
    first = root / "f16" / "plain" / "dense_gemm.ptx"
    first.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}", f"-code=compute_{a.cc}",
        "-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(first), str(SOURCE)
    ]
    version = (b"cudaforge:" + hashlib.sha256(a.ptx_file.read_bytes()).digest()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    profiles = [(dtype, mode) for dtype in DTYPES for mode in MODES]
    stamp = root / "source.sha256"
    def target(dtype, mode):
        return root / dtype / mode / first.name
    if stamp.exists() and stamp.read_text().strip() == identity and all(
        target(dtype, mode).is_file()
        and target(dtype, mode).with_name("manifest.json").is_file()
        and json.loads(target(dtype, mode).with_name("manifest.json").read_text())["sha256"]
            == digest(target(dtype, mode).read_bytes())
        for dtype, mode in profiles
    ):
        print(f"DKR_DENSE_GEMM_REUSED=PASS cc={a.cc}")
        return

    if a.ptx_file:
        shutil.copyfile(a.ptx_file, first)
    else:
        subprocess.run(command, check=True)
    sha = digest(first.read_bytes())
    for dtype, mode in profiles:
        path = target(dtype, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != first:
            shutil.copyfile(first, path)
        args = [
            tensor("input", dtype, 2, "read"),
            tensor("weight", dtype, 2, "read"),
        ]
        if mode == "bias":
            args.append(tensor("bias", dtype, 1, "read"))
        args.extend([
            tensor("output", dtype, 2, "write"),
            scalar("m"), scalar("n"), scalar("k")
        ])
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"linear.gemm.{mode}.t16.v1",
            "implementation_id": f"cuda.dense_gemm.{dtype}.{mode}.t16.v1",
            "artifact": path.name, "sha256": sha,
            "entry": f"dkr_gemm_{dtype}_{mode}",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [16, 16, 1], "shared_mem_bytes": 0,
            "arguments": args
        }
        path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_DENSE_GEMM_PACKAGES=STAGED profiles={len(profiles)} cc={a.cc}")

if __name__ == "__main__":
    main()
