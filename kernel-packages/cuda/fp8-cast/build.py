#!/usr/bin/env python3
"""Package scalar E4M3 <-> F16/BF16/F32 PTX with fixed typed ABI."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("fp8_cast.cu")
DTYPES = ("f16", "bf16", "f32")
BLOCK = 256

def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, access):
    return {"kind": "tensor", "name": name, "dtype": dtype,
            "rank": 1, "access": access}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True, type=int)
    p.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--ptx-file", type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target architecture")
    root = a.output / f"sm{a.cc}" / "fp8-cast"
    first = root / "encode" / "f16" / "fp8_cast.ptx"
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
    fingerprint = hash_bytes(SOURCE.read_bytes() + version + "\0".join(command).encode())
    profiles = [(direction, dtype) for direction in ("encode", "decode") for dtype in DTYPES]
    stamp = root / "source.sha256"
    def artifact(direction, dtype):
        return root / direction / dtype / first.name
    if stamp.exists() and stamp.read_text().strip() == fingerprint and all(
        artifact(d, t).is_file()
        and artifact(d, t).with_name("manifest.json").is_file()
        and json.loads(artifact(d, t).with_name("manifest.json").read_text())["sha256"]
        == hash_bytes(artifact(d, t).read_bytes())
        for d, t in profiles
    ):
        print(f"DKR_FP8_CAST_REUSED=PASS cc={a.cc}")
        return
    if a.ptx_file:
        shutil.copyfile(a.ptx_file, first)
    else:
        subprocess.run(command, check=True)
    sha = hash_bytes(first.read_bytes())
    for direction, dtype in profiles:
        file = artifact(direction, dtype)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != first:
            shutil.copyfile(first, file)
        src, dst = ((dtype, "fp8_e4m3") if direction == "encode"
                    else ("fp8_e4m3", dtype))
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"quant.fp8_e4m3.{direction}.v1",
            "implementation_id": f"cuda.fp8_cast.{direction}.{dtype}.v1",
            "artifact": file.name, "sha256": sha,
            "entry": f"dkr_fp8_{direction}_{dtype}",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [BLOCK, 1, 1], "shared_mem_bytes": 0,
            "arguments": [
                tensor("input", src, "read"),
                tensor("output", dst, "write"),
                {"kind": "scalar", "name": "count", "dtype": "u64"},
            ]
        }
        file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(fingerprint + "\n")
    print(f"DKR_FP8_CAST_PACKAGES=STAGED profiles={len(profiles)} cc={a.cc}")

if __name__ == "__main__":
    main()
