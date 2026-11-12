#!/usr/bin/env python3
"""Stage DKR-R1.21/S5 scaled FP8 paged decode, 3 typed CUDA profiles."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("paged_fp8.cu")
DTYPES = ("f16", "bf16", "f32")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access, representation=None):
    value = {"kind": "tensor", "name": name, "dtype": dtype,
             "rank": rank, "access": access}
    if representation:
        value["representation"] = representation
    return value

def scalar(name, dtype="u32"):
    return {"kind": "scalar", "name": name, "dtype": dtype}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--output", type=Path,
                   default=Path("kernel-packages/build/cuda"))
    p.add_argument("--ptx-file", type=Path)
    p.add_argument("--nvcc", default="nvcc")
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target")
    root = a.output / f"sm{a.cc}" / "paged-fp8"
    base = root / "f16" / "paged_fp8.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}", f"-code=compute_{a.cc}",
        "-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-o", str(base), str(SOURCE),
    ]
    version = (b"cudaforge:" + digest(a.ptx_file.read_bytes()).encode()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    fingerprint = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"
    def artifact(dtype):
        return root / dtype / base.name
    if stamp.is_file() and stamp.read_text().strip() == fingerprint and all(
        artifact(dt).is_file()
        and artifact(dt).with_name("manifest.json").is_file()
        and json.loads(artifact(dt).with_name("manifest.json").read_text())["sha256"]
        == digest(artifact(dt).read_bytes())
        for dt in DTYPES
    ):
        print(f"DKR_PAGED_FP8_REUSED=PASS cc={a.cc}")
        return
    if a.ptx_file:
        shutil.copyfile(a.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    sha = digest(base.read_bytes())
    for dt in DTYPES:
        path = artifact(dt)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != base:
            shutil.copyfile(base, path)
        arguments = [
            tensor("query", dt, 3, "read"),
            tensor("key_cache", "fp8_e4m3", 5, "read", "vllm_k_blocked_x.v1"),
            tensor("value_cache", "fp8_e4m3", 4, "read", "vllm_v_token_strided.v1"),
            tensor("block_tables", "u32", 2, "read"),
            tensor("context_lens", "u32", 1, "read"),
            tensor("k_scale", "f32", 1, "read"),
            tensor("v_scale", "f32", 1, "read"),
            tensor("output", dt, 3, "write"),
        ] + [scalar(name) for name in (
            "seqs", "query_heads", "kv_heads", "head_size",
            "block_size", "x", "table_stride", "max_context",
        )] + [scalar("softmax_scale", "f32")]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": "attention.paged.fp8_decode.online.v1",
            "implementation_id": f"cuda.paged_fp8.{dt}.online128.v1",
            "artifact": path.name, "sha256": sha,
            "entry": f"dkr_paged_fp8_decode_{dt}",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [128, 1, 1], "shared_mem_bytes": 0,
            "arguments": arguments,
        }
        path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(fingerprint + "\n")
    print(f"DKR_PAGED_FP8_PACKAGES=STAGED profiles={len(DTYPES)} cc={a.cc}")

if __name__ == "__main__":
    main()
