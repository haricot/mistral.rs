#!/usr/bin/env python3
"""Package typed VLLM-layout paged-KV gather kernels built via cudaforge."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("kv_gather.cu")
DTYPES = ("f16", "bf16", "f32")

def digest(contents):
    return hashlib.sha256(contents).hexdigest()

def tensor(name, dtype, rank, access, representation=None):
    value = {"kind": "tensor", "name": name, "dtype": dtype,
             "rank": rank, "access": access}
    if representation:
        value["representation"] = representation
    return value

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--ptx-file", type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target")
    root = a.output / f"sm{a.cc}" / "kv-gather"
    base = root / "f16" / "kv_gather.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}", f"-code=compute_{a.cc}",
        "-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-o", str(base), str(SOURCE)
    ]
    version = (b"cudaforge:" + hashlib.sha256(a.ptx_file.read_bytes()).digest()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"
    def target(dtype):
        return root / dtype / base.name

    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        target(t).is_file() and target(t).with_name("manifest.json").is_file()
        and json.loads(target(t).with_name("manifest.json").read_text())["sha256"]
        == digest(target(t).read_bytes())
        for t in DTYPES
    ):
        print(f"DKR_KV_GATHER_REUSED=PASS cc={a.cc}")
        return
    if a.ptx_file:
        shutil.copyfile(a.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    sha = digest(base.read_bytes())
    for dtype in DTYPES:
        file = target(dtype)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != base:
            shutil.copyfile(base, file)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": "kv_cache.gather.vllm.contiguous.v1",
            "implementation_id": f"cuda.kv_gather.{dtype}.v1",
            "artifact": file.name, "sha256": sha,
            "entry": f"dkr_kv_gather_{dtype}",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [256, 1, 1], "shared_mem_bytes": 0,
            "arguments": [
                tensor("keys", dtype, 5, "read", "vllm_k_blocked_x.v1"),
                tensor("values", dtype, 4, "read", "vllm_v_token_strided.v1"),
                tensor("block_table", "i32", 2, "read"),
                tensor("cu_seq_lens", "i32", 1, "read"),
                tensor("key_output", dtype, 3, "write"),
                tensor("value_output", dtype, 3, "write"),
                *[scalar(n) for n in (
                    "tokens", "seqs", "block_size", "block_table_stride",
                    "heads", "head_size", "x"
                )],
            ]
        }
        file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_KV_GATHER_PACKAGES=STAGED profiles={len(DTYPES)} cc={a.cc}")

if __name__ == "__main__":
    main()
