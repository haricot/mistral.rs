#!/usr/bin/env python3
"""DKR-R1.21/S7: package exact-copy and VLLM slot-write CUDA kernels.

NVCC compilation belongs to the Rust cudaforge package builder. This
script stamps a given, already-built PTX with typed ABI manifests.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("kv_cache_ops.cu")
PROFILES = (
    [("copy", "bytes")] +
    [("reshape", dtype) for dtype in ("f16", "bf16", "f32")]
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor(name, dtype, rank, access, representation=None):
    arg = {"kind": "tensor", "name": name, "dtype": dtype,
           "rank": rank, "access": access}
    if representation is not None:
        arg["representation"] = representation
    return arg


def scalar(name, dtype="u32"):
    return {"kind": "scalar", "name": name, "dtype": dtype}


def parameters(op, dtype):
    if op == "copy":
        # Each pointer table contains a device address per layer.
        # The pointed-to K/V caches are modified indirectly in place.
        return [
            tensor("key_layer_ptrs", "i64", 1, "read",
                   "kv_cache.layer_ptr_table.i64.v1"),
            tensor("value_layer_ptrs", "i64", 1, "read",
                   "kv_cache.layer_ptr_table.i64.v1"),
            tensor("block_pairs", "i64", 2, "read",
                   "kv_cache.block_pairs.i64.v1"),
            scalar("layers"), scalar("pairs"),
            scalar("key_block_bytes", "u64"),
            scalar("value_block_bytes", "u64"),
        ]
    return [
        tensor("key", dtype, 3, "read"),
        tensor("value", dtype, 3, "read"),
        tensor("key_cache", dtype, 5, "read_write",
               "vllm_k_blocked_x.v1"),
        tensor("value_cache", dtype, 4, "read_write",
               "vllm_v_token_strided.v1"),
        tensor("slot_mapping", "i64", 1, "read"),
        *[scalar(name) for name in (
            "tokens", "heads", "head_size", "num_blocks", "block_size", "x"
        )],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True, type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--ptx-file", type=Path)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported compute capability")
    root = args.output / f"sm{args.cc}" / "kv-cache-ops"
    base = root / "copy" / "bytes" / "kv_cache_ops.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(base), str(SOURCE),
    ]
    version = (
        b"cudaforge:" + digest(args.ptx_file.read_bytes()).encode()
        if args.ptx_file else subprocess.check_output([args.nvcc, "--version"])
    )
    fingerprint = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def artifact(op, dt):
        return root / op / dt / base.name

    if (stamp.is_file() and stamp.read_text().strip() == fingerprint
            and all(artifact(op, dt).is_file()
                    and artifact(op, dt).with_name("manifest.json").is_file()
                    and json.loads(artifact(op, dt)
                                   .with_name("manifest.json").read_text())["sha256"]
                    == digest(artifact(op, dt).read_bytes())
                    for op, dt in PROFILES)):
        print(f"DKR_KV_CACHE_OPS_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    ptx_hash = digest(base.read_bytes())
    for op, dt in PROFILES:
        file = artifact(op, dt)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != base:
            shutil.copyfile(base, file)
        if op == "copy":
            operation = "kv_cache.copy_blocks.bytes.b256.v1"
            symbol = "dkr_kv_copy_blocks_bytes"
        else:
            operation = "kv_cache.reshape_store.vllm.b256.v1"
            symbol = f"dkr_kv_reshape_store_{dt}"
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": operation,
            "implementation_id": f"cuda.kv_cache_ops.{op}.{dt}.b256.v1",
            "artifact": file.name,
            "sha256": ptx_hash,
            "entry": symbol,
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [256, 1, 1], "shared_mem_bytes": 0,
            "arguments": parameters(op, dt),
        }
        file.with_name("manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    stamp.write_text(fingerprint + "\n")
    print(
        f"DKR_KV_CACHE_OPS_PACKAGES=STAGED profiles={len(PROFILES)} "
        f"operations=2 cc={args.cc}"
    )


if __name__ == "__main__":
    main()
