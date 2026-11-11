#!/usr/bin/env python3
"""Build one external in-place F32 causal masking kernel per architecture."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("causal_mask.cu")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path, help="PTX already built by cudaforge")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")

    root = args.output / f"sm{args.cc}" / "causal-mask"
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "causal_mask.ptx"
    manifest_path = root / "manifest.json"
    stamp = root / "source.sha256"
    command = [
        args.nvcc,
        "-ptx",
        f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}",
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "-o",
        str(artifact),
        str(SOURCE),
    ]
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + " ".join(command).encode())
    if stamp.is_file() and stamp.read_text().strip() == identity:
        if artifact.is_file() and manifest_path.is_file():
            if json.loads(manifest_path.read_text())["sha256"] == digest(artifact.read_bytes()):
                print(f"DKR_CAUSAL_MASK_REUSED=PASS target_cc={args.cc}")
                return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(command, check=True)
    manifest = {
        "schema": "DKR-KERNEL-MODULE-V1",
        "abi": "dkr.typed.v1",
        "operation": "attention.causal_mask_f32.inplace.v1",
        "implementation_id": "cuda.causal_mask.f32.inplace.v1",
        "artifact": artifact.name,
        "sha256": digest(artifact.read_bytes()),
        "entry": "dkr_causal_mask_f32",
        "target_cc": args.cc,
        "min_cc": args.cc,
        "max_cc": 120,
        "block": [256, 1, 1],
        "shared_mem_bytes": 0,
        "arguments": [
            {"kind": "tensor", "name": "scores", "dtype": "f32", "rank": 3, "access": "read_write"},
            {"kind": "scalar", "name": "batch_heads", "dtype": "i32"},
            {"kind": "scalar", "name": "q_len", "dtype": "i32"},
            {"kind": "scalar", "name": "kv_len", "dtype": "i32"},
            {"kind": "scalar", "name": "q_offset", "dtype": "i32"},
            {"kind": "scalar", "name": "prefix_len", "dtype": "i32"},
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_CAUSAL_MASK_PACKAGE=PASS target_cc={args.cc}")


if __name__ == "__main__":
    main()
