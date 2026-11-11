#!/usr/bin/env python3
"""Compile standalone MoE PTX for exactly one CUDA compute capability."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("moe.cu")
ENTRY = "dkr_moe_grouped_gemm"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-dir", type=Path, help="directory with cudaforge f16/bf16 PTX")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")

    nvcc_version = (b"cudaforge:" if args.ptx_dir else subprocess.check_output([args.nvcc, "--version"]))
    for variant, element_type, mode in (("f16", "f16", "0"), ("bf16", "bf16", "1")):
        root = args.output / f"sm{args.cc}" / "moe" / variant
        root.mkdir(parents=True, exist_ok=True)
        artifact = root / "moe.ptx"
        manifest = root / "manifest.json"
        stamp = root / "source.sha256"
        command = [
            args.nvcc,
            "-ptx",
            f"-arch=compute_{args.cc}",
            f"-code=compute_{args.cc}",
            "-std=c++17",
            "-O3",
            f"-DDKR_BF16_MODE={mode}" if mode == "1" else "-UDKR_BF16_MODE",
            "-o",
            str(artifact),
            str(SOURCE),
        ]
        prebuilt = args.ptx_dir / variant / "moe.ptx" if args.ptx_dir else None
        identity = digest(SOURCE.read_bytes() + nvcc_version + " ".join(command).encode()
                          + (bytes.fromhex(digest(prebuilt.read_bytes())) if prebuilt else b""))
        if manifest.is_file() and artifact.is_file() and stamp.is_file() and stamp.read_text().strip() == identity:
            existing = json.loads(manifest.read_text())
            if existing["sha256"] == digest(artifact.read_bytes()):
                print(f"DKR_MOE_REUSED={variant} target_cc={args.cc}")
                continue

        if prebuilt is not None:
            shutil.copyfile(prebuilt, artifact)
        else:
            subprocess.run(command, check=True)
        payload = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": "moe.grouped_gemm.v1",
            "implementation_id": f"cuda.moe.reference.{variant}.v1",
            "artifact": artifact.name,
            "sha256": digest(artifact.read_bytes()),
            "entry": ENTRY,
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [128, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": [
                {"kind": "tensor", "name": "input", "dtype": element_type, "rank": 2, "access": "read"},
                {"kind": "tensor", "name": "weights", "dtype": element_type, "rank": 3, "access": "read"},
                {"kind": "tensor", "name": "sorted", "dtype": "u32", "rank": 1, "access": "read"},
                {"kind": "tensor", "name": "experts", "dtype": "u32", "rank": 1, "access": "read"},
                {"kind": "tensor", "name": "gating", "dtype": "f32", "rank": 1, "access": "read"},
                {"kind": "tensor", "name": "output", "dtype": element_type, "rank": 2, "access": "write"},
                *({"kind": "scalar", "name": name, "dtype": "u32"} for name in (
                    "num_experts", "topk", "size_m", "size_n", "size_k", "has_gating"
                )),
            ],
        }
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
        stamp.write_text(identity + "\n")
        print(f"DKR_MOE_PACKAGE=PASS variant={variant} target_cc={args.cc} artifact_sha256={payload['sha256']}")


if __name__ == "__main__":
    main()
