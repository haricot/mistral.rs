#!/usr/bin/env python3
"""Build external MoE reductions and sink-aware softmax for one CUDA target."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("reduce_norm.cu")
TYPES = ("f16", "bf16", "f32")
SOFTMAX_BLOCKS = (64, 128, 256, 512, 1024)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor(name: str, dtype: str, rank: int, access: str) -> dict:
    return {"kind": "tensor", "name": name, "dtype": dtype, "rank": rank, "access": access}


def scalar(name: str, dtype: str) -> dict:
    return {"kind": "scalar", "name": name, "dtype": dtype}


def variants():
    for input_type in TYPES:
        for output_type in TYPES:
            yield (
                f"weighted/{input_type}/{output_type}",
                "moe.weighted_reduce.v1",
                f"dkr_weighted_{input_type}_{output_type}",
                256,
                [
                    tensor("input", input_type, 3, "read"),
                    tensor("weights", "f32", 2, "read"),
                    tensor("output", output_type, 2, "write"),
                    scalar("tokens", "u64"),
                    scalar("hidden", "u64"),
                    scalar("topk", "u64"),
                ],
            )
    for dtype in TYPES:
        for block in SOFTMAX_BLOCKS:
            yield (
                f"softmax/{dtype}/b{block}",
                f"attention.softmax_sinks.b{block}.v1",
                f"dkr_softmax_sinks_{dtype}",
                block,
                [
                    tensor("logits", dtype, 4, "read"),
                    tensor("sinks", dtype, 1, "read"),
                    tensor("output", dtype, 4, "write"),
                    scalar("num_heads", "u32"),
                    scalar("q_len", "u32"),
                    scalar("k_len", "u32"),
                ],
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")

    root = args.output / f"sm{args.cc}" / "reduce-norm"
    artifact = root / "weighted" / "f16" / "f16" / "reduce_norm.ptx"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    stamp = root / "source.sha256"
    command = [
        args.nvcc,
        "-ptx",
        f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}",
        "-std=c++17",
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--use_fast_math",
        "-o",
        str(artifact),
        str(SOURCE),
    ]
    nvcc_version = subprocess.check_output([args.nvcc, "--version"])
    identity = digest(SOURCE.read_bytes() + nvcc_version + " ".join(command).encode())
    profiles = list(variants())
    if stamp.is_file() and stamp.read_text().strip() == identity:
        if all(
            (folder / "manifest.json").is_file()
            and (folder / artifact.name).is_file()
            and json.loads((folder / "manifest.json").read_text())["sha256"]
            == digest((folder / artifact.name).read_bytes())
            for name, *_ in profiles
            for folder in [root / name]
        ):
            print(f"DKR_REDUCE_NORM_REUSED=PASS target_cc={args.cc}")
            return

    subprocess.run(command, check=True)
    for name, operation, entry, block, signature in profiles:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / artifact.name
        if output != artifact:
            shutil.copyfile(artifact, output)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": operation,
            "implementation_id": f"cuda.reduce_norm.{name.replace('/', '.')}.v1",
            "artifact": output.name,
            "sha256": digest(output.read_bytes()),
            "entry": entry,
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [block, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": signature,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"DKR_REDUCE_NORM_PACKAGE=PASS profile={name} target_cc={args.cc}")
    stamp.write_text(identity + "\n")


if __name__ == "__main__":
    main()
