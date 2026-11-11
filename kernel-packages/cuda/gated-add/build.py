#!/usr/bin/env python3
"""Build one external CUDA module for sigmoid-gated residual addition."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("gated_add.cu")
VARIANTS = (
    ("f16", "flat", "dkr_gated_add_f16"),
    ("bf16", "flat", "dkr_gated_add_bf16"),
    ("f32", "flat", "dkr_gated_add_f32_flat"),
    ("f32", "rows", "dkr_gated_add_f32_rows"),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")

    root = args.output / f"sm{args.cc}" / "gated-add"
    artifact = root / "f16" / "flat" / "gated_add.ptx"
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
    paths = [root / dtype / mode for dtype, mode, _ in VARIANTS]
    if stamp.is_file() and stamp.read_text().strip() == identity:
        if all(
            (path / "manifest.json").is_file()
            and (path / artifact.name).is_file()
            and json.loads((path / "manifest.json").read_text())["sha256"]
            == digest((path / artifact.name).read_bytes())
            for path in paths
        ):
            print(f"DKR_GATED_ADD_REUSED=PASS target_cc={args.cc}")
            return

    subprocess.run(command, check=True)
    for dtype, mode, entry in VARIANTS:
        folder = root / dtype / mode
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / artifact.name
        if output != artifact:
            shutil.copyfile(artifact, output)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": f"tensor.sigmoid_gated_add.{mode}.v1",
            "implementation_id": f"cuda.gated_add.{dtype}.{mode}.v1",
            "artifact": output.name,
            "sha256": digest(output.read_bytes()),
            "entry": entry,
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [256, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": [
                {"kind": "tensor", "name": "residual", "dtype": dtype, "rank": 2, "access": "read"},
                {"kind": "tensor", "name": "value", "dtype": dtype, "rank": 2, "access": "read"},
                {"kind": "tensor", "name": "gate", "dtype": dtype, "rank": 1, "access": "read"},
                {"kind": "tensor", "name": "output", "dtype": dtype, "rank": 2, "access": "write"},
                {"kind": "scalar", "name": "elements", "dtype": "u64"},
                {"kind": "scalar", "name": "columns", "dtype": "u64"},
            ],
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"DKR_GATED_ADD_PACKAGE=PASS dtype={dtype} mode={mode} target_cc={args.cc}")
    stamp.write_text(identity + "\n")


if __name__ == "__main__":
    main()
