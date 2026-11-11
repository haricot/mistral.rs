#!/usr/bin/env python3
"""Build one CUDA PTX module and typed indexed-copy package variants."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("indexed_copy.cu")
OPERATION = "tensor.indexed_row_copy.v1"
VARIANTS = (
    ("f16", "dkr_indexed_copy_u16"),
    ("bf16", "dkr_indexed_copy_u16"),
    ("f32", "dkr_indexed_copy_u32"),
)


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

    root = args.output / f"sm{args.cc}" / "indexed-copy"
    root.mkdir(parents=True, exist_ok=True)
    artifact = root / "f16" / "indexed_copy.ptx"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    stamp = root / "source.sha256"
    command = [
        args.nvcc,
        "-ptx",
        f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}",
        "-std=c++17",
        "-O3",
        "-o",
        str(artifact),
        str(SOURCE),
    ]
    nvcc_version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + nvcc_version + " ".join(command).encode())
    variants_present = all(
        (root / dtype / "indexed_copy.ptx").is_file()
        and (root / dtype / "manifest.json").is_file()
        for dtype, _ in VARIANTS
    )
    if stamp.is_file() and stamp.read_text().strip() == identity and variants_present:
        if all(
            json.loads((root / dtype / "manifest.json").read_text())["sha256"]
            == digest((root / dtype / "indexed_copy.ptx").read_bytes())
            for dtype, _ in VARIANTS
        ):
            print(f"DKR_INDEXED_COPY_REUSED=PASS target_cc={args.cc}")
            return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(command, check=True)
    for dtype, entry in VARIANTS:
        folder = root / dtype
        folder.mkdir(parents=True, exist_ok=True)
        output_artifact = folder / artifact.name
        if output_artifact != artifact:
            shutil.copyfile(artifact, output_artifact)
        payload = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": OPERATION,
            "implementation_id": f"cuda.indexed_copy.{dtype}.v1",
            "artifact": output_artifact.name,
            "sha256": digest(output_artifact.read_bytes()),
            "entry": entry,
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [256, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": [
                {"kind": "tensor", "name": "src", "dtype": dtype, "rank": 2, "access": "read"},
                {"kind": "tensor", "name": "dst", "dtype": dtype, "rank": 2, "access": "write"},
                {"kind": "tensor", "name": "indices", "dtype": "u32", "rank": 1, "access": "read"},
                {"kind": "scalar", "name": "source_rows", "dtype": "u32"},
                {"kind": "scalar", "name": "destination_rows", "dtype": "u32"},
                {"kind": "scalar", "name": "row_elements", "dtype": "u64"},
            ],
        }
        (folder / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"DKR_INDEXED_COPY_PACKAGE=PASS dtype={dtype} target_cc={args.cc}")
    stamp.write_text(identity + "\n")


if __name__ == "__main__":
    main()
