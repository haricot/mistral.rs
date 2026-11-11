#!/usr/bin/env python3
"""Build the DKR external U32 decode input padding package."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("input_pad.cu")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path, help="PTX already built by cudaforge")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported compute capability")
    directory = args.output / f"sm{args.cc}" / "input-pad"
    artifact = directory / "input_pad.ptx"
    manifest_file = directory / "manifest.json"
    stamp = directory / "source.sha256"
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-o", str(artifact), str(SOURCE),
    ]
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    if (stamp.is_file() and stamp.read_text().strip() == identity
            and artifact.is_file() and manifest_file.is_file()
            and json.loads(manifest_file.read_text())["sha256"] == digest(artifact.read_bytes())):
        print(f"DKR_INPUT_PAD_REUSED=PASS target_cc={args.cc}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(command, check=True)
    tensor = lambda name, rank, access: {
        "kind": "tensor", "name": name, "dtype": "u32", "rank": rank, "access": access,
    }
    scalar = lambda name: {"kind": "scalar", "name": name, "dtype": "i32"}
    manifest = {
        "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
        "operation": "layout.pad_decode_u32.v1",
        "implementation_id": "cuda.input_pad.u32.v1",
        "artifact": artifact.name, "sha256": digest(artifact.read_bytes()),
        "entry": "dkr_pad_decode_input_u32",
        "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
        "block": [256, 1, 1], "shared_mem_bytes": 0,
        "arguments": [
            tensor("input", 2, "read"), tensor("output", 2, "write"),
            scalar("input_rows"), scalar("output_rows"), scalar("width"),
        ],
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_INPUT_PAD_PACKAGE=PASS target_cc={args.cc}")


if __name__ == "__main__":
    main()
