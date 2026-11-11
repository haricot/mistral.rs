#!/usr/bin/env python3
"""Build or package contiguous, rounded RMSNorm+RoPE external PTX."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("rounded_rope.cu")
DTYPES = ("f16", "bf16", "f32")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor(name, dtype, rank, access):
    return {"kind": "tensor", "name": name, "dtype": dtype,
            "rank": rank, "access": access}


def scalar(name, dtype):
    return {"kind": "scalar", "name": name, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path, help="PTX already built by cudaforge")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported compute capability")
    root = args.output / f"sm{args.cc}" / "rounded-rope"
    target = root / "f16" / "b32" / "rounded_rope.ptx"
    stamp = root / "source.sha256"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--use_fast_math", "-o", str(target), str(SOURCE),
    ]
    version = (
        b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest()
        if args.ptx_file else subprocess.check_output([args.nvcc, "--version"])
    )
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    profiles = [(dtype, block) for dtype in DTYPES for block in (32, 1024)]

    def folder(dtype, block):
        return root / dtype / f"b{block}"

    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        (folder(dtype, block) / "manifest.json").is_file()
        and (folder(dtype, block) / target.name).is_file()
        and json.loads((folder(dtype, block) / "manifest.json").read_text())["sha256"]
        == digest((folder(dtype, block) / target.name).read_bytes())
        for dtype, block in profiles
    ):
        print(f"DKR_ROUNDED_ROPE_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, target)
    else:
        subprocess.run(command, check=True)
    sha = digest(target.read_bytes())
    for dtype, block in profiles:
        directory = folder(dtype, block)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / target.name
        if output != target:
            shutil.copyfile(target, output)
        argspec = [
            tensor("input", dtype, 4, "read"),
            tensor("weight", dtype, 1, "read"),
            tensor("cos", dtype, 2, "read"),
            tensor("sin", dtype, 2, "read"),
            tensor("output", dtype, 4, "write"),
            scalar("batch", "i32"),
            scalar("sequence", "i32"),
            scalar("heads", "i32"),
            scalar("columns", "i32"),
            scalar("half_rot", "i32"),
            scalar("epsilon", "f32"),
        ]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"rotary.rounded_rms_rope.b{block}.v1",
            "implementation_id": f"cuda.rounded_rms_rope.{dtype}.b{block}.v1",
            "artifact": output.name,
            "sha256": sha,
            "entry": f"dkr_rounded_norm_rope_{dtype}",
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [block, 1, 1], "shared_mem_bytes": 0,
            "arguments": argspec,
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_ROUNDED_ROPE_PACKAGES=PASS cc={args.cc} profiles={len(profiles)}")


if __name__ == "__main__":
    main()
