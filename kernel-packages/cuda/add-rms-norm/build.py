#!/usr/bin/env python3
"""Package external fused add+RMSNorm CUDA kernels for one architecture.

Keep the scalar/vec8 launch widths of the original CUDA wrappers. All
variants share one NVCC compilation and its content-addressed stamp.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


SOURCE = Path(__file__).with_name("add_rms_norm.cu")
DTYPES = ("f16", "bf16", "f32")
VECTOR_BLOCKS = tuple(range(32, 1025, 32))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def variants():
    for dtype in DTYPES:
        for block in (32, 1024):
            yield dtype, "scalar", block
        if dtype != "f32":
            for block in VECTOR_BLOCKS:
                yield dtype, "vec8", block


def tensor(name: str, dtype: str, rank: int, access: str) -> dict:
    return {"kind": "tensor", "name": name, "dtype": dtype, "rank": rank, "access": access}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path, help="PTX already built by cudaforge")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")

    profiles = list(variants())
    root = args.output / f"sm{args.cc}" / "add-rms-norm"
    artifact = root / "f16" / "scalar" / "b32" / "add_rms_norm.ptx"
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
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + " ".join(command).encode())

    def folder(dtype: str, mode: str, block: int) -> Path:
        return root / dtype / mode / f"b{block}"

    if stamp.is_file() and stamp.read_text().strip() == identity:
        if all(
            (folder(dtype, mode, block) / "manifest.json").is_file()
            and (folder(dtype, mode, block) / artifact.name).is_file()
            and json.loads((folder(dtype, mode, block) / "manifest.json").read_text())[
                "sha256"
            ] == digest((folder(dtype, mode, block) / artifact.name).read_bytes())
            for dtype, mode, block in profiles
        ):
            print(f"DKR_ADD_RMS_NORM_REUSED=PASS target_cc={args.cc}")
            return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(command, check=True)
    ptx_bytes = artifact.read_bytes()
    ptx_hash = digest(ptx_bytes)
    for dtype, mode, block in profiles:
        directory = folder(dtype, mode, block)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / artifact.name
        if output != artifact:
            shutil.copyfile(artifact, output)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": f"tensor.add_rms_norm.{mode}.b{block}.v1",
            "implementation_id": f"cuda.add_rms_norm.{dtype}.{mode}.b{block}.v1",
            "artifact": output.name,
            "sha256": ptx_hash,
            "entry": f"dkr_add_rms_norm_{dtype}_{mode}",
            "target_cc": args.cc,
            "min_cc": args.cc,
            "max_cc": 120,
            "block": [block, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": [
                tensor("input", dtype, 2, "read"),
                tensor("residual", dtype, 2, "read"),
                tensor("weight", dtype, 1, "read"),
                tensor("residual_output", dtype, 2, "write"),
                tensor("normalized_output", dtype, 2, "write"),
                {"kind": "scalar", "name": "columns", "dtype": "u32"},
                {"kind": "scalar", "name": "eps", "dtype": "f32"},
            ],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_ADD_RMS_NORM_PACKAGES=PASS profiles={len(profiles)} target_cc={args.cc}")


if __name__ == "__main__":
    main()
