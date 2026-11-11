#!/usr/bin/env python3
"""Emit typed DKR dense-GEMV manifests for prebuilt cudaforge PTX.

No device kernel is compiled from Python when --ptx-file is supplied.
Two fixed ABIs (plain/bias) avoid a nullable pointer on the launch path.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("dense_gemv.cu")
DTYPES = ("f16", "bf16", "f32")
MODES = ("plain", "bias")
BLOCK = 128

def digest(contents):
    return hashlib.sha256(contents).hexdigest()

def tensor(name, dtype, rank, access):
    return {"kind": "tensor", "name": name, "dtype": dtype,
            "rank": rank, "access": access}

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True, type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path)
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported target architecture")
    root = args.output / f"sm{args.cc}" / "dense-gemv"
    source_artifact = root / "f16" / "plain" / "dense_gemv.ptx"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(source_artifact), str(SOURCE)
    ]
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest()
               if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def artifact(dtype, mode):
        return root / dtype / mode / source_artifact.name

    profiles = [(dtype, mode) for dtype in DTYPES for mode in MODES]
    if stamp.exists() and stamp.read_text().strip() == identity and all(
        artifact(dtype, mode).is_file()
        and artifact(dtype, mode).with_name("manifest.json").is_file()
        and json.loads(artifact(dtype, mode).with_name("manifest.json").read_text())["sha256"]
        == digest(artifact(dtype, mode).read_bytes())
        for dtype, mode in profiles
    ):
        print(f"DKR_DENSE_GEMV_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, source_artifact)
    else:
        subprocess.run(command, check=True)
    sha = digest(source_artifact.read_bytes())
    for dtype, mode in profiles:
        path = artifact(dtype, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != source_artifact:
            shutil.copyfile(source_artifact, path)
        arguments = [
            tensor("weight", dtype, 2, "read"),
            tensor("input", dtype, 2, "read"),
        ]
        if mode == "bias":
            arguments.append(tensor("bias", dtype, 1, "read"))
        arguments += [
            tensor("output", dtype, 2, "write"),
            scalar("rows"), scalar("columns"), scalar("batch")
        ]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": f"linear.gemv.{mode}.b128.v1",
            "implementation_id": f"cuda.dense_gemv.{dtype}.{mode}.b128.v1",
            "artifact": path.name,
            "sha256": sha,
            "entry": f"dkr_gemv_{dtype}_{mode}",
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [BLOCK, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": arguments,
        }
        path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_DENSE_GEMV_PACKAGES=STAGED profiles={len(profiles)} cc={args.cc}")

if __name__ == "__main__":
    main()
