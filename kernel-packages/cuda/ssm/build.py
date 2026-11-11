#!/usr/bin/env python3
"""Build external F32 selective-scan entry points for one CUDA target."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("ssm.cu")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor(name, rank, access):
    return {"kind": "tensor", "name": name, "dtype": "f32",
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
        parser.error("unsupported CUDA compute capability")

    root = args.output / f"sm{args.cc}" / "ssm"
    artifact = root / "cf1" / "ssm.ptx"
    stamp = root / "source.sha256"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3", "--use_fast_math",
        "-o", str(artifact), str(SOURCE),
    ]
    identity = digest(SOURCE.read_bytes() + (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"])) + "\0".join(cmd).encode())
    variants = (1, 2, 4, 8)
    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        (root / f"cf{cf}" / "manifest.json").is_file()
        and (root / f"cf{cf}" / artifact.name).is_file()
        and json.loads((root / f"cf{cf}" / "manifest.json").read_text())["sha256"]
        == digest((root / f"cf{cf}" / artifact.name).read_bytes())
        for cf in variants
    ):
        print(f"DKR_SSM_REUSED=PASS target_cc={args.cc}")
        return
    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(cmd, check=True)
    sha = digest(artifact.read_bytes())
    signature = [
        tensor("x", 4, "read"), tensor("dt", 3, "read"),
        tensor("a", 1, "read"), tensor("b", 4, "read"),
        tensor("c", 4, "read"), tensor("d", 1, "read"),
        tensor("dt_bias", 1, "read"), tensor("state", 4, "read_write"),
        tensor("output", 4, "write"),
        scalar("heads", "i32"), scalar("head_dim", "i32"),
        scalar("d_state", "i32"), scalar("seq_len", "i32"),
        scalar("dt_min", "f32"), scalar("dt_max", "f32"),
    ]
    for cf in variants:
        folder = root / f"cf{cf}"
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / artifact.name
        if output != artifact:
            shutil.copyfile(artifact, output)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"state_space.ssm_scan_f32.cf{cf}.v1",
            "implementation_id": f"cuda.ssm_scan.f32.cf{cf}.v1",
            "artifact": output.name, "sha256": sha,
            "entry": f"dkr_ssm_scan_cf{cf}",
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [32, 1, 1], "shared_mem_bytes": 0,
            "arguments": signature,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_SSM_PACKAGES=PASS target_cc={args.cc} variants={len(variants)}")


if __name__ == "__main__":
    main()
