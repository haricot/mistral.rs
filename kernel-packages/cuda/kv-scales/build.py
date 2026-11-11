#!/usr/bin/env python3
"""Build or package DKR FP8 KV-cache scale-update PTX for one CUDA target."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("kv_scales.cu")
DTYPES = ("f16", "bf16", "f32")

def digest(contents):
    return hashlib.sha256(contents).hexdigest()

def tensor(name, dtype, access):
    return {"kind": "tensor", "name": name, "dtype": dtype, "rank": 1,
            "access": access}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True, type=int)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path)
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported compute capability")
    root = args.output / f"sm{args.cc}" / "kv-scales"
    source_artifact = root / "f16" / "kv_scales.ptx"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    stamp = root / "source.sha256"
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(source_artifact), str(SOURCE),
    ]
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest()
               if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())

    def artifact(dtype):
        return root / dtype / source_artifact.name

    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        artifact(dtype).is_file()
        and artifact(dtype).with_name("manifest.json").is_file()
        and json.loads(artifact(dtype).with_name("manifest.json").read_text())["sha256"]
        == digest(artifact(dtype).read_bytes())
        for dtype in DTYPES
    ):
        print(f"DKR_KV_SCALES_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, source_artifact)
    else:
        subprocess.run(command, check=True)
    sha = digest(source_artifact.read_bytes())
    for dtype in DTYPES:
        file = artifact(dtype)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != source_artifact:
            shutil.copyfile(source_artifact, file)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": "kv_cache.scale_update.atomic_max.v1",
            "implementation_id": f"cuda.kv_scales.{dtype}.v1",
            "artifact": file.name, "sha256": sha,
            "entry": f"dkr_kv_scale_update_{dtype}",
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [512, 1, 1], "shared_mem_bytes": 4096,
            "arguments": [
                tensor("key", dtype, "read"),
                tensor("value", dtype, "read"),
                tensor("key_scale", "f32", "read_write"),
                tensor("value_scale", "f32", "read_write"),
                {"kind": "scalar", "name": "elements", "dtype": "u64"},
            ],
        }
        file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_KV_SCALES_PACKAGES=STAGED profiles={len(DTYPES)} cc={args.cc}")

if __name__ == "__main__":
    main()
