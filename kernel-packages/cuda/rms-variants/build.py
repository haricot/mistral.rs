#!/usr/bin/env python3
"""Build fixed-signature residual RMSNorm variants in one CUDA source."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("rms_variants.cu")
DTYPES = ("f16", "bf16", "f32")
BLOCK = 256


def tensor(name, dtype, rank, access):
    return {"kind": "tensor", "name": name, "dtype": dtype,
            "rank": rank, "access": access}


def scalar(name, dtype):
    return {"kind": "scalar", "name": name, "dtype": dtype}


def profiles():
    for dtype in DTYPES:
        for mode in ("plain", "scaled"):
            scaled = mode == "scaled"
            args = [tensor("input", dtype, 2, "read"),
                    tensor("residual", dtype, 2, "read"),
                    tensor("weight", dtype, 1, "read")]
            if scaled:
                args.append(tensor("scale", dtype, 1, "read"))
            yield (f"{dtype}/residual/{mode}",
                   f"tensor.rms_residual.{mode}.v1",
                   f"dkr_rms_residual_{dtype}_{mode}",
                   args + [tensor("output", dtype, 2, "write"),
                           scalar("rows", "u32"), scalar("width", "u32"),
                           scalar("eps", "f32")])
            args = [tensor("input", dtype, 2, "read"),
                    tensor("residual", dtype, 2, "read"),
                    tensor("residual_weight", dtype, 1, "read")]
            if scaled:
                args.append(tensor("scale", dtype, 1, "read"))
            yield (f"{dtype}/residual_then_rms/{mode}",
                   f"tensor.rms_residual_then_rms.{mode}.v1",
                   f"dkr_rms_then_{dtype}_{mode}",
                   args + [tensor("norm_weight", dtype, 1, "read"),
                           tensor("residual_output", dtype, 2, "write"),
                           tensor("normalized_output", dtype, 2, "write"),
                           scalar("rows", "u32"), scalar("width", "u32"),
                           scalar("residual_eps", "f32"), scalar("norm_eps", "f32")])


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
        parser.error("unsupported CUDA compute capability")
    root = args.output / f"sm{args.cc}" / "rms-variants"
    artifact = root / "f16" / "residual" / "plain" / "rms_variants.ptx"
    stamp = root / "source.sha256"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--use_fast_math",
        "-o", str(artifact), str(SOURCE),
    ]
    version = (b"cudaforge:" + hashlib.sha256(args.ptx_file.read_bytes()).digest() if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    items = list(profiles())
    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        (root / name / "manifest.json").is_file()
        and (root / name / artifact.name).is_file()
        and json.loads((root / name / "manifest.json").read_text())["sha256"]
        == digest((root / name / artifact.name).read_bytes())
        for name, *_ in items
    ):
        print(f"DKR_RMS_VARIANTS_REUSED=PASS target_cc={args.cc}")
        return
    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, artifact)
    else:
        subprocess.run(command, check=True)
    ptx_hash = digest(artifact.read_bytes())
    for name, operation, entry, arguments in items:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / artifact.name
        if output != artifact:
            shutil.copyfile(artifact, output)
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": operation,
            "implementation_id": f"cuda.rms_variants.{name.replace('/', '.')}.v1",
            "artifact": output.name, "sha256": ptx_hash,
            "entry": entry, "target_cc": args.cc, "min_cc": args.cc,
            "max_cc": 120, "block": [BLOCK, 1, 1],
            "shared_mem_bytes": 0, "arguments": arguments,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_RMS_VARIANTS_PACKAGES=PASS target_cc={args.cc} entries={len(items)}")


if __name__ == "__main__":
    main()
