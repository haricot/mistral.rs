#!/usr/bin/env python3
"""Two exact, noninterchangeable packed FP4 dequant reference ABI manifests."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("fp4_dequant.cu")
FORMATS = ("nvfp4", "mxfp4")
BLOCK = 256

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access, representation=None):
    result = {"kind": "tensor", "name": name, "dtype": dtype,
              "rank": rank, "access": access}
    if representation:
        result["representation"] = representation
    return result

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--ptx-file", type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target")
    root = a.output / f"sm{a.cc}" / "fp4-dequant"
    first = root / "nvfp4" / "fp4_dequant.ptx"
    first.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}", f"-code=compute_{a.cc}",
        "-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-o", str(first), str(SOURCE)
    ]
    version = (b"cudaforge:" + hashlib.sha256(a.ptx_file.read_bytes()).digest()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"
    def artifact(kind):
        return root / kind / first.name
    if stamp.exists() and stamp.read_text().strip() == identity and all(
        artifact(kind).is_file()
        and artifact(kind).with_name("manifest.json").is_file()
        and json.loads(artifact(kind).with_name("manifest.json").read_text())["sha256"]
            == digest(artifact(kind).read_bytes())
        for kind in FORMATS
    ):
        print(f"DKR_FP4_DEQUANT_REUSED=PASS cc={a.cc}")
        return
    if a.ptx_file:
        shutil.copyfile(a.ptx_file, first)
    else:
        subprocess.run(command, check=True)
    sha = digest(first.read_bytes())
    for kind in FORMATS:
        file = artifact(kind)
        file.parent.mkdir(parents=True, exist_ok=True)
        if file != first:
            shutil.copyfile(first, file)
        if kind == "nvfp4":
            params = [
                tensor("weights", "u8", 2, "read", "nvfp4_e2m1_group16.v1"),
                tensor("scales", "fp8_e4m3", 2, "read", "nvfp4_e4m3_group16.v1"),
                tensor("global", "f32", 1, "read"),
            ]
        else:
            params = [
                tensor("weights", "u8", 2, "read", "mxfp4_e2m1_group32.v1"),
                tensor("scales", "u8", 2, "read", "mxfp4_e8m0_group32.v1"),
            ]
        params += [
            tensor("output", "f32", 2, "write"),
            scalar("rows"), scalar("k")
        ]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"quant.fp4.dequant.{kind}.v1",
            "implementation_id": f"cuda.fp4_dequant.{kind}.f32.v1",
            "artifact": file.name, "sha256": sha,
            "entry": f"dkr_{kind}_dequant_f32",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [BLOCK, 1, 1], "shared_mem_bytes": 0,
            "arguments": params,
        }
        file.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_FP4_DEQUANT_PACKAGES=STAGED formats=2 cc={a.cc}")

if __name__ == "__main__":
    main()
