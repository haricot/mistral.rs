#!/usr/bin/env python3
"""Stage SM61 NVFP4 W4A16 fused decode GEMV PTX: 4 fixed ABI profiles."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("nvfp4_gemv.cu")
PROFILES = [(dt, mode) for dt in ("f16", "bf16") for mode in ("plain", "bias")]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def tensor(name, dtype, rank, access, repr=None):
    result = {"kind": "tensor", "name": name, "dtype": dtype,
              "rank": rank, "access": access}
    if repr:
        result["representation"] = repr
    return result

def scalar(name):
    return {"kind": "scalar", "name": name, "dtype": "u32"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc", required=True, type=int)
    p.add_argument("--output", type=Path,
                   default=Path("kernel-packages/build/cuda"))
    p.add_argument("--ptx-file", type=Path)
    p.add_argument("--nvcc", default="nvcc")
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported target")
    root = a.output / f"sm{a.cc}" / "nvfp4-gemv"
    base = root / "f16" / "plain" / "nvfp4_gemv.ptx"
    base.parent.mkdir(parents=True, exist_ok=True)
    command = [
        a.nvcc, "-ptx", f"-arch=compute_{a.cc}", f"-code=compute_{a.cc}",
        "-std=c++17", "-O3", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-o", str(base), str(SOURCE)
    ]
    version = (b"cudaforge:" + digest(a.ptx_file.read_bytes()).encode()
               if a.ptx_file else subprocess.check_output([a.nvcc, "--version"]))
    fingerprint = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"
    def file(dt, mode):
        return root / dt / mode / base.name
    if stamp.exists() and stamp.read_text().strip() == fingerprint and all(
        file(dt, mode).is_file()
        and file(dt, mode).with_name("manifest.json").is_file()
        and json.loads(file(dt, mode).with_name("manifest.json").read_text())["sha256"]
            == digest(file(dt, mode).read_bytes())
        for dt, mode in PROFILES
    ):
        print(f"DKR_NVFP4_GEMV_REUSED=PASS cc={a.cc}")
        return
    if a.ptx_file:
        shutil.copyfile(a.ptx_file, base)
    else:
        subprocess.run(command, check=True)
    sha = digest(base.read_bytes())
    for dt, mode in PROFILES:
        path = file(dt, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path != base:
            shutil.copyfile(base, path)
        args = [
            tensor("input", dt, 2, "read"),
            tensor("weight", "u8", 2, "read", "nvfp4_e2m1_group16.v1"),
            tensor("block_scale", "fp8_e4m3", 2, "read",
                   "nvfp4_e4m3_group16.v1"),
            tensor("global_scale", "f32", 1, "read"),
        ]
        if mode == "bias":
            args.append(tensor("bias", dt, 1, "read"))
        args += [tensor("output", dt, 2, "write"),
                 scalar("m"), scalar("n"), scalar("k")]
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1", "abi": "dkr.typed.v1",
            "operation": f"quant.nvfp4.gemv.{mode}.b128.v1",
            "implementation_id": f"cuda.nvfp4_gemv.{dt}.{mode}.b128.v1",
            "artifact": path.name, "sha256": sha,
            "entry": f"dkr_nvfp4_gemv_{dt}_{mode}",
            "target_cc": a.cc, "min_cc": a.cc, "max_cc": 120,
            "block": [128, 1, 1], "shared_mem_bytes": 0,
            "arguments": args,
        }
        path.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(fingerprint + "\n")
    print(f"DKR_NVFP4_GEMV_PACKAGES=STAGED profiles={len(PROFILES)} cc={a.cc}")

if __name__ == "__main__":
    main()
