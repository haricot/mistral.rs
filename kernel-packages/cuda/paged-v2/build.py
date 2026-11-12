#!/usr/bin/env python3
"""R1.21/S6: package two fixed ABIs for SM61 long-context paged attention.

Cudaforge owns NVCC compilation. This script only stamps/checks
manifest identities and packages a provided, already compiled PTX.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("paged_v2.cu")
DTYPES = ("f16", "bf16", "f32")
STAGES = ("partition", "merge")


def digest(contents):
    return hashlib.sha256(contents).hexdigest()


def tensor(name, dtype, rank, access, representation=None):
    result = {"kind": "tensor", "name": name, "dtype": dtype,
              "rank": rank, "access": access}
    if representation is not None:
        result["representation"] = representation
    return result


def scalar(name, dtype="u32"):
    return {"kind": "scalar", "name": name, "dtype": dtype}


def parameters(stage, dt):
    if stage == "partition":
        return [
            tensor("query", dt, 3, "read"),
            tensor("key_cache", dt, 5, "read", "vllm_k_blocked_x.v1"),
            tensor("value_cache", dt, 4, "read", "vllm_v_token_strided.v1"),
            tensor("block_table", "u32", 2, "read"),
            tensor("context_lens", "u32", 1, "read"),
            tensor("partial_output", "f32", 4, "write"),
            tensor("partial_logsumexp", "f32", 3, "write"),
            *[scalar(name) for name in (
                "seqs", "heads", "kv_heads", "head_size",
                "block_size", "x", "table_stride", "max_context")],
            scalar("softmax_scale", "f32"),
        ]
    return [
        tensor("partial_output", "f32", 4, "read"),
        tensor("partial_logsumexp", "f32", 3, "read"),
        tensor("output", dt, 3, "write"),
        *[scalar(name) for name in (
            "seqs", "heads", "head_size", "partitions")],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True, type=int)
    parser.add_argument("--output", type=Path,
                        default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--ptx-file", type=Path)
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported target")
    root = args.output / f"sm{args.cc}" / "paged-v2"
    first = root / "partition" / "f16" / "paged_v2.ptx"
    first.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}",
        f"-code=compute_{args.cc}", "-std=c++17", "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-o", str(first), str(SOURCE),
    ]
    version = (b"cudaforge:" + digest(args.ptx_file.read_bytes()).encode()
               if args.ptx_file else subprocess.check_output([args.nvcc, "--version"]))
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    stamp = root / "source.sha256"

    def target(stage, dt):
        return root / stage / dt / first.name

    if (stamp.is_file() and stamp.read_text().strip() == identity
            and all(target(stage, dt).is_file()
                    and target(stage, dt).with_name("manifest.json").is_file()
                    and json.loads(target(stage, dt).with_name(
                        "manifest.json").read_text())["sha256"]
                    == digest(target(stage, dt).read_bytes())
                    for stage in STAGES for dt in DTYPES)):
        print(f"DKR_PAGED_V2_REUSED=PASS cc={args.cc}")
        return

    if args.ptx_file is not None:
        shutil.copyfile(args.ptx_file, first)
    else:
        subprocess.run(command, check=True)
    artifact_hash = digest(first.read_bytes())
    for stage in STAGES:
        for dtype in DTYPES:
            file = target(stage, dtype)
            file.parent.mkdir(parents=True, exist_ok=True)
            if file != first:
                shutil.copyfile(first, file)
            manifest = {
                "schema": "DKR-KERNEL-MODULE-V1",
                "abi": "dkr.typed.v1",
                "operation": f"attention.paged.v2.{stage}.p256.v1",
                "implementation_id":
                    f"cuda.paged_v2.{stage}.{dtype}.p256.b128.v1",
                "artifact": file.name,
                "sha256": artifact_hash,
                "entry": f"dkr_paged_v2_{stage}_{dtype}",
                "target_cc": args.cc,
                "min_cc": args.cc,
                "max_cc": 120,
                "block": [128, 1, 1],
                "shared_mem_bytes": 0,
                "arguments": parameters(stage, dtype),
            }
            file.with_name("manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    stamp.write_text(identity + "\n")
    print(
        f"DKR_PAGED_V2_PACKAGES=STAGED profiles={len(DTYPES) * len(STAGES)} "
        f"stages={len(STAGES)} cc={args.cc}"
    )


if __name__ == "__main__":
    main()
