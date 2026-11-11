#!/usr/bin/env python3
"""Build the external sparse-logits package for one CUDA architecture."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

SOURCE = Path(__file__).with_name("logits.cu")
BLOCK = 256


def tensor(name, dtype, access):
    return {"kind": "tensor", "name": name, "dtype": dtype, "rank": 1, "access": access}


def scalar(name, dtype):
    return {"kind": "scalar", "name": name, "dtype": dtype}


def profiles():
    yield ("copy", "tensor.copy.contiguous.f32.v1", "dkr_copy_f32", [
        tensor("input", "f32", "read"), tensor("output", "f32", "write"),
        scalar("n", "u32"),
    ])
    yield ("penalties", "sampling.logits.penalties_sparse.f32.v1",
           "dkr_logits_penalties_sparse_f32", [
        tensor("logits", "f32", "read_write"), tensor("token_ids", "u32", "read"),
        tensor("counts", "f32", "read"), scalar("vocab_size", "u32"),
        scalar("count", "u32"), scalar("frequency_penalty", "f32"),
        scalar("presence_penalty", "f32"), scalar("repetition_penalty", "f32"),
    ])
    yield ("bias", "sampling.logits.bias_sparse.f32.v1",
           "dkr_logits_bias_sparse_f32", [
        tensor("logits", "f32", "read_write"), tensor("token_ids", "u32", "read"),
        tensor("biases", "f32", "read"), scalar("vocab_size", "u32"),
        scalar("count", "u32"),
    ])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()
    if not 50 <= args.cc <= 120:
        parser.error("unsupported CUDA compute capability")
    root = args.output / f"sm{args.cc}" / "sparse-logits"
    artifact = root / "copy" / "logits.ptx"
    stamp = root / "source.sha256"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    command = [
        args.nvcc, "-ptx", f"-arch=compute_{args.cc}", f"-code=compute_{args.cc}",
        "-std=c++17", "-O3", "-o", str(artifact), str(SOURCE),
    ]
    version = subprocess.check_output([args.nvcc, "--version"])
    identity = digest(SOURCE.read_bytes() + version + "\0".join(command).encode())
    items = list(profiles())
    if stamp.is_file() and stamp.read_text().strip() == identity and all(
        (root / name / "manifest.json").is_file()
        and (root / name / "logits.ptx").is_file()
        and json.loads((root / name / "manifest.json").read_text())["sha256"]
        == digest((root / name / "logits.ptx").read_bytes())
        for name, *_ in items
    ):
        print(f"DKR_SPARSE_LOGITS_REUSED=PASS target_cc={args.cc}")
        return
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
            "operation": operation, "implementation_id": f"cuda.sparse_logit.{name}.v1",
            "artifact": output.name, "sha256": ptx_hash, "entry": entry,
            "target_cc": args.cc, "min_cc": args.cc, "max_cc": 120,
            "block": [BLOCK, 1, 1], "shared_mem_bytes": 0,
            "arguments": arguments,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    stamp.write_text(identity + "\n")
    print(f"DKR_SPARSE_LOGITS_PACKAGES=PASS target_cc={args.cc} entries={len(items)}")


if __name__ == "__main__":
    main()
