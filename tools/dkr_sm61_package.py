#!/usr/bin/env python3
"""Package one prebuilt SM61 PTX into DKR typed manifests; never invoke a compiler."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def tensor(name: str, dtype: str = "f32", rank: int = 1, access: str = "read") -> dict:
    return {"kind": "tensor", "name": name, "dtype": dtype, "rank": rank, "access": access}


def scalar(name: str, dtype: str = "u32") -> dict:
    return {"kind": "scalar", "name": name, "dtype": dtype}


def operation_specs() -> list[tuple[str, str, list[dict], str]]:
    specs = [
        ("copy", "tensor.copy.f32.v1", [tensor("src"), tensor("dst", access="write"), scalar("n")], "elementwise"),
        ("affine", "tensor.affine.f32.v1", [tensor("src"), tensor("dst", access="write"), scalar("mul", "f32"), scalar("add", "f32"), scalar("n")], "elementwise"),
    ]
    for op in ("add", "sub", "mul", "div"):
        specs.append((op, f"tensor.binary.{op}.f32.v1", [tensor("lhs"), tensor("rhs"), tensor("dst", access="write"), scalar("n")], "elementwise"))
    for op in ("exp", "tanh", "silu", "gelu"):
        specs.append((op, f"tensor.unary.{op}.f32.v1", [tensor("src"), tensor("dst", access="write"), scalar("n")], "elementwise"))
    for op in ("reduce_sum", "reduce_max"):
        specs.append((op, f"tensor.{op}.f32.v1", [tensor("src", rank=2), tensor("dst", access="write"), scalar("rows"), scalar("width")], "reduction"))
    specs.append(("softmax", "tensor.softmax.f32.v1", [tensor("src", rank=2), tensor("dst", rank=2, access="write"), scalar("rows"), scalar("width")], "reduction"))
    for residual in (False, True):
        for scaled in (False, True):
            name = ("residual_" if residual else "") + "rms_norm" + ("_scaled" if scaled else "")
            args = [tensor("src", rank=2)]
            if residual:
                args.append(tensor("residual", rank=2))
            if scaled:
                args.append(tensor("scale"))
            args.extend((tensor("dst", rank=2, access="write"), scalar("rows"), scalar("width"), scalar("epsilon", "f32")))
            specs.append((name, f"tensor.norm.{name}.f32.v1", args, "normalization"))
    specs.extend((
        ("causal_mask", "tensor.mask.causal.f32.v1", [tensor("src", rank=2), tensor("dst", rank=2, access="write"), scalar("rows"), scalar("cols"), scalar("past_tokens")], "attention"),
        ("logits_penalty_sparse", "sampling.logits.penalties_sparse.f32.v1", [tensor("logits", access="read_write"), tensor("token_ids", "u32"), tensor("counts", "u32"), scalar("entries"), scalar("vocab"), scalar("frequency", "f32"), scalar("presence", "f32"), scalar("repetition", "f32")], "sampling_and_sort"),
        ("logits_bias_sparse", "sampling.logits.bias_sparse.f32.v1", [tensor("logits", access="read_write"), tensor("token_ids", "u32"), tensor("biases"), scalar("entries"), scalar("vocab")], "sampling_and_sort"),
        ("gather_rows", "tensor.index.gather_rows.f32.v1", [tensor("src", rank=2), tensor("indices", "u32"), tensor("dst", rank=2, access="write"), scalar("rows"), scalar("width"), scalar("source_rows")], "indexing"),
        ("rope_pairs", "tensor.position.rope_pairs.f32.v1", [tensor("src", rank=2), tensor("cosine", rank=2), tensor("sine", rank=2), tensor("dst", rank=2, access="write"), scalar("rows"), scalar("width"), scalar("rotary_width")], "rotary_position"),
    ))
    return specs


def package(ptx_path: Path, output_dir: Path) -> dict:
    code = ptx_path.read_bytes()
    if len(code) > 32 * 1024 * 1024:
        raise ValueError("PTX exceeds DKR artifact limit")
    text = code.decode("utf-8")
    if not re.search(r"(?m)^\s*\.target\s+sm_61\b", text):
        raise ValueError("PTX does not target sm_61")
    if not re.search(r"(?m)^\s*\.address_size\s+64\b", text):
        raise ValueError("PTX does not use 64-bit addresses")
    specs = operation_specs()
    for name, _, _, _ in specs:
        entry = "dkr_" + name + "_f32"
        if not re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\(", text):
            raise ValueError(f"missing PTX entry: {entry}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = "dkr_ops.ptx"
    destination = output_dir / artifact_name
    if ptx_path.resolve() != destination.resolve():
        shutil.copyfile(ptx_path, destination)
    sha256 = hashlib.sha256(code).hexdigest()
    index = {"schema": "DKR-R2-SM61-PACKAGE-INDEX-V1", "ptx_sha256": sha256, "status": "UNQUALIFIED", "packages": []}
    for name, operation, args, family in specs:
        manifest = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "operation": operation,
            "implementation_id": "sm61.f32.scalar",
            "artifact": artifact_name,
            "sha256": sha256,
            "entry": "dkr_" + name + "_f32",
            "target_cc": 61,
            "min_cc": 61,
            "max_cc": 61,
            "block": [256, 1, 1],
            "shared_mem_bytes": 0,
            "arguments": args,
        }
        filename = name + ".json"
        (output_dir / filename).write_text(json.dumps(manifest, indent=2) + "\n")
        index["packages"].append({"manifest": filename, "operation": operation, "family": family})
    (output_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptx", type=Path, required=True, help="one externally compiled sm_61 PTX")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    index = package(args.ptx, args.out_dir)
    print(f"Packaged {len(index['packages'])} unqualified typed entries into {args.out_dir}")


if __name__ == "__main__":
    main()
