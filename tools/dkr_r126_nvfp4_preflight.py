#!/usr/bin/env python3
"""Read-only R1.26 SM61 NVFP4 W4A16 checkpoint and DKR GEMV PTX audit.

No network, Cargo, CUDA, NVCC, ptxas or GPU actions. The check is NOT a
physical GPU qualification, a complete model inference test, nor a DFlash
draft/target pairing test. Reads only the Safetensors HEADER, not weight data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4"
DEFAULT_REVISION = "913d13c49fbbb2b477dce78552a31ab650d5b693"
TARGET_CC = 61
PROFILES = [(dtype, mode) for dtype in ("f16", "bf16") for mode in ("plain", "bias")]


class PreflightError(ValueError):
    pass


def check(condition, reason):
    if not condition:
        raise PreflightError(reason)


def digest(file):
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_safetensors_header(model):
    check(model.is_file(), f"missing model.safetensors: {model}")
    size = model.stat().st_size
    with model.open("rb") as stream:
        size_bytes = stream.read(8)
        check(len(size_bytes) == 8, "truncated safetensors size header")
        header_size = struct.unpack("<Q", size_bytes)[0]
        check(2 <= header_size <= min(size - 8, 64 * 1024 * 1024),
              f"invalid/oversized safetensors header: {header_size}")
        content = stream.read(header_size)
        check(len(content) == header_size, "truncated safetensors header")
    try:
        index = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"unreadable safetensors header: {exc}") from exc
    check(isinstance(index, dict), "safetensors header is not a mapping")
    tensors = {k: v for k, v in index.items() if k != "__metadata__"}
    check(tensors, "safetensors archive contains no tensors")
    counts = Counter()
    samples = []
    quant_candidates = 0
    for name, desc in tensors.items():
        check(isinstance(name, str) and isinstance(desc, dict),
              "invalid safetensors tensor header")
        shape, data_offsets, dtype = (desc.get(k) for k in ("shape", "data_offsets", "dtype"))
        check(isinstance(shape, list) and
              all(isinstance(d, int) and d >= 0 for d in shape),
              f"invalid tensor shape: {name}")
        check(isinstance(data_offsets, list) and len(data_offsets) == 2
              and all(isinstance(x, int) and x >= 0 for x in data_offsets)
              and data_offsets[0] <= data_offsets[1]
              and data_offsets[1] <= size - 8 - header_size,
              f"invalid tensor data offsets: {name}")
        check(isinstance(dtype, str), f"invalid tensor dtype: {name}")
        counts[dtype] += 1
        lower = name.lower()
        if any(w in lower for w in ("scale", "packed", "weight")):
            quant_candidates += 1
            if len(samples) < 12:
                samples.append({"name": name, "dtype": dtype, "shape": shape})
    return {"file_bytes": size, "header_bytes": header_size,
            "tensor_count": len(tensors), "dtype_counts": dict(sorted(counts.items())),
            "quant_related_count": quant_candidates, "samples": samples}


def inspect_checkpoint(snapshot):
    config_path = snapshot / "config.json"
    check(config_path.is_file(), f"missing config.json: {config_path}")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid config.json: {exc}") from exc
    check(isinstance(config, dict), "config.json is not an object")
    quant = config.get("quantization_config") or config.get("compression_config")
    check(isinstance(quant, dict), "missing ModelOpt quantization_config")
    nested = quant.get("quantization")
    q = nested if isinstance(nested, dict) else quant
    algo = q.get("quant_algo", quant.get("quant_algo"))
    group = q.get("group_size", quant.get("group_size"))
    check(algo == "W4A16_NVFP4" and group == 16,
          f"expected ModelOpt W4A16_NVFP4 group_size=16, got {algo!r}, {group!r}")
    architectures = config.get("architectures", [])
    check(isinstance(architectures, list) and all(isinstance(s, str) for s in architectures),
          "invalid checkpoint architectures")
    safetensors = validate_safetensors_header(snapshot / "model.safetensors")
    return {"snapshot": str(snapshot), "model_type": config.get("model_type"),
            "architectures": architectures,
            "is_dflash_draft": "DFlashDraftModel" in architectures,
            "quant_algo": algo, "group_size": group,
            "hf_quant_config_present": (snapshot / "hf_quant_config.json").is_file(),
            "safetensors": safetensors}


def inspect_packages(root, family="nvfp4-gemv"):
    check(family in ("nvfp4-gemv", "nvfp4-gemm"), "unsupported NVFP4 package family")
    op = family.split("-")[1]
    geometry = "t16" if op == "gemm" else "b128"
    block = [16, 16, 1] if op == "gemm" else [128, 1, 1]
    fingerprints = set()
    paired = set()
    items = {}
    any_paired = False
    for dtype, mode in PROFILES:
        path = root / family / dtype / mode / "manifest.json"
        check(path.is_file(), f"missing NVFP4 typed manifest: {path}")
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise PreflightError(f"invalid NVFP4 manifest {path}: {exc}") from exc
        check(isinstance(data, dict), f"invalid manifest object: {path}")
        expected = {
            "schema": "DKR-KERNEL-MODULE-V1",
            "abi": "dkr.typed.v1",
            "target_cc": TARGET_CC,
            "entry": f"dkr_nvfp4_{op}_{dtype}_{mode}",
            "operation": f"quant.nvfp4.{op}.{mode}.{geometry}.v1",
            "block": block,
        }
        for key, value in expected.items():
            check(data.get(key) == value,
                  f"invalid {key} for {dtype}/{mode}: {data.get(key)!r}")
        artifact = data.get("artifact")
        check(isinstance(artifact, str) and artifact and Path(artifact).name == artifact,
              f"unsafe/missing PTX artifact path: {path}")
        ptx = path.parent / artifact
        check(ptx.is_file(), f"missing NVFP4 PTX: {ptx}")
        pt_sha = digest(ptx)
        check(pt_sha == data.get("sha256"), f"PTX digest mismatch: {ptx}")
        code = ptx.read_text(errors="replace")
        check(re.search(r"\.target\s+sm_61(?:\s|$|,)", code) is not None,
              f"wrong NVFP4 PTX target: {ptx}")
        check(re.search(r"\.entry\s+" + re.escape(expected["entry"]) + r"\s*\(", code)
              is not None, f"missing PTX entry {expected['entry']}")
        fingerprints.add(pt_sha)

        cubin_name = data.get("cubin_artifact")
        cubin_sha = data.get("cubin_sha256")
        check((cubin_name is None) == (cubin_sha is None),
              f"partial CUBIN pairing metadata: {path}")
        if cubin_name is not None:
            any_paired = True
            check(isinstance(cubin_name, str) and cubin_name
                  and Path(cubin_name).name == cubin_name,
                  f"unsafe CUBIN artifact path: {path}")
            cubin = path.parent / cubin_name
            check(cubin.is_file() and digest(cubin) == cubin_sha,
                  f"missing/mismatched CUBIN image: {cubin}")
            with cubin.open("rb") as stream:
                header = stream.read(52)
            check(len(header) == 52 and header[:6] == b"\x7fELF\x02\x01",
                  f"invalid ELF64 little-endian CUBIN: {cubin}")
            flags = int.from_bytes(header[48:52], "little")
            check((flags & 255) == TARGET_CC,
                  f"wrong CUBIN ELF architecture: {cubin}")
            paired.add(cubin_sha)
        items[f"{dtype}/{mode}"] = {
            "ptx_sha256": pt_sha, "cubin_sha256": cubin_sha,
            "operation": data["operation"], "entry": data["entry"],
        }
    check(len(fingerprints) == 1,
          f"NVFP4 four manifests must share one PTX image: {len(fingerprints)} found")
    check(not any_paired or all(item["cubin_sha256"] is not None for item in items.values()),
          "partial CUBIN pairing across profiles")
    check(not any_paired or len(paired) == 1,
          "four NVFP4 entries must share one paired CUBIN image")
    return {"family": family, "profiles": len(items), "target_cc": TARGET_CC,
            "ptx_images": len(fingerprints), "ptx_sha256": sorted(fingerprints),
            "cubin_status": "PAIRED" if any_paired else "PTX_ONLY",
            "cubin_images": len(paired), "cubin_sha256": sorted(paired),
            "variants": items}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=(
        Path.home() / ".cache/huggingface/hub" / DEFAULT_MODEL
        / "snapshots" / DEFAULT_REVISION))
    ap.add_argument("--packages", type=Path, required=True,
                    help="existing /sm61 root containing selected NVFP4 family")
    ap.add_argument("--family", choices=("nvfp4-gemv", "nvfp4-gemm"), default="nvfp4-gemv")
    ap.add_argument("--json-out", type=Path, help="optional CREATE-ONLY JSON report")
    args = ap.parse_args()
    try:
        checkpoint = inspect_checkpoint(args.snapshot)
        packages = inspect_packages(args.packages, args.family)
        report = {"schema": "dkr.r126.nvfp4.read-only-preflight.v1",
                  "checkpoint": checkpoint, "packages": packages,
                  "gpu_qualified": False, "standalone_model_qualified": False,
                  "model_logits_qualified": False}
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with args.json_out.open("x") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")
    except (ValueError, OSError, OverflowError) as exc:
        print(f"DKR_R126_NVFP4_PREFLIGHT=BLOCKED reason={exc}")
        raise SystemExit(3) from exc

    weights = checkpoint["safetensors"]
    print(f"DKR_R126_NVFP4_CHECKPOINT=PASS model_type={checkpoint['model_type']} "
          f"architectures={','.join(checkpoint['architectures'])} "
          f"quant={checkpoint['quant_algo']} group_size=16 "
          f"bytes={weights['file_bytes']} tensors={weights['tensor_count']}")
    print(f"DKR_R126_NVFP4_TENSOR_DTYPES={json.dumps(weights['dtype_counts'], sort_keys=True)}")
    for sample in weights["samples"]:
        print(f"DKR_R126_NVFP4_TENSOR_SAMPLE={sample['name']} "
              f"dtype={sample['dtype']} shape={sample['shape']}")
    print(f"DKR_R126_NVFP4_PTX=PASS profiles=4 unique_images=1 target=sm61 "
          f"sha256={packages['ptx_sha256'][0]} family={args.family}")
    print(f"DKR_R126_NVFP4_CUBIN={packages['cubin_status']} "
          f"unique_images={packages['cubin_images']}")
    print("DKR_R126_NVFP4_PREFLIGHT=PASS scope=metadata,weights_header,ptx "
          "gpu_qualified=false standalone_model=false")
    if args.json_out:
        print(f"DKR_R126_NVFP4_REPORT={args.json_out}")


if __name__ == "__main__":
    main()
