#!/usr/bin/env python3
"""Read-only SM61 DKR FP8-blockwise / MXFP4 GEMV+GEMM package preflight.

Checks only existing PTX (and CUBIN, when present): no Cargo, nvcc, ptxas,
git reset, cache cleanup, or GPU execution.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def variants():
    for family in ("quant-gemv", "quant-gemm"):
        for dtype in ("f16", "bf16"):
            yield family, "fp8", dtype, "plain"
            for mode in ("plain", "bias"):
                yield family, "mxfp4", dtype, mode


def operation_name(family, fmt, mode):
    if family == "quant-gemv":
        if fmt == "fp8":
            return "quant.fp8.blockwise.gemv.b128.v1"
        return f"quant.mxfp4.gemv.{mode}.b128.v1"
    if fmt == "fp8":
        return "quant.fp8.blockwise.gemm.t16.v1"
    return f"quant.mxfp4.gemm.{mode}.t16.v1"


def check_manifest(path, family, fmt, dtype, mode, cc):
    reasons = []
    if not path.is_file():
        return ["manifest_missing"]
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return ["manifest_invalid"]
    for key, expected in (("schema", "DKR-KERNEL-MODULE-V1"),
                          ("abi", "dkr.typed.v1"),
                          ("operation", operation_name(family, fmt, mode)),
                          ("target_cc", cc)):
        if raw.get(key) != expected:
            reasons.append("invalid_" + key)
    if not isinstance(raw.get("entry"), str) or not raw["entry"]:
        reasons.append("entry_missing")
    artifact = raw.get("artifact")
    if not isinstance(artifact, str) or not artifact or Path(artifact).name != artifact:
        return reasons + ["unsafe_ptx_path"]
    ptx = path.parent / artifact
    if not ptx.is_file():
        return reasons + ["ptx_missing"]
    content = ptx.read_bytes()
    if sha256(content) != raw.get("sha256"):
        reasons.append("ptx_sha_mismatch")
    text = content.decode("utf-8", "replace")
    if not re.search(r"\.target\s+sm_" + str(cc) + r"(?:\s|$|,)", text):
        reasons.append("wrong_ptx_target")
    if "entry" in raw and isinstance(raw["entry"], str):
        if not re.search(r"\.entry\s+" + re.escape(raw["entry"]) + r"\s*\(", text):
            reasons.append("ptx_entry_missing")
    cubin_file = raw.get("cubin_artifact")
    cubin_sha = raw.get("cubin_sha256")
    if (cubin_file is None) != (cubin_sha is None):
        reasons.append("partial_cubin_pair")
    if cubin_file is not None:
        if not isinstance(cubin_file, str) or Path(cubin_file).name != cubin_file:
            reasons.append("unsafe_cubin_path")
        else:
            cubin = path.parent / cubin_file
            if not cubin.is_file():
                reasons.append("cubin_missing")
            elif sha256(cubin.read_bytes()) != cubin_sha:
                reasons.append("cubin_sha_mismatch")
            else:
                header = cubin.read_bytes()[:52]
                if len(header) < 52 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
                    reasons.append("cubin_elf_invalid")
                elif (int.from_bytes(header[48:52], "little") & 0xff) != cc:
                    reasons.append("cubin_elf_wrong_sm")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", type=int, default=61)
    parser.add_argument("--root", type=Path, default=os.environ.get("DKR_R125_TYPED_PACKAGE_ROOT"))
    parser.add_argument("--json-out", type=Path, help="optional report; never overwrite")
    args = parser.parse_args()
    if args.root is None:
        args.root = Path("kernel-packages/build/cuda/sm61")
    root = args.root.resolve()
    failures, results = {}, {}
    count = 0
    for family, fmt, dtype, mode in variants():
        count += 1
        manifest = root / family / fmt / dtype / mode / "manifest.json"
        problems = check_manifest(manifest, family, fmt, dtype, mode, args.cc)
        key = f"{family}/{fmt}/{dtype}/{mode}"
        results[key] = {"manifest": str(manifest), "problems": problems}
        if problems:
            failures[key] = problems
            print(f"DKR_R125_QUANT_MANIFEST=BLOCKED variant={key} reasons={','.join(problems)}")
    report = {"schema": "dkr.r125.quant.manifest-preflight.v1", "cc": args.cc,
              "root": str(root), "variants": results,
              "ok": count - len(failures), "total": count}
    status = "PASS" if not failures else "BLOCKED"
    print(f"DKR_R125_QUANT_PREFLIGHT={status} valid={count - len(failures)} expected={count} "
          f"families=quant-gemv,quant-gemm device_cc={args.cc}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("x") as output:
            json.dump(report, output, sort_keys=True, indent=2)
            output.write("\n")
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
