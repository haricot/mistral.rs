#!/usr/bin/env python3
"""Validate *recorded* SM61 quant GPU numerical-test & CUBIN evidence.

Pure read-only: does not invoke Rust, CUDA, nvcc, ptxas, or the GPU.
Image admission messages are per unique CUBIN *module*, not per kernel entry:
GEMM and GEMV each package six entry symbols within ONE shared CUDA image.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


CASES = {
    "gemm": ("quant-gemm", "dkr_quantized_gemm_gpu"),
    "gemv": ("quant-gemv", "dkr_quantized_gemv_gpu"),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def manifests_for(root, family):
    expected = {}
    for dtype in ("f16", "bf16"):
        for fmt, variants in (("fp8", ("plain",)), ("mxfp4", ("plain", "bias"))):
            for variant in variants:
                path = root / family / fmt / dtype / variant / "manifest.json"
                if not path.is_file():
                    raise ValueError(f"manifest missing: {path}")
                data = json.loads(path.read_text())
                if data.get("target_cc") != 61:
                    raise ValueError(f"wrong SM61 architecture: {path}")
                cb = data.get("cubin_artifact")
                fingerprint = data.get("cubin_sha256")
                if not isinstance(cb, str) or not cb or Path(cb).name != cb:
                    raise ValueError(f"missing/unsafe CUBIN artifact: {path}")
                if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
                    raise ValueError(f"missing/invalid CUBIN SHA256: {path}")
                binary = path.parent / cb
                if not binary.is_file() or sha256(binary) != fingerprint.lower():
                    raise ValueError(f"CUBIN artifact hash mismatch: {binary}")
                expected[f"{fmt}/{dtype}/{variant}"] = fingerprint.lower()
    return expected


def verify(root, log, scope):
    text = log.read_text()
    if "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in text:
        raise ValueError("PTX fallback detected, strict CUBIN evidence invalid")
    if re.search(r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", text):
        raise ValueError("PTX load found in strict CUBIN log")
    if "DKR_R125_TYPED_POLICY requested=cubin" not in text:
        raise ValueError("strict CUBIN loader policy not logged")
    observed = set(re.findall(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+cubin_sha256=([0-9a-fA-F]{64})",
        text,
    ))
    observed = {x.lower() for x in observed}
    required = ["gemm", "gemv"] if scope == "both" else [scope]
    manifest_hashes = {}
    for test in required:
        family, name = CASES[test]
        # A genuine Nextest PASS from this execution, not merely Rust's
        # inline "test result: ok" nor a staged manifest.
        if not re.search(r"(?m)^\s*PASS\s+\[[^\]]+\]\s+.*\b" + re.escape(name) + r"\s*$", text):
            raise ValueError(f"Nextest PASS not found for {name}")
        hashes = manifests_for(root, family)
        expected = set(hashes.values())
        if not expected.issubset(observed):
            raise ValueError(f"{family} not loaded via recorded SM61 CUBIN: "
                             f"missing SHA256={sorted(expected - observed)}")
        manifest_hashes[family] = {
            "profiles": len(hashes),
            "distinct_cubin_images": len(expected),
            "cubin_sha256": sorted(expected),
        }
    if re.search(r"(?m)^\s*FAIL\s+\[", text):
        raise ValueError("Nextest FAIL found in log")
    return {"scope": scope, "numeric_tests": len(required),
            "cubin_modules": len(observed), "families": manifest_hashes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="paired typed package root /sm61")
    p.add_argument("--log", required=True, type=Path, help="Nextest log from this test run")
    p.add_argument("--scope", required=True, choices=("gemm", "gemv", "both"))
    args = p.parse_args()
    try:
        report = verify(args.root, args.log, args.scope)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"DKR_R125_QUANT_EVIDENCE=FAIL scope={args.scope} reason={e}")
        raise SystemExit(1)
    print(f"DKR_R125_QUANT_EVIDENCE=PASS scope={args.scope} "
          f"numerical_tests={report['numeric_tests']} "
          f"loaded_distinct_cubin_modules={report['cubin_modules']}")
    for family, info in report["families"].items():
        print(f"DKR_R125_QUANT_IMAGE=PASS family={family} "
              f"profiles={info['profiles']} unique_cubins={info['distinct_cubin_images']} "
              f"cubin_sha256={','.join(info['cubin_sha256'])}")
    if args.scope != "both":
        print(f"DKR_R125_QUANT_DIAGNOSTIC=PASS scope={args.scope} "
              "full_qualification=false")
    else:
        print("DKR_R125_QUANT=PASS tests=2 formats=fp8,mxfp4 mode=cubin "
              "verified_by=manifest_image_sha256")


if __name__ == "__main__":
    main()
