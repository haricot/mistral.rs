#!/usr/bin/env python3
"""Join two independently-qualified SM61 quant family results, offline only.

Each Nextest log must prove its own numerical PASS, strict typed CUBIN
module admission, and the exact SHA256 of the on-disk paired binary.
This intentionally DOES NOT assert a shared test session, model qualification,
or model-level accuracy/performance parity.
"""
import argparse
import hashlib
import json
from pathlib import Path

from dkr_r125_quant_evidence import verify


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def verify_pair(root, gemm_log, gemv_log):
    results = {
        "gemm": verify(root, gemm_log, "gemm"),
        "gemv": verify(root, gemv_log, "gemv"),
    }
    images = {
        family: sorted(info["cubin_sha256"])
        for result in results.values()
        for family, info in result["families"].items()
    }
    # Different families must be physically attested, even if each happens
    # to contain six entry symbols pointing to one shared module.
    if set(images) != {"quant-gemm", "quant-gemv"}:
        raise ValueError("both quant-gemm and quant-gemv are required")
    if len(set(sum(images.values(), []))) != 2:
        raise ValueError("expected 2 independently identifiable CUBIN images")
    if sum(info["profiles"] for result in results.values()
           for info in result["families"].values()) != 12:
        raise ValueError("expected 12 typed format/dtype profiles")
    return {
        "schema": "dkr.r125.quant.family-set.evidence.v1",
        "status": "PASS",
        "target": "sm61",
        "scope": "two_separate_numeric_test_runs",
        "cubin_mode": "strict",
        "numerical_tests_passed": 2,
        "manifest_profiles_verified": 12,
        "distinct_cubin_images_verified": 2,
        "families": images,
        "logs": {
            "gemm": {"path": str(gemm_log), "sha256": sha256_file(gemm_log)},
            "gemv": {"path": str(gemv_log), "sha256": sha256_file(gemv_log)},
        },
        "combined_nextest_run": False,
        "model_inference_qualified": False,
        "model_performance_qualified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--gemm-log", required=True, type=Path)
    p.add_argument("--gemv-log", required=True, type=Path)
    p.add_argument("--json-out", type=Path,
                   help="optional NEW report path, never overwritten")
    args = p.parse_args()
    try:
        report = verify_pair(args.root, args.gemm_log, args.gemv_log)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"DKR_R125_QUANT_FAMILY_SET=FAIL reason={exc}")
        raise SystemExit(1)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"DKR_R125_QUANT_FAMILY_EVIDENCE_FILE={args.json_out}")
    print(
        "DKR_R125_QUANT_FAMILY_SET=PASS "
        "numerical_tests=2 manifests=12 cubin_images=2 "
        "separate_runs=true model_qualified=false"
    )
    for family, hashes in report["families"].items():
        print(f"DKR_R125_QUANT_FAMILY_IMAGE=PASS family={family} "
              f"cubin_sha256={','.join(hashes)}")


if __name__ == "__main__":
    main()
