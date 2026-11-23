#!/usr/bin/env python3
"""Offline evidence gate for R1.25 real FP8 blockwise layer-forward routing.

Verifies a Nextest numerical PASS for the production-facing layer method
and exact paired SM61 CUBIN module admissions for BOTH quantized GEMV
(decode) and GEMM (prefill). Does not run Rust, CUDA, Cargo, or the GPU.
Does NOT attest full-model execution, model logits, or MXFP4 model routing.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from dkr_r125_quant_evidence import manifests_for

TEST = "dkr_blockwise_fp8_layer_forward_gpu"
FAMILIES = ("quant-gemv", "quant-gemm")
LOAD_PATTERN = (
    r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+"
    r"format=cubin\s+cubin_sha256=([0-9a-fA-F]{64})"
)


def sha256_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_forward(root, logfile):
    log = logfile.read_text()
    if "DKR_R125_TYPED_POLICY requested=cubin" not in log:
        raise ValueError("strict CUBIN policy banner not found")
    if "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in log:
        raise ValueError("PTX fallback observed")
    if re.search(r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", log):
        raise ValueError("PTX image admitted instead of CUBIN")
    if not re.search(
        r"(?m)^\s*PASS\s+\[[^\]]+\]\s+(?:\(\d+/\d+\)\s+)?"
        r"mistralrs-quant\s+blockwise_fp8::tests::" + TEST + r"\s*$",
        log,
    ):
        raise ValueError("production FP8 layer-forward Nextest PASS missing")
    if re.search(r"(?m)^\s*(?:FAIL|TIMEOUT)\s+\[", log):
        raise ValueError("Nextest failure/timeout found")
    loaded = {s.lower() for s in re.findall(LOAD_PATTERN, log)}
    expected = {}
    for family in FAMILIES:
        images = set(manifests_for(root, family).values())
        if len(images) != 1:
            raise ValueError(f"{family} expected one deduplicated CUBIN, got {len(images)}")
        if not images.issubset(loaded):
            raise ValueError(f"{family} exact CUBIN admission missing: {sorted(images - loaded)}")
        expected[family] = next(iter(images))
    if len(set(expected.values())) != 2:
        raise ValueError("GEMV and GEMM must have independently attested CUBIN identities")
    return {
        "schema": "dkr.r125.fp8.layer-forward.evidence.v1",
        "status": "PASS",
        "target": "sm61",
        "test": TEST,
        "scope": "blockwise_fp8_layer_forward_raw",
        "dtype_cases": ["f16", "bf16"],
        "batch_cases": ["m1_decode", "m17_prefill", "rank3_m17_prefill"],
        "cubin_policy": "strict",
        "verified_cubin_sha256": expected,
        "nextest_log_sha256": sha256_file(logfile),
        "model_inference_qualified": False,
        "model_logits_qualified": False,
        "mxfp4_layer_forward_qualified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--json-out", type=Path, help="optional create-only attestation file")
    args = p.parse_args()
    try:
        report = verify_forward(args.root, args.log)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            with args.json_out.open("x") as output:
                json.dump(report, output, indent=2, sort_keys=True)
                output.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"DKR_R125_FP8_FORWARD_EVIDENCE=FAIL reason={error}")
        raise SystemExit(1)
    print(
        "DKR_R125_FP8_FORWARD_EVIDENCE=PASS "
        "numeric_tests=1 cubin_images=2 dtypes=f16,bf16 "
        "routes=gemv_decode,gemm_prefill layer_forward=true "
        "full_model_qualified=false"
    )
    for family, fingerprint in report["verified_cubin_sha256"].items():
        print(f"DKR_R125_FP8_FORWARD_IMAGE=PASS family={family} cubin_sha256={fingerprint}")
    if args.json_out is not None:
        print(f"DKR_R125_FP8_FORWARD_EVIDENCE_FILE={args.json_out}")


if __name__ == "__main__":
    main()
