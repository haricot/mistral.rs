#!/usr/bin/env python3
"""Verify the S2 NVFP4 GEMM physical test against its own paired CUBIN."""
import argparse
import json
import math
import re
from pathlib import Path

from dkr_r126_nvfp4_preflight import inspect_packages, digest


TEST = "dkr_nvfp4_checkpoint_gemm_gpu"


def verify(root, logfile):
    package = inspect_packages(root, "nvfp4-gemm")
    if package["cubin_status"] != "PAIRED" or package["cubin_images"] != 1:
        raise ValueError("strict paired NVFP4 GEMM CUBIN required")
    expected = package["cubin_sha256"][0]
    log = logfile.read_text()
    if "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in log or re.search(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", log
    ):
        raise ValueError("PTX fallback or PTX image admission")
    if re.search(r"(?m)^\s*(?:FAIL|TIMEOUT|ABORT|LEAK)\s+\[", log):
        raise ValueError("Nextest failure observed")
    match = re.search(
        r"(?ms)^\s*START\s+\[[^\]]+\]\s+[^\n]*::" + TEST + r"\s*$"
        r"(.*?)^\s*PASS\s+\[[^\]]+\]\s+[^\n]*::" + TEST + r"\s*$", log
    )
    if not match:
        raise ValueError("GEMM Nextest START/PASS missing")
    section = match.group(1)
    if "DKR_R125_TYPED_POLICY requested=cubin" not in section:
        raise ValueError("strict CUBIN policy not logged")
    loaded = re.findall(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+"
        r"cubin_sha256=([0-9a-f]{64})\s+ptx_sha256=([0-9a-f]{64})", section
    )
    if set(loaded) != {(expected, package["ptx_sha256"][0])}:
        raise ValueError(f"unexpected GEMM CUBIN/PTX admission: {loaded}")
    geometry = re.search(
        r"DKR_R126_NVFP4_CHECKPOINT_LAYER prefix=layers\.0\.self_attn\.k_proj "
        r"source_n=(\d+) sampled_n=(\d+) k=(\d+)", section
    )
    if not geometry:
        raise ValueError("real projection geometry missing")
    n, sampled, k = map(int, geometry.groups())
    if n <= 0 or sampled != min(n, 37) or k <= 0 or k % 16:
        raise ValueError("unexpected projection geometry")
    real_cases = []
    for dtype, tol in (("F16", 0.0015), ("BF16", 0.008)):
        for rows in (9, 17, 33):
            record = re.search(
                rf"DKR_R126_NVFP4_LAYER_NUMERIC dtype={dtype} rows={rows} "
                r"relative_l2=(\S+) max_abs=(\S+) max_ref=(\S+)", section
            )
            if not record:
                raise ValueError(f"missing real projection case {dtype}/{rows}")
            rel, abs_err, ref = map(float, record.groups())
            if not all(math.isfinite(v) and v >= 0 for v in (rel, abs_err, ref)):
                raise ValueError("invalid numerical metrics")
            # The Rust assertion compares before formatting; allow rounding
            # of the eight displayed decimal places in the recorded evidence.
            if rel > tol + 1e-8 or abs_err > 2 * tol * ref + 1e-5 + 1e-8:
                raise ValueError(f"real projection tolerance exceeded {dtype}/{rows}")
            real_cases.append(dict(dtype=dtype, rows=rows, relative_l2=rel,
                                   max_abs=abs_err, max_ref=ref))
        for bias in ("false", "true"):
            marker = (f"DKR_R126_NVFP4_GEMM_SYNTHETIC=PASS dtype={dtype} "
                      f"bias={bias} m=17 n=19 k=48")
            if marker not in section:
                raise ValueError(f"missing synthetic case {dtype}/{bias}")
    if "DKR_R126_NVFP4_LAYER=PASS real_projection=true prefill_gemm=true" not in section:
        raise ValueError("real checkpoint GEMM result missing")
    return {
        "schema": "dkr.r126.nvfp4.gemm.evidence.v1", "status": "PASS",
        "target_cc": 61, "test": TEST, "profiles": 4,
        "cubin_sha256": expected, "ptx_sha256": package["ptx_sha256"][0],
        "nextest_log_sha256": digest(logfile),
        "source_output_rows": n, "sampled_output_rows": sampled,
        "input_width": k, "real_cases": real_cases, "synthetic_cases": 4,
        "real_checkpoint_projection": True,
        "nvfp4_prefill_gemm_qualified": True,
        "qualification_scope": "sampled_dense_w4a16_layer_functional",
        "full_model_qualified": False, "dflash_qualified": False,
        "performance_qualified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--json-out", type=Path)
    args = p.parse_args()
    try:
        report = verify(args.packages, args.log)
        if args.json_out:
            with args.json_out.open("x") as f:
                json.dump(report, f, indent=2, sort_keys=True)
                f.write("\n")
    except (ValueError, OSError) as exc:
        print(f"DKR_R126_NVFP4_GEMM_EVIDENCE=FAIL reason={exc}")
        raise SystemExit(1)
    print("DKR_R126_NVFP4_GEMM_EVIDENCE=PASS profiles=4 cubin_images=1 "
          "real_cases=6 synthetic_cases=4 full_model=false performance=false")
    print(f"DKR_R126_NVFP4_GEMM_IMAGE_SHA256={report['cubin_sha256']}")
    if args.json_out:
        print(f"DKR_R126_NVFP4_GEMM_REPORT={args.json_out}")


if __name__ == "__main__":
    main()
