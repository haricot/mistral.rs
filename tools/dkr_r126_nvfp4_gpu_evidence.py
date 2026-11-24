#!/usr/bin/env python3
"""Attest existing NVFP4 GEMV tests and their shared strict CUBIN offline."""
import argparse
import json
import re
from pathlib import Path

from dkr_r126_nvfp4_preflight import inspect_packages, digest

TESTS = ("dkr_nvfp4_gemv_gpu", "dkr_nvfp4_checkpoint_layer_gpu")


def verify(root, logfile):
    package = inspect_packages(root)
    if package["cubin_status"] != "PAIRED" or package["cubin_images"] != 1:
        raise ValueError("strict paired NVFP4 CUBIN required")
    expected = package["cubin_sha256"][0]
    text = logfile.read_text()
    if "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in text or re.search(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", text
    ):
        raise ValueError("PTX fallback or PTX image admission")
    if re.search(r"(?m)^\s*(?:FAIL|TIMEOUT|ABORT|LEAK)\s+\[", text):
        raise ValueError("Nextest failure observed")
    # Each test runs as a separate process. Require its own strict CUBIN
    # admission within its START/PASS interval, not a mere global count.
    for name in TESTS:
        pattern = (r"(?ms)^\s*START\s+\[[^\]]+\]\s+[^\n]*::"
                   + re.escape(name) + r"\s*$"
                   + r"(.*?)^\s*PASS\s+\[[^\]]+\]\s+[^\n]*::"
                   + re.escape(name) + r"\s*$")
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Nextest START/PASS missing for {name}")
        section = match.group(1)
        if "DKR_R125_TYPED_POLICY requested=cubin" not in section:
            raise ValueError(f"strict CUBIN policy not logged for {name}")
        loaded = re.findall(
            r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+"
            r"cubin_sha256=([0-9a-f]{64})", section
        )
        if set(loaded) != {expected}:
            raise ValueError(f"exact NVFP4 CUBIN admission missing for {name}: {loaded}")
        if name == TESTS[1]:
            geometry = re.search(
                r"DKR_R126_NVFP4_CHECKPOINT_LAYER prefix=layers\.0\.self_attn\.k_proj "
                r"source_n=(\d+) sampled_n=(\d+) k=(\d+)", section
            )
            if not geometry:
                raise ValueError("real checkpoint projection geometry missing")
            n, sampled_n, k = map(int, geometry.groups())
            if n <= 0 or sampled_n != min(n, 32) or k <= 0 or k % 16:
                raise ValueError("unexpected checkpoint projection geometry")
            for dtype in ("F16", "BF16"):
                for rows in (1, 3, 8):
                    if not re.search(r"DKR_R126_NVFP4_LAYER_NUMERIC dtype="
                                     + dtype + rf" rows={rows}\b", section):
                        raise ValueError(f"missing real checkpoint case {dtype}/{rows}")
            if "DKR_R126_NVFP4_LAYER=PASS real_projection=true prefill_gemm=false" not in section:
                raise ValueError("real checkpoint layer result missing")
    return {"schema": "dkr.r126.nvfp4.gemv.evidence.v1", "status": "PASS",
            "target_cc": 61, "tests": list(TESTS), "profiles": 4,
            "cubin_sha256": expected, "ptx_sha256": package["ptx_sha256"][0],
            "nextest_log_sha256": digest(logfile),
            "real_checkpoint_projection": True, "sampled_output_rows": sampled_n,
            "source_output_rows": n, "input_width": k,
            "activation_rows": [1, 3, 8], "activation_dtypes": ["f16", "bf16"],
            "nvfp4_prefill_gemm_qualified": False,
            "full_model_qualified": False, "dflash_qualified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--json-out", type=Path)
    a = p.parse_args()
    try:
        report = verify(a.packages, a.log)
        if a.json_out:
            with a.json_out.open("x") as f:
                json.dump(report, f, indent=2, sort_keys=True)
                f.write("\n")
    except (ValueError, OSError) as exc:
        print(f"DKR_R126_NVFP4_GPU_EVIDENCE=FAIL reason={exc}")
        raise SystemExit(1)
    print("DKR_R126_NVFP4_GPU_EVIDENCE=PASS numeric_tests=2 profiles=4 "
          "cubin_images=1 real_checkpoint_projection=true "
          "prefill_gemm=false full_model=false")
    print(f"DKR_R126_NVFP4_IMAGE_SHA256={report['cubin_sha256']}")
    if a.json_out:
        print(f"DKR_R126_NVFP4_GPU_REPORT={a.json_out}")


if __name__ == "__main__":
    main()
