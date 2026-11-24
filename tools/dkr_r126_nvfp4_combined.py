#!/usr/bin/env python3
"""Stage qualified NVFP4 packages, and attest one combined full-projection test.
No compiler or GPU is invoked by this tool.
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

from dkr_r126_nvfp4_preflight import inspect_packages, digest

IMAGES = {
    "nvfp4-gemv": (
        "38fd99ef14140818b0485d5a70569e6402532c628ec18d4a1dfe272f7a709165",
        "e44c78486c280abc942dc8dad79c41adb9090b5a5f99eb537fe4ba570b7157ca",
    ),
    "nvfp4-gemm": (
        "cefd5eb2aecfb8262eb82ed71cf0a364e4857ebad5d3d3659124917e3aeec1ef",
        "5b71508ffcb5fe7f07ad0802bdebc0d6393dc86d1137eda34be545df494b9fde",
    ),
}
TEST = "dkr_nvfp4_checkpoint_combined_gpu"
ROWS = (17, 8, 9, 1, 17)


def check_family(root, family):
    p = inspect_packages(root, family)
    cubin, ptx = IMAGES[family]
    if p["cubin_sha256"] != [cubin] or p["ptx_sha256"] != [ptx]:
        raise ValueError(f"{family}: qualified artifact identity changed")
    return p


def check_catalog(root):
    for family in IMAGES:
        check_family(root, family)
    if len(list(root.rglob("manifest.json"))) != 8:
        raise ValueError("combined NVFP4 catalogue must contain exactly eight manifests")


def stage(root, gemv, gemm):
    sources = {"nvfp4-gemv": gemv, "nvfp4-gemm": gemm}
    for family, src in sources.items():
        check_family(src, family)
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        for family, src in sources.items():
            shutil.copytree(src / family, root / family)
    check_catalog(root)
    print("DKR_R126_NVFP4_COMBINED_CATALOG=PASS profiles=8 cubin_images=2 reused=true")


def verify(root, log):
    check_catalog(root)
    text = log.read_text()
    if "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in text or re.search(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", text
    ):
        raise ValueError("PTX admission or fallback")
    if re.search(r"(?m)^\s*(?:FAIL|TIMEOUT|ABORT|LEAK)\s+\[", text):
        raise ValueError("Nextest failure")
    match = re.search(
        r"(?ms)^\s*START\s+\[[^\]]+\]\s+[^\n]*::" + TEST + r"\s*$"
        r"(.*?)^\s*PASS\s+\[[^\]]+\]\s+[^\n]*::" + TEST + r"\s*$", text
    )
    if not match:
        raise ValueError("combined Nextest START/PASS missing")
    section = match.group(1)
    if "DKR_R125_TYPED_POLICY requested=cubin" not in section:
        raise ValueError("strict CUBIN mode missing")
    if "MISTRAL_DKR_CATALOG_VALIDATED manifests=8" not in section:
        raise ValueError("combined catalogue admission missing")
    if "DKR_R126_NVFP4_CHECKPOINT_LAYER prefix=layers.0.self_attn.k_proj source_n=1024 sampled_n=1024 k=2560" not in section:
        raise ValueError("full projection geometry missing")
    loaded = set(re.findall(
        r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+"
        r"cubin_sha256=([0-9a-f]{64})\s+ptx_sha256=([0-9a-f]{64})", section
    ))
    if loaded != set(IMAGES.values()):
        raise ValueError("expected exact qualified GEMV and GEMM admissions")
    steps = re.findall(
        r"DKR_R126_NVFP4_COMBINED_STEP=PASS dtype=(\w+) step=(\d+) rows=(\d+) route=(\w+)", section
    )
    expected_steps = [
        (dtype, str(i), str(m), "gemv" if m <= 8 else "gemm")
        for dtype in ("F16", "BF16") for i, m in enumerate(ROWS)
    ]
    if steps != expected_steps:
        raise ValueError("combined transition sequence missing or out of order")
    cross = re.findall(
        r"DKR_R126_NVFP4_CROSS dtype=(\w+) step=(\d+) rows=(\d+) relative_l2=(\S+) max_abs=(\S+)", section
    )
    expected_cross = [x[:3] for x in expected_steps if x[1] != "0"]
    if [x[:3] for x in cross] != expected_cross:
        raise ValueError("cross-route numeric records missing")
    metrics = []
    for dtype, step, rows, rel, maximum in cross:
        rel, maximum = float(rel), float(maximum)
        if not all(math.isfinite(x) and x >= 0 for x in (rel, maximum)):
            raise ValueError("invalid cross-route metric")
        if rel > (0.003 if dtype == "F16" else 0.016) + 1e-8:
            raise ValueError("cross-route tolerance exceeded")
        metrics.append(dict(dtype=dtype, step=int(step), rows=int(rows),
                            relative_l2=rel, max_abs=maximum))
    if "DKR_R126_NVFP4_COMBINED=PASS full_projection=true same_layer=true" not in section:
        raise ValueError("combined completion marker missing")
    return dict(schema="dkr.r126.nvfp4.combined.evidence.v1", status="PASS",
                target_cc=61, profiles=8, cubin_images=2,
                image_pairs=IMAGES, projection="layers.0.self_attn.k_proj",
                n=1024, k=2560, steps_per_dtype=list(ROWS),
                cross_route_metrics=metrics, nextest_log_sha256=digest(log),
                same_layer=True, full_projection=True, full_model_qualified=False,
                dflash_qualified=False, performance_qualified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--gemv", type=Path, required=True)
    s.add_argument("--gemm", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--log", type=Path, required=True)
    v.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == "stage":
            stage(a.root, a.gemv, a.gemm)
        else:
            report = verify(a.root, a.log)
            with a.json_out.open("x") as f:
                json.dump(report, f, indent=2, sort_keys=True)
                f.write("\n")
            print("DKR_R126_NVFP4_COMBINED_EVIDENCE=PASS profiles=8 cubin_images=2 steps=10 full_projection=true same_layer=true full_model=false")
            print(f"DKR_R126_NVFP4_COMBINED_REPORT={a.json_out}")
    except (ValueError, OSError) as exc:
        print(f"DKR_R126_NVFP4_COMBINED_EVIDENCE=FAIL reason={exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
