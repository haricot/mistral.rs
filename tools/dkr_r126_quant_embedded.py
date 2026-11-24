#!/usr/bin/env python3
"""Stage a disposable embedded build input and verify strict GPU evidence."""
import argparse
import json
import re
from pathlib import Path

from dkr_r126_quant_catalog import IMAGES, collect, stage
from dkr_r126_nvfp4_preflight import digest

TESTS = {
    "dkr_quantized_gemv_gpu": ("quant-gemv",),
    "dkr_quantized_gemm_gpu": ("quant-gemm",),
    "dkr_blockwise_fp8_layer_forward_gpu": ("quant-gemv", "quant-gemm"),
    "dkr_nvfp4_gemv_gpu": ("nvfp4-gemv",),
    "dkr_nvfp4_checkpoint_gemm_gpu": ("nvfp4-gemm",),
    "dkr_nvfp4_checkpoint_combined_gpu": ("nvfp4-gemv", "nvfp4-gemm"),
}


def validate(root, report):
    expected = collect(root, root)
    data = json.loads(report.read_text())
    if (data.get("schema") != "dkr.r126.quant.catalog.evidence.v1"
            or data.get("status") != "PASS" or data.get("profiles") != 20
            or data.get("cubin_images") != 4 or data.get("target_cc") != 61
            or data.get("prior_numeric_evidence_reverified") is not True
            or data.get("files") != expected
            or data.get("image_pairs") != {k: list(v) for k, v in IMAGES.items()}
            or len(list(root.rglob("manifest.json"))) != 20):
        raise ValueError("S4 catalogue/evidence mismatch")
    return expected


def prepare(root, report, output):
    expected = validate(root, report)
    if root.resolve() == output.resolve() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("embedded build input must be separate from S4")
    stage(output, root, root, expected)
    print("DKR_R126_EMBED_INPUT=PASS profiles=20 cubin_images=4")


def verify(root, report, log):
    validate(root, report)
    text = log.read_text()
    if ("CANDLE_DKR_TYPED_IMAGE_FALLBACK" in text
            or re.search(r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", text)
            or "MISTRAL_DKR_CATALOG_VALIDATED" in text
            or re.search(r"(?m)^\s*(?:FAIL|TIMEOUT|ABORT|LEAK)\s+\[", text)):
        raise ValueError("filesystem admission, PTX fallback or failed test")
    if "DKR_R126_EMBED_INPUT_ABSENT=PASS" not in text or "DKR_R126_EMBED_INPUT_STILL_ABSENT=PASS" not in text:
        raise ValueError("missing build-input isolation evidence")
    for test, families in TESTS.items():
        match = re.search(
            r"(?ms)^\s*START\s+\[[^\]]+\]\s+[^\n]*::" + re.escape(test)
            + r"\s*$" + r"(.*?)^\s*PASS\s+\[[^\]]+\]\s+[^\n]*::"
            + re.escape(test) + r"\s*$", text
        )
        if not match:
            raise ValueError(f"Nextest START/PASS missing: {test}")
        section = match.group(1)
        if ("MISTRAL_DKR_CATALOG_EMBEDDED manifests=20" not in section
                or "DKR_R125_TYPED_POLICY requested=cubin" not in section):
            raise ValueError(f"strict embedded catalogue admission missing: {test}")
        loaded = set(re.findall(
            r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+"
            r"cubin_sha256=([0-9a-f]{64})\s+ptx_sha256=([0-9a-f]{64})", section
        ))
        if loaded != {IMAGES[f] for f in families}:
            raise ValueError(f"unexpected image admission in {test}: {loaded}")
        if test == "dkr_nvfp4_checkpoint_combined_gpu":
            if "DKR_R126_NVFP4_COMBINED=PASS full_projection=true same_layer=true" not in section:
                raise ValueError("full projection result missing")
        if test == "dkr_nvfp4_checkpoint_gemm_gpu":
            for dtype in ("F16", "BF16"):
                for bias in ("false", "true"):
                    if f"DKR_R126_NVFP4_GEMM_SYNTHETIC=PASS dtype={dtype} bias={bias} " not in section:
                        raise ValueError("NVFP4 GEMM bias case missing")
    return dict(schema="dkr.r126.quant.embedded.evidence.v1", status="PASS",
                target_cc=61, profiles=20, cubin_images=4, image_pairs=IMAGES,
                tests=list(TESTS), combined_four_family_gpu_run=True,
                same_process_all_four_families=False,
                embedded_runtime_qualified=True, build_input_absent_during_run=True,
                filesystem_catalogue_admission=False, strict_cubin=True,
                source_catalogue_evidence_sha256=digest(report),
                nextest_log_sha256=digest(log), full_model_qualified=False,
                performance_qualified=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("stage", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, required=True)
        s.add_argument("--evidence", type=Path, required=True)
        if name == "stage":
            s.add_argument("--output", type=Path, required=True)
        else:
            s.add_argument("--log", type=Path, required=True)
            s.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    try:
        if a.action == "stage":
            prepare(a.root, a.evidence, a.output)
        else:
            data = verify(a.root, a.evidence, a.log)
            with a.json_out.open("x") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            print("DKR_R126_QUANT_EMBEDDED_EVIDENCE=PASS profiles=20 cubin_images=4 tests=6 embedded=true build_input_absent=true full_model=false")
            print(f"DKR_R126_QUANT_EMBEDDED_REPORT={a.json_out}")
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"DKR_R126_QUANT_EMBEDDED_EVIDENCE=FAIL reason={e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
