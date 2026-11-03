"""Verify the synthetic SM61 KV measurements separately from model evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics

CONTEXTS = (128, 2048, 8192)
METRICS = ("fp16_write", "fp8_write", "fp16_decode", "fp8_decode")


def verify(path, node, run_id, attempt):
    report = json.loads(Path(path).read_text())
    if (report.get("schema") != 1 or report.get("status") != "PASS"
            or report.get("scope") != "synthetic-paged-kv-cache" or report.get("full_model_benchmark") is not False
            or report.get("candidate") != node["candidate"] or report.get("component") != node["component"]
            or report.get("run_id") != str(run_id) or report.get("run_attempt") != str(attempt)
            or report.get("samples") != 9 or report.get("iterations_per_sample") != 8
            or report.get("warmup") != 4 or report.get("performance_threshold") is not None
            or [x.get("context") for x in report.get("cases", [])] != list(CONTEXTS)):
        raise RuntimeError("FP8 KV measurement identity or protocol mismatch")
    for case in report["cases"]:
        elements = case["context"] * 8 * 64 * 2
        if (case.get("fp16_cache_bytes") != elements * 2 or case.get("fp8_cache_bytes") != elements
                or case.get("exact_scaled_cache_roundtrip") is not True
                or any(case.get(k) != v for k, v in (("query_heads", 32), ("kv_heads", 8), ("head_dim", 64)))):
            raise RuntimeError("FP8 KV cache layout or roundtrip proof missing")
        for name in ("fp16_reference_max_abs_error", "fp8_reference_max_abs_error", "fp8_fp16_max_abs_error"):
            value = case.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 0.01:
                raise RuntimeError("FP8 KV deterministic attention correctness failed")
        for name in METRICS:
            values = case.get(name + "_us", [])
            if (len(values) != report["samples"]
                    or not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in values)
                    or case.get(name + "_median_us") != statistics.median(values)):
                raise RuntimeError("FP8 KV raw timings or median missing")
        ratio = case["fp16_decode_median_us"] / case["fp8_decode_median_us"]
        if not math.isclose(case.get("decode_speed_ratio_fp16_over_fp8", -1), ratio, rel_tol=1e-12):
            raise RuntimeError("FP8 KV performance ratio does not match measurements")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_campaign(plan, directory, gpu, run_id):
    evidence = gpu.get("fp8_kv", {})
    for node in plan["matrix"]:
        name = node["component"]
        record = evidence.get(name, {})
        path = Path(directory) / (name + "-fp8-kv.json")
        if (record.get("test_count", 0) < 1 or record.get("production_layout_test_count", 0) < 1
                or verify(path, node, run_id, gpu.get("run_attempt")) != record.get("sha256")):
            raise RuntimeError("final-campaign physical FP8 KV proof is missing")
