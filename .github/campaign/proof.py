#!/usr/bin/env python3
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from campaign import ROOT, file_hash, verify_artifact, write_json

PORTABLE_RUSTFLAGS = "-C target-cpu=x86-64"


def node(plan, component):
    matches = [n for n in plan["matrix"] if n["component"] == component]
    if len(matches) != 1:
        raise RuntimeError("unknown or ambiguous campaign component")
    return matches[0]


def test_count(path, required):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    passed = [c for c in cases if c.find("skipped") is None]
    if any(c.find("failure") is not None or c.find("error") is not None for c in cases):
        raise RuntimeError("JUnit contains failing tests")
    if not passed or not any(required in c.get("name", "") for c in passed):
        raise RuntimeError(f"required test did not execute: {required}")
    return len(passed)


def build(directory):
    directory = Path(directory)
    proof = json.loads((directory / "node.json").read_text())
    proof["cpu_test_count"] = test_count(directory / "cpu-junit.xml", "campaign_cpu_backend")
    test_count(directory / "cpu-junit.xml", "campaign_force_cpu_selects_cpu")
    test_count(directory / "cpu-junit.xml", "campaign_cpu_flag_is_available")
    if os.environ.get("MISTRALRS_LFM3_REQUIRED") == "true":
        for required in (
            "models::lfm3::tests::",
            "gguf::lfm3_vl_config::tests::",
            "lfm3_conv::tests::cpu_prefill_decode_and_cached_chunks_match_reference",
        ):
            test_count(directory / "cpu-junit.xml", required)
        proof["lfm3_cpu_qualified"] = True
    proof["status"] = "BUILD_AND_CPU_PASSED"
    proof["features"] = ["cuda-legacy-bf16", "cuda-legacy-fp8", "cuda-legacy-fp4"]
    proof["profile"] = "campaign"
    proof["rustflags"] = (directory / "rustflags.txt").read_text().strip()
    if proof["rustflags"] != PORTABLE_RUSTFLAGS:
        raise RuntimeError("non-portable Rust compiler flags")
    proof["gpu_executed"] = False
    proof["files"] = {n: file_hash(directory / n) for n in [
        "tests.tar.zst", "cargo-nextest", "cpu-junit.xml", "Cargo.lock", "rustflags.txt", "rustc-version.txt"]}
    if proof["files"]["Cargo.lock"] != proof["lock"]:
        raise RuntimeError("built lockfile differs from preparation")
    write_json(directory / "build.json", proof)


def verify_build(directory, expected):
    proof = verify_artifact(directory, expected)
    if proof.get("rustflags") != PORTABLE_RUSTFLAGS:
        raise RuntimeError("non-portable Rust compiler flags")
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["identity", "build", "verify", "gpu-count", "gpu-required", "config"])
    parser.add_argument("path", type=Path)
    parser.add_argument("component", nargs="?")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--junit", type=Path)
    args = parser.parse_args()
    if args.mode == "config":
        text = (ROOT / "nextest.toml").read_text()
        marker = f'junit.path = "{args.component}-junit.xml"'
        if text.count(marker) != 1 or args.junit is None:
            raise RuntimeError("invalid JUnit output configuration")
        args.path.write_text(text.replace(marker, "junit.path = " + json.dumps(str(args.junit.resolve()))))
    elif args.mode == "identity":
        print(json.dumps(node(json.loads(args.path.read_text()), args.component)))
    elif args.mode == "build":
        build(args.path)
    elif args.mode == "gpu-count":
        print(test_count(args.path, "campaign_cuda_backend"))
    elif args.mode == "gpu-required":
        if args.component not in ("asd_sm61_smoke", "cuda_prefill_decode_and_cached_chunks_match_reference"):
            raise RuntimeError("unsupported required GPU test")
        print(test_count(args.path, args.component))
    else:
        plan = json.loads(args.plan.read_text())
        verify_build(args.path, node(plan, args.component))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"PROOF_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
