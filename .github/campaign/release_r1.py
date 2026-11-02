#!/usr/bin/env python3
"""Verify the one-and-only optimized Extended binary before reuse or publication."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import statistics
from pathlib import Path

SCHEMA = "haricot-extended-release-r1"
PORTABLE_RUSTFLAGS = "-C target-cpu=x86-64"
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
ROOT = Path(__file__).resolve().parent
IMAGE_SCHEMA = "haricot-extended-release-image-r1"
GPU_SCHEMA = "haricot-extended-release-gpu-r1"
PAGED_BENCH_TOKENS = 32


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected(plan):
    if plan.get("target") != "extended" or not SHA.fullmatch(plan.get("candidate", "")):
        raise ValueError("Extended candidate must be pinned to a commit")
    matrix = [n for n in plan["matrix"] if n["component"] == "extended"]
    if len(matrix) != 1 or matrix[0]["candidate"] != plan["candidate"]:
        raise ValueError("Extended matrix candidate differs from the plan")
    image = plan["image"]
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("compiler image is not immutable")
    return {
        "schema": SCHEMA,
        "candidate": plan["candidate"],
        "recipe": plan["recipe"],
        "image": image,
        "target": "sm_61",
        "toolchain": plan["config"]["toolchain"],
        "profile": "release",
        "rustflags": PORTABLE_RUSTFLAGS,
        "features": sorted(set(plan["config"]["features"] + ["code-execution"])),
        "lock_sha256": matrix[0]["lock"],
    }


def inspect(plan_file, directory):
    plan = json.loads(Path(plan_file).read_text())
    directory = Path(directory)
    binary = directory / "mistralrs"
    manifest = directory / "release.json"
    if binary.is_symlink() or manifest.is_symlink():
        raise ValueError("release artifact must not contain symlinks")
    if not binary.is_file() or not manifest.is_file():
        raise ValueError("release binary or manifest is missing")
    if not os.access(binary, os.X_OK):
        raise ValueError("release binary is not executable")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("release executable is not ELF")
    details = json.loads(manifest.read_text())
    for key, value in expected(plan).items():
        if details.get(key) != value:
            raise ValueError(f"release identity mismatch: {key}")
    if not DIGEST.fullmatch(details.get("binary_sha256", "")):
        raise ValueError("invalid binary SHA-256")
    if hash_file(binary) != details["binary_sha256"]:
        raise ValueError("release binary SHA-256 mismatch")
    if details.get("binary_bytes") != binary.stat().st_size:
        raise ValueError("release binary size mismatch")
    return details


def model_module():
    path = ROOT.parent / "lfm3/qualification.py"
    spec = importlib.util.spec_from_file_location("release_model_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def image_record(plan_file, directory, image_directory, archive=False):
    manifest = inspect(plan_file, directory)
    image_directory = Path(image_directory)
    record = json.loads((image_directory / "oci.json").read_text())
    if (record.get("schema") != IMAGE_SCHEMA or record.get("release") != manifest
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", record.get("image_id", ""))
            or record.get("binary_sha256") != manifest["binary_sha256"]
            or record.get("containerfile_sha256") != hash_file(ROOT / "Containerfile.release-r1")
            or record.get("loader_smoke") != "PASS" or record.get("gpu_executed") is not False
            or record.get("archive") != "image.tar.zst" or not DIGEST.fullmatch(record.get("archive_sha256", ""))):
        raise ValueError("runtime image identity or loader qualification mismatch")
    if archive and hash_file(image_directory / record["archive"]) != record["archive_sha256"]:
        raise ValueError("qualified image archive SHA256 mismatch")
    for name, key in (("loader-help.txt", "loader_help_sha256"), ("loader-ldd.txt", "loader_ldd_sha256")):
        if not DIGEST.fullmatch(record.get(key, "")) or hash_file(image_directory / name) != record[key]:
            raise ValueError("runtime image loader evidence SHA256 mismatch")
    return record


def collect_gpu(plan_file, directory, proofs, image_directory):
    manifest = inspect(plan_file, directory)
    image_record(plan_file, directory, image_directory)
    proofs = Path(proofs)
    kernel = json.loads((proofs / "report.json").read_text())
    if (kernel.get("candidate") != manifest["candidate"] or kernel.get("run_id") != os.environ.get("GITHUB_RUN_ID")
            or kernel.get("status") != "GPU_PASSED" or kernel.get("asd_required") is not True
            or kernel.get("gpu_executed") is not True or kernel.get("target") != "sm_61"
            or os.environ.get("EXPECTED_RUN_ID") != kernel["run_id"]
            or os.environ.get("EXPECTED_CANDIDATE_SHA") != manifest["candidate"]
            or os.environ.get("RUNNER_REPO") != "haricot/mistral.rs"
            or os.environ.get("RUNNER_NO_DEFAULT_LABELS") != "true"):
        raise ValueError("release qualification requires the dedicated candidate GPU and ASD proofs")
    for path, name in (("/run/haricot/asd/fallbacks/current.json", "current.json"),
                       ("/run/haricot/asd/profiles/current.asd", "current.asd")):
        if hash_file(Path(path)) != kernel["asd_profiles"][name]:
            raise ValueError("release ASD profiles changed after kernel qualification")
    q = model_module()
    policy = q.read(q.ROOT / "models.json")
    if policy["image"] != manifest["image"]:
        raise ValueError("release model policy uses a different compiler image")
    paths = q.model_paths(policy)
    rows = q.run("nvidia-smi", "--query-gpu=uuid,compute_cap,driver_version", "--format=csv,noheader").splitlines()
    if len(rows) != 1 or rows[0].split(",")[1].strip() != policy["sm"]:
        raise ValueError("release inference requires one physical SM61 GPU")
    uuid, sm, driver = [x.strip() for x in rows[0].split(",")]
    destination = proofs / "release-r1"
    destination.mkdir(exist_ok=True)
    report = {"schema": GPU_SCHEMA, "release": manifest, "run_id": kernel["run_id"],
              "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "gpu_uuid": uuid, "sm": sm, "driver": driver,
              "kernel_report_sha256": hash_file(proofs / "report.json"),
              "image_proof_sha256": hash_file(Path(image_directory) / "oci.json"),
              "harness_sha256": hash_file(q.ROOT / "qualification.py"),
              "policy_sha256": hash_file(q.ROOT / "models.json"), "models": {}, "full_models_executed": False}
    for mode in q.MODES:
        pair, errors = {}, {}
        for device in ("cpu", "cuda"):
            name = mode + "-" + device
            print("RELEASE_MODEL_EXECUTE=" + name, flush=True)
            try:
                pair[device] = q.infer(Path(directory) / "mistralrs", mode, device, paths, policy, destination / (name + ".log"))
                q.write(destination / (name + ".json"), pair[device])
            except q.MODEL_ERRORS as error:
                errors[device] = str(error)
        result = {"status": "ERROR", "errors": errors, "weights": policy["models"][mode]}
        if len(pair) == 2:
            try:
                result.update(q.measure(pair["cpu"], pair["cuda"], policy))
            except q.MODEL_ERRORS as error:
                errors["references"] = str(error)
        report["models"][mode] = result
        q.write(destination / "report.json", report)
    complete = all(not x["errors"] for x in report["models"].values())
    paged_complete = True
    if json.loads(Path(plan_file).read_text())["config"].get("fp8_kv_qualification"):
        benchmark = {"status": "ERROR", "model": "text", "weights": policy["models"]["text"],
                     "binary_sha256": manifest["binary_sha256"], "cases": {}, "errors": {},
                     "performance_threshold": None, "numeric_acceptance": "separate unchanged Text/VL/GGUF policy"}
        for cache in ("auto", "f8e4m3"):
            print("RELEASE_PAGED_KV_EXECUTE=" + cache, flush=True)
            try:
                observation = q.infer(Path(directory) / "mistralrs", "text", "cuda", paths,
                                      {**policy, "max_tokens": PAGED_BENCH_TOKENS},
                                      destination / ("paged-kv-" + cache + ".log"), paged_cache_type=cache)
                q.write(destination / ("paged-kv-" + cache + ".json"), observation)
                benchmark["cases"][cache] = paged_summary(observation, cache, q)
            except q.MODEL_ERRORS as error:
                benchmark["errors"][cache] = str(error)
        paged_complete = not benchmark["errors"] and len(benchmark["cases"]) == 2
        if paged_complete:
            benchmark["status"] = "PASS"
            benchmark["request_speed_ratio_fp16_over_fp8"] = (benchmark["cases"]["auto"]["median_request_seconds"]
                                                               / benchmark["cases"]["f8e4m3"]["median_request_seconds"])
        report["paged_fp8"] = benchmark
    passed = complete and all(x["status"] == "PASS" for x in report["models"].values())
    report.update(full_models_executed=complete, status="PASS" if passed and paged_complete else "COLLECTED" if complete and paged_complete else "ERROR")
    report["files"] = {p.name: hash_file(p) for p in destination.iterdir() if p.name != "report.json"}
    q.write(destination / "report.json", report)
    summary = "\nRelease binary SHA256: " + manifest["binary_sha256"] + "\n\n"
    summary += "| Release model | Numeric verdict | Max logprob error |\n| --- | --- | --- |\n"
    summary += "".join(f"| {mode} | {case['status']} | {case.get('max_logprob_error', 'unavailable')} |\n" for mode, case in report["models"].items())
    summary += "\nCollection is not numeric acceptance. Publication requires every release-model comparison to pass.\n"
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream: stream.write(summary)
    print(summary)
    if not complete:
        raise ValueError("release model execution incomplete; all three models were attempted")
    if not paged_complete:
        raise ValueError("real Text FP16/FP8 paged KV benchmark incomplete; both cache modes were attempted")
    return report


def paged_summary(observation, cache, q):
    measured = observation["paged_kv"]
    command = measured["command"]
    if (observation.get("device") != "cuda" or measured["cache_type"] != cache
            or measured["warmup"] != q.KV_BENCH_WARMUP
            or observation["request"]["max_tokens"] != PAGED_BENCH_TOKENS
            or any(command[command.index(flag) + 1] != value for flag, value in (
                ("--paged-attn", "on"), ("--pa-cache-type", cache), ("--kv-cache", "2048tokens"),
                ("--prefix-cache-n", "0"), ("--max-num-batched-tokens", "128"),
                ("--max-prefill-chunk-tokens", "128")))):
        raise ValueError("paged KV benchmark command or protocol mismatch")
    samples = measured["samples"]
    timings = [x["wall_seconds"] for x in samples]
    if (len(timings) != q.KV_BENCH_REPEATS or any(not math.isfinite(x) or x <= 0 for x in timings)
            or any(x["response"]["usage"].get(name, 0) < 1 for x in samples
                   for name in ("prompt_tokens", "completion_tokens"))):
        raise ValueError("missing real-model paged KV timing samples")
    return {"median_request_seconds": statistics.median(timings),
            "completion_tokens": [x["response"]["usage"]["completion_tokens"] for x in samples],
            "prompt_tokens": [x["response"]["usage"]["prompt_tokens"] for x in samples]}


def verify_gpu(plan_file, directory, proofs, image_directory):
    manifest = inspect(plan_file, directory)
    image_record(plan_file, directory, image_directory)
    proofs = Path(proofs)
    destination = proofs / "release-r1"
    report = json.loads((destination / "report.json").read_text())
    kernel = json.loads((proofs / "report.json").read_text())
    q = model_module()
    policy = q.read(q.ROOT / "models.json")
    if (policy["image"] != manifest["image"] or report.get("schema") != GPU_SCHEMA or report.get("release") != manifest
            or report.get("status") != "PASS" or report.get("full_models_executed") is not True
            or report.get("run_id") != kernel.get("run_id") or kernel.get("candidate") != manifest["candidate"]
            or kernel.get("status") != "GPU_PASSED" or kernel.get("asd_required") is not True
            or kernel.get("gpu_executed") is not True or kernel.get("target") != "sm_61"
            or report.get("kernel_report_sha256") != hash_file(proofs / "report.json")
            or report.get("image_proof_sha256") != hash_file(Path(image_directory) / "oci.json")
            or report.get("harness_sha256") != hash_file(q.ROOT / "qualification.py")
            or report.get("policy_sha256") != hash_file(q.ROOT / "models.json")
            or report.get("sm") != "6.1" or not report.get("gpu_uuid") or not report.get("driver")
            or set(report.get("models", {})) != set(q.MODES)):
        raise ValueError("accepted release-specific model and image qualification is missing")
    mandatory = {f"{m}-{d}.json" for m in q.MODES for d in ("cpu", "cuda")}
    if json.loads(Path(plan_file).read_text())["config"].get("fp8_kv_qualification"):
        benchmark = report.get("paged_fp8", {})
        if (benchmark.get("status") != "PASS" or benchmark.get("errors")
                or benchmark.get("binary_sha256") != manifest["binary_sha256"]):
            raise ValueError("real Text FP16/FP8 paged KV benchmark is missing")
        mandatory.update({"paged-kv-auto.json", "paged-kv-f8e4m3.json"})
    if not mandatory.issubset(report.get("files", {})):
        raise ValueError("raw release CPU/CUDA references are missing")
    for name, expected_hash in report["files"].items():
        if Path(name).name != name or not DIGEST.fullmatch(expected_hash) or hash_file(destination / name) != expected_hash:
            raise ValueError("release model evidence SHA256 mismatch")
    if json.loads(Path(plan_file).read_text())["config"].get("fp8_kv_qualification"):
        benchmark = report["paged_fp8"]
        for cache in ("auto", "f8e4m3"):
            actual = paged_summary(q.read(destination / ("paged-kv-" + cache + ".json")), cache, q)
            if benchmark.get("cases", {}).get(cache) != actual:
                raise ValueError("paged KV summary differs from raw measurements")
        ratio = benchmark["cases"]["auto"]["median_request_seconds"] / benchmark["cases"]["f8e4m3"]["median_request_seconds"]
        if benchmark.get("request_speed_ratio_fp16_over_fp8") != ratio:
            raise ValueError("paged KV speed ratio differs from raw measurements")
    for mode in q.MODES:
        case = report["models"][mode]
        numeric = q.compare(q.read(destination / f"{mode}-cpu.json"), q.read(destination / f"{mode}-cuda.json"), policy)
        if case.get("errors") or case.get("weights") != policy["models"][mode] or any(case.get(k) != v for k, v in numeric.items()):
            raise ValueError("release numeric evidence does not match the raw references")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "verify", "gpu", "verify-gpu"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--proofs", type=Path)
    parser.add_argument("--image-dir", type=Path)
    args = parser.parse_args()
    if args.command in ("gpu", "verify-gpu"):
        if not args.proofs or not args.image_dir: parser.error("--proofs and --image-dir are required")
        (collect_gpu if args.command == "gpu" else verify_gpu)(args.plan, args.dir, args.proofs, args.image_dir)
        return
    if args.command == "record":
        plan = json.loads(args.plan.read_text())
        binary = args.dir / "mistralrs"
        if not binary.is_file():
            raise ValueError("missing release binary")
        details = {**expected(plan), "binary_sha256": hash_file(binary),
                   "binary_bytes": binary.stat().st_size}
        (args.dir / "release.json").write_text(json.dumps(details, sort_keys=True, indent=2) + "\n")
    manifest = inspect(args.plan, args.dir)
    print(f"EXTENDED_RELEASE_R1=PASS candidate={manifest['candidate']} binary_sha256={manifest['binary_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("EXTENDED_RELEASE_R1_REJECT=" + str(error))

