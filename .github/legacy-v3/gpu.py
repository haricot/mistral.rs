import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import campaign as c


def run_archived_tests(builds, report, workspace):
    report.mkdir(parents=True, exist_ok=True)
    target = report / "target"
    target.mkdir(parents=True, exist_ok=True)
    config = report / "nextest.toml"
    text = (c.ROOT / "nextest.toml").read_text().replace('junit.path = "gpu-junit.xml"', "junit.path = " + json.dumps(str((report / "gpu-junit.xml").resolve())))
    config.write_text(text)
    log = report / "gpu-tests.log"
    with log.open("w") as stream:
        result = subprocess.run([str(builds / "cargo-nextest"), "nextest", "run", "--archive-file", str(builds / "tests.tar.zst"),
                                 "--workspace-remap", str(workspace), "--extract-to", str(target),
                                 "--config-file", str(config), "--profile", "gpu", "--no-tests=fail"],
                                stdout=stream, stderr=subprocess.STDOUT)
    print(log.read_text(), end="", flush=True)
    result.check_returncode()
    return c.junit(report / "gpu-junit.xml", c.GPU_REQUIRED)


def execute(prep, builds, report, workspace):
    plan = c.verify_plan(prep)
    build = c.verify_build(plan, builds)
    required = {"EXPECTED_RUN_ID": os.environ["GITHUB_RUN_ID"], "RUNNER_REPO": c.REPOSITORY,
                "RUNNER_NO_DEFAULT_LABELS": "true", "EXPECTED_CANDIDATE_SHA": plan["candidate"]}
    if any(os.environ.get(name) != value for name, value in required.items()):
        raise ValueError("runner run/repository/candidate isolation mismatch")
    if plan["identity"]["cpu_profile"] == "avx2":
        flags = set(Path("/proc/cpuinfo").read_text().split())
        if not {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe", "xsave"} <= flags or not ({"abm", "lzcnt"} & flags):
            raise ValueError("host does not support x86-64-v3")
    rows = list(csv.reader(c.command("nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader").splitlines(), skipinitialspace=True))
    if not rows or any(row[2].strip() != "6.1" for row in rows):
        raise ValueError("physical Pascal SM61 required")
    report.mkdir(parents=True, exist_ok=True)
    c.git(workspace, "fetch", "--no-tags", prep / "candidate.bundle", plan["candidate_ref"])
    c.git(workspace, "switch", "--detach", "--force", plan["candidate"])
    if c.git(workspace, "rev-parse", "HEAD^{tree}") != plan["identity"]["tree"]:
        raise ValueError("executed source tree mismatch")
    for name in ("cargo-nextest", "mistralrs"):
        (builds / name).chmod(0o755)
    tests = run_archived_tests(builds, report, workspace)
    loader = importlib.util.spec_from_file_location("model_measurements", c.ROOT.parent / "lfm3/qualification.py")
    models = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(models)
    policy = c.read(c.ROOT / "models.json")
    paths = models.model_paths(policy)
    results = {}
    for mode in models.MODES:
        pair = {}
        result = results[mode] = {"status": "ERROR", "cpu_executed": False, "cuda_executed": False,
                                  "errors": {}, "weights": policy["models"][mode], "elapsed_seconds": {}}
        for device in ("cpu", "cuda"):
            started = time.monotonic()
            try:
                pair[device] = models.infer(builds / "mistralrs", mode, device, paths, policy, report / f"{mode}-{device}.log")
                if not pair[device].get("response", {}).get("choices"):
                    raise RuntimeError("model response contains no inference result")
                c.write(report / f"{mode}-{device}.json", pair[device])
                result[device + "_executed"] = True
            except models.MODEL_ERRORS as error:
                result["errors"][device] = str(error)
            result["elapsed_seconds"][device] = time.monotonic() - started
        if len(pair) == 2:
            result.update(c.measure_pair(models, pair["cpu"], pair["cuda"], policy))
        c.write(report / "models.json", results)
    complete = all(c.numeric_accepted(mode, result) for mode, result in results.items())
    proof = {"schema": 1, "status": "GPU_AND_MODELS_PASSED" if complete else "FAILED",
             "key": plan["key"], "identity": plan["identity"], "origin": c.origin(),
             "compute_capability": "6.1", "gpu": rows, "tests": tests, "models": results,
             "binary_sha256": build["files"]["mistralrs"], "archive_sha256": build["files"]["tests.tar.zst"],
             "full_models_executed": all(r["cpu_executed"] and r["cuda_executed"] and not r["errors"] for r in results.values()),
             "candle_scope": "consumed BF16/FP8/FP4 paths qualified through this exact integrated binary"}
    proof["files"] = {p.name: c.sha(p) for p in report.iterdir() if p.is_file() and p.name != "gpu.json"}
    c.write(report / "gpu.json", proof)
    c.verify_gpu(plan, report, build)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--builds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(args.preparation.resolve(), args.builds.resolve(), args.output.resolve(), Path.cwd())
