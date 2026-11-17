#!/usr/bin/env python3
"""Compare a shared typed DKR catalogue with independent discovery and the static archive."""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import dkr_r123_s0 as s0
import dkr_r123_s1a as s1a
from dkr_r122_s5_evidence import merge
from dkr_r122_s683_audit import need, sha
from dkr_r122_s684b_gate import trace, typed_index
from dkr_r123_s1a_catalog import FULL_COUNT, compare_phases, observe, phase_durations, route_counts

RUNTIME_SOURCES = ("mistralrs-kernel-runtime/src/universal.rs", "mistralrs-kernel-runtime/src/universal_cuda.rs")
EXAMPLE = "dkr_s684_model"
FEATURES = "dkr-sm61-experimental,mistralrs-core/dkr-fp4-dequant"
SHARING_ENV = "MISTRALRS_DKR_SHARED_CATALOG"
CATALOGUE_MARKER = "MISTRAL_DKR_CATALOG_VALIDATED manifests="
ORDER = ("static", "independent", "shared", "shared", "independent", "static")
PROVIDERS = {"static": "A", "independent": "B", "shared": "B"}
REPORTED_PHASES = (
    "process", "model_load", "first_raw_to_exit", "raw_total",
    "short.generation=1", "long.generation=1", "first_generations_short_long", "long.pass_to_exit",
)


def prepared_candidate(base, revision):
    saved = base / "build"
    binary = saved / "candidate-bin"
    report_path = saved / "build.json"
    need(report_path.is_file(), "candidate build absent; run with --build")
    report = s0.read(report_path)
    need(report["revision"] == revision and report["features"] == FEATURES
         and os.access(binary, os.X_OK) and sha(binary) == report["binary_sha256"],
         f"candidate build evidence differs: {report_path}")
    return binary, report


def build_candidate(original, base, revision):
    if (base / "build").exists():
        binary, _ = prepared_candidate(base, revision)
        print(f"DKR_R123_S1A_SHARED_BUILD=REUSED binary={binary}")
        return
    stage = Path(tempfile.mkdtemp(prefix="build-", dir=base))
    log = stage / "build.log"
    root = Path(__file__).resolve().parents[1]
    target = original / "target/dkr-r122-s6-cargo"
    cache = original / "target/dkr-cuda-cache/sm61"
    env = dict(os.environ, CARGO_TARGET_DIR=str(target), MISTRALRS_CUDA_BUILD_ROOT=str(cache))
    env.pop("CUDA_COMPUTE_CAP", None)
    command = ["cargo", "build", "--locked", "-p", "mistralrs", "--example", EXAMPLE, "--features", FEATURES]
    print(f"DKR_R123_S1A_SHARED_BUILD_LOG={log}", flush=True)
    with log.open("xb") as output:
        with subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as cargo:
            for line in cargo.stdout:
                output.write(line)
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
            rc = cargo.wait()
    need(rc == 0, f"candidate build failed (exit={rc}); inspect {log}")
    binary = stage / "candidate-bin"
    shutil.copy2(target / "debug/examples" / EXAMPLE, binary)
    s1a.write_checkpoint(stage / "build.json", {
        "revision": revision, "binary_sha256": sha(binary), "features": FEATURES,
        "command": command, "cuda_build_root": str(cache),
    })
    stage.rename(base / "build")
    print(f"DKR_R123_S1A_SHARED_BUILD=PASS binary={base / 'build/candidate-bin'} log={base / 'build/build.log'}")


def catalogue_validations(log):
    return [int(line.removeprefix(CATALOGUE_MARKER)) for line in log.splitlines()
            if line.startswith(CATALOGUE_MARKER)]


def compare_trials(timed):
    need(tuple(trial["variant"] for trial in timed) == ORDER, "unexpected timing order")
    phases = {variant: [t["phase_seconds"] for t in timed if t["variant"] == variant] for variant in PROVIDERS}
    return {
        "independent_to_shared": compare_phases(phases["independent"], phases["shared"], "independent", "shared"),
        "static_to_shared": compare_phases(phases["static"], phases["shared"], "static", "shared"),
    }


def benchmark(base, s0_report, evidence):
    revision, modelrun, prior, native, typed, seal, _, model, refs, binaries = evidence
    candidate, build = prepared_candidate(base, revision)
    s0_digest = sha(s0_report)
    need(len(seal["typed_manifest_sha256"]) == FULL_COUNT, "unexpected sealed catalogue size")
    index = typed_index(typed)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    print(f"DKR_R123_S1A_SHARED_RUN={run}", flush=True)

    def run_trial(stem, variant, telemetry=False):
        need(not s1a.compute_apps(), f"other GPU compute process active before {stem}")
        provider = PROVIDERS[variant]
        binary = binaries["A"] if provider == "A" else candidate
        dest, log = run / f"{stem}.json", run / f"{stem}.log"
        env = s0.environment(os.environ, model, prior, native, typed, True, provider, dest)
        if provider == "B":
            env[SHARING_ENV] = "1" if variant == "shared" else "0"
        if telemetry:
            env.update(DKR_TELEMETRY="1", CANDLE_DKR_TRACE="1")
        before = s0.gpu()
        rc, start, finish, markers = observe(binary, env, log)
        (run / f"{stem}.exit").write_text(f"{rc}\n")
        need(rc == 0, f"{stem} failed (exit={rc}); inspect {log}")
        after = s0.gpu()
        data = s0.read(dest)
        ref = refs["paged"][provider]
        need(data["mode"] == ref["mode"] and data["model"] == model and data["paged_requested"] is True,
             f"provider/model mismatch {stem}")
        need(s0.exact({**ref, "mode": "legacy"}, {**data, "mode": "external"}), f"numerical mismatch {stem}")
        trial = {
            "stem": stem, "variant": variant, "parity_exact": True,
            "gpu_before": before, "gpu_after": after, "json_sha256": sha(dest), "log_sha256": sha(log),
        }
        metrics = s0.extract(data)
        if telemetry:
            trial["telemetry"] = trace(run, stem, index, native, provider == "A")
            content = log.read_text()
            trial["runtime_counters"] = merge(content.splitlines())
            trial["catalogue_validations"] = catalogue_validations(content)
        else:
            phases = phase_durations(start, finish, markers, data["load_seconds"])
            phases["first_raw_to_exit"] = phases["process"] - phases["before_first_raw"]
            phases["first_generations_short_long"] = phases["short.generation=1"] + phases["long.generation=1"]
            trial.update(phase_seconds=phases, generation_metrics=metrics)
        s1a.write_checkpoint(run / f"{stem}.metrics.json", trial)
        print(f"{stem} exact=True", flush=True)
        return trial

    timed = [run_trial(f"{i:02d}-{variant}", variant) for i, variant in enumerate(ORDER)]
    audited = {variant: run_trial(f"{variant}-telemetry", variant, telemetry=True) for variant in PROVIDERS}
    validation = {variant: t["catalogue_validations"] for variant, t in audited.items()}
    validation_counts = {variant: len(value) for variant, value in validation.items()}
    need(validation["static"] == [] and validation["shared"] == [FULL_COUNT]
         and len(validation["independent"]) > 1 and all(n == FULL_COUNT for n in validation["independent"]),
         f"unexpected catalogue validations {validation}; inspect telemetry logs in {run}")
    before = route_counts(audited["independent"]["runtime_counters"])
    after = route_counts(audited["shared"]["runtime_counters"])
    archived = route_counts(merge((modelrun / "external-paged.log").read_text().splitlines()))
    need(before == after, "external route identity or submission count changed between catalogue modes")
    need(before.keys() == archived.keys(), "candidate external routes differ from the qualified paged archive")
    need(s0.model_gate(modelrun, prior, native) == seal, "qualified archive changed during the comparison")
    need(sha(s0_report) == s0_digest and sha(candidate) == build["binary_sha256"],
         "comparison input changed during the benchmark")

    comparison = compare_trials(timed)
    dest = run / "shared-catalogue-comparison.json"
    s1a.write_checkpoint(dest, {
        "status": "SHARED_CATALOGUE_COMPARISON_COMPLETE", "revision": revision, "layout": "paged",
        "s0_report_sha256": s0_digest, "candidate_build": build, "static_binary_sha256": sha(binaries["A"]),
        "model_seal": seal, "numeric_exact_vs_sealed": True, "same_external_routes_and_counts": True,
        "catalogue_validations": validation, "trials": timed, "telemetry": audited, "comparison": comparison,
        "timing_method": "fresh processes; phase boundaries observed on the archive's existing stderr markers",
        "performance_promotion_approved": False,
    })
    for phase in REPORTED_PHASES:
        ab = comparison["independent_to_shared"][phase]
        ac = comparison["static_to_shared"][phase]
        print(f"{phase} static_ms={ac['static_median_ms']:.3f} independent_ms={ab['independent_median_ms']:.3f} "
              f"shared_ms={ab['shared_median_ms']:.3f} shared_saved_ms={ab['saved_ms']:.3f}")
    print(f"DKR_R123_S1A_SHARED=COMPLETE validations={validation_counts}")
    print(f"DKR_R123_S1A_SHARED_REPORT={dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--s0-report", required=True, type=Path)
    parser.add_argument("--build", action="store_true", help="prepare just the candidate binary for the benchmark")
    args = parser.parse_args()
    original = args.original.resolve()
    s0_report = args.s0_report.resolve()
    with (original / "target/dkr-r122-s684/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = s1a.prep(original, s0_report, allowed_source_changes=RUNTIME_SOURCES)
        revision = evidence[0]
        s0.gpu()
        base = original / "target/dkr-r123-s1a-shared-catalog" / revision[:12]
        base.mkdir(parents=True, exist_ok=True)
        if args.build:
            build_candidate(original, base, revision)
        else:
            benchmark(base, s0_report, evidence)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        print(f"DKR_R123_S1A_SHARED=REVIEW_REQUIRED {error}", file=sys.stderr)
        sys.exit(4)
