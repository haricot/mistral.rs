#!/usr/bin/env python3
"""Compare qualified DKR catalogue loading modes and the static archive."""
import argparse
import contextlib
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
from dkr_r123_s1a_catalog import verify_catalogue_copy
from dkr_r122_s6_evidence import native_audit

RUNTIME_SOURCES = (
    "docs/DKR_BUILD_EMBEDDED_CATALOGUES.md",
    "Cargo.toml", "Cargo.lock", "mistralrs-kernel-runtime/Cargo.toml", "mistralrs-kernel-runtime/build.rs",
    "mistralrs-kernel-runtime/src/lib.rs", "mistralrs-kernel-runtime/src/embedded.rs",
    "mistralrs-kernel-runtime/src/universal.rs", "mistralrs-kernel-runtime/src/universal_cuda.rs",
    "mistralrs-core/src/cuda/kernel_runtime.rs", "mistralrs-quant/src/dkr_cuda.rs",
    "mistralrs-paged-attn/src/cuda/backend/kv_cache_ops_dynamic.rs",
    "mistralrs-paged-attn/src/cuda/backend/kv_gather_dynamic.rs",
    "mistralrs-paged-attn/src/cuda/backend/kv_scales_dynamic.rs",
    "mistralrs-paged-attn/src/cuda/backend/paged_decode_dynamic.rs",
    "mistralrs-paged-attn/src/cuda/backend/paged_fp8_dynamic.rs",
    "mistralrs-paged-attn/src/cuda/backend/paged_v2_dynamic.rs",
)
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


def build_candidate(original, base, evidence, *, embedded=False):
    revision, _, _, native, typed, seal, *_ = evidence
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
    env.pop("DKR_EMBED_NATIVE_DIR", None)
    env.pop("DKR_EMBED_TYPED_DIR", None)
    inputs = None
    if embedded:
        inputs = base / "inputs"
        if not inputs.exists():
            with tempfile.TemporaryDirectory(prefix="inputs-", dir=base) as temp:
                staged = Path(temp) / "inputs"
                shutil.copytree(native, staged / "native")
                shutil.copytree(typed, staged / "typed")
                staged.rename(inputs)
        verify_catalogue_copy(inputs / "typed", seal["typed_manifest_sha256"])
        native_audit(inputs / "native")
        need(tree_hashes(native) == tree_hashes(inputs / "native"), "native build input differs from archive")
        env.update(DKR_EMBED_NATIVE_DIR=str(inputs / "native"), DKR_EMBED_TYPED_DIR=str(inputs / "typed"))
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
        "embedded_inputs": str(inputs) if inputs else None,
    })
    stage.rename(base / "build")
    print(f"DKR_R123_S1A_SHARED_BUILD=PASS binary={base / 'build/candidate-bin'} log={base / 'build/build.log'}")


def catalogue_validations(log):
    return [int(line.removeprefix(CATALOGUE_MARKER)) for line in log.splitlines()
            if line.startswith(CATALOGUE_MARKER)]


def tree_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()}


@contextlib.contextmanager
def unavailable_inputs(inputs, hide):
    if not hide:
        yield
        return
    hidden = inputs.with_name("inputs-unavailable")
    inputs.rename(hidden)
    try:
        yield
    finally:
        hidden.rename(inputs)


def compare_trials(timed, *, embedded=False):
    variants = ("static", "filesystem", "embedded") if embedded else tuple(PROVIDERS)
    order = (*variants, *variants[::-1])
    need(tuple(trial["variant"] for trial in timed) == order, "unexpected timing order")
    phases = {variant: [t["phase_seconds"] for t in timed if t["variant"] == variant] for variant in variants}
    reference, candidate = variants[1:]
    return {
        f"{reference}_to_{candidate}": compare_phases(phases[reference], phases[candidate], reference, candidate),
        f"static_to_{candidate}": compare_phases(phases["static"], phases[candidate], "static", candidate),
    }


def benchmark(base, s0_report, evidence, *, embedded=False):
    revision, modelrun, prior, native, typed, seal, _, model, refs, binaries = evidence
    candidate, build = prepared_candidate(base, revision)
    variants = ("static", "filesystem", "embedded") if embedded else tuple(PROVIDERS)
    order = (*variants, *variants[::-1])
    inputs = Path(build["embedded_inputs"]) if embedded else None
    if inputs:
        hidden = inputs.with_name("inputs-unavailable")
        if hidden.exists() and not inputs.exists():
            hidden.rename(inputs)
        verify_catalogue_copy(inputs / "typed", seal["typed_manifest_sha256"])
        need(tree_hashes(native) == tree_hashes(inputs / "native"), "native copied input changed")
    s0_digest = sha(s0_report)
    need(len(seal["typed_manifest_sha256"]) == FULL_COUNT, "unexpected sealed catalogue size")
    index = typed_index(typed)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    print(f"DKR_R123_S1A_SHARED_RUN={run}", flush=True)

    def run_trial(stem, variant, telemetry=False):
        need(not s1a.compute_apps(), f"other GPU compute process active before {stem}")
        provider = "A" if variant == "static" else "B"
        binary = binaries["A"] if provider == "A" else candidate
        dest, log = run / f"{stem}.json", run / f"{stem}.log"
        env = s0.environment(os.environ, model, prior, native, typed, True, provider, dest)
        if provider == "B":
            env[SHARING_ENV] = "0" if variant == "independent" else "1"
            env["MISTRALRS_DKR_ARTIFACT_SOURCE"] = variant if embedded else "filesystem"
            if embedded:
                env.update(CANDLE_DKR_S31_DIR=str(inputs / "native"), DKR_KERNEL_PACKAGES=str(inputs / "typed"))
        if telemetry:
            env.update(DKR_TELEMETRY="1", CANDLE_DKR_TRACE="1")
        before = s0.gpu()
        with unavailable_inputs(inputs, embedded and variant == "embedded"):
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
            trial["embedded_catalogue_loads"] = [
                int(line.split("manifests=", 1)[1]) for line in content.splitlines()
                if line.startswith("MISTRAL_DKR_CATALOG_EMBEDDED manifests=")
            ]
            trial["native_catalogues"] = [line for line in content.splitlines()
                                         if line.startswith("CANDLE_DKR_NATIVE_CATALOG ")]
        else:
            phases = phase_durations(start, finish, markers, data["load_seconds"])
            phases["first_raw_to_exit"] = phases["process"] - phases["before_first_raw"]
            phases["first_generations_short_long"] = phases["short.generation=1"] + phases["long.generation=1"]
            trial.update(phase_seconds=phases, generation_metrics=metrics)
        trial["embedded_inputs_unavailable"] = embedded and variant == "embedded"
        s1a.write_checkpoint(run / f"{stem}.metrics.json", trial)
        print(f"{stem} exact=True", flush=True)
        return trial

    timed = [run_trial(f"{i:02d}-{variant}", variant) for i, variant in enumerate(order)]
    audited = {variant: run_trial(f"{variant}-telemetry", variant, telemetry=True) for variant in variants}
    validation = {variant: t["catalogue_validations"] for variant, t in audited.items()}
    validation_counts = {variant: len(value) for variant, value in validation.items()}
    if embedded:
        need(validation["static"] == [] and validation["filesystem"] == [FULL_COUNT]
             and validation["embedded"] == [] and audited["embedded"]["embedded_catalogue_loads"] == [FULL_COUNT],
             f"unexpected embedded catalogue evidence {validation}")
        for variant in variants[1:]:
            rows = audited[variant]["native_catalogues"]
            need(rows and all(line == f"CANDLE_DKR_NATIVE_CATALOG source={variant} manifests=25" for line in rows),
                 f"unexpected native catalogue evidence {variant}: {rows}")
    else:
        need(validation["static"] == [] and validation["shared"] == [FULL_COUNT]
             and len(validation["independent"]) > 1 and all(n == FULL_COUNT for n in validation["independent"]),
             f"unexpected catalogue validations {validation}; inspect telemetry logs in {run}")
    before = route_counts(audited[variants[1]]["runtime_counters"])
    after = route_counts(audited[variants[2]]["runtime_counters"])
    archived = route_counts(merge((modelrun / "external-paged.log").read_text().splitlines()))
    need(before == after, "external route identity or submission count changed between catalogue modes")
    need(before.keys() == archived.keys(), "candidate external routes differ from the qualified paged archive")
    need(s0.model_gate(modelrun, prior, native) == seal, "qualified archive changed during the comparison")
    need(sha(s0_report) == s0_digest and sha(candidate) == build["binary_sha256"],
         "comparison input changed during the benchmark")
    if embedded:
        verify_catalogue_copy(inputs / "typed", seal["typed_manifest_sha256"])
        need(tree_hashes(native) == tree_hashes(inputs / "native"), "native build input changed during benchmark")

    comparison = compare_trials(timed, embedded=embedded)
    dest = run / ("embedded-catalogue-comparison.json" if embedded else "shared-catalogue-comparison.json")
    s1a.write_checkpoint(dest, {
        "status": ("EMBEDDED_CATALOGUE_COMPARISON_COMPLETE" if embedded else "SHARED_CATALOGUE_COMPARISON_COMPLETE"),
        "revision": revision, "layout": "paged",
        "s0_report_sha256": s0_digest, "candidate_build": build, "static_binary_sha256": sha(binaries["A"]),
        "model_seal": seal, "numeric_exact_vs_sealed": True, "same_external_routes_and_counts": True,
        "catalogue_validations": validation, "trials": timed, "telemetry": audited, "comparison": comparison,
        "timing_method": "fresh processes; phase boundaries observed on the archive's existing stderr markers",
        "performance_promotion_approved": False,
    })
    reference, label = variants[1:]
    for phase in REPORTED_PHASES:
        ab = comparison[f"{reference}_to_{label}"][phase]
        ac = comparison[f"static_to_{label}"][phase]
        print(f"{phase} static_ms={ac['static_median_ms']:.3f} {reference}_ms={ab[f'{reference}_median_ms']:.3f} "
              f"{label}_ms={ab[f'{label}_median_ms']:.3f} saved_ms={ab['saved_ms']:.3f}")
    print(f"DKR_R123_S1A_SHARED=COMPLETE validations={validation_counts}")
    print(f"DKR_R123_S1A_SHARED_REPORT={dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--s0-report", required=True, type=Path)
    parser.add_argument("--build", action="store_true", help="prepare just the candidate binary for the benchmark")
    parser.add_argument("--embedded", action="store_true", help="embed qualified PTX and compare with filesystem loading")
    args = parser.parse_args()
    original = args.original.resolve()
    s0_report = args.s0_report.resolve()
    with (original / "target/dkr-r122-s684/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = s1a.prep(original, s0_report, allowed_source_changes=RUNTIME_SOURCES)
        revision = evidence[0]
        s0.gpu()
        name = "dkr-r123-s1a-embedded" if args.embedded else "dkr-r123-s1a-shared-catalog"
        base = original / "target" / name / revision[:12]
        base.mkdir(parents=True, exist_ok=True)
        if args.build:
            build_candidate(original, base, evidence, embedded=args.embedded)
        else:
            benchmark(base, s0_report, evidence, embedded=args.embedded)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        print(f"DKR_R123_S1A_SHARED=REVIEW_REQUIRED {error}", file=sys.stderr)
        sys.exit(4)
