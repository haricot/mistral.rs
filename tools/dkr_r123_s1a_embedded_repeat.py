#!/usr/bin/env python3
"""Repeat static versus embedded startup using an existing sealed candidate."""
import argparse
import fcntl
import math
import os
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

import dkr_r123_s1a_shared_catalog as shared

s0, s1a = shared.s0, shared.s1a
need, sha = shared.need, shared.sha
ORDER = ("static", "embedded", "embedded", "static")
VARIANTS = ("static", "embedded")
STABILITY_PHASES = ("process", "model_load", "before_first_raw", "first_raw_to_exit")
MIN_ROUNDS = 3
DEFAULT_MAX_DRIFT_PCT = 5.0
REPORT_NAME = "embedded-startup-repeat.json"
SOURCE_EXCLUSIONS = (":!tools/**", ":!docs/DKR_BUILD_EMBEDDED_CATALOGUES.md")


def compare(trials, max_drift_pct):
    need(len(trials) >= MIN_ROUNDS * len(ORDER) and len(trials) % len(ORDER) == 0,
         "at least three complete ABBA rounds required")
    need(tuple(t["variant"] for t in trials) == ORDER * (len(trials) // len(ORDER)), "unexpected trial order")
    phases = tuple(trials[0]["phase_seconds"])
    need(all(set(t["phase_seconds"]) == set(phases) for t in trials), "phase inventory changed")
    spread = {}
    for variant in VARIANTS:
        spread[variant] = {}
        for phase in STABILITY_PHASES:
            values = [t["phase_seconds"][phase] * 1000 for t in trials if t["variant"] == variant]
            need(all(math.isfinite(v) and v > 0 for v in values), "invalid stability timing")
            spread[variant][phase] = {
                "min_ms": min(values), "max_ms": max(values), "median_ms": st.median(values),
                "spread_pct": 100 * (max(values) - min(values)) / st.median(values),
            }
    brackets, paired = [], {phase: [] for phase in phases}
    for offset in range(0, len(trials), len(ORDER)):
        a, b, c, d = trials[offset:offset + len(ORDER)]
        drift = {}
        for phase in STABILITY_PHASES:
            left, right = (t["phase_seconds"][phase] for t in (a, d))
            drift[phase] = 100 * abs(right - left) / ((left + right) / 2)
        brackets.append({"round": offset // len(ORDER), "static_drift_pct": drift})
        left_tick, right_tick = ((t["start_monotonic_ns"] + t["finish_monotonic_ns"]) / 2 for t in (a, d))
        for trial in (b, c):
            tick = (trial["start_monotonic_ns"] + trial["finish_monotonic_ns"]) / 2
            need(left_tick < tick < right_tick, "candidate outside static control bracket")
            weight = (tick - left_tick) / (right_tick - left_tick)
            for phase in phases:
                reference = (1 - weight) * a["phase_seconds"][phase] + weight * d["phase_seconds"][phase]
                candidate = trial["phase_seconds"][phase]
                need(math.isfinite(reference) and math.isfinite(candidate), "non-finite paired timing")
                paired[phase].append({
                    "stem": trial["stem"], "static_interpolated_ms": reference * 1000,
                    "embedded_ms": candidate * 1000, "delta_ms": (candidate - reference) * 1000,
                    "delta_pct": 100 * (candidate / reference - 1) if reference > 0 else None,
                })
    comparison = {}
    for phase, rows in paired.items():
        deltas = [r["delta_pct"] for r in rows if r["delta_pct"] is not None]
        comparison[phase] = {
            "static_interpolated_median_ms": st.median(r["static_interpolated_ms"] for r in rows),
            "embedded_median_ms": st.median(r["embedded_ms"] for r in rows),
            "paired_delta_median_ms": st.median(r["delta_ms"] for r in rows),
            "paired_delta_median_pct": st.median(deltas) if deltas else None,
            "paired_delta_min_pct": min(deltas) if deltas else None,
            "paired_delta_max_pct": max(deltas) if deltas else None,
            "pairs": rows,
        }
    reference_stable = all(v["spread_pct"] <= max_drift_pct for v in spread["static"].values())
    reference_stable = reference_stable and all(
        v <= max_drift_pct for b in brackets for v in b["static_drift_pct"].values())
    candidate_stable = all(v["spread_pct"] <= max_drift_pct for v in spread["embedded"].values())
    return {"timing_stable": reference_stable and candidate_stable,
            "reference_stable": reference_stable, "candidate_stable": candidate_stable,
            "max_drift_pct": max_drift_pct,
            "spread": spread, "brackets": brackets, "comparison": comparison}


def reuse_candidate(original, report_path, evidence):
    revision, _, _, _, _, seal, _, _, _, binaries = evidence
    previous = s0.read(report_path)
    need(previous["status"] == "EMBEDDED_CATALOGUE_COMPARISON_COMPLETE"
         and previous["numeric_exact_vs_sealed"] and previous["same_external_routes_and_counts"]
         and previous["layout"] == "paged" and previous["model_seal"] == seal,
         "previous embedded comparison does not match the qualified archive")
    candidate_revision = previous["revision"]
    need(re.fullmatch(r"[0-9a-f]{40}", candidate_revision), "invalid candidate revision")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", candidate_revision, revision,
                    "--", ".", *SOURCE_EXCLUSIONS], check=True)
    base = original / "target/dkr-r123-s1a-embedded" / candidate_revision[:12]
    need(report_path.parent.parent == base, "previous report must remain in its original run directory")
    candidate, build = shared.prepared_candidate(base, candidate_revision)
    need(build == previous["candidate_build"] and sha(binaries["A"]) == previous["static_binary_sha256"],
         "candidate or static binary differs from the completed comparison")
    need(previous["catalogue_validations"] == {"static": [], "filesystem": [shared.FULL_COUNT], "embedded": []},
         "unexpected previous catalogue validation evidence")
    for trial in [*previous["trials"], *previous["telemetry"].values()]:
        stem = trial["stem"]
        need(Path(stem).name == stem and trial["parity_exact"], "invalid previous trial")
        need(sha(report_path.parent / f"{stem}.json") == trial["json_sha256"]
             and sha(report_path.parent / f"{stem}.log") == trial["log_sha256"], "previous raw evidence changed")
    need(previous["telemetry"]["embedded"]["embedded_inputs_unavailable"], "previous inputs were not hidden")
    inputs = Path(build["embedded_inputs"])
    need(inputs == base / "inputs", "embedded input path differs from the candidate build directory")
    return candidate, build, inputs, previous


class Campaign:
    def __init__(self, evidence, candidate, inputs, run):
        (_, _, self.prior, self.native, self.typed, _, _, self.model, self.refs, self.binaries) = evidence
        self.candidate, self.inputs, self.run = candidate, inputs, run
        self.index = shared.typed_index(self.typed)

    def trial(self, stem, variant, kind):
        need(not s1a.compute_apps(), f"other GPU compute process active before {stem}")
        provider = "A" if variant == "static" else "B"
        binary = self.binaries["A"] if provider == "A" else self.candidate
        dest, log = self.run / f"{stem}.json", self.run / f"{stem}.log"
        env = s0.environment(os.environ, self.model, self.prior, self.native, self.typed, True, provider, dest)
        if variant == "embedded":
            env.update(MISTRALRS_DKR_ARTIFACT_SOURCE="embedded", MISTRALRS_DKR_SHARED_CATALOG="1",
                       CANDLE_DKR_S31_DIR=str(self.inputs / "native"), DKR_KERNEL_PACKAGES=str(self.inputs / "typed"))
        if kind == "telemetry":
            env.update(DKR_TELEMETRY="1", CANDLE_DKR_TRACE="1")
        before = s0.gpu()
        with shared.unavailable_inputs(self.inputs, variant == "embedded"):
            rc, start, finish, markers = shared.observe(binary, env, log)
        (self.run / f"{stem}.exit").write_text(f"{rc}\n")
        need(rc == 0, f"{stem} failed (exit={rc}); inspect {log}")
        after = s0.gpu()
        data, reference = s0.read(dest), self.refs["paged"][provider]
        need(data["mode"] == reference["mode"] and data["model"] == self.model and data["paged_requested"] is True,
             f"provider/model mismatch {stem}")
        need(s0.exact({**reference, "mode": "legacy"}, {**data, "mode": "external"}), f"numerical mismatch {stem}")
        trial = {"stem": stem, "variant": variant, "kind": kind, "parity_exact": True,
                 "gpu_before": before, "gpu_after": after, "json_sha256": sha(dest), "log_sha256": sha(log),
                 "start_monotonic_ns": start, "finish_monotonic_ns": finish,
                 "embedded_inputs_unavailable": variant == "embedded"}
        metrics = s0.extract(data)
        if kind == "telemetry":
            trial["telemetry"] = shared.trace(self.run, stem, self.index, self.native, provider == "A")
            content = log.read_text()
            trial["runtime_counters"] = shared.merge(content.splitlines())
            trial["catalogue_validations"] = shared.catalogue_validations(content)
            trial["embedded_catalogue_loads"] = [
                int(line.split("manifests=", 1)[1]) for line in content.splitlines()
                if line.startswith("MISTRAL_DKR_CATALOG_EMBEDDED manifests=")]
            trial["native_catalogues"] = [line for line in content.splitlines()
                                          if line.startswith("CANDLE_DKR_NATIVE_CATALOG ")]
        else:
            phases = shared.phase_durations(start, finish, markers, data["load_seconds"])
            phases["first_raw_to_exit"] = phases["process"] - phases["before_first_raw"]
            phases["first_generations_short_long"] = phases["short.generation=1"] + phases["long.generation=1"]
            trial.update(phase_seconds=phases, generation_metrics=metrics)
        s1a.write_checkpoint(self.run / f"{stem}.metrics.json", trial)
        print(f"{stem} exact=True", flush=True)
        return trial


def campaign(original, s0_report, report_path, rounds, max_drift_pct):
    evidence = s1a.prep(original, s0_report, allowed_source_changes=shared.RUNTIME_SOURCES)
    revision, modelrun, prior, native, _, seal, _, _, _, binaries = evidence
    candidate, build, inputs, previous = reuse_candidate(original, report_path, evidence)
    s0_digest, previous_digest = sha(s0_report), sha(report_path)
    need(previous["s0_report_sha256"] == s0_digest, "S0 report differs from the completed comparison")
    hidden = inputs.with_name("inputs-unavailable")
    if hidden.exists() and not inputs.exists():
        hidden.rename(inputs)

    def verify_inputs():
        need(len(seal["typed_manifest_sha256"]) == shared.FULL_COUNT, "unexpected typed catalogue size")
        shared.verify_catalogue_copy(inputs / "typed", seal["typed_manifest_sha256"])
        need(shared.tree_hashes(native) == shared.tree_hashes(inputs / "native"), "native build inputs changed")

    verify_inputs()
    base = original / "target/dkr-r123-s1a-embedded-repeat" / revision[:12]
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    runner = Campaign(evidence, candidate, inputs, run)
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT_RUN={run}", flush=True)
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT_BINARY=REUSED revision={build['revision']} sha256={sha(candidate)}", flush=True)
    warmups = [runner.trial(f"warmup-{v}", v, "warmup") for v in VARIANTS]
    trials = [runner.trial(f"{r:02d}-{i}-{v}", v, "timed") for r in range(rounds) for i, v in enumerate(ORDER)]
    audited = {v: runner.trial(f"{v}-telemetry", v, "telemetry") for v in VARIANTS}
    static, embedded = (audited[v] for v in VARIANTS)
    need(static["catalogue_validations"] == [] and embedded["catalogue_validations"] == []
         and embedded["embedded_catalogue_loads"] == [shared.FULL_COUNT]
         and embedded["native_catalogues"] == ["CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25"],
         "unexpected catalogue source evidence")
    routes = shared.route_counts(embedded["runtime_counters"])
    need(routes == shared.route_counts(previous["telemetry"]["embedded"]["runtime_counters"]),
         "embedded routes or submission counts changed from the completed comparison")
    archived = shared.route_counts(shared.merge((modelrun / "external-paged.log").read_text().splitlines()))
    need(routes.keys() == archived.keys(), "embedded route identity differs from the qualified archive")
    need(s0.model_gate(modelrun, prior, native) == seal, "qualified archive changed during repetition")
    need(sha(s0_report) == s0_digest and sha(report_path) == previous_digest
         and sha(candidate) == build["binary_sha256"] and sha(binaries["A"]) == previous["static_binary_sha256"],
         "comparison input changed during repetition")
    verify_inputs()
    analysis = compare(trials, max_drift_pct)
    stable = analysis["timing_stable"]
    status = "EMBEDDED_STARTUP_REPEAT_COMPLETE" if stable else "EMBEDDED_STARTUP_REPEAT_REVIEW_REQUIRED"
    dest = run / REPORT_NAME
    s1a.write_checkpoint(dest, {
        "status": status, "revision": revision, "layout": "paged", "rounds": rounds, "order": list(ORDER),
        "s0_report_sha256": s0_digest, "previous_report": str(report_path), "previous_report_sha256": previous_digest,
        "candidate_build": build, "static_binary_sha256": previous["static_binary_sha256"], "model_seal": seal,
        "build_performed": False, "numeric_exact_vs_sealed": True, "same_external_routes_and_counts": True,
        "warmups": warmups, "warmups_excluded_from_timing": True, "trials": trials, "telemetry": audited,
        "analysis": analysis, "performance_promotion_approved": False,
        "timing_method": "fresh processes; ABBA; linear interpolation between static process midpoint timestamps",
    })
    for phase in ("process", "before_first_raw", "model_load", "first_raw_to_exit"):
        row = analysis["comparison"][phase]
        print(f"{phase} static_ms={row['static_interpolated_median_ms']:.3f} embedded_ms={row['embedded_median_ms']:.3f} "
              f"paired_delta_pct={row['paired_delta_median_pct']:.3f}")
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT_REFERENCE={'STABLE' if analysis['reference_stable'] else 'UNSTABLE'}")
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT_TIMING={'STABLE' if stable else 'UNSTABLE'} max_drift_pct={max_drift_pct}")
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT_REPORT={dest}")
    print(f"DKR_R123_S1A_EMBEDDED_REPEAT={'COMPLETE' if stable else 'REVIEW_REQUIRED'}")
    return 0 if stable else 4


def self_test():
    import copy

    fixtures = []
    for i, variant in enumerate(ORDER * MIN_ROUNDS):
        elapsed = 6.8 if variant == "static" else 6.75
        fixtures.append({"stem": str(i), "variant": variant, "kind": "timed",
                         "start_monotonic_ns": i * 10**10, "finish_monotonic_ns": i * 10**10 + elapsed * 10**9,
                         "phase_seconds": dict.fromkeys(STABILITY_PHASES, elapsed)})
    result = compare(fixtures, DEFAULT_MAX_DRIFT_PCT)
    need(result["timing_stable"] and result["comparison"]["process"]["paired_delta_median_pct"] < 0,
         "stable comparison failed")
    sloped = copy.deepcopy(fixtures)
    for trial in sloped:
        value = 0.5 + (trial["start_monotonic_ns"] + trial["finish_monotonic_ns"]) / (2 * 10**13)
        trial["phase_seconds"]["model_load"] = value - (0.01 if trial["variant"] == "embedded" else 0)
    pairs = compare(sloped, DEFAULT_MAX_DRIFT_PCT)["comparison"]["model_load"]["pairs"]
    need(all(math.isclose(p["delta_ms"], -10, abs_tol=1e-8) for p in pairs), "timestamp interpolation failed")
    unstable = copy.deepcopy(fixtures)
    unstable[0]["phase_seconds"]["model_load"] = 1.815
    need(not compare(unstable, DEFAULT_MAX_DRIFT_PCT)["reference_stable"], "unstable model load accepted")
    unstable = copy.deepcopy(fixtures)
    unstable[1]["phase_seconds"]["process"] = 9
    result = compare(unstable, DEFAULT_MAX_DRIFT_PCT)
    need(result["reference_stable"] and not result["timing_stable"], "unstable candidate accepted")
    regressed = copy.deepcopy(fixtures)
    for trial in regressed:
        if trial["variant"] == "embedded":
            trial["phase_seconds"] = dict.fromkeys(STABILITY_PHASES, 7.5)
    need(compare(regressed, DEFAULT_MAX_DRIFT_PCT)["comparison"]["process"]["paired_delta_median_pct"] > 10,
         "real regression hidden")
    try:
        compare(fixtures[:-1], DEFAULT_MAX_DRIFT_PCT)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete bracket accepted")
    print("DKR_R123_S1A_EMBEDDED_REPEAT_SELF_TEST=PASS")


def nonempty_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--original", type=nonempty_path)
    parser.add_argument("--s0-report", type=nonempty_path)
    parser.add_argument("--prior-report", type=nonempty_path)
    parser.add_argument("--rounds", type=int, default=MIN_ROUNDS)
    parser.add_argument("--max-drift-pct", type=float, default=DEFAULT_MAX_DRIFT_PCT)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    need(args.original and args.s0_report and args.prior_report, "--original, --s0-report and --prior-report are required")
    need(args.rounds >= MIN_ROUNDS and math.isfinite(args.max_drift_pct) and 0 < args.max_drift_pct <= 100,
         "at least three rounds and a finite drift threshold in (0, 100] are required")
    original = args.original.resolve()
    with (original / "target/dkr-r122-s684/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return campaign(original, args.s0_report.resolve(), args.prior_report.resolve(), args.rounds, args.max_drift_pct)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        print(f"DKR_R123_S1A_EMBEDDED_REPEAT=REVIEW_REQUIRED {error}", file=sys.stderr)
        sys.exit(4)
