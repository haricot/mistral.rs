#!/usr/bin/env python3
"""Measure the paged archive with its full and reduced typed DKR catalogues."""
import argparse
import fcntl
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import dkr_r123_s0 as s0
import dkr_r123_s1a as s1a
from dkr_r122_s683_audit import need, sha
from dkr_r122_s684b_gate import trace, typed_index
from dkr_r122_s5_evidence import merge

FAMILIES = ("causal-mask", "indexed-copy", "kv-cache-ops", "paged-decode", "paged-v2", "rms-variants")
FULL_COUNT = 212
REDUCED_COUNT = 29
ORDER = ("full", "reduced", "reduced", "full")
REQUESTS = ("raw=1", "raw=2", "generation=1", "generation=2")
CASE_PREFIX = b"DKR_S684_CASE_"


def reduced_inventory(sealed):
    selected = {name: digest for name, digest in sealed.items() if Path(name).parts[0] in FAMILIES}
    need(len(sealed) == FULL_COUNT and len(selected) == REDUCED_COUNT, "unexpected catalogue inventory")
    need({Path(name).parts[0] for name in selected} == set(FAMILIES), "missing reduced catalogue family")
    return selected


def manifest_digests(root):
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(root.rglob("manifest.json"))}


def stage_catalogue(full, reduced, inventory):
    for name, digest in sorted(inventory.items()):
        path = Path(name)
        need(not path.is_absolute() and ".." not in path.parts, "invalid sealed manifest path")
        source = full / path
        need(sha(source) == digest, f"sealed manifest changed: {name}")
        artifact = Path(s0.read(source)["artifact"])
        need(not artifact.is_absolute() and ".." not in artifact.parts and artifact != Path("manifest.json"),
             f"invalid artifact path: {name}")
        destination = reduced / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        (destination.parent / artifact).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        shutil.copy2(source.parent / artifact, destination.parent / artifact)


def verify_catalogue_copy(root, inventory):
    need(manifest_digests(root) == inventory, "staged catalogue differs from sealed manifests")
    index = set()
    for name in inventory:
        path = root / name
        manifest = s0.read(path)
        artifact = path.parent / manifest["artifact"]
        need(sha(artifact) == manifest["sha256"], f"staged PTX differs from sealed artifact: {name}")
        index.add(tuple(manifest[key] for key in ("operation", "entry", "sha256", "target_cc")))
    return index


def phase_durations(start, finish, markers, load_seconds):
    expected = {}
    for case in s0.KEYS:
        for request in REQUESTS:
            name = f"DKR_S684_CASE_BEGIN {case} {request}".encode()
            expected[name] = f"{case}.{request}"
        next_phase = "pass_to_exit" if case == s0.KEYS[-1] else "pass_to_next_case"
        expected[f"DKR_S684_CASE_PASS {case}".encode()] = f"{case}.{next_phase}"
    need([name for name, _ in markers] == list(expected), "unexpected model phase marker sequence")
    ticks = [start, *(tick for _, tick in markers), finish]
    need(ticks == sorted(ticks) and finish > start, "invalid phase timestamps")
    need(math.isfinite(load_seconds) and load_seconds > 0, "invalid model load time")
    labels = ["before_first_raw", *expected.values()]
    seconds = {name: (end - begin) / 1e9 for name, begin, end in zip(labels, ticks, ticks[1:])}
    seconds["process"] = (finish - start) / 1e9
    seconds["model_load"] = load_seconds
    seconds["before_first_raw_minus_model_load"] = seconds["before_first_raw"] - load_seconds
    seconds["raw_total"] = sum(seconds[f"{case}.raw={repeat}"] for case in s0.KEYS for repeat in (1, 2))
    return seconds


def observe(binary, env, log):
    markers = []
    with log.open("xb") as output:
        start = time.monotonic_ns()
        with subprocess.Popen([str(binary)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
            for line in child.stdout:
                if line.startswith(CASE_PREFIX):
                    markers.append((line.rstrip(b"\r\n"), time.monotonic_ns()))
                output.write(line)
            rc = child.wait()
            finish = time.monotonic_ns()
    return rc, start, finish, markers


def route_counts(runtime):
    counts = {}
    for row in runtime["rows"]:
        abi = row["abi"]
        if abi == "dkr.typed.v1":
            count = row["adapter_ok"]
        elif abi == "candle.native.ptx.v1":
            count = row["submitted"]
        else:
            continue
        if count:
            key = (abi, row["operation"], row["entry"], row["sha256"], row["target_cc"])
            counts[key] = count
    return counts


def compare(trials):
    need(tuple(trial["catalogue"] for trial in trials) == ORDER, "unexpected paired trial order")
    full = [trial["phase_seconds"] for trial in trials if trial["catalogue"] == "full"]
    reduced = [trial["phase_seconds"] for trial in trials if trial["catalogue"] == "reduced"]
    result = {}
    for phase in full[0]:
        a = [leg[phase] * 1000 for leg in full]
        b = [leg[phase] * 1000 for leg in reduced]
        result[phase] = {
            "full_median_ms": statistics.median(a),
            "reduced_median_ms": statistics.median(b),
            "saved_ms": statistics.median(a) - statistics.median(b),
            "paired_saved_ms": [x - y for x, y in zip(a, b)],
        }
    return result


def self_test():
    markers = []
    for case in s0.KEYS:
        markers.extend(f"DKR_S684_CASE_BEGIN {case} {request}".encode() for request in REQUESTS)
        markers.append(f"DKR_S684_CASE_PASS {case}".encode())
    stamped = [(name, i * 10**9) for i, name in enumerate(markers, 1)]
    seconds = phase_durations(0, 16 * 10**9, stamped, 0.75)
    need(seconds["process"] == 16 and seconds["before_first_raw_minus_model_load"] == 0.25
         and seconds["raw_total"] == 6 and seconds["long.pass_to_exit"] == 1, "phase timing regression")
    fixtures = [{"catalogue": mode, "phase_seconds": {"process": elapsed}}
                for mode, elapsed in zip(ORDER, (8, 7, 6, 9))]
    process = compare(fixtures)["process"]
    need(process["saved_ms"] == 2000 and process["paired_saved_ms"] == [1000, 3000], "paired timing regression")
    print("DKR_R123_S1A_CATALOG_SELF_TEST=PASS")


def campaign(orig, s0_report, s1a_report):
    rev, modelrun, prior, native, full, seal, prereq, model, refs, binaries = s1a.prep(orig, s0_report)
    s0_sha, s1a_sha = sha(s0_report), sha(s1a_report)
    baseline = s0.read(s1a_report)
    need(baseline["status"] == "ATTRIBUTION_CAPTURE_COMPLETE" and baseline["numeric_exact_vs_sealed"]
         and baseline["model_seal"] == seal and baseline["prerequisites"] == prereq
         and baseline["s0_report_sha256"] == s0_sha, "S1A evidence does not match the qualified archive")
    witnesses = [t for t in baseline["trials"] if t["stem"] == "paged-B-telemetry"]
    need(len(witnesses) == 1 and witnesses[0]["parity_exact"], "missing qualified paged telemetry")
    qualified_routes = route_counts(witnesses[0]["runtime_counters"])
    sealed = seal["typed_manifest_sha256"]
    selected = reduced_inventory(sealed)
    full_index = typed_index(full)

    base = orig / "target/dkr-r123-s1a-catalog" / rev[:12]
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    print(f"DKR_R123_S1A_CATALOG_RUN={run}", flush=True)
    reduced = run / "typed-reduced"
    stage_catalogue(full, reduced, selected)
    reduced_index = verify_catalogue_copy(reduced, selected)
    needed = {key[1:] for key in qualified_routes if key[0] == "dkr.typed.v1"}
    need(needed and needed <= reduced_index <= full_index, "reduced catalogue misses a qualified paged route")
    print(f"DKR_R123_S1A_CATALOG_PREFLIGHT=PASS full={len(sealed)} reduced={len(selected)}", flush=True)
    catalogues = {"full": (full, full_index), "reduced": (reduced, reduced_index)}

    def run_trial(stem, kind, catalogue):
        need(not s1a.compute_apps(), f"other GPU compute process active before {stem}")
        root, index = catalogues[catalogue]
        dest, log = run / f"{stem}.json", run / f"{stem}.log"
        env = s0.environment(os.environ, model, prior, native, root, True, "B", dest)
        if kind == "telemetry":
            env.update(DKR_TELEMETRY="1", CANDLE_DKR_TRACE="1")
        before = s0.gpu()
        rc, start, finish, markers = observe(binaries["B"], env, log)
        (run / f"{stem}.exit").write_text(f"{rc}\n")
        need(rc == 0, f"{stem} failed (exit={rc}); inspect {log}")
        after = s0.gpu()
        data = s0.read(dest)
        ref = refs["paged"]["B"]
        need(data["mode"] == "external" and data["model"] == model and data["paged_requested"] is True,
             f"provider/model mismatch {stem}")
        need(s0.exact({**ref, "mode": "legacy"}, data), f"numerical mismatch {stem}")
        trial = {
            "stem": stem, "catalogue": catalogue, "kind": kind, "parity_exact": True,
            "gpu_before": before, "gpu_after": after, "json_sha256": sha(dest), "log_sha256": sha(log),
        }
        metrics = s0.extract(data)
        if kind == "timing":
            trial.update(phase_seconds=phase_durations(start, finish, markers, data["load_seconds"]),
                         generation_metrics=metrics)
        else:
            trial["telemetry"] = trace(run, stem, index, native, False)
            runtime = merge(log.read_text().splitlines())
            need(route_counts(runtime).keys() == qualified_routes.keys(), f"external route changed in {stem}")
            trial["runtime_counters"] = runtime
        s1a.write_checkpoint(run / f"{stem}.metrics.json", trial)
        print(f"{stem} exact=True", flush=True)
        return trial

    timed = [run_trial(f"{i:02d}-{mode}", "timing", mode) for i, mode in enumerate(ORDER)]
    audited = {mode: run_trial(f"{mode}-telemetry", "telemetry", mode) for mode in catalogues}
    counts_equal = route_counts(audited["full"]["runtime_counters"]) == route_counts(
        audited["reduced"]["runtime_counters"]
    )
    need(s0.model_gate(modelrun, prior, native) == seal, "qualified archive changed during the comparison")
    need(sha(s0_report) == s0_sha and sha(s1a_report) == s1a_sha, "qualification report changed")
    need(verify_catalogue_copy(reduced, selected) == reduced_index, "reduced catalogue changed")

    summary = compare(timed)
    report = {
        "status": "CATALOGUE_COMPARISON_COMPLETE", "revision": rev, "s0_report_sha256": s0_sha,
        "s1a_report_sha256": s1a_sha, "external_binary_sha256": sha(binaries["B"]),
        "model_seal": seal, "layout": "paged", "full_manifest_count": len(sealed),
        "reduced_manifest_count": len(selected), "reduced_manifest_sha256": selected,
        "numeric_exact_vs_sealed": True, "new_compilation": False,
        "same_external_route_identities": True, "same_external_route_counts": counts_equal,
        "performance_promotion_approved": False,
        "timing_method": "fresh processes; phase boundaries observed on the archive's existing stderr markers",
        "trials": timed + list(audited.values()), "comparison": summary,
    }
    dest = run / "catalogue-comparison.json"
    s1a.write_checkpoint(dest, report)
    for key in ("process", "before_first_raw", "model_load", "before_first_raw_minus_model_load", "raw_total",
                "short.generation=1", "long.generation=1", "long.pass_to_exit"):
        value = summary[key]
        print(f"{key} full_ms={value['full_median_ms']:.3f} reduced_ms={value['reduced_median_ms']:.3f} "
              f"saved_ms={value['saved_ms']:.3f}")
    print(f"DKR_R123_S1A_CATALOG={report['status']} route_counts_equal={counts_equal}")
    print(f"DKR_R123_S1A_CATALOG_REPORT={dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--s0-report", type=Path)
    parser.add_argument("--s1a-report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    need(args.original and args.s0_report and args.s1a_report,
         "--original, --s0-report and --s1a-report are required")
    orig = args.original.resolve()
    with (orig / "target/dkr-r122-s684/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign(orig, args.s0_report.resolve(), args.s1a_report.resolve())


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        print(f"DKR_R123_S1A_CATALOG=REVIEW_REQUIRED {error}", file=sys.stderr)
        sys.exit(4)
