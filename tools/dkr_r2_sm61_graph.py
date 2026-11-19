#!/usr/bin/env python3
"""Prove DKR paged graph replay, then compare LFM3 streaming on the same SM61 binary."""

import argparse
import fcntl
import math
import os
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import dkr_r123_s1a_shared_catalog as shared


s0, s1a = shared.s0, shared.s1a
need, sha = shared.need, shared.sha
EXAMPLE = "dkr_sm61_graph"
ORDER = ("off", "on", "on", "off")
STREAM_TOKENS = 128
PROBE_ROUTES = {"kv_cache.reshape_store.vllm.b256.v1", "attention.paged.decode.online.v1"}
GRAPH_SOURCES = shared.RUNTIME_SOURCES + (
    "mistralrs/Cargo.toml",
    "mistralrs/examples/advanced/dkr_sm61_graph/main.rs",
    "mistralrs/examples/advanced/dkr_sm61_graph/probe.rs",
    "mistralrs/examples/advanced/dkr_sm61_graph/graph_metrics.rs",
    "mistralrs-core/src/perf_flags.rs",
    "mistralrs-core/src/models/lfm3.rs",
    "mistralrs-core/src/pipeline/cuda_graph.rs",
    "mistralrs-core/src/pipeline/normal.rs",
)


def stream_key(stream):
    return stream["tokens"], stream["finish_reason"]


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def stream_metrics(stream, graphs, count):
    need(stream["token_count"] == count and len(stream["tokens"]) == count, "stream token count differs")
    arrivals = stream["token_receipt_ms"]
    need(len(arrivals) == count and all(math.isfinite(x) and x >= 0 for x in arrivals)
         and arrivals == sorted(arrivals), "nonmonotonic token receipt")
    gaps = stream["inter_token_receipt_ms"]
    need(gaps == [b - a for a, b in zip(arrivals, arrivals[1:])]
         and stream["ttft_logprob_ms"] == arrivals[0], "stream timings disagree with observed tokens")
    events = stream["graph"]
    need(events["replays"] == (count - 1 if graphs else 0),
         f"expected {'graph replay' if graphs else 'eager decode'} on all {count - 1} decode steps, got {events}")
    need(events["failures"] == events["eager_fallbacks"] == 0, f"graph failures or fallback: {events}")
    need(len(events["capture_ms"]) == events["captures"], "stream capture timing inventory differs")
    host = events["decode_host_ms"]
    need(len(host) == count - 1 and all(math.isfinite(x) and x > 0 for x in host),
         "host forward timing inventory differs from real decode steps")
    return {"ttft_ms": arrivals[0], "tpot_ms": (arrivals[-1] - arrivals[0]) / (count - 1),
            "itl_p50_ms": st.median(gaps), "itl_p95_ms": p95(gaps),
            "host_forward_p50_ms": st.median(host), "host_forward_p95_ms": p95(host)}


def model_metrics(data, graphs, model, sealed):
    need(data["schema"] == "DKR-SM61-GRAPH-STREAMING-V1" and data["mode"] == "external"
         and data["paged_requested"] is True and data["model"] == model and data["graphs_requested"] == graphs,
         "model or graph mode differs from the requested leg")
    need(s0.exact({**sealed, "mode": "legacy"}, {**data, "schema": sealed["schema"]}),
         "raw logits or short generation differ from sealed LFM3")
    total = data["graph_total"]
    need(total["failures"] == total["eager_fallbacks"] == 0
         and len(total["capture_ms"]) == total["captures"], f"graph evidence incomplete: {total}")
    need((total["captures"] > 0) == graphs, "requested graph mode did not match actual captures")
    need(graphs or total["replays"] == 0, "graphs replayed in the eager control")
    metrics, tokens = {}, {}
    for prompt in s0.KEYS:
        case = data["cases"][prompt]
        short = case["streaming_24"]
        stream_metrics(short, graphs, 24)
        need(short["tokens"] == case["generation"][1]["tokens"], f"{prompt}: streaming parity failed")
        first, second = case["streaming_128"]
        need(stream_key(first) == stream_key(second), f"{prompt}: repeated 128-token streams differ")
        tokens[prompt] = stream_key(second)
        for name, stream in (("first", first), ("repeat", second)):
            for key, value in stream_metrics(stream, graphs, STREAM_TOKENS).items():
                metrics[f"{prompt}.{name}.{key}"] = value
    return metrics, tokens


def summarize(trials):
    need(len(trials) % len(ORDER) == 0 and trials
         and tuple(t["mode"] for t in trials) == ORDER * (len(trials) // len(ORDER)), "incomplete timing brackets")
    result = {}
    for metric in trials[0]["stream_metrics"]:
        ratios, drift = [], []
        eager_samples, graph_samples = [], []
        for start in range(0, len(trials), len(ORDER)):
            a, b, c, d = [t["stream_metrics"][metric] for t in trials[start:start + len(ORDER)]]
            mid = (a + d) / 2
            need(mid > 0 and min(a, b, c, d) > 0, f"invalid SDK timing for {metric}")
            eager_samples.extend((a, d))
            graph_samples.extend((b, c))
            ratios.extend((100 * (b / mid - 1), 100 * (c / mid - 1)))
            drift.append(100 * abs(a - d) / mid)
        result[metric] = {
            "off_median_ms": st.median(eager_samples), "on_median_ms": st.median(graph_samples),
            "paired_delta_median_pct": st.median(ratios), "paired_delta_pct": ratios,
            "max_eager_bracket_drift_pct": max(drift),
        }
    return result


def embedded_routes(log, typed_index, native):
    lines = log.read_text().splitlines()
    need([line for line in lines if line.startswith("MISTRAL_DKR_CATALOG_EMBEDDED ")]
         == [f"MISTRAL_DKR_CATALOG_EMBEDDED manifests={shared.FULL_COUNT}"],
         "missing typed embedded catalogue evidence")
    native_catalogues = [line for line in lines if line.startswith("CANDLE_DKR_NATIVE_CATALOG ")]
    need(native_catalogues and all(line == "CANDLE_DKR_NATIVE_CATALOG source=embedded manifests=25"
                                  for line in native_catalogues), "missing native embedded catalogue evidence")
    need(not shared.catalogue_validations("\n".join(lines)), "candidate validated a disk catalogue")
    counters = shared.merge(lines)
    need(counters["processes"] == 1, "expected a single DKR telemetry process")
    need(not any(r[k] for r in counters["rows"] for k in ("adapter_error", "prepare_error", "rejected")),
         "DKR dispatch errors during qualification")
    need(not any(r["fallback_selected"] for r in counters["rows"] if r["abi"] == "candle.native.ptx.v1"),
         "Candle native fallback during qualification")
    routes = shared.route_counts(counters)
    for abi, op, entry, digest, cc in routes:
        need(cc == 61, "non-SM61 route")
        if abi == "dkr.typed.v1":
            need((op, entry, digest, cc) in typed_index, f"unqualified typed route: {op}")
        else:
            need(s0.read(native / op / "manifest.json")["sha256"] == digest, f"unqualified native route: {op}")
    return sorted(routes)


def verify_inputs(inputs, native, seal):
    need(len(seal["typed_manifest_sha256"]) == shared.FULL_COUNT, "unexpected qualified typed inventory")
    shared.verify_catalogue_copy(inputs / "typed", seal["typed_manifest_sha256"])
    need(shared.tree_hashes(inputs / "native") == shared.tree_hashes(native), "native build inputs differ from seal")


def experiment(base, s0_report, evidence, rounds):
    revision, modelrun, prior, native, typed, seal, _, model, refs, _ = evidence
    candidate, build = shared.prepared_candidate(base, revision)
    need(build["command"][build["command"].index("--example") + 1] == EXAMPLE, "wrong archived candidate target")
    inputs = base / "inputs"
    need(Path(build["embedded_inputs"]) == inputs, "wrong embedded input root")
    hidden = inputs.with_name("inputs-unavailable")
    if hidden.exists() and not inputs.exists():
        hidden.rename(inputs)
    verify_inputs(inputs, native, seal)
    typed_index = shared.typed_index(typed)
    s0_digest = sha(s0_report)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    print(f"DKR_R2_SM61_GRAPH_RUN={run}", flush=True)

    def launch(stem, mode, *, proof=False, audit=False):
        need(not s1a.compute_apps(), f"other GPU compute process active before {stem}")
        dest, log = run / f"{stem}.json", run / f"{stem}.log"
        graphs = mode == "on"
        env = s0.environment(os.environ, model, prior, native, typed, True, "B", dest)
        env.update(MISTRALRS_CUDA_GRAPHS=str(int(graphs)), MISTRALRS_DKR_LFM3_CUDA_GRAPHS="1",
                   MISTRALRS_DKR_ARTIFACT_SOURCE="embedded", MISTRALRS_DKR_SHARED_CATALOG="1",
                   CANDLE_DKR_S31_DIR=str(inputs / "native"), DKR_KERNEL_PACKAGES=str(inputs / "typed"))
        if proof or audit:
            env.update(DKR_TELEMETRY="1", CANDLE_DKR_TRACE="1")
        before = s0.gpu()
        samples, errors, stop = [], [], threading.Event()
        sampler = None
        if audit:
            sampler = threading.Thread(target=s1a.memory_sample, args=(stop, samples, errors), daemon=True)
            sampler.start()
        try:
            command = [str(candidate), "--probe"] if proof else [str(candidate)]
            started = time.monotonic_ns()
            with shared.unavailable_inputs(inputs, True), log.open("xb") as output:
                rc = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT).returncode
            elapsed = (time.monotonic_ns() - started) / 1e9
        finally:
            stop.set()
            if sampler:
                sampler.join()
        (run / f"{stem}.exit").write_text(f"{rc}\n")
        need(rc == 0, f"{stem} failed (exit={rc}); inspect {log}")
        data = s0.read(dest)
        detail = {"stem": stem, "mode": mode, "process_seconds": elapsed,
                  "gpu_before": before, "gpu_after": s0.gpu(), "inputs_hidden": True,
                  "json_sha256": sha(dest), "log_sha256": sha(log)}
        if proof or audit:
            detail["routes"] = embedded_routes(log, typed_index, native)
        if audit:
            need(not errors and samples, f"{stem} memory sampling failed: {errors}")
            peak = max(sample["device_used_mib"] for sample in samples)
            detail["vram"] = {
                "baseline_mib": int(before["memory.used"]), "sampled_peak_mib": peak,
                "sampled_extra_mib": peak - int(before["memory.used"]), "samples": samples,
            }
        return data, detail

    proof, proof_detail = launch("paged-replay-proof", "on", proof=True)
    need(proof["schema"] == "DKR-SM61-PAGED-GRAPH-PROOF-V1" and proof["exact_output_and_kv"] is True
         and proof["batch"] == 1 and len(proof["replays"]) == 4
         and set(proof["captured_routes"]) == PROBE_ROUTES
         and PROBE_ROUTES <= {route[1] for route in proof_detail["routes"]}, "incomplete paged replay proof")
    print(f"DKR_R2_SM61_GRAPH_PROOF=PASS replays={len(proof['replays'])} capture_ms={proof['capture_ms']:.3f}", flush=True)

    reference_tokens = None

    def leg(stem, mode, *, audit=False):
        nonlocal reference_tokens
        data, detail = launch(stem, mode, audit=audit)
        metrics, tokens = model_metrics(data, mode == "on", model, refs["paged"]["B"])
        if reference_tokens is None:
            need(mode == "off", "the first comparison leg must run eager")
            reference_tokens = tokens
        need(tokens == reference_tokens, f"{stem}: 128-token output or logprobs differ from eager")
        detail.update(stream_metrics=metrics, graph_at_load=data["graph_at_load"],
                      graph_total=data["graph_total"], load_seconds=data["load_seconds"], exact=True)
        print(f"{stem} exact=True captures={data['graph_total']['captures']} replays={data['graph_total']['replays']}", flush=True)
        return detail

    audited = {mode: leg(f"{mode}-vram", mode, audit=True) for mode in ("off", "on")}
    need(audited["off"]["routes"] == audited["on"]["routes"], "graph and eager use different DKR route identities")
    need(PROBE_ROUTES <= {route[1] for route in audited["on"]["routes"]},
         "true decode did not exercise the required paged DKR routes")
    need(any(route[0] == "candle.native.ptx.v1" for route in audited["on"]["routes"]),
         "true decode did not exercise the qualified native Candle routes")
    trials = [leg(f"{r:02d}-{i}-{mode}", mode) for r in range(rounds) for i, mode in enumerate(ORDER)]
    comparison = summarize(trials)
    verify_inputs(inputs, native, seal)
    need(s0.model_gate(modelrun, prior, native) == seal and sha(s0_report) == s0_digest
         and sha(candidate) == build["binary_sha256"], "qualification inputs changed during measurement")
    output = run / "dkr-sm61-graph-comparison.json"
    s1a.write_checkpoint(output, {
        "status": "DKR_SM61_GRAPH_COMPARISON_COMPLETE", "revision": revision,
        "s0_report_sha256": s0_digest, "model_seal": seal, "candidate_build": build,
        "probe": {**proof, **proof_detail}, "audit": audited, "trials": trials, "comparison": comparison,
        "same_dkr_route_identities": True, "exact_vs_eager_and_seal": True,
        "timing_scope": "SDK token receipt; host paged decode forward including preparation and submission; "
                        "headless ABBA; VRAM sampling and DKR telemetry in separate legs",
    })
    for prompt in s0.KEYS:
        for metric in ("ttft_ms", "tpot_ms", "itl_p95_ms", "host_forward_p50_ms"):
            row = comparison[f"{prompt}.repeat.{metric}"]
            print(f"{prompt}.{metric} off={row['off_median_ms']:.3f} on={row['on_median_ms']:.3f} "
                  f"paired_delta_pct={row['paired_delta_median_pct']:+.2f} "
                  f"eager_drift_pct={row['max_eager_bracket_drift_pct']:.2f}")
    print(f"vram_sampled_peak_mib off={audited['off']['vram']['sampled_peak_mib']} "
          f"on={audited['on']['vram']['sampled_peak_mib']}")
    capture_totals = [sum(trial["graph_total"]["capture_ms"]) for trial in trials if trial["mode"] == "on"]
    print(f"capture_host_total_ms_median={st.median(capture_totals):.3f}")
    print(f"DKR_R2_SM61_GRAPH_REPORT={output}")
    print("DKR_R2_SM61_GRAPH=COMPLETE")


def path_arg(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=path_arg, required=True)
    parser.add_argument("--s0-report", type=path_arg, required=True)
    parser.add_argument("--build", action="store_true", help="prepare one embedded binary using the qualified PTX")
    parser.add_argument("--rounds", type=int, default=3, help="number of timing brackets")
    args = parser.parse_args()
    need(args.rounds >= 1, "--rounds must be positive")
    with (args.original / "target/dkr-r122-s684/.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evidence = s1a.prep(args.original, args.s0_report, allowed_source_changes=GRAPH_SOURCES)
        s0.gpu()
        base = args.original / "target/dkr-r2-sm61-graph" / evidence[0][:12]
        base.mkdir(parents=True, exist_ok=True)
        if args.build:
            shared.build_candidate(args.original, base, evidence, embedded=True, example=EXAMPLE)
        else:
            experiment(base, args.s0_report, evidence, args.rounds)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f"DKR_R2_SM61_GRAPH=REVIEW_REQUIRED {error}", file=sys.stderr)
        sys.exit(4)
