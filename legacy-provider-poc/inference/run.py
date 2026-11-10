#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from fixture import create_model, HIDDEN, INTERMEDIATE, MODEL_SHAPES, OPERATOR_SHAPE, ROW_COUNTS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILE_ENV = "MISTRALRS_ASD_F32_PROFILE"
EXAMPLE = "provider_f32_inference"
TOLERANCE = 1e-4


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def cargo(*args):
    return ["cargo", *args, "--locked", "--manifest-path", ROOT / "Cargo.toml"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_builder():
    spec = importlib.util.spec_from_file_location("f32_artifacts", HERE.parent / "f32/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(manifest_path, output, label):
    manifest = json.loads(manifest_path.read_text())
    decisions = []
    for n, k in [*MODEL_SHAPES, OPERATOR_SHAPE]:
        for m in ROW_COUNTS:
            decisions.append(dict(id=f"linear-{m}-{n}-{k}", shape=[m, n, k],
                implementation_id=manifest["implementation_id"],
                manifest=os.path.relpath(manifest_path, output.parent),
                manifest_sha256=sha(manifest_path), artifact_sha256=manifest["sha256"]))
    result = dict(schema="ASD-LINEAR-PROFILE-V1", profile_id=f"manual-f32-{label}",
        abi="mistral.linear.f32.v1", purpose="validation_only", target_cc=manifest["target_cc"],
        decisions=decisions)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_values(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected), "output cardinality differs"
        for a, b in zip(actual, expected):
            compare_values(a, b)
    else:
        assert math.isfinite(actual) and math.isfinite(expected)
        assert abs(actual - expected) <= TOLERANCE * (1 + abs(expected)), (actual, expected)


def compare_models(actual, expected):
    assert actual["prompt_tokens"] == expected["prompt_tokens"]
    compare_values(actual["logits"], expected["logits"])
    a, b = actual["response"]["choices"][0], expected["response"]["choices"][0]
    assert a["message"]["content"] == b["message"]["content"]
    assert a["finish_reason"] == b["finish_reason"]
    aa, bb = a["logprobs"]["content"], b["logprobs"]["content"]
    assert len(aa) == len(bb) == 4
    assert [x["token"] for x in aa] == [x["token"] for x in bb]
    compare_values([x["logprob"] for x in aa], [x["logprob"] for x in bb])


def check_provider(report, selected, engine):
    evidence = report["provider"]
    assert evidence["purpose"] == "validation_only" and evidence["fallback_used"] is False
    assert evidence["resolved_modules"] == 1
    before = {d["decision_id"]: d["submissions"]
              for d in report.get("provider_after_load", {}).get("decisions", [])}
    seen = [d for d in evidence["decisions"]
            if d["submissions"] > before.get(d["decision_id"], 0)]
    assert seen and all(d["artifact_sha256"] == selected["decisions"][0]["artifact_sha256"] for d in seen)
    assert evidence["devices"] and all(d["compute_capability"] == selected["target_cc"] for d in evidence["devices"])
    shapes = {tuple(d["shape"]) for d in seen}
    if engine:
        assert evidence["prepared_layers"] >= 8
        assert report["provider_after_load"]["prepared_layers"] >= 8
        # The fixture packs the gate/up projections (2 * INTERMEDIATE, HIDDEN).
        # Individual (INTERMEDIATE, HIDDEN) decisions must remain available in the
        # exact ASD profile, but are not submitted when the packed path is used.
        active_pairs = [(HIDDEN, HIDDEN), (HIDDEN, INTERMEDIATE),
                        (2 * INTERMEDIATE, HIDDEN)]
        required = {(m, n, k) for m in (1, 3) for n, k in active_pairs}
        missing = required - shapes
        assert not missing, f"model did not execute required provider shapes: {sorted(missing)}"
        declared = {tuple(d["shape"]) for d in selected["decisions"]}
        required_profile = {(m, n, k) for m in (1, 3) for n, k in MODEL_SHAPES}
        assert not (required_profile - declared), "profile omits exact model decisions"
    else:
        assert all((m, *OPERATOR_SHAPE) in shapes for m in [1, 3, 6])


def reuse_existing(source, output):
    """Replay previously compiled SM61 artifacts without invoking Cargo or NVCC."""
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("reuse source and output must be separate directories")
    if not source.is_dir():
        raise FileNotFoundError(f"missing previous evidence directory: {source}")
    if any(output.iterdir()):
        raise FileExistsError(f"reuse output must be empty: {output}")

    binary = source / "target/debug/examples" / EXAMPLE
    if not binary.is_file() or binary.stat().st_size == 0:
        raise FileNotFoundError(f"missing previously compiled executable: {binary}")
    binary_hash = sha(binary)

    # Confirm that this is the same executable which produced earlier PASS reports.
    for label in ("cpu-model", "baseline-model", "A-operator"):
        report_path = source / f"{label}.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing prior PASS checkpoint: {report_path}")
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS" or report.get("binary_sha256") != binary_hash:
            raise RuntimeError(f"prior checkpoint does not match executable: {report_path}")

    manifests = []
    package_shas = {}
    for name, entry in (("a", "linear_simple"), ("b", "linear_tiled")):
        directory = source / "packages" / name
        manifest_path = directory / "module.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing prior module manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        expected = {"schema": 1, "abi": "mistral.linear.f32.v1",
                    "target_cc": 61, "min_cc": 61, "max_cc": 61,
                    "artifact": "kernel.ptx", "entry": entry, "dtype": "f32"}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"prior manifest is not the expected SM61 module: {manifest_path}")
        artifact = directory / "kernel.ptx"
        if not artifact.is_file() or sha(artifact) != manifest.get("sha256"):
            raise RuntimeError(f"PTX hash mismatch or artifact missing: {artifact}")
        ptx = artifact.read_text()
        if not re.search(r"\.target\s+sm_61\b", ptx) or not re.search(
                rf"\.entry\s+{entry}\s*\(", ptx):
            raise RuntimeError(f"PTX target or entry mismatch: {artifact}")
        destination = output / "packages" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(directory, destination)
        if sha(destination / "kernel.ptx") != manifest["sha256"]:
            raise RuntimeError(f"copied PTX differs from source: {destination}")
        manifests.append(destination / "module.json")
        package_shas[name] = manifest["sha256"]
    print(f"PROVIDER_F32_REUSED_BINARY_SHA256={binary_hash}", flush=True)
    print("PROVIDER_F32_REUSED_PTX_SM61=PASS", flush=True)
    return binary, manifests, dict(source=str(source), package_sha256=package_shas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "cuda-compile", "gpu"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-from", type=Path, help="prior GPU output; skip all compilation")
    args = parser.parse_args()
    if args.reuse_from and args.mode != "gpu":
        parser.error("--reuse-from is only available for gpu mode")
    builder = artifact_builder()
    builder.source_scope()
    output = (args.output or Path(tempfile.mkdtemp(prefix="mistral-provider-f32-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT={output}", flush=True)
    env = os.environ.copy()
    env.pop(PROFILE_ENV, None)
    env["HF_HUB_OFFLINE"] = "1"
    reuse_details = None
    if args.reuse_from:
        binary, reused_manifests, reuse_details = reuse_existing(args.reuse_from, output)
    else:
        target = Path(env.get("CARGO_TARGET_DIR", str(output / "target"))).resolve()
        env["CARGO_TARGET_DIR"] = str(target)
        env.setdefault("CARGO_BUILD_JOBS", "2")
        env.setdefault("CARGO_PROFILE_DEV_OPT_LEVEL", "1")
        env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
        features = ["--features", "cuda"] if args.mode != "cpu" else []
        if args.mode == "cpu":
            run(["cargo", "fmt", "--manifest-path", ROOT / "Cargo.toml", "-p", "mistralrs-kernel-runtime",
                 "-p", "mistralrs-quant", "-p", "mistralrs", "--", "--check"], env=env)
            run(cargo("check", "-p", "mistralrs", "--example", EXAMPLE), env=env)
            run(cargo("test", "-p", "mistralrs-kernel-runtime"), env=env)
            run(cargo("test", "-p", "mistralrs-quant", "--lib", "unquantized::tests"), env=env)
        run(cargo("build", "-p", "mistralrs", "--example", EXAMPLE, *features), env=env)
        binary = target / "debug/examples" / EXAMPLE
    binary_hash = sha(binary)
    model_path = create_model(output / "model")
    reports = {}

    def execute(label, selected_profile=None, device="cuda", remove=False):
        selected_env = env.copy()
        if selected_profile:
            selected_env[PROFILE_ENV] = str(selected_profile)
        for command in ["operator", "model"]:
            path = output / f"{label}-{command}.json"
            argv = [binary, command, device, path]
            if command == "model":
                argv += [model_path]
                if remove:
                    argv += ["await-control-plane-removal"]
            if command == "model" and remove:
                child = subprocess.Popen(list(map(str, argv)), env=selected_env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                while True:
                    line = child.stdout.readline()
                    if not line:
                        raise RuntimeError(f"model exited before control-plane handshake: {child.wait()}")
                    print(line, end="", flush=True)
                    if line.strip() == "PROVIDER_F32_MODEL_PREPARED":
                        break
                owned_files = [selected_profile, selected_profile.parent / "a/module.json",
                               selected_profile.parent / "a/kernel.ptx"]
                for owned in owned_files:
                    owned.unlink()
                rest, _ = child.communicate("continue\n", timeout=180)
                print(rest, end="", flush=True)
                assert child.returncode == 0 and all(not p.exists() for p in owned_files)
            else:
                run(argv, env=selected_env)
            report = json.loads(path.read_text())
            assert report["binary_sha256"] == binary_hash and report["status"] == "PASS"
            reports[f"{label}-{command}"] = report

    if args.mode == "cpu":
        execute("cpu", device="cpu")
    else:
        if args.reuse_from:
            a, b = reused_manifests
        else:
            cc = env.get("CUDA_COMPUTE_CAP", "61")
            assert cc.isdecimal(), "CUDA_COMPUTE_CAP must be numeric"
            a, b = builder.packages(output / "packages", int(cc))
        selected = {}
        for label, manifest in [("A", a), ("B", b)]:
            selected[label] = profile(manifest, output / f"{label}.profile.json", label)
        if args.mode == "gpu":
            execute("cpu", device="cpu")
            execute("baseline")
            compare_models(reports["baseline-model"], reports["cpu-model"])
            for label in ["A", "B"]:
                execute(label, output / f"{label}.profile.json")
                for kind in ["operator", "model"]:
                    check_provider(reports[f"{label}-{kind}"], selected[label], kind == "model")
                compare_models(reports[f"{label}-model"], reports["baseline-model"])
            incomplete = json.loads((output / "A.profile.json").read_text())
            incomplete["decisions"] = [d for d in incomplete["decisions"] if d["shape"] != [6, *OPERATOR_SHAPE]]
            negative_profile = output / "missing-decision.profile.json"
            negative_profile.write_text(json.dumps(incomplete))
            negative_env = env.copy()
            negative_env[PROFILE_ENV] = str(negative_profile)
            rejected = subprocess.run([str(binary), "operator", "cuda", str(output / "unexpected.json")],
                                      env=negative_env, capture_output=True, text=True)
            (output / "missing-decision.log").write_text(rejected.stdout + rejected.stderr)
            assert rejected.returncode != 0 and "no exact linear decision" in rejected.stderr, rejected.stderr
            print("PROVIDER_F32_MISSING_DECISION=REFUSED_AS_EXPECTED", flush=True)
            replay = output / "replay"
            replay.mkdir()
            shutil.copytree(a.parent, replay / "a")
            replay_profile = replay / "A2.profile.json"
            replay_selection = profile(replay / "a/module.json", replay_profile, "A2")
            execute("A2", replay_profile, remove=True)
            for kind in ["operator", "model"]:
                check_provider(reports[f"A2-{kind}"], replay_selection, kind == "model")
            compare_models(reports["A2-model"], reports["A-model"])
            assert reports["A2-model"]["logits"] == reports["A-model"]["logits"]
            assert reports["A2-model"]["control_plane_files_removed_before_requests"] is True
    assert sha(binary) == binary_hash, "binary changed while replacing profiles/artifacts"
    evidence = dict(status="PASS", mode=args.mode, binary_sha256=binary_hash,
        candle_revision=builder.CANDLE_REV, tuner_used=False,
        gpu_executed=args.mode == "gpu", model_source="generated local deterministic Llama fixture",
        checkpoint_sha256=sha(model_path / "model.safetensors"),
        scope="F32 UnquantLinear integration, not quantized providers or performance qualification",
        reports=sorted(reports))
    if reuse_details is not None:
        evidence["reused_from"] = reuse_details["source"]
        evidence["reused_package_sha256"] = reuse_details["package_sha256"]
        evidence["cargo_invoked"] = False
        evidence["nvcc_invoked"] = False
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
    print(f"PROVIDER_F32_INFERENCE_{args.mode.upper().replace('-', '_')}=PASS", flush=True)


if __name__ == "__main__":
    main()

