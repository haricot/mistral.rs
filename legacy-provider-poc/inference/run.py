#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from fixture import create_model, MODEL_SHAPES, OPERATOR_SHAPE, ROW_COUNTS

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
    seen = [d for d in evidence["decisions"] if d["submissions"]]
    assert seen and all(d["artifact_sha256"] == selected["decisions"][0]["artifact_sha256"] for d in seen)
    assert evidence["devices"] and all(d["compute_capability"] == selected["target_cc"] for d in evidence["devices"])
    shapes = {tuple(d["shape"]) for d in seen}
    if engine:
        assert evidence["prepared_layers"] >= 8
        assert report["provider_after_load"]["prepared_layers"] >= 8
        assert all((m, n, k) in shapes for m in [1, 3] for n, k in MODEL_SHAPES)
    else:
        assert all((m, *OPERATOR_SHAPE) in shapes for m in [1, 3, 6])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "cuda-compile", "gpu"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    builder = artifact_builder()
    builder.source_scope()
    output = (args.output or Path(tempfile.mkdtemp(prefix="mistral-provider-f32-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT={output}", flush=True)
    env = os.environ.copy()
    env.pop(PROFILE_ENV, None)
    env["HF_HUB_OFFLINE"] = "1"
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
                    argv += ["remove-control-plane"]
            run(argv, env=selected_env)
            report = json.loads(path.read_text())
            assert report["binary_sha256"] == binary_hash and report["status"] == "PASS"
            reports[f"{label}-{command}"] = report

    if args.mode == "cpu":
        execute("cpu", device="cpu")
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
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)
    print(f"PROVIDER_F32_INFERENCE_{args.mode.upper().replace('-', '_')}=PASS", flush=True)


if __name__ == "__main__":
    main()
