#!/usr/bin/env python3
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import time
import urllib.request
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
REPOSITORY = "haricot/mistral.rs"
WORKFLOW = ".github/workflows/lfm3-model-sm61-qualification.yml"
MODES = ("text", "vl", "gguf")
RUNTIMES = ("legacy", "extended")
READY_TIMEOUT = 900
REQUEST_TIMEOUT = 600
IMAGE_EDGE = 64
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
MODEL_ERRORS = (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError, subprocess.CalledProcessError)
MEASUREMENT_SOURCE_SHA = "b804e46bb4322f1c3b64887c50feda83f0947ade"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def run(*args, **kwargs):
    return subprocess.check_output([str(x) for x in args], text=True, **kwargs).strip()


def api(path):
    return json.loads(run("gh", "api", "repos/" + REPOSITORY + path))


def download_artifact(artifact, destination):
    if artifact.get("expired") is not False or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", "")):
        raise RuntimeError("artifact expired or missing SHA256")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "download.zip"
    with archive.open("wb") as stream:
        subprocess.run(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{artifact['id']}/zip"], stdout=stream, check=True)
    if sha(archive) != artifact["digest"].split(":")[1]:
        raise RuntimeError("GitHub artifact SHA256 mismatch")
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise RuntimeError("unsafe artifact member")
        zipped.extractall(destination)
    archive.unlink()


def artifacts(run_id):
    data = api(f"/actions/runs/{run_id}/artifacts?per_page=100")
    if len(data["artifacts"]) != data["total_count"]:
        raise RuntimeError("incomplete artifact inventory")
    return data["artifacts"]


def artifact_for(run_id, prefix):
    found = [a for a in artifacts(run_id) if a["name"].startswith(prefix) and not a["expired"]]
    if len(found) != 1:
        raise RuntimeError("expected one immutable artifact: " + prefix)
    return found[0]


def source_fingerprint(repo, candidate, plan, policy):
    entries = run("git", "-C", repo, "ls-tree", "-r", candidate).splitlines()
    retained = []
    for entry in entries:
        path = entry.split("\t", 1)[1]
        if path.startswith((".github/", "docs/", "mistral-integration/")) or path.endswith(".md"):
            continue
        retained.append(entry)
    return digest({"source": retained, "image": plan["image"], "features": plan["config"]["features"],
                   "rust": plan["config"]["toolchain"], "policy": policy, "harness": sha(__file__),
                   "compiler_env": sha(ROOT.parent / "campaign/compiler-env.sh"), "build_profile": "campaign"})


def import_bundle(repo, directory):
    plan = read(directory / "plan.json")
    if sha(directory / "candidate.bundle") != plan["bundle_sha256"]:
        raise RuntimeError("immutable preparation bundle changed")
    run("git", "-C", repo, "fetch", str(directory / "candidate.bundle"),
        "refs/heads/integration/campaign/*:refs/remotes/lfm3/*")
    return plan


def fetch_campaign(repo, target, run_id, destination):
    original = api(f"/actions/runs/{run_id}")
    expected = ".github/workflows/" + ("mistral-cuda-legacy-sm61.yml" if target == "legacy" else "mistral-extended-sm61.yml")
    if (original["head_branch"] != "mirror_orch" or original["path"] != expected
            or original["event"] not in ("push", "workflow_dispatch") or original["status"] != "completed"):
        raise RuntimeError("original campaign identity is invalid or still active")
    jobs = api(f"/actions/runs/{run_id}/jobs?per_page=100")
    if len(jobs["jobs"]) != jobs["total_count"]:
        raise RuntimeError("incomplete original job inventory")
    physical = [j for j in jobs["jobs"] if (j["name"].startswith("Physical NVIDIA Pascal SM61 smoke proof (")
                                         or j["name"].startswith("SM61 Mistral Extended runtime ("))]
    if len(physical) != 1 or physical[0]["conclusion"] != "success":
        raise RuntimeError("original physical GPU proof is not PASS")
    for job in jobs["jobs"]:
        if job["id"] != physical[0]["id"] and job["name"] != "promote" and job["conclusion"] != "success":
            raise RuntimeError("original hosted preparation is not entirely PASS")
    prefix = "mistral-legacy-" if target == "legacy" else "mistral-asd-"
    prep = artifact_for(run_id, prefix + "prep-")
    download_artifact(prep, destination)
    plan = import_bundle(repo, destination)
    if plan["target"] != target:
        raise RuntimeError("original preparation target mismatch")
    proof_dir = destination / "physical"
    gpu_artifact = artifact_for(run_id, prefix + "gpu-")
    download_artifact(gpu_artifact, proof_dir)
    gpu = read(proof_dir / "report.json")
    if (gpu["candidate"] != plan["candidate"] or gpu["run_id"] != str(run_id)
            or gpu["status"] != "GPU_PASSED" or gpu.get("gpu_executed") is not True):
        raise RuntimeError("original physical proof does not bind the candidate")
    if target == "extended" and (gpu.get("asd_required") is not True
            or any(gpu.get("asd_test_counts", {}).get(n, 0) < 1 for n in ("asd", "extended"))
            or any(not HEX256.fullmatch(gpu.get("asd_profiles", {}).get(n, "")) for n in ("current.json", "current.asd"))):
        raise RuntimeError("original ASD execution and hashes are required")
    if any(gpu.get("test_counts", {}).get(n["component"], 0) < 1 for n in plan["matrix"]):
        raise RuntimeError("original physical execution counters are missing")
    write(destination / "origin.json", {"run_id": int(run_id), "run_attempt": original["run_attempt"],
          "prep_artifact": prep, "gpu_artifact": gpu_artifact, "gpu_job_id": physical[0]["id"],
          "build_artifact": artifact_for(run_id, "mistral-builds-")})
    return plan


def check_report(directory, context):
    directory = Path(directory)
    report = read(directory / "report.json")
    if context["policy"] != read(ROOT / "models.json"):
        raise RuntimeError("model policy changed; new qualification is required")
    if context["fingerprint"] != digest({"runtimes": {r: context["runtimes"][r]["fingerprint"] for r in RUNTIMES},
                                         "asd_profiles": context["asd_profiles"]}):
        raise RuntimeError("combined runtime fingerprint mismatch")
    if (context.get("collection_only") is True or report.get("collection_only") is True
            or report.get("schema") != 1 or report.get("fingerprint") != context["fingerprint"]
            or report.get("full_models_executed") is not True or report.get("status") != "PASS"
            or report.get("sm") != "6.1" or not report.get("gpu_uuid") or not report.get("driver")
            or report.get("asd_profiles") != context["asd_profiles"]):
        raise RuntimeError("complete physical LFM3 qualification is missing")
    if set(report.get("runtimes", {})) != set(RUNTIMES):
        raise RuntimeError("both final runtimes must be qualified")
    for runtime in RUNTIMES:
        actual = report["runtimes"][runtime]
        expected = context["runtimes"][runtime]
        if actual.get("fingerprint") != expected["fingerprint"] or not HEX256.fullmatch(actual.get("binary_sha256", "")):
            raise RuntimeError("model proof runtime fingerprint mismatch")
        if set(actual.get("models", {})) != set(MODES):
            raise RuntimeError("Text, VL and GGUF are all mandatory")
        for mode in MODES:
            result = actual["models"][mode]
            if (result.get("status") != "PASS" or result.get("weights") != context["policy"]["models"][mode]
                    or result.get("cpu_executed") is not True or result.get("cuda_executed") is not True
                    or result.get("prefill_tokens", 0) < 1 or result.get("decode_tokens", 0) < 2
                    or result.get("max_logprob_error", math.inf) > context["policy"]["logprob_abs_tolerance"]):
                raise RuntimeError("incomplete real-model numeric evidence: " + runtime + "/" + mode)
    files = report.get("files", {})
    mandatory = {f"{r}-{m}-{d}.json" for r in RUNTIMES for m in MODES for d in ("cpu", "cuda")}
    if not mandatory.issubset(files):
        raise RuntimeError("raw CPU and CUDA references are missing")
    for name, expected in files.items():
        if Path(name).name != name or not HEX256.fullmatch(expected) or sha(directory / name) != expected:
            raise RuntimeError("model evidence file SHA256 mismatch")
    for runtime in RUNTIMES:
        for mode in MODES:
            numeric = compare(read(directory / f"{runtime}-{mode}-cpu.json"),
                              read(directory / f"{runtime}-{mode}-cuda.json"), context["policy"])
            claimed = report["runtimes"][runtime]["models"][mode]
            if any(claimed.get(k) != v for k, v in numeric.items()):
                raise RuntimeError("claimed model counters do not match raw references")
    return report


def prepare_measurement_reuse(output, ctx, run_id):
    if not ctx["collection_only"]:
        raise RuntimeError("partial measurement reuse is limited to collection-only campaigns")
    prior = api(f"/actions/runs/{run_id}")
    if (prior["path"] != WORKFLOW or prior["event"] != "workflow_dispatch" or prior["head_branch"] != "mirror_orch"
            or prior["status"] != "completed" or prior["head_sha"] != MEASUREMENT_SOURCE_SHA):
        raise RuntimeError("unrecognized original measurement harness or active campaign")
    jobs = api(f"/actions/runs/{run_id}/jobs?per_page=100")
    if len(jobs["jobs"]) != jobs["total_count"]:
        raise RuntimeError("incomplete measurement job inventory")
    for name in ("prepare", "cpu", "cuda_compile"):
        found = [j for j in jobs["jobs"] if j["name"] == name]
        if len(found) != 1 or found[0]["conclusion"] != "success":
            raise RuntimeError("original measurement hosted gates did not pass")
    prep = artifact_for(run_id, "mistral-asd-prep-")
    previous = output / "previous-preparation"
    download_artifact(prep, previous)
    old = read(previous / "context.json")
    if old.get("policy") != ctx["policy"] or old.get("asd_profiles") != ctx["asd_profiles"] or not old.get("collection_only"):
        raise RuntimeError("measurement policy, weights or ASD profiles changed")
    artifact = artifact_for(run_id, "lfm3-model-measurements-")
    directory = output / "reused-measurements"
    download_artifact(artifact, directory)
    report = read(directory / "report.json")
    if (report.get("run_id") != str(run_id) or report.get("run_attempt") != str(prior["run_attempt"])
            or report.get("fingerprint") != old["fingerprint"] or report.get("asd_profiles") != ctx["asd_profiles"]
            or report.get("sm") != ctx["policy"]["sm"] or not report.get("gpu_uuid") or not report.get("driver")):
        raise RuntimeError("measurement report provenance mismatch")
    physical = [j for j in jobs["jobs"] if j["name"] == "SM61 Mistral Extended runtime (candidate=" + ctx["runtimes"]["extended"]["candidate"] + ")"]
    if len(physical) != 1 or physical[0]["status"] != "completed":
        raise RuntimeError("original measurement physical job is missing or active")
    for name, expected in report["files"].items():
        if Path(name).name != name or not HEX256.fullmatch(expected) or sha(directory / name) != expected:
            raise RuntimeError("reused measurement SHA256 mismatch")
    for runtime in RUNTIMES:
        origin = report["runtimes"][runtime]
        if (origin["candidate"] != ctx["runtimes"][runtime]["candidate"]
                or origin["fingerprint"] != old["runtimes"][runtime]["fingerprint"]
                or origin["image"] != ctx["policy"]["image"] or origin["cargo_check"] != "PASS"
                or not HEX256.fullmatch(origin["binary_sha256"])):
            raise RuntimeError("reused executable provenance changed")
        for mode in ("text", "vl"):
            case = origin["models"][mode]
            numeric = measure(read(directory / f"{runtime}-{mode}-cpu.json"),
                              read(directory / f"{runtime}-{mode}-cuda.json"), ctx["policy"])
            if case.get("errors") or case["weights"] != ctx["policy"]["models"][mode] or any(case.get(k) != v for k, v in numeric.items()):
                raise RuntimeError("reused Text/VL references are incomplete or changed")
    ctx["resumed_measurements"] = {"run_id": run_id, "head_sha": prior["head_sha"], "artifact": artifact,
                                    "preparation_artifact": prep, "gpu_job_id": physical[0]["id"]}
    shutil.rmtree(previous)
    print(f"LFM3_MEASUREMENTS_REUSED={run_id} modes=text,vl original numeric verdicts retained")


def prepare(repo, output, legacy_run, extended_run, collection_only=False, resume_measurements_run=0):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    policy = read(ROOT / "models.json")
    ctx = {"schema": 1, "policy": policy, "runtimes": {}, "collection_only": collection_only}
    for target, run_id in (("legacy", legacy_run), ("extended", extended_run)):
        directory = output / "campaigns" / target
        plan = fetch_campaign(repo, target, run_id, directory)
        if plan["image"] != policy["image"]:
            raise RuntimeError("model builder image must match the original CUDA image")
        ctx["runtimes"][target] = {"candidate": plan["candidate"], "source_run_id": int(run_id),
                                   "fingerprint": source_fingerprint(repo, plan["candidate"], plan, policy)}
    # Candidate identities stay in provenance; cache identity uses executable inputs.
    ctx["asd_profiles"] = read(output / "campaigns/extended/physical/report.json")["asd_profiles"]
    ctx["fingerprint"] = digest({"runtimes": {r: ctx["runtimes"][r]["fingerprint"] for r in RUNTIMES},
                                  "asd_profiles": ctx["asd_profiles"]})
    ctx["reuse_models"] = False
    if resume_measurements_run:
        prepare_measurement_reuse(output, ctx, resume_measurements_run)
    name = "lfm3-model-proof-" + ctx["fingerprint"]
    data = {"artifacts": []} if collection_only else api("/actions/artifacts?name=" + name + "&per_page=100")
    for artifact in data["artifacts"]:
        if artifact["expired"]:
            continue
        prior = api("/actions/runs/" + str(artifact["workflow_run"]["id"]))
        if (prior["path"] != WORKFLOW or prior["event"] != "workflow_dispatch"
                or prior["head_branch"] != "mirror_orch" or prior["conclusion"] != "success"):
            continue
        download_artifact(artifact, output / "model-proof")
        check_report(output / "model-proof", ctx)
        ctx["reuse_models"] = True
        ctx["reused_artifact"] = artifact
        break
    write(output / "context.json", ctx)
    shutil.copy2(output / "campaigns/extended/plan.json", output / "plan.json")
    shutil.copy2(output / "campaigns/extended/candidate.bundle", output / "candidate.bundle")
    shutil.copytree(ROOT, output / "lfm3", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT.parent / "campaign", output / "campaign", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    extended_plan = read(output / "plan.json")
    env_key = hashlib.sha256(json.dumps([extended_plan["image"], extended_plan["config"]["toolchain"],
                                      extended_plan["config"]["features"], "sm61", "bf16,fp8"], sort_keys=True).encode()).hexdigest()
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in {"candidate": ctx["runtimes"]["extended"]["candidate"], "fingerprint": ctx["fingerprint"],
                           "reuse_models": str(ctx["reuse_models"]).lower(),
                           "environment_key": env_key, "cache_key": extended_plan["input_key"]}.items():
            stream.write(f"{key}={value}\n")
    print("LFM3_PREPARED=" + ctx["fingerprint"] + " REUSE_MODELS=" + str(ctx["reuse_models"]))


def build(repo, output, target):
    output = Path(output).resolve()
    ctx = read(output / "context.json")
    plan = import_bundle(repo, output / "campaigns" / target)
    run("git", "-C", repo, "switch", "--detach", "--force", plan["candidate"])
    if source_fingerprint(repo, plan["candidate"], plan, ctx["policy"]) != ctx["runtimes"][target]["fingerprint"]:
        raise RuntimeError("runtime inputs changed before compilation")
    env = dict(os.environ)
    env.update(CUDA_COMPUTE_CAP="61", ALLOW_LEGACY="bf16,fp8", CARGO_BUILD_JOBS="2", CARGO_INCREMENTAL="0")
    kernel_tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "HEAD", "--",
        "mistralrs-core/build.rs", "mistralrs-core/src/cuda", "mistralrs-paged-attn/build.rs",
        "mistralrs-paged-attn/src/cuda", "mistralrs-quant/build.rs", "mistralrs-quant/kernels"])
    env["MISTRALRS_CUDA_BUILD_ROOT"] = str(Path(env["CAMPAIGN_CACHE"]) / "cuda" / hashlib.sha256(kernel_tree).hexdigest())
    features = ",".join(plan["config"]["features"])
    args = ["--locked", "--profile", "campaign", "-p", "mistralrs-cli", "--no-default-features", "--features", features]
    subprocess.run(["cargo", "check", *args], env=env, cwd=repo, check=True)
    subprocess.run(["cargo", "build", *args, "--bin", "mistralrs"], env=env, cwd=repo, check=True)
    dest = output / "binaries" / target
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "campaign/mistralrs", dest / "mistralrs")
    if ctx.get("resumed_measurements"):
        previous = read(output / "reused-measurements/report.json")
        if sha(dest / "mistralrs") != previous["runtimes"][target]["binary_sha256"]:
            raise RuntimeError("compiled executable changed; previous model measurements cannot be reused")
    subprocess.run(["python3", ROOT.parent / "campaign/cpu-runtime.py", "--", dest / "mistralrs", "serve", "--help"],
                   check=True, stdout=subprocess.DEVNULL)
    write(dest / "build.json", {"schema": 1, **ctx["runtimes"][target], "binary_sha256": sha(dest / "mistralrs"),
                                "image": plan["image"], "cargo_check": "PASS"})


def image_data():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = b"".join(b"\0" + b"".join(bytes((255, 0, 0) if x < IMAGE_EDGE // 2 else (0, 0, 255))
                                                   for x in range(IMAGE_EDGE)) for _ in range(IMAGE_EDGE))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", IMAGE_EDGE, IMAGE_EDGE, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def model_paths(policy):
    paths = {}
    for mode in MODES:
        path = Path(os.environ["HARICOT_LFM3_" + mode.upper() + "_SNAPSHOT"])
        spec = policy["models"][mode]
        if path.name != spec["revision"] or path.parent.name != "snapshots":
            raise RuntimeError("snapshot revision mismatch: " + mode)
        for name, expected in spec["files"].items():
            if sha(path / name) != expected:
                raise RuntimeError("model file SHA256 mismatch: " + mode + "/" + name)
        paths[mode] = path
    return paths


def http(port, endpoint, payload=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}" + endpoint,
                                    data=json.dumps(payload).encode() if payload else None,
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.load(response)


def infer(binary, mode, device, paths, policy, log):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = [str(binary), "serve", "-m", str(paths[mode]), "--dtype", policy["dtype"],
            "--host", "127.0.0.1", "--port", str(port), "--no-ui", "--paged-attn", "off",
            "--max-model-len", str(policy["max_model_len"])]
    if mode == "vl":
        args += ["--tokenizer", str(paths["vl"] / "tokenizer.json"), "--max-edge", "128", "--max-num-images", "1"]
    elif mode == "text":
        args += ["--tokenizer", str(paths["text"] / "tokenizer.json")]
    if mode == "gguf":
        args += ["--format", "gguf", "-f", "LFM3-1.7B-Q4_K_M.gguf", "--tok-model-id", str(paths["text"])]
    if device == "cpu":
        args += ["--cpu"]
    else:
        config = read(paths["vl" if mode == "vl" else "text"] / "config.json")
        config = config.get("text_config", config)
        args += ["--device-layers", "0:" + str(config["num_hidden_layers"])]
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_TOKEN="", HUGGING_FACE_HUB_TOKEN="",
               HF_TOKEN_PATH="/dev/null", HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
               ALL_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost", CUDA_VISIBLE_DEVICES="0")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with Path(log).open("w") as stream:
        process = subprocess.Popen(args, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + READY_TIMEOUT
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"model server failed ({mode}/{device}); see {log}")
                try:
                    models = http(port, "/v1/models")["data"]
                    model = models[0]["id"]
                    break
                except (OSError, ValueError, KeyError, IndexError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("model loading timed out")
                    time.sleep(1)
            content = "Continue this list with a few words: one, two, three,"
            if mode == "vl":
                content = [{"type": "text", "text": "Describe the two colors in this image briefly."},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_data()).decode()}}]
            payload = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                       "max_tokens": policy["max_tokens"], "logprobs": True, "top_logprobs": policy["top_logprobs"],
                       "enable_thinking": False, "stream": False, "seed": 44}
            response = http(port, "/v1/chat/completions", payload)
            return {"device": device, "request": payload, "response": response,
                    "fixture_sha256": hashlib.sha256(image_data()).hexdigest() if mode == "vl" else None}
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def token_key(token):
    return json.dumps([token["token"], token.get("bytes")], sort_keys=True)


def measure(cpu, cuda, policy):
    if (cpu.get("device") != "cpu" or cuda.get("device") != "cuda" or cpu.get("request") != cuda.get("request")
            or cpu.get("fixture_sha256") != cuda.get("fixture_sha256")):
        raise RuntimeError("CPU/CUDA requests, image fixture or device identities differ")
    observations = []
    minimum = policy["max_tokens"]
    for evidence in (cpu, cuda):
        response = evidence["response"]
        if response.get("usage", {}).get("prompt_tokens", 0) < 1:
            raise RuntimeError("prefill execution count is missing")
        if response.get("usage", {}).get("completion_tokens", 0) < minimum:
            raise RuntimeError("decode execution is shorter than the required sample")
        tokens = response["choices"][0].get("logprobs", {}).get("content", [])
        if len(tokens) < minimum:
            raise RuntimeError("numeric token references are missing")
        observations.append(tokens)
    if len(observations[0]) != len(observations[1]):
        raise RuntimeError("CPU/CUDA decode lengths differ")
    violations, positions, generated_errors, top_errors, probability_errors = [], [], [], [], []
    common_counts = []
    same_prefix = True
    for index, (left, right) in enumerate(zip(*observations, strict=True)):
        equal = token_key(left) == token_key(right)
        if not equal:
            violations.append("CPU/CUDA greedy decode diverged")
        for item in (left, right):
            if not math.isfinite(item["logprob"]) or len(item.get("top_logprobs", [])) < policy["top_logprobs"]:
                raise RuntimeError("non-finite or incomplete numeric reference")
        generated_error = abs(left["logprob"] - right["logprob"])
        generated_errors.append(generated_error)
        a = {token_key(t): t["logprob"] for t in left["top_logprobs"]}
        b = {token_key(t): t["logprob"] for t in right["top_logprobs"]}
        if (len(a) != policy["top_logprobs"] or len(b) != policy["top_logprobs"]
                or any(not math.isfinite(x) for x in [*a.values(), *b.values()])):
            raise RuntimeError("non-finite, duplicate or incomplete top token references")
        common = a.keys() & b.keys()
        common_counts.append(len(common))
        if len(common) < policy["minimum_common_top_tokens"]:
            violations.append("CPU/CUDA top token overlap is insufficient")
        if a.keys() != b.keys():
            for current, other in ((a, b), (b, a)):
                cutoff = min(current.values())
                for key in current.keys() - other.keys():
                    value = current[key]
                    if (value > math.log(policy["top_boundary_max_probability"])
                            or value - cutoff > policy["logprob_abs_tolerance"]):
                        violations.append("CPU/CUDA significant top token set diverged")
        top = []
        for key in sorted(common):
            error = abs(a[key] - b[key])
            probability_error = abs(math.exp(a[key]) - math.exp(b[key]))
            top_errors.append(error)
            probability_errors.append(probability_error)
            top.append({"token": json.loads(key), "cpu_logprob": a[key], "cuda_logprob": b[key],
                        "logprob_error": error, "cpu_probability": math.exp(a[key]),
                        "cuda_probability": math.exp(b[key]), "probability_error": probability_error})
        positions.append({"index": index, "same_prefix": same_prefix, "greedy_equal": equal,
                          "cpu_token": json.loads(token_key(left)), "cuda_token": json.loads(token_key(right)),
                          "generated_logprob_error": generated_error, "common_top_tokens": top,
                          "cpu_only": [{"token": json.loads(k), "logprob": a[k]} for k in sorted(a.keys() - b.keys())],
                          "cuda_only": [{"token": json.loads(k), "logprob": b[k]} for k in sorted(b.keys() - a.keys())]})
        same_prefix = same_prefix and equal
    maximum = max(generated_errors + top_errors)
    if maximum > policy["logprob_abs_tolerance"]:
        violations.append(f"model numeric parity failed: max logprob error {maximum}")
    return {"status": "FAIL" if violations else "PASS", "cpu_executed": True, "cuda_executed": True,
            "prefill_tokens": cpu["response"]["usage"]["prompt_tokens"],
            "decode_tokens": len(observations[0]), "max_logprob_error": maximum,
            "violations": list(dict.fromkeys(violations)),
            "metrics": {"greedy_tokens_equal": same_prefix, "max_generated_logprob_error": max(generated_errors),
                        "max_common_top_logprob_error": max(top_errors, default=0),
                        "max_common_top_probability_error": max(probability_errors, default=0),
                        "minimum_common_top_tokens": min(common_counts), "positions": positions}}


def compare(cpu, cuda, policy):
    result = measure(cpu, cuda, policy)
    if result["status"] != "PASS":
        raise RuntimeError("; ".join(result["violations"]))
    return result


def save_measurements(dest, report):
    report["files"] = {p.name: sha(p) for p in dest.iterdir()
                       if p.suffix in (".json", ".log", ".md") and p.name != "report.json"}
    write(dest / "report.json", report)


def reuse_case(output, ctx, runtime, mode, build_proof):
    directory = output / "reused-measurements"
    old = read(directory / "report.json")
    origin = old["runtimes"][runtime]
    if (origin["candidate"] != build_proof["candidate"] or origin["binary_sha256"] != build_proof["binary_sha256"]
            or old["asd_profiles"] != ctx["asd_profiles"]):
        raise RuntimeError("reused model observations do not match the exact executable or ASD profile")
    pair = {}
    for device in ("cpu", "cuda"):
        name = f"{runtime}-{mode}-{device}"
        for suffix in (".json", ".log"):
            filename = name + suffix
            if filename not in old["files"] or sha(directory / filename) != old["files"][filename]:
                raise RuntimeError("reused raw observation or server log changed")
            shutil.copy2(directory / filename, output / "model-proof" / filename)
        pair[device] = read(directory / (name + ".json"))
    case = measure(pair["cpu"], pair["cuda"], ctx["policy"])
    case.update(weights=ctx["policy"]["models"][mode], errors={}, evidence_origin={
        "run_id": old["run_id"], "run_attempt": old["run_attempt"], "gpu_uuid": old["gpu_uuid"],
        "driver": old["driver"], "candidate": origin["candidate"], "binary_sha256": origin["binary_sha256"],
        "artifact_sha256": ctx["resumed_measurements"]["artifact"]["digest"], "reused": True})
    return case


def collect_models(output, ctx, paths, report):
    dest = output / "model-proof"
    dest.mkdir(exist_ok=True)
    for runtime in RUNTIMES:
        directory = output / "binaries" / runtime
        build_proof = read(directory / "build.json")
        binary = directory / "mistralrs"
        if (sha(binary) != build_proof["binary_sha256"] or build_proof["fingerprint"] != ctx["runtimes"][runtime]["fingerprint"]
                or build_proof["cargo_check"] != "PASS" or build_proof["candidate"] != ctx["runtimes"][runtime]["candidate"]):
            raise RuntimeError("compiled runtime verification failed")
        binary.chmod(0o755)
        actual = report["runtimes"][runtime] = {**build_proof, "models": {}}
        for mode in MODES:
            if ctx.get("resumed_measurements") and mode in ("text", "vl"):
                actual["models"][mode] = reuse_case(output, ctx, runtime, mode, build_proof)
                print(f"LFM3_MODEL_REUSED={runtime}-{mode} original_run={ctx['resumed_measurements']['run_id']} "
                      f"status={actual['models'][mode]['status']}", flush=True)
                save_measurements(dest, report)
                continue
            pair = {}
            result = actual["models"][mode] = {"status": "COLLECTING", "cpu_executed": False,
                "cuda_executed": False, "weights": ctx["policy"]["models"][mode], "errors": {},
                "evidence_origin": {"run_id": report.get("run_id"), "run_attempt": report.get("run_attempt"),
                                    "gpu_uuid": report.get("gpu_uuid"), "driver": report.get("driver"), "reused": False}}
            for device in ("cpu", "cuda"):
                name = f"{runtime}-{mode}-{device}"
                print("LFM3_MODEL_EXECUTE=" + name, flush=True)
                try:
                    pair[device] = infer(binary, mode, device, paths, ctx["policy"], dest / (name + ".log"))
                    write(dest / (name + ".json"), pair[device])
                    result[device + "_executed"] = True
                except MODEL_ERRORS as error:
                    result["errors"][device] = str(error)
                    print(f"LFM3_MODEL_EXECUTION_ERROR={name} reason={error}", flush=True)
                save_measurements(dest, report)
            if len(pair) == 2:
                try:
                    result.update(measure(pair["cpu"], pair["cuda"], ctx["policy"]))
                except MODEL_ERRORS as error:
                    result["errors"]["references"] = str(error)
            if result["errors"]:
                result["status"] = "ERROR"
            print(f"LFM3_MODEL_RESULT={runtime}-{mode} status={result['status']} "
                  f"max_logprob_error={result.get('max_logprob_error', 'unavailable')}", flush=True)
            save_measurements(dest, report)
    results = [report["runtimes"][r]["models"][m] for r in RUNTIMES for m in MODES]
    complete = all(x["cpu_executed"] and x["cuda_executed"] and not x["errors"] for x in results)
    passed = all(x["status"] == "PASS" for x in results)
    report.update(collection_complete=complete, full_models_executed=complete,
                  collection_only=ctx.get("collection_only", False), numeric_status="PASS" if passed else "FAIL",
                  promotion_eligible=passed and not ctx.get("collection_only", False))
    report["status"] = ("COLLECTED" if report["collection_only"] else "PASS") if complete and (passed or report["collection_only"]) else "FAIL"
    lines = ["# LFM3 physical measurements", "", f"Collection complete: {complete}. Numeric verdict: {report['numeric_status']}.",
             f"Collection only: {report['collection_only']}. Promotion eligible: {report['promotion_eligible']}.", "",
             f"Unchanged absolute logprob tolerance: {ctx['policy']['logprob_abs_tolerance']}.", "",
             "| Runtime | Model | CPU/CUDA executed | Greedy equal | Max generated logprob error | Max common top logprob error | Max common top probability error | Evidence run | Verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for runtime in RUNTIMES:
        for mode in MODES:
            x = report["runtimes"][runtime]["models"][mode]
            metrics = x.get("metrics", {})
            values = [runtime, mode, f"{x['cpu_executed']}/{x['cuda_executed']}", metrics.get("greedy_tokens_equal", "n/a")]
            values += [f"{metrics[k]:.7g}" if k in metrics else "n/a" for k in
                       ("max_generated_logprob_error", "max_common_top_logprob_error", "max_common_top_probability_error")]
            evidence = x["evidence_origin"]
            origin = str(evidence["run_id"]) + (" (reused)" if evidence["reused"] else "")
            lines.append("| " + " | ".join(map(str, values + [origin, x["status"]])) + " |")
    for runtime in RUNTIMES:
        for mode in MODES:
            x = report["runtimes"][runtime]["models"][mode]
            for reason in [*x.get("violations", []), *x["errors"].values()]:
                lines.append(f"\n{runtime}/{mode}: {reason}")
    lines += ["", "Raw JSON and logs are retained per device. Probability errors are absolute, not percentages.",
              "After a greedy divergence, later positions use different prefixes and cannot calibrate parity.",
              "A collection-only report cannot authorize promotion, even when all numeric checks pass.", ""]
    summary = "\n".join(lines)
    (dest / "summary.md").write_text(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(summary)
    save_measurements(dest, report)
    print(summary, flush=True)
    return complete, passed


def execute(output):
    output = Path(output).resolve()
    ctx = read(output / "context.json")
    if os.environ.get("EXPECTED_RUN_ID") != os.environ.get("GITHUB_RUN_ID"):
        raise RuntimeError("qualification run identity mismatch")
    if os.environ.get("EXPECTED_CANDIDATE_SHA") != ctx["runtimes"]["extended"]["candidate"]:
        raise RuntimeError("qualification candidate identity mismatch")
    if os.environ.get("RUNNER_REPO") != REPOSITORY or os.environ.get("ASD_REQUIRED") != "true":
        raise RuntimeError("dedicated Extended runner and ASD isolation are required")
    for path, name in (("/run/haricot/asd/fallbacks/current.json", "current.json"),
                       ("/run/haricot/asd/profiles/current.asd", "current.asd")):
        if sha(path) != ctx["asd_profiles"][name]:
            raise RuntimeError("ASD read-only profile differs from the qualified original campaign")
    gpu = run("nvidia-smi", "--query-gpu=uuid,compute_cap,driver_version", "--format=csv,noheader").splitlines()
    if len(gpu) != 1 or gpu[0].split(",")[1].strip() != ctx["policy"]["sm"]:
        raise RuntimeError("one physical Pascal SM61 device is required")
    uuid, sm, driver = [x.strip() for x in gpu[0].split(",")]
    paths = model_paths(ctx["policy"])
    dest = output / "model-proof"
    dest.mkdir(exist_ok=True)
    report = {"schema": 1, "status": "COLLECTING", "full_models_executed": False, "sm": sm,
              "gpu_uuid": uuid, "driver": driver, "run_id": os.environ["GITHUB_RUN_ID"],
              "run_attempt": os.environ["GITHUB_RUN_ATTEMPT"], "fingerprint": ctx["fingerprint"],
              "asd_profiles": ctx["asd_profiles"], "runtimes": {}, "files": {}}
    complete, passed = collect_models(output, ctx, paths, report)
    if not complete:
        raise RuntimeError("all six model cases attempted; collection incomplete, see model-proof/summary.md")
    print("LFM3_COLLECTION_COMPLETE=6/6", flush=True)
    if ctx.get("collection_only"):
        print("LFM3_MEASUREMENTS_COLLECTED=PASS LFM3_PROMOTION_ALLOWED=false", flush=True)
        return
    if not passed:
        raise RuntimeError("all six model cases collected; numeric qualification refused, see model-proof/summary.md")
    check_report(dest, ctx)
    print("LFM3_FULL_MODEL_QUALIFICATION=PASS")


def attach(repo, plan_path, proofs):
    plan = read(plan_path)
    proofs = Path(proofs)
    context = read(proofs / "model-evidence/context.json")
    report = check_report(proofs / "model-evidence/model-proof", context)
    target = plan["target"]
    if target == "extended" and read(proofs / "gpu/report.json").get("asd_profiles") != context["asd_profiles"]:
        raise RuntimeError("supplementary ASD fingerprints do not match the original physical proof")
    expected = source_fingerprint(repo, plan["candidate"], plan, context["policy"])
    if report["runtimes"][target]["fingerprint"] != expected:
        raise RuntimeError("model qualification is stale for this source/backend/environment")
    return report


def promote(repo, output, evidence_run):
    if not evidence_run:
        raise RuntimeError("explicit successful qualification run is required")
    original = api(f"/actions/runs/{evidence_run}")
    if (original["path"] != WORKFLOW or original["event"] != "workflow_dispatch"
            or original["head_branch"] != "mirror_orch" or original["conclusion"] != "success"):
        raise RuntimeError("qualification workflow is not successfully completed")
    output = Path(output).resolve()
    download_artifact(artifact_for(evidence_run, "mistral-asd-prep-"), output)
    ctx = read(output / "context.json")
    if ctx.get("collection_only"):
        raise RuntimeError("measurement collection is not an accepted numeric qualification")
    model_dir = output / "model-evidence"
    model_dir.mkdir(exist_ok=True)
    shutil.copy2(output / "context.json", model_dir / "context.json")
    artifact = artifact_for(evidence_run, "lfm3-model-proof-")
    if artifact["name"] != "lfm3-model-proof-" + ctx["fingerprint"]:
        raise RuntimeError("qualification artifact name does not bind its fingerprint")
    download_artifact(artifact, model_dir / "model-proof")
    check_report(model_dir / "model-proof", ctx)
    spec = importlib.util.spec_from_file_location("campaign", ROOT.parent / "campaign/campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    for target in RUNTIMES:
        directory = output / "campaigns" / target
        origin = read(directory / "origin.json")
        plan = import_bundle(repo, directory)
        if plan["candidate"] != ctx["runtimes"][target]["candidate"]:
            raise RuntimeError("qualification references another original candidate")
        proof_dir = output / "promotion" / target
        download_artifact(origin["build_artifact"], proof_dir)
        shutil.copytree(directory / "physical", proof_dir / "gpu", dirs_exist_ok=True)
        shutil.copytree(model_dir, proof_dir / "model-evidence", dirs_exist_ok=True)
        campaign.promote(repo, directory / "plan.json", proof_dir, origin["run_id"])
        print("LFM3_PROMOTION=PASS target=" + target + " candidate=" + plan["candidate"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "build", "execute", "verify", "promote"))
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--legacy-run", type=int, default=41080696084)
    p.add_argument("--extended-run", type=int, default=41080753574)
    p.add_argument("--target", choices=RUNTIMES)
    p.add_argument("--evidence-run", type=int)
    p.add_argument("--collection-only", action="store_true")
    p.add_argument("--resume-measurements-run", type=int, default=0)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.repo, a.output, a.legacy_run, a.extended_run, a.collection_only, a.resume_measurements_run)
    elif a.action == "build":
        build(a.repo, a.output, a.target)
    elif a.action == "execute":
        execute(a.output)
    elif a.action == "verify":
        check_report(a.output / "model-proof", read(a.output / "context.json"))
    elif a.action == "promote":
        promote(a.repo, a.output, a.evidence_run)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("LFM3_QUALIFICATION_REJECT=" + str(error))


