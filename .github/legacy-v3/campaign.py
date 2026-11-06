import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent
REPOSITORY = "haricot/mistral.rs"
WORKFLOW = ".github/workflows/mistral-legacy-v3.yml"
BRANCH = "mirror_orch"
SHA = re.compile(r"[0-9a-f]{40}\Z")
SDK = re.compile(r"ghcr.io/[a-z0-9/._-]+@sha256:[0-9a-f]{64}\Z")
CPU_REQUIRED = ["cpu_backend_remains_available", "campaign_force_cpu_selects_cpu",
                "campaign_cpu_flag_is_available", "campaign_cpu_fp8_kv",
                "row_multipliers_precede_bias_and_restore_batch_shape", "fp4_cpu_mxfp4_embedding"]
GPU_REQUIRED = ["legacy_bf16_unquantized_linear_avoids_unsupported_dispatch",
                "sm61_bf16_gemm_preserves_values_outside_f16_range",
                "sm61_fp8_blockwise_packed_f16_w8a16", "sm61_fp8_blockwise_packed_bf16_preserves_range",
                "sm61_single_packed_cuda_resident_copy", "sm61_moe_experts_single_resident_weight_each",
                "sm61_fp8_moe_groups_experts_without_dense_weights", "fp4_cuda_nvfp4_sm61_w4a16",
                "legacy_sm61_moe_f16_bf16_prefill", "campaign_cuda_fp8_kv_sm61_correctness_and_performance"]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def command(*args, cwd=None, env=None):
    return subprocess.check_output([str(a) for a in args], cwd=cwd, env=env, text=True).strip()


def git(repo, *args):
    return command("git", "-C", repo, *args)


def api(path):
    return json.loads(command("gh", "api", "repos/" + REPOSITORY + path))


def emit(**values):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for name, value in values.items():
                stream.write(f"{name}={value}\n")


def origin():
    return {"repository": REPOSITORY, "workflow": WORKFLOW, "branch": os.environ.get("GITHUB_REF_NAME", "local"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")),
            "control_sha": os.environ.get("GITHUB_SHA", "local")}


def recipe_files(repo):
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.sh")) + list(ROOT.glob("*.toml"))
    files += [ROOT / "config.json", ROOT / "models.json", repo / WORKFLOW,
              repo / ".github/campaign/cpu-runtime.py", repo / ".github/lfm3/qualification.py"]
    return {str(p.relative_to(repo)): sha(p) for p in sorted(files)}


def identity(candidate, tree, lock, source_contract, support, image, environment_key, cpu_profile, recipe):
    if cpu_profile not in ("portable", "avx2") or not SDK.fullmatch(image):
        raise ValueError("invalid compiler profile or immutable SDK")
    return {"schema": 1, "candidate": candidate, "tree": tree, "lock": lock,
            "sources": source_contract, "support": support, "image": image,
            "environment_key": environment_key, "cpu_profile": cpu_profile,
            "rustflags": "-C target-cpu=" + ("x86-64" if cpu_profile == "portable" else "x86-64-v3"),
            "rust": "1.99.0", "profile": "release", "cuda_compute_cap": "61",
            "features": ["mistralrs-cli/cuda-legacy", "mistralrs-cli/code-execution"], "recipe": recipe}


def verify_plan(directory):
    directory = Path(directory)
    plan = read(directory / "plan.json")
    if plan["key"] != digest(plan["identity"]) or plan["candidate"] != plan["identity"]["candidate"]:
        raise ValueError("plan identity mismatch")
    if sha(directory / "candidate.bundle") != plan["bundle_sha256"]:
        raise ValueError("candidate bundle mismatch")
    for name, expected in plan["identity"]["recipe"].items():
        if sha(directory / "orchestrator" / name) != expected:
            raise ValueError("recipe changed: " + name)
    return plan


def prepare(repo, output, image, environment_key, cpu_profile):
    config = read(ROOT / "config.json")
    support = config["support_sha"]
    if not SHA.fullmatch(support) or output.exists():
        raise ValueError("invalid support SHA or existing output")
    for proof in config["source_cpu_proofs"]:
        run = api("/actions/runs/" + str(proof["run_id"]))
        if run.get("head_sha") != proof["sha"] or run.get("status") != "completed" or run.get("conclusion") != "success":
            raise ValueError("source CPU qualification is not exact and successful")
    output.mkdir(parents=True)
    work = output / "source"
    source_contract = json.loads(git(repo, "show", support + ":legacy-v3-support/sources.json"))
    source_refs = [support, source_contract["base_master_sha"], source_contract["source_legacy_sha"],
                   *[s["sha"] for s in source_contract["sources"]]]
    for ref in source_refs:
        git(repo, "cat-file", "-e", ref + "^{commit}")
    fixed_epoch = max(int(git(repo, "show", "-s", "--format=%ct", ref)) for ref in source_refs)
    with tempfile.TemporaryDirectory() as temporary:
        module = Path(temporary) / "assemble.py"
        module.write_text(git(repo, "show", support + ":legacy-v3-support/assemble.py"))
        loader = importlib.util.spec_from_file_location("legacy_v3_assembly", module)
        assembly = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(assembly)
        previous = {name: os.environ.get(name) for name in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE")}
        try:
            for name in previous:
                os.environ[name] = f"@{fixed_epoch} +0000"
            assembly.compose(repo, work, source_contract, support)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    candidate = git(work, "rev-parse", "HEAD")
    tree = git(work, "rev-parse", "HEAD^{tree}")
    locked = identity(candidate, tree, sha(work / "Cargo.lock"), source_contract, support,
                      image, environment_key, cpu_profile, recipe_files(repo))
    key = digest(locked)
    ref = "refs/heads/integration/legacy-v3/" + key
    git(repo, "update-ref", ref, candidate)
    git(repo, "bundle", "create", output / "candidate.bundle", ref)
    target = command("git", "ls-remote", "--heads", "origin", "refs/heads/legacy_v3", cwd=repo)
    lease = target.split()[0] if target else None
    plan = {"schema": 1, "kind": "legacy-v3-pilot", "target": "legacy_v3", "candidate": candidate,
            "candidate_ref": ref, "key": key, "identity": locked, "origin": origin(),
            "target_lease": lease, "bundle_sha256": sha(output / "candidate.bundle"),
            "numeric_policy": {"text": "blocking", "vl": "informative", "gguf": "informative"}}
    for name in locked["recipe"]:
        dest = output / "orchestrator" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, dest)
    write(output / "plan.json", plan)
    verify_plan(output)
    emit(candidate=candidate, key=key, rustflags=locked["rustflags"], cpu_profile=cpu_profile)
    return plan


def junit(path, required):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if any(c.find("failure") is not None or c.find("error") is not None for c in cases):
        raise ValueError("failing test in JUnit")
    passed = [c for c in cases if c.find("skipped") is None]
    counts = {name: sum(name in c.get("name", "") for c in passed) for name in required}
    if any(value < 1 for value in counts.values()):
        raise ValueError("required test not executed: " + str(counts))
    return {"passed": len(passed), "required": counts}


def verify_files(directory, proof):
    for name, expected in proof["files"].items():
        if Path(name).name != name or sha(Path(directory) / name) != expected:
            raise ValueError("artifact payload mismatch: " + name)


def cpu_proof(plan, directory):
    directory = Path(directory)
    value = {"schema": 1, "status": "CPU_BASELINE_PASSED", "identity": plan["identity"], "key": plan["key"],
             "origin": origin(), "cuda_enabled": False, "command": "cargo check --locked -p mistralrs-cli --no-default-features --features code-execution",
             "files": {name: sha(directory / name) for name in ("Cargo.lock", "check.log")}}
    if value["files"]["Cargo.lock"] != plan["identity"]["lock"]:
        raise ValueError("CPU baseline lock mismatch")
    write(directory / "cpu.json", value)


def verify_cpu(plan, directory):
    proof = read(Path(directory) / "cpu.json")
    if (proof.get("status") != "CPU_BASELINE_PASSED" or proof.get("key") != plan["key"]
            or proof.get("identity") != plan["identity"] or proof.get("cuda_enabled") is not False
            or set(proof["files"]) != {"Cargo.lock", "check.log"}
            or proof["files"]["Cargo.lock"] != plan["identity"]["lock"]):
        raise ValueError("CPU-only baseline proof mismatch")
    verify_files(directory, proof)
    return proof


def build_proof(plan, directory):
    directory = Path(directory)
    report = {"schema": 1, "status": "BUILD_AND_CPU_PASSED", "key": plan["key"],
              "identity": plan["identity"], "origin": origin(), "gpu_executed": False,
              "cpu": junit(directory / "cpu-junit.xml", CPU_REQUIRED),
              "files": {name: sha(directory / name) for name in ["tests.tar.zst", "mistralrs", "cargo-nextest",
                        "cpu-junit.xml", "Cargo.lock", "compiler.txt", "nextest.toml"]}}
    if report["files"]["Cargo.lock"] != plan["identity"]["lock"]:
        raise ValueError("compiled lockfile differs")
    write(directory / "build.json", report)
    return report


def verify_build(plan, directory):
    proof = read(Path(directory) / "build.json")
    if proof.get("status") != "BUILD_AND_CPU_PASSED" or proof.get("key") != plan["key"] or proof.get("identity") != plan["identity"]:
        raise ValueError("build belongs to different inputs")
    verify_files(directory, proof)
    if set(proof["files"]) != {"tests.tar.zst", "mistralrs", "cargo-nextest", "cpu-junit.xml", "Cargo.lock", "compiler.txt", "nextest.toml"}:
        raise ValueError("incomplete build payload manifest")
    if proof["files"]["Cargo.lock"] != plan["identity"]["lock"] or proof["cpu"] != junit(Path(directory) / "cpu-junit.xml", CPU_REQUIRED):
        raise ValueError("invalid CPU build proof")
    return proof


def numeric_accepted(mode, result):
    if not result.get("cpu_executed") or not result.get("cuda_executed") or result.get("errors"):
        return False
    return result.get("status") in ("PASS", "FAIL") and (mode in ("vl", "gguf") or result.get("status") == "PASS")


def measure_pair(helper, cpu, cuda, policy):
    try:
        return helper.measure(cpu, cuda, policy)
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError, ArithmeticError) as error:
        return {"status": "FAIL", "numeric_error": str(error)}


def verify_gpu(plan, directory, build):
    proof = read(Path(directory) / "gpu.json")
    if (proof.get("status") != "GPU_AND_MODELS_PASSED" or proof.get("key") != plan["key"]
            or proof.get("identity") != plan["identity"] or proof.get("binary_sha256") != build["files"]["mistralrs"]
            or proof.get("archive_sha256") != build["files"]["tests.tar.zst"]
            or proof.get("compute_capability") != "6.1" or proof.get("full_models_executed") is not True):
        raise ValueError("physical evidence identity mismatch")
    verify_files(directory, proof)
    if proof["tests"] != junit(Path(directory) / "gpu-junit.xml", GPU_REQUIRED):
        raise ValueError("GPU contract evidence incomplete")
    if set(proof["models"]) != {"text", "vl", "gguf"} or any(not numeric_accepted(mode, result) for mode, result in proof["models"].items()):
        raise ValueError("real model execution or blocking text comparison failed")
    if proof["models"] != read(Path(directory) / "models.json"):
        raise ValueError("model summary differs from raw report")
    loader = importlib.util.spec_from_file_location("legacy_v3_numeric_verification", ROOT.parent / "lfm3/qualification.py")
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    policy = read(ROOT / "models.json")
    for mode, result in proof["models"].items():
        if result["weights"] != policy["models"][mode]:
            raise ValueError("model weight identity mismatch")
        names = [f"{mode}-cpu.json", f"{mode}-cuda.json"]
        if not set(names) <= set(proof["files"]):
            raise ValueError("model execution responses are missing")
        measured = measure_pair(helper, read(Path(directory) / names[0]), read(Path(directory) / names[1]), policy)
        if any(result.get(name) != value for name, value in measured.items()):
            raise ValueError("recorded numeric observations differ from raw responses")
    return proof


def safe_extract(archive, destination):
    with zipfile.ZipFile(archive) as payload:
        for entry in payload.infolist():
            name = Path(entry.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("unsafe artifact archive member")
        payload.extractall(destination)


def download(artifact_id, directory):
    entry = api("/actions/artifacts/" + str(int(artifact_id)))
    if entry.get("expired") is not False or entry.get("size_in_bytes", 0) <= 0:
        raise ValueError("artifact absent or expired")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "artifact.zip"
        with archive.open("wb") as stream:
            subprocess.run(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{int(artifact_id)}/zip"], stdout=stream, check=True)
        expected = entry.get("digest")
        if expected and expected != "sha256:" + sha(archive):
            raise ValueError("GitHub artifact digest mismatch")
        safe_extract(archive, directory)
    return entry


def verify_producer(proof, role, entry):
    source = proof["origin"]
    run = api("/actions/runs/" + str(source["run_id"]))
    if (run.get("path") != WORKFLOW or run.get("head_branch") != BRANCH or run.get("head_sha") != source["control_sha"]
            or run.get("head_repository", {}).get("full_name") != REPOSITORY
            or str(entry.get("workflow_run", {}).get("id")) != str(source["run_id"])):
        raise ValueError("artifact producer provenance mismatch")
    response = api(f"/actions/runs/{source['run_id']}/attempts/{source['run_attempt']}/jobs?per_page=100")
    if len(response.get("jobs", [])) != response.get("total_count"):
        raise ValueError("producer job list incomplete")
    jobs = [job for job in response["jobs"] if job["name"] == role or (role == "Legacy V3 SM61" and job["name"].startswith(role + " (candidate="))]
    if len(jobs) != 1 or jobs[0]["conclusion"] != "success":
        raise ValueError("original producer job did not succeed")


def restore(plan, directory, kind, artifact_id=0, build=None):
    prefix = "legacy-v3-" + kind + "-" + plan["key"]
    if artifact_id:
        entries = [api("/actions/artifacts/" + str(int(artifact_id)))]
    else:
        response = api("/actions/artifacts?name=" + prefix + "&per_page=100")
        if response.get("total_count", 0) > len(response.get("artifacts", [])):
            raise ValueError("artifact discovery truncated")
        entries = response.get("artifacts", [])
    for entry in entries:
        if entry.get("name") != prefix or entry.get("expired") is not False:
            continue
        with tempfile.TemporaryDirectory() as temporary:
            try:
                downloaded = download(entry["id"], temporary)
                proof = verify_cpu(plan, temporary) if kind == "cpu" else (verify_build(plan, temporary) if kind == "build" else verify_gpu(plan, temporary, build))
                role = {"cpu": "Legacy V3 CPU baseline", "build": "Hosted Legacy V3 CUDA and CPU", "evidence": "Legacy V3 SM61"}[kind]
                verify_producer(proof, role, downloaded)
            except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
                if artifact_id:
                    raise
                print(f"Rejected reusable artifact {entry['id']}: {error}", file=sys.stderr)
                continue
            shutil.copytree(temporary, directory, dirs_exist_ok=True)
            emit(reused="true", artifact_id=entry["id"], origin_run=proof["origin"]["run_id"])
            return entry["id"]
    if artifact_id:
        raise ValueError("requested exact artifact was not verified")
    emit(reused="false", artifact_id="0")
    return 0


def promote(repo, plan, builds, evidence, output, cpu_baseline):
    cpu = verify_cpu(plan, cpu_baseline)
    build = verify_build(plan, builds)
    gpu = verify_gpu(plan, evidence, build)
    before = command("git", "ls-remote", "--heads", "origin", "refs/heads/legacy_v3", cwd=repo)
    before = before.split()[0] if before else None
    if before != plan["target_lease"]:
        raise ValueError("legacy_v3 changed after preparation; preserve concurrent work")
    git(repo, "fetch", "--no-tags", Path(output).parent / "candidate.bundle", plan["candidate_ref"])
    refs = [plan["candidate"] + ":refs/heads/legacy_v3"]
    if before:
        refs.append(before + ":refs/tags/archive/legacy-v3/" + str(origin()["run_id"]) + "-" + str(origin()["run_attempt"]))
    git(repo, "push", "--atomic", "--force-with-lease=refs/heads/legacy_v3:" + (before or ""), "origin", *refs)
    receipt = {"schema": 1, "status": "PILOT_QUALIFIED", "target": "legacy_v3", "before": before,
               "after": plan["candidate"], "key": plan["key"], "origin": origin(),
               "cpu_origin": cpu["origin"], "build_origin": build["origin"], "gpu_origin": gpu["origin"], "binary_sha256": build["files"]["mistralrs"],
               "numeric_status": {mode: value["status"] for mode, value in gpu["models"].items()},
               "definitive_workflow_validated": False}
    write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "verify-plan", "cpu-proof", "restore-cpu", "build-proof", "restore-build", "restore-evidence", "verify-build", "promote", "config"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--environment-key")
    parser.add_argument("--cpu-profile", choices=["portable", "avx2"], default="portable")
    parser.add_argument("--artifact-id", type=int, default=0)
    parser.add_argument("--builds", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--cpu-baseline", type=Path)
    parser.add_argument("--profile", choices=["cpu", "gpu"])
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(args.repo.resolve(), args.directory.resolve(), args.image, args.environment_key, args.cpu_profile)))
        return
    plan = verify_plan(args.directory)
    if args.mode == "verify-plan":
        return
    if args.mode == "config":
        text = (ROOT / "nextest.toml").read_text()
        args.output.write_text(text.replace(f'junit.path = "{args.profile}-junit.xml"', "junit.path = " + json.dumps(str((args.output.parent / (args.profile + "-junit.xml")).resolve()))))
    elif args.mode == "build-proof":
        build_proof(plan, args.output)
    elif args.mode == "cpu-proof":
        cpu_proof(plan, args.output)
    elif args.mode == "restore-cpu":
        restore(plan, args.output, "cpu", args.artifact_id)
    elif args.mode == "verify-build":
        verify_build(plan, args.builds)
    elif args.mode == "restore-build":
        restore(plan, args.output, "build", args.artifact_id)
    elif args.mode == "restore-evidence":
        restore(plan, args.output, "evidence", args.artifact_id, verify_build(plan, args.builds))
    else:
        promote(args.repo.resolve(), plan, args.builds, args.evidence, args.output, args.cpu_baseline)


if __name__ == "__main__":
    main()
