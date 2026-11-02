"""Publish the exact tested release without compiling or running model inference."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

REPOSITORY = "haricot/mistral.rs"
APPROVER = "haricot"
GATES = (
    "cuda_image / Independently validate CUDA environment reuse",
    "cuda_image / Resolve or publish the verified CUDA environment",
    "prepare", "cpu", "cuda_compile", "Extended-Release-R1 single build",
    "Assemble and verify the exact release runtime image",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def api(path):
    return json.loads(run("gh", "api", "/repos/" + REPOSITORY + path))


def verify_run(receipt, source, jobs, artifacts):
    require(source["id"] == receipt["run_id"] and source["run_attempt"] == receipt["run_attempt"]
            and source["head_sha"] == receipt["source_sha"] and source["conclusion"] == "success"
            and source["status"] == "completed" and source["event"] == "push"
            and source["head_repository"]["full_name"] == REPOSITORY,
            "source qualification run identity or success changed")
    require(len(jobs["jobs"]) == jobs["total_count"], "incomplete job listing")
    for name in GATES:
        matches = [j for j in jobs["jobs"] if j["name"] == name and j["run_attempt"] == receipt["run_attempt"]]
        require(len(matches) == 1 and matches[0]["conclusion"] == "success", "hosted gate missing: " + name)
    gpu = [j for j in jobs["jobs"] if j["id"] == receipt["gpu_job_id"]]
    require(len(gpu) == 1 and gpu[0]["conclusion"] == "success"
            and gpu[0]["run_attempt"] == receipt["run_attempt"]
            and "candidate=" + receipt["candidate"] in gpu[0]["name"]
            and set(gpu[0]["labels"]) == {"mistral-extended", "mistral-extended-" + str(receipt["run_id"])},
            "dedicated physical GPU job mismatch")
    require(len(artifacts["artifacts"]) == artifacts["total_count"], "incomplete artifact listing")
    indexed = {a["id"]: a for a in artifacts["artifacts"]}
    for expected in receipt["artifacts"]:
        actual = indexed.get(expected["id"], {})
        require(all(actual.get(k) == expected[k] for k in ("name", "digest"))
                and actual.get("expired") is False, "qualification artifact changed")


def unpack(archive, destination, fingerprint):
    require(sha(archive) == fingerprint.removeprefix("sha256:"), "artifact ZIP SHA256 mismatch")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            target = (destination / entry.filename).resolve()
            require(target.is_relative_to(destination) and not stat.S_ISLNK(entry.external_attr >> 16),
                    "unsafe artifact entry")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def verify_models(report, kernel, manifest, image, policy, qualification, receipt, directory):
    require(report["schema"] == "haricot-extended-release-gpu-r1" and report["release"] == manifest
            and report["full_models_executed"] is True and report["status"] in ("PASS", "COLLECTED")
            and report["run_id"] == kernel["run_id"] == str(receipt["run_id"])
            and report["run_attempt"] == kernel["run_attempt"] == str(receipt["run_attempt"])
            and report["sm"] == "6.1" and report["gpu_uuid"] and report["driver"]
            and set(report["models"]) == {"text", "vl", "gguf"}, "complete real-model evidence missing")
    require(report["kernel_report_sha256"] == sha(directory.parent / "report.json")
            and report["image_proof_sha256"] == sha(image / "oci.json")
            and report["policy_sha256"] == sha(qualification.ROOT / "models.json")
            and report["harness_sha256"] == sha(qualification.ROOT / "qualification.py"), "model proof linkage changed")
    require(set(receipt["non_blocking_numeric_models"]) == {"vl", "gguf"}, "unexpected numeric publication policy")
    for name, fingerprint in report["files"].items():
        require(Path(name).name == name and sha(directory / name) == fingerprint, "raw model evidence changed")
    for mode, case in report["models"].items():
        require(not case["errors"] and case["weights"] == policy["models"][mode], "model execution/weights mismatch")
        actual = qualification.measure(read(directory / (mode + "-cpu.json")),
                                       read(directory / (mode + "-cuda.json")), policy)
        require(all(case.get(k) == v for k, v in actual.items()), "numeric measurements differ from raw observations")
        require(actual["cpu_executed"] and actual["cuda_executed"], "model execution incomplete")
        require(mode in receipt["non_blocking_numeric_models"] or actual["status"] == "PASS",
                "required numeric model check failed")
    require(report["paged_fp8"]["status"] == "PASS" and not report["paged_fp8"]["errors"]
            and report["paged_fp8"]["binary_sha256"] == manifest["binary_sha256"], "real FP8 benchmark missing")


def branch_updates(plan):
    changes = {plan["config"]["components"][name]["branch"]: value
               for name, value in plan["components"].items() if name != "extended"}
    changes[plan["config"]["targets"][plan["target"]]["branch"]] = plan["candidate"]
    return changes


def push_arguments(repo, plan, receipt, current):
    changes = branch_updates(plan)
    for branch, expected in plan["leases"].items():
        require(current[branch] in (expected, changes.get(branch)), "source lease changed: " + branch)
    arguments = ["git", "-C", str(repo), "push", "--atomic", "origin"]
    for branch, value in changes.items():
        arguments += ["--force-with-lease=refs/heads/" + branch + ":" + current[branch],
                      value + ":refs/heads/" + branch,
                      plan["leases"][branch] + ":refs/tags/archive/campaign/" + str(receipt["run_id"]) + "/" + branch]
    return arguments


def write_source_helper(destination, plan, receipt):
    config = {"changes":branch_updates(plan), "leases":plan["leases"], "candle":plan["candle"],
              "run_id":receipt["run_id"], "candidate":plan["candidate"]}
    source = '''#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path
CONFIG=json.loads(REPLACE_CONFIG)
GIT=['git','-c','credential.helper=','-c','credential.helper=!gh auth git-credential']
def run(*args):
 return subprocess.check_output(list(map(str,args)),text=True).strip()
subprocess.run(['gh','auth','status'],check=True)
with tempfile.TemporaryDirectory(prefix='haricot-source-promotion-') as temporary:
 repo=Path(temporary)/'repo.git'
 subprocess.run([*GIT,'init','--bare','-q',str(repo)],check=True)
 subprocess.run([*GIT,'-C',str(repo),'remote','add','origin','https://github.com/haricot/mistral.rs.git'],check=True)
 subprocess.run([*GIT,'-C',str(repo),'fetch','origin',CONFIG['candidate']],check=True)
 current={}
 for branch,expected in CONFIG['leases'].items():
  current[branch]=run(*GIT,'-C',repo,'ls-remote','--heads','origin','refs/heads/'+branch).split()[0]
  if current[branch] not in (expected,CONFIG['changes'].get(branch)):
   raise SystemExit('Source lease changed: '+branch)
 for branch,expected in CONFIG['candle'].items():
  actual=run(*GIT,'ls-remote','--heads','https://github.com/haricot/candle.git','refs/heads/'+branch).split()[0]
  if actual!=expected['sha']:raise SystemExit('Candle lease changed: '+branch)
 args=[*GIT,'-C',str(repo),'push','--atomic','origin']
 for branch,candidate in CONFIG['changes'].items():
  args += ['--force-with-lease=refs/heads/'+branch+':'+current[branch],candidate+':refs/heads/'+branch,
           CONFIG['leases'][branch]+':refs/tags/archive/campaign/'+str(CONFIG['run_id'])+'/'+branch]
 subprocess.run(args,check=True)
 print('SOURCE_PROMOTION=PASS')
'''.replace('REPLACE_CONFIG', repr(json.dumps(config,sort_keys=True)))
    destination.write_text(source)


def verify_payloads(work, qualified, receipt):
    sys.path.insert(0, str(qualified / ".github/campaign"))
    campaign = importlib.import_module("campaign")
    release = importlib.import_module("release_r1")
    image_module = importlib.import_module("release_image_r1")
    fp8 = importlib.import_module("fp8_kv")
    plan_file = work / "preparation/plan.json"
    plan = read(plan_file)
    require(plan["candidate"] == receipt["candidate"] and plan["orchestrator"] == receipt["source_sha"],
            "preparation candidate mismatch")
    require(sha(work / "preparation/candidate.bundle") == plan["bundle_sha256"], "candidate bundle mismatch")
    (work / "release/mistralrs").chmod(0o755)
    manifest = release.inspect(plan_file, work / "release")
    require(manifest["binary_sha256"] == receipt["binary_sha256"], "release binary differs from qualified binary")
    image = release.image_record(plan_file, work / "release", work / "image", archive=True)
    kernel = read(work / "gpu/report.json")
    require(kernel["candidate"] == plan["candidate"] and kernel["status"] == "GPU_PASSED"
            and kernel["gpu_executed"] is True and kernel["target"] == "sm_61"
            and kernel["asd_required"] is True, "physical GPU/ASD proof missing")
    require(all(kernel["asd_test_counts"].get(n, 0) > 0 for n in ("asd", "extended"))
            and all(campaign.SHA256.fullmatch(kernel["asd_profiles"].get(n, "")) for n in ("current.json", "current.asd")),
            "ASD profile proof missing")
    fp8.verify_campaign(plan, work / "gpu", kernel, receipt["run_id"])
    for node in plan["matrix"]:
        name = node["component"]
        proof = campaign.verify_artifact(work / "builds" / name, node)
        require(kernel["archives"][name] == proof["files"]["tests.tar.zst"]
                and kernel["test_counts"][name] > 0 and kernel["fp4_test_counts"][name] > 0
                and kernel["lfm3_conv_gpu_counts"][name] > 0, "component physical proof mismatch")
    qualification = release.model_module()
    report = read(work / "gpu/release-r1/report.json")
    verify_models(report, kernel, manifest, work / "image", read(qualification.ROOT / "models.json"),
                  qualification, receipt, work / "gpu/release-r1")
    for cache in ("auto", "f8e4m3"):
        measured = release.paged_summary(read(work / "gpu/release-r1" / ("paged-kv-" + cache + ".json")), cache, qualification)
        require(measured == report["paged_fp8"]["cases"][cache], "FP8 benchmark summary mismatch")
    require(report["paged_fp8"]["request_speed_ratio_fp16_over_fp8"] ==
            report["paged_fp8"]["cases"]["auto"]["median_request_seconds"] /
            report["paged_fp8"]["cases"]["f8e4m3"]["median_request_seconds"], "FP8 timing ratio mismatch")
    return plan, manifest, image, image_module


def publish(repo, qualified, receipt_file, work):
    receipt = read(receipt_file)
    require(receipt["repository"] == REPOSITORY and receipt["requested_by"] == APPROVER
            and os.environ.get("GITHUB_ACTOR") == APPROVER
            and os.environ.get("GITHUB_REF") == "refs/heads/mirror_orch", "publication authority mismatch")
    require(run("git", "-C", qualified, "rev-parse", "HEAD") == receipt["source_sha"], "qualified source changed")
    prefix = "/actions/runs/" + str(receipt["run_id"])
    verify_run(receipt, api(prefix), api(prefix + "/jobs?filter=all&per_page=100"), api(prefix + "/artifacts?per_page=100"))
    work.mkdir(parents=True, exist_ok=True)
    for item in receipt["artifacts"]:
        archive = work / (str(item["id"]) + ".zip")
        with archive.open("wb") as stream:
            subprocess.run(["gh", "api", "/repos/" + REPOSITORY + "/actions/artifacts/" + str(item["id"]) + "/zip"],
                           check=True, stdout=stream)
        unpack(archive, work / item["destination"], item["digest"])
        archive.unlink()
    plan, manifest, image, publisher = verify_payloads(work, qualified, receipt)
    run("git", "-C", repo, "fetch", str(work / "preparation/candidate.bundle"),
        "refs/heads/integration/campaign/*:refs/remotes/qualified/*")
    for node in plan["matrix"]:
        run("git", "-C", repo, "merge-base", "--is-ancestor", plan["master"], node["candidate"])
    current = {branch: run("git", "-C", repo, "ls-remote", "--heads", "origin", "refs/heads/" + branch).split()[0]
               for branch in plan["leases"]}
    for branch, candle in plan["candle"].items():
        actual = run("git", "ls-remote", "--heads", "https://github.com/haricot/candle.git", "refs/heads/" + branch).split()[0]
        require(actual == candle["sha"], "Candle changed after qualification")
    output = work / "publication"
    output.mkdir(exist_ok=True)
    transaction = {"changes":branch_updates(plan), "previous":current, "run_id":receipt["run_id"]}
    try:
        subprocess.run(push_arguments(repo, plan, receipt, current), check=True, capture_output=True, text=True)
        transaction["status"] = "PASS"
    except subprocess.CalledProcessError as error:
        if 'without `workflows` permission' not in error.stderr:
            raise
        transaction.update(status="PENDING_GITHUB_WORKFLOWS_PERMISSION", reason=error.stderr)
    (output / "source-promotion.json").write_text(json.dumps(transaction,sort_keys=True,indent=2)+'\n')
    write_source_helper(output / "promote-source-r1.py", plan, receipt)
    print("SOURCE_PROMOTION=" + transaction["status"], flush=True)
    publisher.publish(work / "preparation/plan.json", work / "release", work / "image")
    native = output / "mistralrs-extended-sm61-cuda12.9.2-x86_64.tar.zst"
    run("tar", "--sort=name", "--mtime=@0", "--owner=0", "--group=0", "--numeric-owner",
        "-C", work / "release", "-I", "zstd -T2", "-cf", native, "mistralrs", "release.json")
    (output / (native.name + ".sha256")).write_text(sha(native) + "  " + native.name + "\n")
    shutil.copy2(work / "image/published.json", output / "published.json")
    shutil.copy2(receipt_file, output / "promotion.json")
    tag = "extended-r1-" + plan["candidate"][:12]
    notes = output / "release-notes.md"
    notes.write_text("Extended Release R1 for CUDA SM61.\n\nQualified workflow: https://github.com/" + REPOSITORY +
                     "/actions/runs/" + str(receipt["run_id"]) + "\n\nBinary SHA256: " + manifest["binary_sha256"] +
                     "\n\n" + run("cat", output / "published.json") +
                     "\n\nCPU, CUDA, ASD and real Text/VL/GGUF executions are retained in the qualification artifacts. "
                     "VL/GGUF numerical observations are informational under the repository owner's publication decision.\n")
    previous = subprocess.run(["gh", "api", "/repos/" + REPOSITORY + "/releases/tags/" + tag], capture_output=True, text=True)
    files = [native, output / (native.name + ".sha256"), output / "published.json", output / "promotion.json",
             output / "source-promotion.json", output / "promote-source-r1.py"]
    if previous.returncode:
        run("gh", "release", "create", tag, "--repo", REPOSITORY, "--target", plan["candidate"],
            "--title", "Extended Release R1 (CUDA SM61)", "--notes-file", notes, "--latest=false", *files)
    else:
        release = json.loads(previous.stdout)
        assets = {a["name"]: a for a in release["assets"]}
        for path in files:
            if path.name in assets:
                require(assets[path.name].get("digest") == "sha256:" + sha(path), "existing release asset differs")
            else:
                run("gh", "release", "upload", tag, path, "--repo", REPOSITORY)
    print("R1_RELEASE=https://github.com/" + REPOSITORY + "/releases/tag/" + tag, flush=True)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--repo", type=Path, required=True)
    arguments.add_argument("--qualified", type=Path, required=True)
    arguments.add_argument("--receipt", type=Path, required=True)
    arguments.add_argument("--work", type=Path, required=True)
    options = arguments.parse_args()
    publish(options.repo.resolve(), options.qualified.resolve(), options.receipt.resolve(), options.work.resolve())
