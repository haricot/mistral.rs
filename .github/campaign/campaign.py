#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{40}\Z")
CANDLE_CRATES = ("candle-core", "candle-nn", "candle-flash-attn-v3", "candle-metal-kernels")
ROOT = Path(__file__).resolve().parent
META = "mistral-integration/sources.json"
METADATA_CONFLICTS = {META, "mistral-integration/README.md", "Cargo.lock"}
IDENTITY = {"GIT_AUTHOR_NAME": "Haricot Campaign", "GIT_COMMITTER_NAME": "Haricot Campaign",
            "GIT_AUTHOR_EMAIL": "344493+haricot@users.noreply.github.com",
            "GIT_COMMITTER_EMAIL": "344493+haricot@users.noreply.github.com"}


def command(args, cwd=None, check=True, env=None):
    result = subprocess.run([str(a) for a in args], cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {args}\n{result.stdout}\n{result.stderr}")
    return result


def git(repo, *args, **kwargs):
    return command(["git", "-C", repo, *args], **kwargs).stdout.strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote_head(repo, remote, branch):
    output = git(repo, "ls-remote", "--heads", remote, f"refs/heads/{branch}")
    if not output or not SHA.fullmatch(output.split()[0]):
        raise RuntimeError(f"missing branch: {remote}:{branch}")
    return output.split()[0]


def recipe():
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in paths})


def snapshot(repo, target, image, output):
    config = json.loads((ROOT / "config.json").read_text())
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise RuntimeError("the CUDA compiler image must have an immutable SHA-256 identity")
    selected = config["targets"][target]
    branches = {config["master_branch"], selected["branch"]}
    branches.update(config["components"][name]["branch"] for name in selected["components"])
    heads = {b: remote_head(repo, "origin", b) for b in sorted(branches)}
    for branch, sha in heads.items():
        git(repo, "fetch", "--no-tags", "origin", f"refs/heads/{branch}:refs/remotes/campaign/{branch}")
        if git(repo, "rev-parse", f"refs/remotes/campaign/{branch}") != sha:
            raise RuntimeError(f"branch moved while snapshotting: {branch}")
    sources = {name: heads[config["components"][name]["branch"]] for name in selected["components"]}
    bootstrap_path = ROOT / "bootstrap.json"
    if bootstrap_path.exists():
        bootstrap = json.loads(bootstrap_path.read_text())
        if bootstrap["master"] == heads[config["master_branch"]]:
            for name in sources:
                entry = bootstrap.get("components", {}).get(name)
                if entry and sources[name] == entry["before"]:
                    prepared = entry["prepared"]
                    git(repo, "fetch", "--no-tags", "origin", prepared)
                    if not ancestor(repo, sources[name], prepared) or not ancestor(repo, bootstrap["master"], prepared):
                        raise RuntimeError("bootstrap source provenance mismatch")
                    sources[name] = prepared
    candle_repo = "https://github.com/haricot/candle.git"
    candle = {}
    for branch in sorted({config["components"][n]["candle_branch"] for n in selected["components"]}):
        sha = remote_head(repo, candle_repo, branch)
        kind = "legacy" if branch == "cuda_legacy" else "extended"
        refs = git(repo, "ls-remote", "--heads", "--tags", candle_repo)
        mapping = {ref: oid for oid, ref in (line.split() for line in refs.splitlines())}
        matches = [ref for ref, oid in mapping.items()
                   if "/archive/" in ref and f"/{kind}/sm61-validated-" in ref
                   and not ref.endswith("^{}") and mapping.get(ref + "^{}", oid) == sha]
        if not matches:
            raise RuntimeError(f"Candle {branch} has no matching physical validation ref")
        candle[branch] = {"sha": sha, "validation_ref": sorted(matches)[-1]}
    value = {"schema_version": 2, "target": target, "config": config,
             "master": heads[config["master_branch"]], "leases": heads, "sources": sources, "candle": candle,
             "image": image, "recipe": recipe(), "orchestrator": git(repo, "rev-parse", "HEAD")}
    value["input_key"] = digest({k: v for k, v in value.items() if k != "orchestrator"})
    write_json(output, value)
    return value


def pin_candle(repo, sha):
    path = repo / "Cargo.toml"
    text = path.read_text()
    dependencies = tomllib.loads(text)["workspace"]["dependencies"]
    for name in CANDLE_CRATES:
        entry = dependencies.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"expected one workspace dependency: {name}")
        version = entry.get("version")
        if not isinstance(version, str):
            raise RuntimeError(f"missing Candle workspace version: {name}")
        entry = {**entry, "git": "https://github.com/haricot/candle.git", "rev": sha}
        for key in ("branch", "tag", "path", "registry"):
            entry.pop(key, None)
        fields = ", ".join(f"{key} = {json.dumps(value)}" for key, value in entry.items())
        text, count = re.subn(
            rf"^{re.escape(name)} = .*?$",
            f"{name} = {{ {fields} }}",
            text, flags=re.M)
        if count != 1:
            raise RuntimeError(f"expected one workspace dependency: {name}")
    path.write_text(text)


def install_contract(repo, contract):
    destination = repo / "mistralrs-core/tests/campaign_backends.rs"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "campaign_backends.rs", destination)
    for source, target, marker in (
        ("force_cpu_test.rs", "mistralrs-server-core/src/mistralrs_for_server_builder.rs", "mod campaign_runtime_backend_tests"),
        ("cpu_flag_test.rs", "mistralrs-cli/src/args/mod.rs", "mod campaign_cpu_flag_tests"),
    ):
        path = repo / target
        if path.exists() and marker not in path.read_text():
            path.write_text(path.read_text() + "\n" + (ROOT / source).read_text())
    for package in ("mistralrs", "mistralrs-pyo3"):
        manifest = repo / package / "Cargo.toml"
        if not manifest.exists():
            continue
        text = manifest.read_text()
        for feature in contract["features"]:
            if not re.search(rf"^{re.escape(feature)}\s*=", text, flags=re.M):
                line = f'{feature} = ["cuda", "mistralrs-core/{feature}"]\n'
                text = text.replace("[features]\n", "[features]\n" + line, 1)
        manifest.write_text(text)
    (repo / "rust-toolchain.toml").write_text(
        f'[toolchain]\nchannel = "{contract["toolchain"]}"\nprofile = "minimal"\ncomponents = ["rustfmt"]\n')
    cargo = repo / "Cargo.toml"
    text = cargo.read_text()
    if "[profile.campaign]" not in text:
        text += ('\n[profile.campaign]\ninherits = "dev"\nopt-level = 1\ndebug = 0\n'
                 'incremental = false\ncodegen-units = 8\n'
                 '\n[profile.campaign.package.mistralrs-core]\nopt-level = 0\n')
        cargo.write_text(text)
    write_json(repo / META, contract)


def resolve_candle_conflicts(text):
    def resolve(match):
        try:
            sides = [tomllib.loads(match.group(n)) for n in (1, 2)]
        except tomllib.TOMLDecodeError:
            return match.group(0)
        for side in sides:
            if not side or not set(side).issubset(CANDLE_CRATES):
                return match.group(0)
            for value in side.values():
                if not isinstance(value, dict):
                    return match.group(0)
                value.pop("git", None)
                value.pop("rev", None)
        return match.group(1) if sides[0] == sides[1] else match.group(0)
    return re.sub(r"^<<<<<<< [^\n]*\n(.*?)^=======\n(.*?)^>>>>>>> [^\n]*\n", resolve, text, flags=re.M | re.S)


def merge(repo, source, lock_source):
    result = command(["git", "-C", repo, "-c", "rerere.enabled=true", "-c",
                      "rerere.autoupdate=true", "merge", "--no-commit", "--no-ff", source], check=False)
    conflicts = git(repo, "diff", "--name-only", "--diff-filter=U").splitlines()
    if "Cargo.toml" in conflicts:
        path = repo / "Cargo.toml"
        resolved = resolve_candle_conflicts(path.read_text())
        if "<<<<<<< " not in resolved:
            path.write_text(resolved)
            git(repo, "add", "Cargo.toml")
    # Scope: only LFM2.5 DSpark against the newer LFM3 master. Never
    # replace entire upstream attention/CLI files or silently choose ours.
    if Path(repo).name == "lfm25_dspark":
        from resolve_dspark import resolve_known_conflicts
        resolve_known_conflicts(repo, set(conflicts))
    for path in set(conflicts) & METADATA_CONFLICTS:
        version = git(repo, "show", f"{lock_source if path == 'Cargo.lock' else 'HEAD'}:{path}", check=False)
        dest = repo / path
        if version:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(version + "\n")
            git(repo, "add", path)
        else:
            git(repo, "rm", "-f", "--ignore-unmatch", path)
    remaining = git(repo, "diff", "--name-only", "--diff-filter=U")
    if remaining or (result.returncode and not conflicts and not git(repo, "rev-parse", "-q", "--verify", "MERGE_HEAD", check=False)):
        raise RuntimeError(f"unresolved functional merge into {repo}:\n{remaining}\n{result.stderr}")


def commit(repo, parents, message):
    git(repo, "add", "-A")
    tree = git(repo, "write-tree")
    unique_parents = list(dict.fromkeys(parents))
    date = max(int(git(repo, "show", "-s", "--format=%ct", p)) for p in unique_parents) + 1
    env = {**os.environ, **IDENTITY, "GIT_AUTHOR_DATE": f"{date} +0000", "GIT_COMMITTER_DATE": f"{date} +0000"}
    args = ["git", "-C", repo, "commit-tree", tree, "-m", message]
    for parent in unique_parents:
        args.extend(["-p", parent])
    sha = command(args, env=env).stdout.strip()
    git(repo, "reset", "--hard", sha)
    return sha


def source_contract(repo, sha):
    text = git(repo, "show", f"{sha}:{META}", check=False)
    try:
        return json.loads(text)
    except ValueError:
        return {}


def ancestor(repo, older, newer):
    return command(["git", "-C", repo, "merge-base", "--is-ancestor", older, newer], check=False).returncode == 0


def prepared_ref(repo, name, inputs):
    ref = f"integration/prepared/{digest(inputs)}/{name}"
    found = git(repo, "ls-remote", "origin", f"refs/heads/{ref}")
    if found:
        sha = found.split()[0]
        git(repo, "fetch", "--no-tags", "origin", ref)
        return ref, sha
    return ref, None


def publish_ref(repo, ref, sha):
    current = git(repo, "ls-remote", "origin", f"refs/heads/{ref}")
    if current and current.split()[0] != sha:
        raise RuntimeError("immutable candidate ref collision")
    git(repo, "push", "origin", f"{sha}:refs/heads/{ref}")


def checkout(repo, node, original, scope):
    command(["git", "clone", "--shared", "--no-checkout", repo, node])
    git(node, "switch", "--detach", original)
    git(node, "config", "user.name", IDENTITY["GIT_AUTHOR_NAME"])
    git(node, "config", "user.email", IDENTITY["GIT_AUTHOR_EMAIL"])
    cache = ROOT / "rr-cache" / scope
    if cache.exists():
        shutil.copytree(cache, node / ".git/rr-cache", dirs_exist_ok=True)


def import_checkout(repo, node):
    objects = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")) / "objects"
    for shard in (node / ".git/objects").iterdir():
        if shard.is_dir() and (re.fullmatch(r"[0-9a-f]{2}", shard.name) or shard.name == "pack"):
            shutil.copytree(shard, objects / shard.name, dirs_exist_ok=True)
    git(repo, "cat-file", "-e", git(node, "rev-parse", "HEAD") + "^{commit}")
    shutil.rmtree(node)


def resolve_lock(repo, toolchain):
    args = ["cargo", f"+{toolchain}", "update", *sum((["-p", name] for name in CANDLE_CRATES), [])]
    result = command([*args, "--offline"], cwd=repo, check=False)
    if result.returncode:
        command(args, cwd=repo)


def prepare(repo, snapshot_path, output, publish=False):
    repo = Path(repo).resolve()
    state = json.loads(Path(snapshot_path).read_text())
    cfg = state["config"]
    target = cfg["targets"][state["target"]]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_root = output / "worktrees"
    work_root.mkdir(exist_ok=True)
    objects = {}
    metadata = {}
    for name in target["components"]:
        component = cfg["components"][name]
        original = state.get("sources", {}).get(name, state["leases"][component["branch"]])
        prerequisites = {n: objects[n] for n in component["requires"]}
        candle = state["candle"][component["candle_branch"]]
        identity = {"schema_version": 2, "kind": "mistral-standalone", "component": name,
                    "target_branch": component["branch"], "base_master_sha": state["master"],
                    "prerequisites": prerequisites, "candle_repository": "haricot/candle",
                    "candle_branch": component["candle_branch"], "candle_sha": candle["sha"],
                    "candle_validation_ref": candle["validation_ref"],
                    "features": cfg["features"], "toolchain": cfg["toolchain"],
                    "contract_hash": digest({n: file_hash(ROOT / n) for n in ["campaign_backends.rs", "force_cpu_test.rs", "cpu_flag_test.rs", "config.json", "campaign.py"]})}
        previous = source_contract(repo, original)
        ref, cached = prepared_ref(repo, name, {"source": original, "contract": identity})
        if cached:
            if source_contract(repo, cached) != identity or not ancestor(repo, original, cached):
                raise RuntimeError("prepared standalone provenance mismatch")
            sha = cached
        elif previous == identity and ancestor(repo, state["master"], original):
            sha = original
        else:
            node = work_root / name
            checkout(repo, node, original, name)
            parents = [original, state["master"], *prerequisites.values()]
            for parent in parents[1:]:
                merge(node, parent, state["master"])
                if git(node, "rev-parse", "-q", "--verify", "MERGE_HEAD", check=False):
                    commit(node, [git(node, "rev-parse", "HEAD"), parent], f"merge standalone prerequisite {parent}")
            pin_candle(node, candle["sha"])
            install_contract(node, identity)
            resolve_lock(node, cfg["toolchain"])
            command(["cargo", f'+{cfg["toolchain"]}', "fmt", "--all", "--", "--check"], cwd=node)
            sha = commit(node, [git(node, "rev-parse", "HEAD")], f"build: pin independent {name} campaign inputs")
            import_checkout(repo, node)
        if publish:
            publish_ref(repo, ref, sha)
        objects[name] = sha
        metadata[name] = identity
    if state["target"] == "legacy":
        candidate = objects["legacy"]
    else:
        contract = {"schema_version": 2, "kind": "mistral-extended", "target_branch": "extended",
                    "base_master_sha": state["master"], "components": objects,
                    "candle_repository": "haricot/candle", "candle_branch": "extended",
                    "candle_sha": state["candle"]["extended"]["sha"],
                    "candle_validation_ref": state["candle"]["extended"]["validation_ref"],
                    "features": cfg["features"], "toolchain": cfg["toolchain"], "asd_required": True,
                    "contract_hash": metadata["legacy"]["contract_hash"]}
        existing = state["leases"]["extended"]
        ref, cached = prepared_ref(repo, "extended", contract)
        if cached:
            if source_contract(repo, cached) != contract:
                raise RuntimeError("prepared composition provenance mismatch")
            candidate = cached
        elif source_contract(repo, existing) == contract:
            candidate = existing
        else:
            node = work_root / "extended"
            checkout(repo, node, state["master"], "extended")
            for sha in objects.values():
                merge(node, sha, objects["legacy"])
                commit(node, [git(node, "rev-parse", "HEAD"), sha], f"merge independent component {sha}")
            pin_candle(node, contract["candle_sha"])
            install_contract(node, contract)
            resolve_lock(node, cfg["toolchain"])
            candidate = commit(node, [git(node, "rev-parse", "HEAD")], "build: lock Extended standalone composition")
            import_checkout(repo, node)
        if publish:
            publish_ref(repo, ref, candidate)
        if ("asd" in cfg["targets"]["extended"]["components"]
                and ("asd" not in objects or not contract["asd_required"]
                or contract["components"].get("asd") != objects["asd"]
                or not ancestor(repo, objects["asd"], candidate))):
            raise RuntimeError("Extended candidate lacks mandatory ASD ancestry")
        if "asd" in cfg["targets"]["extended"]["components"] and "mistralrs-core/tests/asd_sm61_smoke.rs" not in git(
            repo, "ls-tree", "-r", "--name-only", candidate
        ).splitlines():
            raise RuntimeError("Extended candidate lacks ASD GPU smoke test")
        objects["extended"] = candidate
    matrix = []
    for name, sha in objects.items():
        contract = source_contract(repo, sha)
        key = digest({"candidate": sha, "image": state["image"], "recipe": state["recipe"]})
        ref = f"integration/campaign/{key}/{name}"
        git(repo, "update-ref", f"refs/heads/{ref}", sha)
        if publish:
            publish_ref(repo, ref, sha)
        matrix.append({"component": name, "candidate": sha, "ref": ref, "build_key": key,
                       "candle": contract["candle_sha"], "candle_branch": contract["candle_branch"],
                       "builder_image": state["image"],
                       "lock": git(repo, "show", f"{sha}:Cargo.lock")})
    for item in matrix:
        item["lock"] = hashlib.sha256((item["lock"] + "\n").encode()).hexdigest()
    value = {**state, "candidate": candidate, "components": objects, "matrix": matrix}
    write_json(output / "plan.json", value)
    refs = [item["ref"] for item in matrix]
    git(repo, "bundle", "create", output / "candidate.bundle", *refs)
    value["bundle_sha256"] = file_hash(output / "candidate.bundle")
    write_json(output / "plan.json", value)
    return value


def verify_artifact(directory, expected):
    directory = Path(directory)
    proof = json.loads((directory / "build.json").read_text())
    for key in ("candidate", "build_key", "component", "lock"):
        if proof.get(key) != expected[key]:
            raise RuntimeError(f"artifact {key} mismatch")
    for name, expected_hash in proof["files"].items():
        if Path(name).name != name or file_hash(directory / name) != expected_hash:
            raise RuntimeError(f"artifact digest mismatch: {name}")
    if proof.get("status") != "BUILD_AND_CPU_PASSED" or proof.get("cpu_test_count", 0) < 1:
        raise RuntimeError("CPU execution proof is missing")
    return proof


def promote(repo, plan_path, proofs, run_id):
    plan = json.loads(Path(plan_path).read_text())
    gpu = json.loads((Path(proofs) / "gpu/report.json").read_text())
    if (gpu.get("candidate") != plan["candidate"] or gpu.get("run_id") != str(run_id)
            or gpu.get("status") != "GPU_PASSED" or gpu.get("target") != "sm_61"
            or gpu.get("gpu_executed") is not True):
        raise RuntimeError("physical GPU proof does not match campaign")
    if (plan["target"] == "extended"
            and "asd" in plan["config"]["targets"]["extended"]["components"]):
        matrices = {item["component"] for item in plan["matrix"]}
        if ("asd" not in plan["config"]["targets"]["extended"]["components"]
                or "asd" not in plan["components"]
                or not {"asd", "extended"}.issubset(matrices)
                or gpu.get("asd_required") is not True):
            raise RuntimeError("Extended ASD provenance/runtime evidence is missing")
        if any(gpu.get("asd_test_counts", {}).get(n, 0) < 1 for n in ("asd", "extended")):
            raise RuntimeError("Extended ASD GPU smoke proof missing")
        if any(not SHA.fullmatch(gpu.get("asd_profiles", {}).get(n, ""))
               for n in ("current.json", "current.asd")):
            raise RuntimeError("Extended ASD profile SHA256 evidence missing")
    for node in plan["matrix"]:
        proof = verify_artifact(Path(proofs) / node["component"], node)
        if gpu.get("archives", {}).get(node["component"]) != proof["files"]["tests.tar.zst"]:
            raise RuntimeError("physical GPU did not execute the hosted archive")
        if gpu.get("test_counts", {}).get(node["component"], 0) < 1:
            raise RuntimeError("physical GPU test execution is missing")
    for branch, expected in plan["leases"].items():
        if remote_head(repo, "origin", branch) != expected:
            raise RuntimeError(f"source lease changed: {branch}")
    for branch, candle in plan["candle"].items():
        if remote_head(repo, "https://github.com/haricot/candle.git", branch) != candle["sha"]:
            raise RuntimeError("Candle changed during qualification")
    changes = {plan["config"]["components"][n]["branch"]: sha
               for n, sha in plan["components"].items() if n != "extended"}
    changes[plan["config"]["targets"][plan["target"]]["branch"]] = plan["candidate"]
    args = ["git", "-C", repo, "push", "--atomic", "origin"]
    for branch, sha in changes.items():
        old = plan["leases"][branch]
        args.extend([f"--force-with-lease=refs/heads/{branch}:{old}", f"{sha}:refs/heads/{branch}"])
        args.append(f"{old}:refs/tags/archive/campaign/{run_id}/{branch}")
    command(args)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--repo", type=Path, required=True)
    snap.add_argument("--target", choices=["legacy", "extended"], required=True)
    snap.add_argument("--image", required=True)
    snap.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--repo", type=Path, required=True)
    prep.add_argument("--snapshot", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--publish", action="store_true")
    prom = sub.add_parser("promote")
    prom.add_argument("--repo", type=Path, required=True)
    prom.add_argument("--plan", type=Path, required=True)
    prom.add_argument("--proofs", type=Path, required=True)
    prom.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.mode == "snapshot":
        snapshot(args.repo, args.target, args.image, args.output)
    elif args.mode == "prepare":
        prepare(args.repo, args.snapshot, args.output, args.publish)
    else:
        promote(args.repo, args.plan, args.proofs, args.run_id)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"CAMPAIGN_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
