#!/usr/bin/env python3
"""S4 local dependency closure. Only Cargo metadata is run, never build/check/test.

Run in the isolated integration worktree. Cargo may fetch dependencies and
update its local Cargo.lock; no NVCC, no build scripts, no CI, no cache deletion.
"""
import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from dkr_r122_candle_integration import CANDIDATE, CANDLE_PINS, PROFILE, candidate_gate

ROOT = Path(__file__).resolve().parents[1]
FEATURE = "cuda-dkr-s31"
REQUIRED = {"candle-core", "candle-nn", "candle-kernels",
            "dkr-kernel-registry", "cudarc", "mistralrs-kernel-runtime", "mistralrs-core"}
PINNED = set(CANDLE_PINS) | {"candle-kernels", "dkr-kernel-registry"}
GIT_URL = "https://github.com/haricot/candle.git"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def candidate_source(source):
    if not isinstance(source, str) or not source.startswith("git+"):
        return False
    url = urlsplit(source[4:])
    return (f"{url.scheme}://{url.netloc}{url.path}" == GIT_URL
            and parse_qs(url.query).get("rev") == [CANDIDATE]
            and url.fragment == CANDIDATE)


def inspect(data, lock):
    packages = {p["id"]: p for p in data["packages"]}
    nodes = {n["id"]: n for n in data["resolve"]["nodes"]}
    members = set(data["workspace_members"])
    roots = [p["id"] for p in packages.values() if p["name"] == "mistralrs-core" and p["id"] in members]
    require(len(roots) == 1, "expected one workspace mistralrs-core")
    reachable = set()
    todo = roots[:]
    while todo:
        item = todo.pop()
        if item in reachable:
            continue
        reachable.add(item)
        todo.extend(nodes[item]["dependencies"])
    active = {}
    for name in sorted(REQUIRED):
        ids = [i for i in reachable if packages[i]["name"] == name]
        require(len(ids) == 1, f"{name}: expected exactly one reachable package, found {ids}")
        active[name] = ids[0]
    # Reject duplicate selected packages even outside the reachable core closure.
    for name in sorted(PINNED | {"cudarc"}):
        selected = [p for p in packages.values() if p["name"] == name and p["id"] in nodes]
        require(len(selected) <= 1, f"{name}: duplicate resolved packages")
        if name in PINNED:
            for p in selected:
                require(candidate_source(p["source"]), f"{name}: wrong git source {p['source']}")
    # Cargo.lock contains optional cross-platform deps too: all Candle pins must agree.
    for name in sorted(PINNED):
        entries = [p for p in lock["package"] if p["name"] == name]
        require(len(entries) == 1, f"lockfile {name}: expected one package, got {len(entries)}")
        require(candidate_source(entries[0].get("source")), f"lockfile {name}: candidate revision mismatch")
    cudarc = packages[active["cudarc"]]
    require(cudarc["version"].startswith("0.19."), f"unexpected cudarc version {cudarc['version']}")
    expected_features = {"mistralrs-core": {PROFILE}, "candle-core": {"cuda", FEATURE},
                         "candle-nn": {"cuda", FEATURE}, "mistralrs-kernel-runtime": {"cuda", "shared-registry"}}
    for name, expected in expected_features.items():
        seen = set(nodes[active[name]]["features"])
        require(expected <= seen, f"{name}: missing features {sorted(expected-seen)}")
    reg = active["dkr-kernel-registry"]
    for name in ("candle-core", "mistralrs-kernel-runtime"):
        require(reg in nodes[active[name]]["dependencies"], f"{name}: not using identical registry package ID")
    require(active["candle-core"] in nodes[active["candle-nn"]]["dependencies"], "Candle-NN core identity mismatch")
    require(active["candle-core"] in nodes[active["mistralrs-kernel-runtime"]]["dependencies"], "Mistral runtime core identity mismatch")
    require(active["cudarc"] in nodes[active["candle-core"]]["dependencies"], "Candle CUDA driver identity mismatch")
    registry_deps = sorted({packages[i]["name"] for i in nodes[reg]["dependencies"]})
    require(registry_deps == ["sha2"], f"neutral registry dependency boundary changed: {registry_deps}")
    return {"active_packages": {name: {"id": i, "version": packages[i]["version"],
                                      "features": sorted(nodes[i]["features"])} for name,i in active.items()},
            "registry_dependency_names": registry_deps,
            "cudarc_version": cudarc["version"],
            "reachable_packages": len(reachable),
            "candle_checkout": str(Path(packages[active["candle-core"]]["manifest_path"]).parent.parent)}


def self_test():
    source = f"git+{GIT_URL}?rev={CANDIDATE}#{CANDIDATE}"
    names = sorted(REQUIRED | PINNED | {"sha2"})
    ps = [{"id": n, "name": n, "version": "0.19.10" if n == "cudarc" else "0.1.0",
           "source": source if n in PINNED else None,
           "manifest_path": "/tmp/candle/candle-core/Cargo.toml"} for n in names]
    edges = {"mistralrs-core": ["candle-core","candle-nn","mistralrs-kernel-runtime"],
             "candle-core": ["candle-kernels","cudarc","dkr-kernel-registry"],
             "candle-nn": ["candle-core"],
             "mistralrs-kernel-runtime": ["candle-core","dkr-kernel-registry"],
             "dkr-kernel-registry": ["sha2"]}
    feats = {"mistralrs-core": [PROFILE], "candle-core": ["cuda",FEATURE],
             "candle-nn": ["cuda",FEATURE], "mistralrs-kernel-runtime": ["cuda","shared-registry"]}
    ns = [{"id": n, "dependencies": edges.get(n,[]), "features": feats.get(n,[])} for n in names]
    data = {"packages": ps, "workspace_members": ["mistralrs-core"], "resolve": {"nodes": ns}}
    lock = {"package": [{"name": n,"source":source} for n in PINNED]}
    assert inspect(data,lock)["registry_dependency_names"] == ["sha2"]
    for obj,key,bad in [(next(p for p in ps if p["name"]=="candle-core"),"source",source.replace(CANDIDATE,"0"*40)),
                        (next(n for n in ns if n["id"]=="candle-nn"),"features",["cuda"]),
                        (next(n for n in ns if n["id"]=="mistralrs-kernel-runtime"),"dependencies",["candle-core"])]:
        before = obj[key]
        obj[key] = bad
        try:
            inspect(data,lock)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted bad {key}")
        finally:
            obj[key] = before
    lock["package"].append(dict(lock["package"][0]))
    try:
        inspect(data,lock)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted duplicate lockfile package")
    print("DKR_S4_SELF_TEST=PASS no_cuda_compilation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--locked", action="store_true", help="require already resolved lockfile")
    p.add_argument("--output", type=Path, default=Path("/tmp/dkr-s4-closure.json"))
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    deps = tomllib.loads((ROOT/"Cargo.toml").read_text())["workspace"]["dependencies"]
    for name in (*CANDLE_PINS, "dkr-kernel-registry"):
        require(deps[name].get("rev") == CANDIDATE and deps[name].get("git") == GIT_URL,
                f"refusing non-candidate workspace: {name}")
    # Only allow changes to Cargo.lock from a previous attempt. Never overwrite user changes.
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).splitlines()
    require(all(line[3:] == "Cargo.lock" for line in dirty), "use isolated worktree; tracked changes other than Cargo.lock")
    require(not args.output.resolve().is_relative_to(ROOT.resolve()), "write report outside worktree")
    before = (ROOT/"Cargo.lock").read_bytes()
    cmd = ["cargo","metadata","--format-version","1","--filter-platform","x86_64-unknown-linux-gnu",
           "--features",f"mistralrs-core/{PROFILE}"]
    if args.offline:
        cmd.append("--offline")
    if args.locked:
        cmd.append("--locked")
    print("DKR_S4_METADATA_START no_cuda_compilation", flush=True)
    result = subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,check=True)
    data = json.loads(result.stdout)
    lock = tomllib.loads((ROOT/"Cargo.lock").read_text())
    report = inspect(data,lock)
    changed = candidate_gate(Path(report["candle_checkout"]))
    report.update({"status":"DEPENDENCY_CLOSURE_PASS_GPU_UNVERIFIED",
                   "candidate":CANDIDATE,"lockfile_changed":before!=(ROOT/"Cargo.lock").read_bytes(),
                   "candle_changed_files":changed,"cuda_compilation":False,
                   "rust_compilation_verified":False,"gpu_qualified":False})
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S4_DEPENDENCY_CLOSURE=PASS no_cuda_compilation")


if __name__=="__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, AssertionError, subprocess.CalledProcessError) as e:
        print(f"DKR_S4_DEPENDENCY_CLOSURE=FAIL {e}",file=sys.stderr)
        sys.exit(1)
