#!/usr/bin/env python3
"""Prepare the Candle DKR integration patch, without Cargo/NVCC or repo writes.

The existing Mistral pin remains qualified. Emit a patch for a separate
integration worktree only after checking the exact standalone Candle candidate.
No package loading or GPU qualification is implied by this source gate.
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "d99f6d77e507688bd9e4b654fbe20ac9e18238fc"
CANDIDATE = "c8a75a5256f4f3b855be8649fdbd2ff71f955904"
FAMILIES = {"affine", "binary", "cast", "fill", "indexing", "reduce", "ternary", "unary"}
PROFILE = "dkr-sm61-experimental"
CANDLE_FEATURE = "cuda-dkr-s31"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def candidate_gate(candle):
    if git(candle, "rev-parse", "HEAD") != CANDIDATE:
        raise ValueError(f"expected exact Candle candidate {CANDIDATE}")
    if git(candle, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Candle tracked tree is dirty")
    git(candle, "merge-base", "--is-ancestor", BASE, CANDIDATE)
    changed = git(candle, "diff", "--name-only", BASE, CANDIDATE).splitlines()
    # Current stage reuses the original CUDA algorithms/headers.
    changed_cuda = [p for p in changed if p.startswith("candle-kernels/src/")
                    and Path(p).suffix in {".cu", ".cuh", ".h", ".hpp"}]
    if changed_cuda:
        raise ValueError(f"unexpected CUDA source modifications: {changed_cuda}")
    core = tomllib.loads((candle / "candle-core/Cargo.toml").read_text())
    if CANDLE_FEATURE not in core["features"]:
        raise ValueError("Candle provider feature missing")
    contract = (candle / "candle-kernels/dkr_s31_contract.rs").read_text()
    line = re.search(r'pub const FAMILIES:.*?=\s*&\[(.*?)\];', contract, re.S)
    if not line or set(re.findall(r'"([^"]+)"', line[1])) != FAMILIES:
        raise ValueError("Candle native module set differs from integration contract")
    loader = (candle / "candle-core/src/cuda_backend/dkr_s31.rs").read_text()
    for family in FAMILIES:
        if f'(kernels::{family.upper()}.index(), "{family}")' not in loader:
            raise ValueError(f"missing native dispatch route {family}")
    if "CANDLE_DKR_S31_DIR" not in loader or "CANDLE_DKR_MODULE_LOADED" not in loader:
        raise ValueError("Candle activation or module trace contract missing")
    return changed


def patch_sources(workspace, core):
    deps = tomllib.loads(workspace)["workspace"]["dependencies"]
    for name in ("candle-core", "candle-nn"):
        if deps[name].get("rev") != BASE or deps[name].get("git") != "https://github.com/haricot/candle.git":
            raise ValueError(f"{name}: qualified source pin changed; review before promotion")
    # Change only the two active workspace dependency lines.
    pattern = re.compile(r'^(candle-(?:core|nn)\s*=\s*\{[^\n]*\brev\s*=\s*")'
                         + BASE + r'(".*)$', re.M)
    new_workspace, count = pattern.subn(lambda m: m[1] + CANDIDATE + m[2], workspace)
    if count != 2:
        raise ValueError("expected exactly two active Candle pins")
    features = tomllib.loads(core)["features"]
    if PROFILE not in features or f"candle-core/{CANDLE_FEATURE}" in features[PROFILE]:
        raise ValueError("unexpected experimental profile")
    anchor = f'{PROFILE} = [\n'
    if core.count(anchor) != 1:
        raise ValueError("experimental feature layout changed")
    new_core = core.replace(anchor, anchor + f'    "candle-core/{CANDLE_FEATURE}",\n', 1)
    assert tomllib.loads(new_core)["features"][PROFILE] == [
        f"candle-core/{CANDLE_FEATURE}", *features[PROFILE]
    ]
    return new_workspace, new_core


def emit_patch(output):
    if git(ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("use a clean Mistral audit worktree")
    paths = ["Cargo.toml", "mistralrs-core/Cargo.toml"]
    old = [(ROOT / p).read_text() for p in paths]
    new = patch_sources(*old)
    patch = "".join(
        "".join(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True),
                                    fromfile=f"a/{p}", tofile=f"b/{p}"))
        for p, a, b in zip(paths, old, new)
    )
    if output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("write patch outside repository")
    output.write_text(patch)


def self_test():
    workspace = ('[workspace.dependencies]\n'
                 f'candle-core = {{ git = "https://github.com/haricot/candle.git", rev = "{BASE}" }}\n'
                 f'candle-nn = {{ git = "https://github.com/haricot/candle.git", rev = "{BASE}" }}\n')
    core = f'[features]\n{PROFILE} = [\n    "cuda-legacy",\n]\n'
    new_workspace, new_core = patch_sources(workspace, core)
    assert new_workspace.count(CANDIDATE) == 2 and BASE not in new_workspace
    assert f'"candle-core/{CANDLE_FEATURE}"' in new_core
    try:
        patch_sources(new_workspace, core)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unexpected existing pin")
    print("DKR_CANDLE_INTEGRATION_SELF_TEST=PASS no_cuda_compilation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candle-dir", type=Path)
    p.add_argument("--emit-patch", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.candle_dir:
        p.error("--candle-dir required")
    changed = candidate_gate(a.candle_dir.resolve())
    patch_sources((ROOT / "Cargo.toml").read_text(), (ROOT / "mistralrs-core/Cargo.toml").read_text())
    if a.emit_patch:
        emit_patch(a.emit_patch)
    print(json.dumps({
        "status": "SOURCE_INTEGRATION_READY_GPU_QUALIFICATION_REQUIRED",
        "qualified_candle_pin": BASE, "candidate_candle_pin": CANDIDATE,
        "native_module_families": sorted(FAMILIES),
        "mistral_profile": PROFILE, "candle_feature": CANDLE_FEATURE,
        "activation": {"CANDLE_DKR_S31_DIR": "absolute Candle package root",
                       "CANDLE_DKR_TRACE": "1", "DKR_KERNEL_PACKAGES": "existing Mistral package root"},
        "candidate_changed_files": changed,
        "patch": str(a.emit_patch) if a.emit_patch else None,
        "requires_lockfile_resolution": True,
        "cargo_build_verified": False,
        "gpu_qualified": False,
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, AssertionError, subprocess.CalledProcessError) as e:
        print(f"DKR_CANDLE_INTEGRATION=FAIL {e}", file=sys.stderr)
        sys.exit(1)
