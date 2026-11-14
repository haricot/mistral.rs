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
CANDIDATE = "4970f252cb4028a2031a811c777f949837816430"
INTEGRATION_BRANCH = "feature/dkr-r122-s35-shared-registry"
CANDLE_PINS = ("candle-core", "candle-nn", "candle-flash-attn-v3", "candle-metal-kernels")
MMQ_FAMILIES = {"mmq_quantize"} | {f"mmq_instance_{codec}" for codec in (
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k")}
FAMILIES = {"affine", "binary", "cast", "conv", "fill", "indexing", "reduce", "sort", "ternary", "unary", "quantized", "mmvq_gguf"} | MMQ_FAMILIES | {"moe_gguf", "moe_simt_f16"}
MOE_FAMILIES = {"moe_gguf", "moe_simt_f16"}
REMAINING = {
    "moe/moe_wmma.cu": "EXCLUDED_PRE_VOLTA",
    "moe/moe_wmma_gguf.cu": "EXCLUDED_PRE_VOLTA",
    "moe/moe_align.cu": "CUTILE_ONLY",
}


def expected_cuda_units():
    external = {(f"mmq_gguf/{family}.cu" if family in MMQ_FAMILIES else
                 f"moe/{family}.cu" if family in MOE_FAMILIES else f"{family}.cu")
                for family in FAMILIES}
    assert not external.intersection(REMAINING)
    return external | set(REMAINING)
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
    # Only the reviewed warp-local tail correction and revision marker are allowed.
    changed_cuda = [p for p in changed if p.startswith("candle-kernels/src/")
                    and Path(p).suffix in {".cu", ".cuh", ".h", ".hpp"}]
    allowed = "candle-kernels/src/moe/moe_gguf.cu"
    if changed_cuda != [allowed]:
        raise ValueError(f"unexpected CUDA source modifications: {changed_cuda}")
    original = git(candle, "show", f"{BASE}:{allowed}")
    expected = original.replace("    __syncthreads();",
        "    // Each warp owns its own shared row. Other warps may return on the N tail.\n    __syncwarp();")
    expected = expected.replace("constexpr int MATRIX_ROW_PADDING = 512;",
        'constexpr int MATRIX_ROW_PADDING = 512;\n\n// DKR package revision marker for the reviewed warp-local tail synchronization.\nextern "C" __global__ void candle_dkr_moe_gguf_warp_tail_v1() {}')
    if (candle / allowed).read_text().strip() != expected.strip():
        raise ValueError("GGUF CUDA source differs from reviewed synchronization correction")
    core = tomllib.loads((candle / "candle-core/Cargo.toml").read_text())
    nn = tomllib.loads((candle / "candle-nn/Cargo.toml").read_text())
    if CANDLE_FEATURE not in core["features"] or f"candle/{CANDLE_FEATURE}" not in nn["features"].get(CANDLE_FEATURE, []):
        raise ValueError("Candle core/NN provider feature missing")
    contract = (candle / "candle-kernels/dkr_s31_contract.rs").read_text()
    line = re.search(r'pub const FAMILIES:.*?=\s*&\[(.*?)\];', contract, re.S)
    if not line or set(re.findall(r'"([^"]+)"', line[1])) != FAMILIES:
        raise ValueError("Candle native module set differs from integration contract")
    prefix = "candle-kernels/src/"
    actual = {p.removeprefix(prefix) for p in git(candle, "ls-files", prefix).splitlines()
              if p.startswith(prefix) and p.endswith(".cu")}
    expected = expected_cuda_units()
    if actual != expected:
        raise ValueError(f"Candle CUDA source inventory drift: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    for name, disposition in REMAINING.items():
        if f'("{name}", "{disposition}")' not in contract:
            raise ValueError(f"remaining source classification changed: {name}")
    loader = (candle / "candle-core/src/cuda_backend/dkr_s31.rs").read_text()
    for family in FAMILIES - {"mmvq_gguf"} - MMQ_FAMILIES - MOE_FAMILIES:
        if f'(kernels::{family.upper()}.index(), "{family}")' not in loader:
            raise ValueError(f"missing native dispatch route {family}")
    if "CANDLE_DKR_S31_DIR" not in loader or "CANDLE_DKR_MODULE_LOADED" not in loader:
        raise ValueError("Candle activation or module trace contract missing")
    mmvq = (candle / "candle-core/src/quantized/fast_mmvq.rs").read_text()
    device = (candle / "candle-core/src/cuda_backend/device.rs").read_text()
    if not all(fragment in mmvq for fragment in ("try_dkr_launch", "dev.dkr_mmvq_func", "plain_cuda{batch}")) or "fn dkr_mmvq_func" not in device:
        raise ValueError("fast MMVQ external dispatch wiring missing")
    mmq = (candle / "candle-core/src/quantized/fast_mmq.rs").read_text()
    if not all(fragment in mmq for fragment in ("external::launch", "dev.dkr_mmq_func", "contract::mmq_tile", "PLAN_CAPACITY: usize = 64")) or "fn dkr_mmq_func" not in device:
        raise ValueError("fast MMQ external launch-plan wiring missing")
    for family in MMQ_FAMILIES:
        if not (candle / "candle-kernels/src/mmq_gguf" / f"{family}.cu").is_file():
            raise ValueError(f"missing MMQ original compilation unit: {family}")
    moe = (candle / "candle-nn/src/moe/dkr_moe.rs").read_text()
    nn_dispatch = (candle / "candle-nn/src/moe.rs").read_text()
    if not all(fragment in moe for fragment in ("dkr_moe_function", "moe_simt_f16", "moe_gguf", "device_ptr_with_guard")) or not all(fragment in nn_dispatch for fragment in ("dkr_moe::simt", "dkr_moe::gguf")):
        raise ValueError("Candle-NN external MoE source wiring missing")
    registry = (candle / "dkr-kernel-registry/src/lib.rs").read_text()
    registry_manifest = tomllib.loads((candle / "dkr-kernel-registry/Cargo.toml").read_text())
    if set(registry_manifest.get("dependencies", {})) != {"sha2"}:
        raise ValueError("neutral registry dependency boundary changed")
    if not all(fragment in registry for fragment in ("pub struct Registry<", "pub struct VerifiedArtifact", "pub struct PlanCache<")) or "get_or_load_dkr_func" not in device or "dkr_kernel_registry::PlanCache" not in mmq:
        raise ValueError("shared registry/cache source wiring missing")
    return changed


def patch_sources(workspace, core):
    deps = tomllib.loads(workspace)["workspace"]["dependencies"]
    for name in CANDLE_PINS:
        if deps[name].get("rev") != BASE or deps[name].get("git") != "https://github.com/haricot/candle.git":
            raise ValueError(f"{name}: qualified source pin changed; review before promotion")
    # Align all four Candle workspace dependencies in the integration candidate.
    pattern = re.compile(r'^(candle-(?:core|nn|flash-attn-v3|metal-kernels)\s*=\s*\{[^\n]*\brev\s*=\s*")'
                         + BASE + r'(".*)$', re.M)
    new_workspace, count = pattern.subn(lambda m: m[1] + CANDIDATE + m[2], workspace)
    if count != len(CANDLE_PINS):
        raise ValueError("expected exactly four active Candle pins")
    features = tomllib.loads(core)["features"]
    forwarding = [f"candle-core/{CANDLE_FEATURE}", f"candle-nn/{CANDLE_FEATURE}"]
    if PROFILE not in features or set(forwarding).intersection(features[PROFILE]):
        raise ValueError("unexpected experimental profile")
    anchor = f'{PROFILE} = [\n'
    if core.count(anchor) != 1:
        raise ValueError("experimental feature layout changed")
    new_core = core.replace(anchor, anchor + "".join(f'    "{f}",\n' for f in forwarding), 1)
    assert tomllib.loads(new_core)["features"][PROFILE] == [*forwarding, *features[PROFILE]]
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
                 f'candle-nn = {{ git = "https://github.com/haricot/candle.git", rev = "{BASE}" }}\n'
                 f'candle-flash-attn-v3 = {{ git = "https://github.com/haricot/candle.git", rev = "{BASE}" }}\n'
                 f'candle-metal-kernels = {{ git = "https://github.com/haricot/candle.git", rev = "{BASE}" }}\n')
    core = f'[features]\n{PROFILE} = [\n    "cuda-legacy",\n]\n'
    new_workspace, new_core = patch_sources(workspace, core)
    assert new_workspace.count(CANDIDATE) == 4 and BASE not in new_workspace
    assert f'"candle-core/{CANDLE_FEATURE}"' in new_core
    try:
        patch_sources(new_workspace, core)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unexpected existing pin")
    assert f'"candle-nn/{CANDLE_FEATURE}"' in new_core
    assert len(FAMILIES) == 25 and len(expected_cuda_units()) == 28
    assert len(REMAINING) == 3 and len(MMQ_FAMILIES) == 11 and len(MOE_FAMILIES) == 2
    print("DKR_CANDLE_INTEGRATION_SELF_TEST=PASS no_cuda_compilation")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candle-dir", type=Path)
    p.add_argument("--emit-patch", type=Path, help="pin/feature-only compatibility patch; use integration branch for full shared-runtime code")
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
        "cuda_source_inventory": {"total": len(expected_cuda_units()),
            "external_staged": len(FAMILIES), "remaining": REMAINING},
        "conv_expected_entries": 66,
        "sort_expected_entries": 14,
        "moe_native_dispatch": "staged_runtime_unverified",
        "moe_operational_entries": 10,
        "moe_gguf_revision_marker": "candle_dkr_moe_gguf_warp_tail_v1",
        "moe_wmma_static_link_verified": False,
        "fast_mmvq_expected_entries": 243,
        "fast_mmq": "external_native_ptx_cached_sm61_tiling_runtime_unverified",
        "fast_mmq_expected_entries": 163,
        "fast_mmq_plan_capacity_per_thread": 64,
        "mistral_profile": PROFILE, "candle_feature": CANDLE_FEATURE,
        "candle_feature_forwarding": ["candle-core", "candle-nn"],
        "activation": {"CANDLE_DKR_S31_DIR": "absolute Candle package root",
                       "CANDLE_DKR_TRACE": "1", "DKR_KERNEL_PACKAGES": "existing Mistral package root"},
        "candidate_changed_files": changed,
        "patch": str(a.emit_patch) if a.emit_patch else None,
        "integration_branch": INTEGRATION_BRANCH,
        "patch_scope": "pin_and_feature_only; full shared registry adapters are on integration branch",
        "aligned_candle_pins": list(CANDLE_PINS),
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
