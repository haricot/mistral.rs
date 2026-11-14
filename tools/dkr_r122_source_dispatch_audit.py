#!/usr/bin/env python3
"""R1.22/S1: source-only inventory and production dispatch discovery.

No Cargo, NVCC, GPU or cache writes. A lexical call-site candidate is not a
reachable runtime route. A compiled manifest is not a performance certificate.
Run with the pinned, clean Candle checkout; never silently accept another rev.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from dkr_r1_inventory import (
    ROOT, mistral_family, candle_family, BACKEND_METHOD_GROUPS,
)
from dkr_s9_ptx_manifest_audit import builder_packages, verify, PACKAGE_OUTPUT_FAMILY
from dkr_operation_matrix import STAGED_OPERATIONS

# Package output family -> qualification feature (core forwards to leaf crates).
PACKAGE_FEATURE = {
    name: "dkr-" + name for name in (
        "add-rms-norm", "causal-mask", "gated-add", "indexed-copy",
        "input-pad", "moe", "reduce-norm", "rms-variants", "ssm",
        "rounded-rope", "kv-scales", "fp8-cast", "kv-gather",
        "fp4-dequant", "quant-gemv", "paged-decode", "nvfp4-gemv",
        "quant-gemm", "paged-fp8", "paged-v2", "kv-cache-ops",
        "routed-quant",
    )
}
PACKAGE_FEATURE.update({
    "sparse-logits": "dkr-logits", "dense-gemv": "dkr-gemv",
    "dense-gemm": "dkr-gemm",
})

# Reviewed entry-point evidence for the feature routes missed by the
# proximity scan. All fragments must be present in the stated tracked file.
# These prove source wiring, NOT execution by a particular model.
REVIEWED_ROUTES = {
    "add-rms-norm": ("mistralrs-core/src/ops.rs", (
        '#[cfg(all(feature = "cuda", feature = "dkr-add-rms-norm"))]',
        'pub use crate::cuda::add_rms_norm_dynamic::cuda_add_rms_norm;',
    )),
    "causal-mask": ("mistralrs-core/src/ops.rs", (
        '#[cfg(all(feature = "cuda", feature = "dkr-causal-mask"))]',
        'pub(crate) use crate::cuda::causal_mask_dynamic::cuda_apply_causal_mask_f32;',
    )),
    "gated-add": ("mistralrs-core/src/cuda/gated_add.rs", (
        '#[cfg(feature = "dkr-gated-add")]',
        '#[path = "gated_add_dynamic.rs"]',
        'pub(crate) use provider::try_sigmoid_gated_add;',
    )),
    "indexed-copy": ("mistralrs-core/src/cuda/indexed_copy.rs", (
        '#[cfg(feature = "dkr-indexed-copy")]',
        '#[path = "indexed_copy_dynamic.rs"]',
        'pub(crate) use provider::copy_rows;',
    )),
    "moe": ("mistralrs-core/src/cuda/moe.rs", (
        '#[cfg(feature = "dkr-moe")]',
        '#[path = "moe_dynamic.rs"]',
        'pub use provider::moe_gemm;',
    )),
    "nvfp4-gemv": ("mistralrs-quant/src/nvfp4/mod.rs", (
        '#[cfg(feature = "dkr-nvfp4-gemv")]',
        'if let Some(output) = gemv_dynamic::try_gemv(',
    )),
    "rms-variants": ("mistralrs-core/src/ops.rs", (
        '#[cfg(feature = "dkr-rms-variants")]',
        'crate::cuda::rms_variants_dynamic::with_rms_plan(',
    )),
    "sparse-logits": ("mistralrs-core/src/ops.rs", (
        '#[cfg(feature = "dkr-logits")]',
        'crate::cuda::logits_dynamic::with_logits_plans(',
    )),
    "ssm": ("mistralrs-core/src/cuda/ssm.rs", (
        '#[cfg(all(feature = "cuda", feature = "dkr-ssm"))]',
        'super::ssm_dynamic::selective_scan_cuda(',
    )),
}


# S2: inspect the other fifteen package entry points, including both
# codec routes and both KV-cache operations. Do not equate this with
# model-level reachability or certify a performance improvement.
REVIEWED_ROUTES.update({
    "dense-gemm": ("mistralrs-quant/src/unquantized/mod.rs", (
        '#[cfg(feature = "dkr-gemm")]',
        'dense_gemm_dynamic::try_gemm(&input, &self.w, self.b.as_ref())?',
    )),
    "dense-gemv": ("mistralrs-quant/src/gemv/mod.rs", (
        '#[cfg(feature = "dkr-gemv")]',
        'if let Some(result) = dynamic::try_gemv(',
    )),
    "fp8-cast": ("mistralrs-quant/src/scalar_fp8/ops.rs", (
        '#[cfg(feature = "dkr-fp8-cast")]',
        'input_s, input_l, DType::F8E4M3, self.target_dtype,',
        'input_s, input_l, self.source_dtype, DType::F8E4M3,',
        'super::dynamic::try_cast(',
    )),
    "input-pad": ("mistralrs-core/src/cuda/input_packing.rs", (
        '#[cfg(feature = "dkr-input-pad")]',
        'super::input_pad_dynamic::with_pad_plan(',
    )),
    "kv-cache-ops": ("mistralrs-paged-attn/src/cuda/backend/cache.rs", (
        '#[cfg(feature = "dkr-kv-cache-ops")]',
        '&& super::kv_cache_ops_dynamic::try_copy_blocks(',
    )),
    "kv-gather": ("mistralrs-paged-attn/src/cuda/backend/gather_kv.rs", (
        '#[cfg(feature = "dkr-kv-gather")]',
        '&& super::kv_gather_dynamic::try_gather(',
    )),
    "kv-scales": ("mistralrs-paged-attn/src/cuda/backend/scale_update.rs", (
        '#[cfg(feature = "dkr-kv-scales")]',
        'if super::kv_scales_dynamic::update_if_supported(',
    )),
    "paged-decode": ("mistralrs-paged-attn/src/cuda/backend/paged_attention.rs", (
        '#[cfg(feature = "dkr-paged-decode")]',
        'if let Some(result) = super::paged_decode_dynamic::try_decode(',
    )),
    "paged-fp8": ("mistralrs-paged-attn/src/cuda/backend/paged_attention.rs", (
        '#[cfg(feature = "dkr-paged-fp8")]',
        'if let Some(result) = super::paged_fp8_dynamic::try_decode(',
    )),
    "paged-v2": ("mistralrs-paged-attn/src/cuda/backend/paged_attention.rs", (
        '#[cfg(feature = "dkr-paged-v2")]',
        'if let Some(result) = super::paged_v2_dynamic::try_decode(',
    )),
    "quant-gemm": ("mistralrs-quant/src/blockwise_fp8/ops.rs", (
        '#[cfg(feature = "dkr-quant-gemm")]',
        'if let Some(output) = crate::quantized_gemm_dynamic::try_fp8(',
    )),
    "quant-gemv": ("mistralrs-quant/src/blockwise_fp8/ops.rs", (
        '#[cfg(feature = "dkr-quant-gemv")]',
        'if let Some(output) = crate::quantized_gemv_dynamic::try_fp8(',
    )),
    "reduce-norm": ("mistralrs-quant/src/utils/ops.rs", (
        '#[cfg(all(feature = "cuda", feature = "dkr-reduce-norm"))]',
        'crate::softmax_sinks_dynamic::cuda_fwd(',
    )),
    "rounded-rope": ("mistralrs-core/src/cuda/rounded_rope.rs", (
        '#[cfg(feature = "dkr-rounded-rope")]',
        'return super::rounded_rope_dynamic::run(',
    )),
    "routed-quant": ("mistralrs-quant/src/routed_gemv/mod.rs", (
        '#[cfg(feature = "dkr-routed-quant")]',
        'if let Some(output) = dynamic::try_gather(x, indices, weights)?',
    )),
})
EXTRA_ROUTES = {
    "kv-cache-ops": [("mistralrs-paged-attn/src/cuda/backend/paged_attention.rs", (
        '#[cfg(feature = "dkr-kv-cache-ops")]',
        'if super::kv_cache_ops_dynamic::try_reshape(',
    ))],
    "quant-gemv": [("mistralrs-quant/src/mxfp4/ops.rs", (
        '#[cfg(feature = "dkr-quant-gemv")]',
        'if let Some(output) = crate::quantized_gemv_dynamic::try_mxfp4(',
    ))],
    "quant-gemm": [("mistralrs-quant/src/mxfp4/ops.rs", (
        '#[cfg(feature = "dkr-quant-gemm")]',
        'if let Some(output) = crate::quantized_gemm_dynamic::try_mxfp4(',
    ))],
    "reduce-norm": [("mistralrs-quant/src/moe/mod.rs", (
        '#[cfg(feature = "dkr-reduce-norm")]',
        '#[path = "weighted_reduce_dynamic.rs"]',
        'pub use weighted_reduce::moe_weighted_reduce;',
    ))],
}

# Explicit unresolved constraints; not a runtime fallback guarantee.
ROUTE_REVIEW_NOTES = {
    "quant-gemm": "Outer FP8/MXFP4 layer dispatch still checks static kernel capability flags; reference GEMM performance not qualified.",
    "quant-gemv": "Outer FP8/MXFP4 layer dispatch still checks static kernel capability flags; SM61 legacy FP8 enables the FP8 flag.",
    "routed-quant": "Runtime opt-in MISTRALRS_DKR_ROUTED_QUANT=1; routed gather wiring does not prove all fused FFN/GLU paths use DKR.",
    "dense-gemm": "Reference external GEMM is not a performance-qualified substitute for cuBLAS.",
    "paged-v2": "Selection follows FP8 and short-context routes; model shapes and fallback paths still require runtime evidence.",
    "reduce-norm": "Feature-selected implementations are not necessarily runtime-switchable to the static route.",
}



def reviewed_route(root, family):
    if family not in REVIEWED_ROUTES:
        return None
    routes = []
    for name, fragments in [REVIEWED_ROUTES[family], *EXTRA_ROUTES.get(family, [])]:
        lines = (root / name).read_text().splitlines()
        evidence = []
        for fragment in fragments:
            matches = [i for i, line in enumerate(lines, 1) if fragment in line
                       and not line.lstrip().startswith("//")]
            if not matches:
                raise ValueError(f"{family}: reviewed source route changed: {fragment}")
            evidence.append({"fragment": fragment, "lines": matches})
        routes.append({"path": name, "evidence": evidence})
    return {"routes": routes, "review_note": ROUTE_REVIEW_NOTES.get(family)}


SOURCE_ROOTS = (
    "mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn",
    "mistralrs-vision", "mistralrs-flash-attn", "legacy-provider-poc",
)
QUAL_SCRIPT = "tools/dkr_sm61_final_qualification.sh"
FEATURE_PATTERN = re.compile(r'feature\s*=\s*"(?P<feature>dkr-[^"]+)"')
CALL_PATTERN = re.compile(r'\b(?:[a-zA-Z_]\w*::)+[a-zA-Z_]\w*\s*\(')
METHOD_PATTERN = re.compile(
    r"^\s*(?:unsafe\s+)?fn\s+([a-z][a-z_0-9]*)\s*(?:<[^>]+>)?\s*\(",
    re.MULTILINE,
)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def tracked(root):
    # ls-files is NUL-delimited; avoid including generated build/cache trees.
    return [Path(p) for p in git(root, "ls-files", "-z").split("\0") if p]


def require_clean(root, label):
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"{label}: tracked worktree/index is dirty")
    if git(root, "rev-parse", "--show-toplevel") != str(root.resolve()):
        raise ValueError(f"{label}: expected repository root")


def pinned_candle(root):
    workspace = tomllib.loads((root / "Cargo.toml").read_text())
    deps = workspace["workspace"]["dependencies"]
    pins = {deps[name]["rev"] for name in ("candle-core", "candle-nn")}
    if len(pins) != 1:
        raise ValueError(f"Candle dependency revisions disagree: {pins}")
    pin = pins.pop()
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError(f"not a full Candle commit SHA: {pin}")
    return pin


def rust_candidates(path, source):
    """Return feature markers and nearby qualified calls, NOT a Rust AST.

    A module declaration or import alone must not count as a call. Keep
    test-looking files/sections separate; cfg expressions and macro expansion
    still require manual review. Lines are evidence pointers, not proof.
    """
    lines = source.splitlines()
    test_hint = path.name in ("tests.rs", "test.rs") or path.stem.endswith("_tests")
    out = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("//"):
            continue
        if re.search(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]", line):
            test_hint = True
        for feature in sorted({m["feature"] for m in FEATURE_PATTERN.finditer(line)}):
            calls = []
            # Nearby lexical calls are deliberately only candidates; even
            # a feature marker on a module does not prove a call is gated.
            for j in range(i + 1, min(i + 9, len(lines))):
                s = lines[j].strip()
                if s.startswith(("#[", "mod ", "pub mod ", "pub(crate) mod ", "use ", "pub use ")):
                    break
                if s.startswith("//"):
                    continue
                for match in CALL_PATTERN.finditer(s):
                    calls.append({"line": j + 1, "call": match.group().rstrip("(").strip()})
            out.append({
                "feature": feature, "path": str(path), "line": i + 1,
                "test_hint": test_hint, "nearby_calls": calls,
            })
    return out


def exact_function_refs(root, paths, names):
    """Specific FP4 gap: qualified references to adapter functions."""
    refs = []
    pattern = re.compile(r"\bdequant_dynamic::(" + "|".join(names) + r")\s*\(")
    for path in paths:
        if path.name == "dequant_dynamic.rs":
            continue
        for i, line in enumerate((root / path).read_text().splitlines(), 1):
            if not line.lstrip().startswith("//") and pattern.search(line):
                refs.append({"path": str(path), "line": i, "text": line.strip()})
    return refs


def source_records(root, paths, candle=False):
    rows = []
    for path in paths:
        if path.suffix != ".cu":
            continue
        if candle:
            if path.parts[:2] != ("candle-kernels", "src"):
                continue
            family = candle_family(Path(*path.parts[2:]))
        else:
            if not path.parts or path.parts[0] not in SOURCE_ROOTS:
                continue
            family = mistral_family(path)
        rows.append({"path": str(path), "family": family})
    return rows


def audit(root, candle, packages):
    require_clean(root, "mistral.rs")
    require_clean(candle, "Candle")
    pin = pinned_candle(root)
    if git(candle, "rev-parse", "HEAD") != pin:
        raise ValueError(f"Candle checkout must be exactly {pin}")
    paths, candle_paths = tracked(root), tracked(candle)
    mist_units = source_records(root, paths)
    candle_units = source_records(candle, candle_paths, candle=True)
    backend_path = candle / "candle-core/src/backend.rs"
    methods = [
        {"method": name, "family": BACKEND_METHOD_GROUPS.get(name)}
        for name in METHOD_PATTERN.findall(backend_path.read_text())
    ]
    if not mist_units or not candle_units or not methods:
        raise ValueError("empty source or backend inventory")
    families = sorted(builder_packages())
    if set(families) != set(PACKAGE_FEATURE):
        raise ValueError("package/feature map differs from current cudaforge builder")
    external_units = [
        p for p in paths if p.suffix == ".cu" and p.parts[:2] == ("kernel-packages", "cuda")
    ]
    external_rows = [{"path": str(p), "package": PACKAGE_OUTPUT_FAMILY.get(p.parts[2], p.parts[2])}
                     for p in external_units]
    if {r["package"] for r in external_rows} != set(families):
        raise ValueError("external source families differ from builder packages")
    excluded_units = [str(p) for p in paths if p.suffix == ".cu"
                      and p.parts[0] not in SOURCE_ROOTS and p not in external_units]
    count, manifests, r121 = verify(packages, 61)
    if count != len(families):
        raise ValueError("manifest family count mismatch")
    by_family = collections.defaultdict(list)
    for path in sorted(packages.rglob("manifest.json")):
        data = json.loads(path.read_text())
        by_family[path.relative_to(packages).parts[0]].append({
            "operation": data["operation"], "entry": data["entry"],
            "implementation_id": data.get("implementation_id"),
            "sha256": data["sha256"], "manifest": str(path.relative_to(packages)),
        })
    rust_paths = [
        p for p in paths if p.suffix == ".rs" and len(p.parts) > 2
        and p.parts[0] in SOURCE_ROOTS and p.parts[1] == "src"
    ]
    candidate_map = collections.defaultdict(list)
    for path in rust_paths:
        for row in rust_candidates(path, (root / path).read_text()):
            candidate_map[row["feature"]].append(row)
    core = tomllib.loads((root / "mistralrs-core/Cargo.toml").read_text())["features"]
    wrappers = {
        crate: tomllib.loads((root / crate / "Cargo.toml").read_text())["features"]
        for crate in ("mistralrs", "mistralrs-cli", "mistralrs-server-core")
    }
    profile = "dkr-sm61-experimental"
    expected_profile = {"cuda-legacy"} | (set(PACKAGE_FEATURE.values()) - {"dkr-fp4-dequant"})
    if set(core.get(profile, [])) != expected_profile:
        raise ValueError("experimental SM61 aggregate must select precisely the production feature set")
    wrapper_profile = {
        "mistralrs": {"cuda-legacy", f"mistralrs-core/{profile}"},
        "mistralrs-server-core": {"cuda-legacy", f"mistralrs-core/{profile}"},
        "mistralrs-cli": {"cuda-legacy", f"mistralrs-core/{profile}",
                           f"mistralrs/{profile}", f"mistralrs-server-core/{profile}"},
    }
    for crate, expected in wrapper_profile.items():
        if set(wrappers[crate].get(profile, [])) != expected:
            raise ValueError(f"{crate}: experimental profile forwarding mismatch")
    if any(profile in table.get("default", []) for table in [core, *wrappers.values()]):
        raise ValueError("experimental profile must not be enabled by default")
    qualification = (root / QUAL_SCRIPT).read_text()
    package_rows = []
    for family in families:
        feature = PACKAGE_FEATURE[family]
        if feature not in core or f"mistralrs-core/{feature}" not in qualification:
            raise ValueError(f"{family}: missing core feature or qualification selection")
        candidates = candidate_map[feature]
        production = [r for r in candidates if not r["test_hint"] and r["nearby_calls"]]
        reviewed = reviewed_route(root, family)
        package_rows.append({
            "family": family, "feature": feature,
            "core_feature_dependencies": core[feature],
            "wrapper_explicit_same_name_feature": {
                name: feature in features for name, features in wrappers.items()
            },
            "profiles": by_family[family],
            "operations": sorted({p["operation"] for p in by_family[family]}),
            "feature_markers": candidates,
            "non_test_nearby_call_candidates": len(production),
            "reviewed_source_route": reviewed,
            "dispatch_status": "REVIEWED_SOURCE_WIRING_RUNTIME_UNVERIFIED" if reviewed
                else "LEXICAL_CANDIDATES_REVIEW_REQUIRED" if production
                else "NO_NON_TEST_NEARBY_CALL_CANDIDATE",
            "model_dispatch_verified": False,
            "performance_qualified": False,
        })
    unknown = [r for r in mist_units + candle_units + methods if r["family"] is None]
    fp4 = exact_function_refs(root, rust_paths, ("dequant_nvfp4", "dequant_mxfp4"))
    # FP4 dequantization is explicitly a parity/debug reference; do not
    # materialize dense weights in the inference hot path just for coverage.
    fp4_source = (root / "mistralrs-quant/src/nvfp4/dequant_dynamic.rs").read_text()
    if "Debug/parity dequantization only; no conversion is inserted into inference." in fp4_source and not fp4:
        for row in package_rows:
            if row["family"] == "fp4-dequant":
                row["dispatch_status"] = "DIAGNOSTIC_REFERENCE_ONLY"
    compiled_ops = {p["operation"] for rows in by_family.values() for p in rows}
    matrix_ops = {op for ops in STAGED_OPERATIONS.values() for op in ops}
    return {
        "schema": "DKR-R122-SOURCE-DISPATCH-AUDIT-V3",
        "mistral_revision": git(root, "rev-parse", "HEAD"),
        "candle_revision": pin,
        "target_cc": 61,
        "source_inventory": {
            "mistral_units": mist_units, "candle_units": candle_units,
            "backend_methods": methods, "unclassified": unknown,
            "external_units": external_rows, "out_of_scope_mistral_cuda_units": excluded_units,
        },
        "ptx_audit": {"families": count, "manifests": manifests, "r121_profiles": r121},
        "compiled_operations": len(compiled_ops),
        "operation_matrix_delta": {
            "compiled_not_listed": sorted(compiled_ops - matrix_ops),
            "listed_not_compiled": sorted(matrix_ops - compiled_ops),
        },
        "packages": package_rows,
        "experimental_feature_profile": {
            "name": profile, "core_features": sorted(expected_profile),
            "wrapper_forwarding": {name: sorted(value) for name, value in wrapper_profile.items()},
            "default_enabled": False, "cargo_build_verified": False,
            "runtime_note": "Routed quant requires MISTRALRS_DKR_ROUTED_QUANT=1; other runtime controls remain unchanged.",
        },
        "fp4_dequant_qualified_name_call_refs": fp4,
        "limitations": [
            "Feature markers and nearby calls are lexical evidence, not AST or reachability.",
            "An absent wrapper feature does not prevent explicit dependency feature selection.",
            "FP4 reference scan does not resolve imports, aliases, macros or indirection.",
            "GPU qualification logs and model-level route telemetry are separate evidence.",
            "External PTX integrity does not prove correspondence to current source.",
            "Inventory classifies source units, not every operation or specialization.",
        ],
        "status": "FAIL_UNCLASSIFIED" if unknown or excluded_units else (
            "FAIL_OPERATION_MATRIX_DELTA" if compiled_ops != matrix_ops
            else "SOURCE_DISCOVERY_PASS_DISPATCH_REVIEW_REQUIRED"
        ),
    }


def self_test():
    assert len(PACKAGE_FEATURE) == 25
    assert set(REVIEWED_ROUTES) == set(PACKAGE_FEATURE) - {"fp4-dequant"}
    assert set(EXTRA_ROUTES) <= set(REVIEWED_ROUTES)
    assert len({op for ops in STAGED_OPERATIONS.values() for op in ops}) == 91
    for family in REVIEWED_ROUTES:
        assert reviewed_route(ROOT, family) is not None
    sample = '''// #[cfg(feature = "dkr-no")]
#[cfg(feature = "dkr-fp4-dequant")]
mod dequant_dynamic;

#[cfg(feature = "dkr-quant-gemv")]
if let Some(out) = crate::quantized_gemv_dynamic::try_fp8(x)? {
    return Ok(out);
}
#[cfg(test)]
mod tests {
    #[cfg(feature = "dkr-gemm")]
    crate::dynamic::run();
}
'''
    rows = rust_candidates(Path("src/ops.rs"), sample)
    assert [r["feature"] for r in rows] == ["dkr-fp4-dequant", "dkr-quant-gemv", "dkr-gemm"]
    assert not rows[0]["nearby_calls"]
    assert not rows[1]["test_hint"] and rows[2]["test_hint"]
    assert any(c["call"] == "crate::quantized_gemv_dynamic::try_fp8" for c in rows[1]["nearby_calls"])
    assert source_records(ROOT, [Path("mistralrs-core/src/cuda/sort.cu")])[0]["family"] == "sampling_and_sort"
    assert source_records(ROOT, [Path("candle-kernels/src/cast.cu")], True)[0]["family"] == "cast"
    print("DKR_R122_SOURCE_AUDIT_SELF_TEST=PASS no_cuda_compilation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candle-dir", type=Path)
    parser.add_argument("--packages", type=Path, default=ROOT / "kernel-packages/build/cuda/sm61")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.candle_dir is None:
        parser.error("--candle-dir is required for R1.22")
    report = audit(ROOT, args.candle_dir.resolve(), args.packages.resolve())
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    print(f"DKR_R122_SOURCE_AUDIT={report['status']} "
          f"mistral_units={len(report['source_inventory']['mistral_units'])} "
          f"candle_units={len(report['source_inventory']['candle_units'])} "
          f"backend_methods={len(report['source_inventory']['backend_methods'])} "
          f"packages={report['ptx_audit']['families']} "
          f"operations={report['compiled_operations']}", file=sys.stderr)
    sys.exit(1 if report["status"].startswith("FAIL_") else 0)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, AssertionError, subprocess.CalledProcessError) as error:
        print(f"DKR_R122_SOURCE_AUDIT=FAIL {error}", file=sys.stderr)
        sys.exit(1)
