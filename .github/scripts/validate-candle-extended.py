#!/usr/bin/env python3
"""Validate the source contract of haricot/candle:extended.

This checks MANIFEST STRUCTURE and feature requirements only. It does NOT
declare a candidate GPU-validated. The caller must verify the immutable
Candle SHA against a completed, physically validated SM61 promotion archive
and retain the matching CPU/CUDA/GPU proof artifacts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KIND = re.compile(r"^extended(?:-[a-z][a-z0-9-]*)?$")
_FEATURE = re.compile(r"^[a-z][a-z0-9_]*$")

# Requirements are defined by the CONSUMER (Mistral ASD), not by how many
# additional features Candle happens to compose.
ASD_REQUIRED_FEATURES = (
    "asd_core_v3_standalone",
    "bf16_candle",
    "cudnn_fallback_candle",
    "fp4_candle",
    "fp8_candle",
    "moe_simt_f16_candle",
)


def validate_manifest(doc: object, required_features=ASD_REQUIRED_FEATURES) -> dict:
    def fail(message: str) -> None:
        raise ValueError(message)

    if not isinstance(doc, dict):
        fail("manifest must be a JSON object")
    if doc.get("schema_version") != 1:
        fail("unsupported Candle manifest schema_version (expected 1)")
    if not isinstance(doc.get("kind"), str) or not _KIND.fullmatch(doc["kind"]):
        fail("kind must be 'extended' or an existing 'extended-*' label")
    if doc.get("integration_target") != "extended":
        fail("integration_target must be 'extended'")
    if doc.get("strategy") not in ("exact-delta", "merge"):
        fail("unknown integration strategy")
    for key in ("main_sha",):
        if not isinstance(doc.get(key), str) or not _SHA40.fullmatch(doc[key]):
            fail(f"{key} is not a full Git SHA")
    if not isinstance(doc.get("source_config_sha256"), str) or not _SHA256.fullmatch(doc["source_config_sha256"]):
        fail("source_config_sha256 is missing or malformed")
    features = doc.get("features")
    if not isinstance(features, list) or not features:
        fail("features must be a non-empty array")
    names: set[str] = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            fail(f"features[{index}] must be an object")
        name = feature.get("feature")
        if not isinstance(name, str) or not _FEATURE.fullmatch(name):
            fail(f"features[{index}].feature invalid")
        if name in names:
            fail(f"duplicate feature {name}")
        names.add(name)
        for key in ("source_sha", "review_source_sha", "prepared_sha", "common_base_sha"):
            value = feature.get(key)
            if not isinstance(value, str) or not _SHA40.fullmatch(value):
                fail(f"{name}.{key} is not a full Git SHA")
        for key in ("source_patch_sha256", "prepared_patch_sha256"):
            value = feature.get(key)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                fail(f"{name}.{key} is not a SHA-256 digest")
        for key in ("source_ref", "target_ref", "preparation_ref"):
            if not isinstance(feature.get(key), str) or not feature[key].strip():
                fail(f"{name}.{key} missing")
        if doc["strategy"] == "exact-delta" and (
            feature["source_patch_sha256"] != feature["prepared_patch_sha256"]
        ):
            fail(f"{name} exact-delta patch hash mismatch")
    required = set(required_features)
    missing = sorted(required - names)
    if missing:
        fail("missing required Mistral ASD features: " + ", ".join(missing))
    state = doc.get("validation_state")
    if not isinstance(state, str) or not state:
        fail("validation_state missing")
    return {
        "schema_version": doc["schema_version"],
        "kind": doc["kind"],
        "feature_count": len(names),
        "features": sorted(names),
        "required_features": sorted(required),
        "validation_state_in_manifest": state,
        "validated_by_this_check": False,
        "physical_proof": "EXTERNAL_PROOF_REQUIRED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--required-feature", action="append", default=None,
        help="Consumer-required feature; repeat to override default ASD feature set",
    )
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        required = (
            args.required_feature if args.required_feature is not None
            else ASD_REQUIRED_FEATURES
        )
        report = validate_manifest(manifest, required)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"CANDLE_MANIFEST_REJECTED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
