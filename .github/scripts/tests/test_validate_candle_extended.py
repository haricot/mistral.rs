"""Count-independent Candle extended contract regressions (no GPU required)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

validator_path = Path(__file__).resolve().parents[1] / "validate-candle-extended.py"
spec = importlib.util.spec_from_file_location("candle_extended_validator", validator_path)
validator = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validator)

_REQUIRED = list(validator.ASD_REQUIRED_FEATURES)
_EXTRAS = ["embeddinggemma2", "future_kernel_v2", "future_backend_v3"]


def sample_feature(name: str, index: int) -> dict:
    return {
        "feature": name,
        "source_ref": name,
        "target_ref": name,
        "preparation_ref": f"prepare/extended/example/{name}",
        "source_sha": f"{index + 1:040x}",
        "review_source_sha": f"{index + 1:040x}",
        "prepared_sha": f"{index + 10:040x}",
        "common_base_sha": "1" * 40,
        "source_patch_sha256": "a" * 64,
        "prepared_patch_sha256": "a" * 64,
    }


def manifest(extra_count: int = 0) -> dict:
    names = _REQUIRED + _EXTRAS[:extra_count]
    labels = {0: "six", 1: "seven", 2: "eight", 3: "nine"}
    return {
        "schema_version": 1,
        "kind": "extended-" + labels[extra_count],
        "integration_target": "extended",
        "strategy": "exact-delta",
        "main_sha": "1" * 40,
        "source_config_sha256": "a" * 64,
        "validation_state": "UNVALIDATED_CANDIDATE",
        "features": [sample_feature(name, i) for i, name in enumerate(names)],
    }


class CandleExtendedContractTest(unittest.TestCase):
    def test_accept_six_seven_eight_and_nine_without_changing_consumer(self):
        for extras in range(4):
            with self.subTest(feature_count=6 + extras):
                report = validator.validate_manifest(manifest(extras))
                self.assertEqual(report["feature_count"], 6 + extras)
                self.assertFalse(report["validated_by_this_check"])
                self.assertEqual(report["physical_proof"], "EXTERNAL_PROOF_REQUIRED")

    def test_accept_new_kind_without_number(self):
        doc = manifest(3)
        doc["kind"] = "extended"
        self.assertEqual(validator.validate_manifest(doc)["feature_count"], 9)

    def test_missing_asd_dependency_is_rejected(self):
        doc = manifest(1)
        doc["features"] = [f for f in doc["features"] if f["feature"] != "asd_core_v3_standalone"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            validator.validate_manifest(doc)

    def test_duplicate_is_rejected(self):
        doc = manifest(2)
        doc["features"].append(dict(doc["features"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate feature"):
            validator.validate_manifest(doc)

    def test_incorrect_hash_and_patch_mismatch_are_rejected(self):
        doc = manifest(1)
        doc["features"][0]["source_sha"] = "bad"
        with self.assertRaisesRegex(ValueError, "source_sha"):
            validator.validate_manifest(doc)
        doc = manifest(1)
        doc["features"][0]["prepared_patch_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "patch hash mismatch"):
            validator.validate_manifest(doc)

    def test_wrong_target_and_schema_are_rejected(self):
        doc = manifest()
        doc["integration_target"] = "release"
        with self.assertRaisesRegex(ValueError, "integration_target"):
            validator.validate_manifest(doc)
        doc = manifest()
        doc["schema_version"] = 99
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validator.validate_manifest(doc)

    def test_manifest_state_is_not_a_physical_proof(self):
        doc = manifest(1)
        doc["validation_state"] = "VALIDATED"
        self.assertFalse(validator.validate_manifest(doc)["validated_by_this_check"])


if __name__ == "__main__":
    unittest.main()
