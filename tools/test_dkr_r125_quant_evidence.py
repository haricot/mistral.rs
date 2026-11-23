#!/usr/bin/env python3
"""Offline regression tests for quantified GPU CUBIN evidence admission."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("dkr_r125_quant_evidence.py")
SPEC = importlib.util.spec_from_file_location("quant_evidence", SCRIPT)
quant_evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quant_evidence)


class QuantEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "sm61"
        self.root.mkdir()
        self.log = Path(self.temp.name) / "nextest.log"
        self.hashes = {}
        for scope, (family, _) in quant_evidence.CASES.items():
            binary = b"CUBIN FIXTURE " + scope.encode("ascii")
            digest = hashlib.sha256(binary).hexdigest()
            self.hashes[scope] = digest
            for dtype in ("f16", "bf16"):
                for fmt, variants in (("fp8", ("plain",)), ("mxfp4", ("plain", "bias"))):
                    for mode in variants:
                        folder = self.root / family / fmt / dtype / mode
                        folder.mkdir(parents=True)
                        (folder / "image.cubin").write_bytes(binary)
                        (folder / "manifest.json").write_text(json.dumps({
                            "target_cc": 61,
                            "cubin_artifact": "image.cubin",
                            "cubin_sha256": digest,
                        }))

    def make_log(self, scopes=("gemm",)):
        rows = ["DKR_R125_TYPED_POLICY requested=cubin source=verified_external_pair"]
        for scope in scopes:
            rows.append(
                f"CANDLE_DKR_TYPED_IMAGE_LOADED target=sm61 format=cubin "
                f"cubin_sha256={self.hashes[scope]} ptx_sha256=" + "0" * 64
            )
            _, test = quant_evidence.CASES[scope]
            rows.append(f"        PASS [   0.230s] (1/1) mistralrs-quant "
                        f"quantized_{scope}_dynamic::tests::{test}")
        self.log.write_text("\n".join(rows) + "\n")

    def test_one_image_for_six_gemm_profiles_is_sufficient(self):
        self.make_log()
        result = quant_evidence.verify(self.root, self.log, "gemm")
        self.assertEqual(result["families"]["quant-gemm"]["profiles"], 6)
        self.assertEqual(result["families"]["quant-gemm"]["distinct_cubin_images"], 1)

    def test_combined_evidence_requires_two_distinct_families(self):
        self.make_log(("gemm", "gemv"))
        result = quant_evidence.verify(self.root, self.log, "both")
        self.assertEqual(result["numeric_tests"], 2)
        self.assertEqual(result["cubin_modules"], 2)

    def test_one_family_cannot_claim_two(self):
        self.make_log(("gemm",))
        with self.assertRaisesRegex(ValueError, "Nextest PASS not found"):
            quant_evidence.verify(self.root, self.log, "both")

    def test_missing_load_is_not_a_pass(self):
        self.make_log(("gemm",))
        self.log.write_text(self.log.read_text().replace(
            "CANDLE_DKR_TYPED_IMAGE_LOADED", "CANDLE_DKR_TYPED_IMAGE_IGNORED"))
        with self.assertRaisesRegex(ValueError, "not loaded"):
            quant_evidence.verify(self.root, self.log, "gemm")

    def test_wrong_hash_is_not_a_pass(self):
        self.make_log(("gemm",))
        self.log.write_text(self.log.read_text().replace(
            self.hashes["gemm"], "f" * 64))
        with self.assertRaisesRegex(ValueError, "not loaded"):
            quant_evidence.verify(self.root, self.log, "gemm")

    def test_ptx_fallback_is_not_a_pass(self):
        self.make_log(("gemm",))
        with self.log.open("a") as f:
            f.write("CANDLE_DKR_TYPED_IMAGE_FALLBACK target=sm61\n")
        with self.assertRaisesRegex(ValueError, "PTX fallback"):
            quant_evidence.verify(self.root, self.log, "gemm")


if __name__ == "__main__":
    unittest.main()
