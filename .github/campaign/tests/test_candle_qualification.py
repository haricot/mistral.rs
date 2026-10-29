import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("candle_proof", Path(__file__).resolve().parents[1] / "verify-candle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandleProofTests(unittest.TestCase):
    def setUp(self):
        self.ref = "refs/heads/archive/legacy/sm61-validated-123-1"
        self.sha = "a" * 40
        self.plan = {"target": "legacy", "candle": {"cuda_legacy": {"sha": self.sha, "validation_ref": self.ref}}}
        self.responses = [{"ref": self.ref, "object": {"type": "commit", "sha": self.sha}},
                          {"status": "completed", "conclusion": "success"}]

    def test_exact_physical_commit_is_required(self):
        with patch.object(module, "get", side_effect=self.responses):
            module.verify(self.plan)
        self.responses[0]["object"]["sha"] = "b" * 40
        with patch.object(module, "get", side_effect=self.responses):
            with self.assertRaisesRegex(RuntimeError, "another commit"):
                module.verify(self.plan)

    def test_failed_physical_run_is_rejected(self):
        self.responses[1]["conclusion"] = "failure"
        with patch.object(module, "get", side_effect=self.responses):
            with self.assertRaisesRegex(RuntimeError, "not successful"):
                module.verify(self.plan)

    def test_extended_requires_both_candle_variants(self):
        self.plan["target"] = "extended"
        with self.assertRaisesRegex(RuntimeError, "missing pinned"):
            module.verify(self.plan)

    def test_legacy_cannot_use_an_extended_proof_name(self):
        self.plan["candle"]["cuda_legacy"]["validation_ref"] = self.ref.replace("/legacy/", "/extended/")
        with self.assertRaisesRegex(RuntimeError, "invalid Candle"):
            module.verify(self.plan)
