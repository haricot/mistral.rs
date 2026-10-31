"""Offline contracts for Extended-Release-R1, without building Rust or pulling images."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_r1 as r


class ReleaseR1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = {
            "target": "extended", "candidate": "a" * 40, "recipe": "b" * 64,
            "image": "ghcr.io/haricot/mistral-ci-sm61@sha256:" + "c" * 64,
            "config": {"toolchain": "1.99.0", "features": ["cuda-legacy-fp8", "cuda-legacy-bf16"]},
            "matrix": [{"component": "extended", "candidate": "a" * 40, "lock": "d" * 64}],
        }
        self.plan_file = self.root / "plan.json"
        self.plan_file.write_text(json.dumps(self.plan))
        self.executable = self.root / "mistralrs"
        self.executable.write_bytes(b"\x7fELF" + b"test")
        self.executable.chmod(0o755)

    def record(self):
        details = {**r.expected(self.plan),
                   "binary_sha256": r.hash_file(self.executable),
                   "binary_bytes": self.executable.stat().st_size}
        (self.root / "release.json").write_text(json.dumps(details))
        return details

    def test_valid_release_binary(self):
        details = self.record()
        self.assertEqual(r.inspect(self.plan_file, self.root), details)
        self.assertIn("code-execution", details["features"])

    def test_refuses_wrong_candidate(self):
        self.record()
        self.plan["candidate"] = "e" * 40
        self.plan_file.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, "candidate"):
            r.inspect(self.plan_file, self.root)

    def test_refuses_modified_binary(self):
        self.record()
        self.executable.write_bytes(self.executable.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            r.inspect(self.plan_file, self.root)

    def test_refuses_invalid_image(self):
        self.plan["image"] = "ghcr.io/haricot/mistral-ci-sm61:latest"
        with self.assertRaisesRegex(ValueError, "immutable"):
            r.expected(self.plan)

    def test_refuses_unqualified_target(self):
        self.plan["target"] = "legacy"
        with self.assertRaisesRegex(ValueError, "Extended"):
            r.expected(self.plan)


if __name__ == "__main__":
    unittest.main()
