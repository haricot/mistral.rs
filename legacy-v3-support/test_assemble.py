import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("assemble", Path(__file__).with_name("assemble.py"))
assemble = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble)


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        assemble.git(self.repo, "init", "-q")
        assemble.git(self.repo, "config", "user.name", "Test")
        assemble.git(self.repo, "config", "user.email", "test@localhost")
        (self.repo / "base").write_text("base")
        self.base = self.commit()
        (self.repo / "old-legacy").write_text("unrelated legacy")
        self.legacy = self.commit()
        assemble.git(self.repo, "checkout", "--detach", self.base)
        self.contract = {"component": "fp4", "base_master_sha": self.base,
                         "requires_mistral_branches": [], "candle": {"revision": "locked"},
                         "allowed_paths": ["payload", "contract.json"],
                         "payload_sha256": {"payload": hashlib.sha256(b"approved").hexdigest()}}
        (self.repo / "payload").write_text("approved")
        self.write_contract()
        self.source = {"sha": self.commit(), "component": "fp4", "manifest": "contract.json"}
        self.meta = {"base_master_sha": self.base, "source_legacy_sha": self.legacy, "candle_sha": "locked"}

    def commit(self):
        assemble.git(self.repo, "add", ".")
        assemble.git(self.repo, "commit", "-qm", "fixture")
        return assemble.git(self.repo, "rev-parse", "HEAD").stdout.decode().strip()

    def write_contract(self):
        (self.repo / "contract.json").write_text(json.dumps(self.contract))

    def test_accepts_exact_reviewed_input(self):
        self.assertEqual(set(assemble.verify_source(self.repo, self.meta, self.source)),
                         {"payload", "contract.json"})

    def test_rejects_corrupt_payload(self):
        (self.repo / "payload").write_text("changed")
        self.source["sha"] = self.commit()
        with self.assertRaisesRegex(ValueError, "payload changed"):
            assemble.verify_source(self.repo, self.meta, self.source)

    def test_rejects_extra_scope(self):
        (self.repo / "hidden-format").write_text("unreviewed")
        self.source["sha"] = self.commit()
        with self.assertRaisesRegex(ValueError, "scope changed"):
            assemble.verify_source(self.repo, self.meta, self.source)

    def test_rejects_implicit_legacy_dependency(self):
        self.contract["requires_mistral_branches"] = ["legacy"]
        self.write_contract()
        self.source["sha"] = self.commit()
        with self.assertRaisesRegex(ValueError, "dependency"):
            assemble.verify_source(self.repo, self.meta, self.source)

    def test_rejects_legacy_ancestry(self):
        assemble.git(self.repo, "merge", "--no-edit", self.legacy)
        self.source["sha"] = assemble.git(self.repo, "rev-parse", "HEAD").stdout.decode().strip()
        with self.assertRaisesRegex(ValueError, "inherit old Legacy"):
            assemble.verify_source(self.repo, self.meta, self.source)

    def test_rejects_changed_resolution_preimage(self):
        with self.assertRaises(ValueError):
            assemble.replace_once("new upstream code", "old preimage", "replacement")
        with self.assertRaises(ValueError):
            assemble.replace_once("old old", "old", "replacement")


if __name__ == "__main__":
    unittest.main()
