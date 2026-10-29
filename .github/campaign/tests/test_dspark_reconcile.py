import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resolve_dspark import transplant, ALLOWED

PATH = "mistralrs-cli/src/args/mod.rs"

class DsparkReconcileTests(unittest.TestCase):
    def test_preserves_upstream_and_feature_additions(self):
        base = ["header\n", "before\n", "scope\n", "last\n", "footer\n"]
        feature = ["header\n", "before\n", "new dspark\n", "scope\n", "last\n", "footer\n"]
        upstream = ["header\n", "before\n", "scope\n", "last\n", "new LFM3\n", "footer\n"]
        merged, decisions = transplant(base, feature, upstream, PATH)
        self.assertIn("new dspark\n", merged)
        self.assertIn("new LFM3\n", merged)
        self.assertEqual(len(decisions), 1)

    def test_refuses_modified_upstream_replacement(self):
        base = ["header\n", "before\n", "replace me\n", "after\n", "footer\n"]
        feature = ["header\n", "before\n", "DSpark\n", "after\n", "footer\n"]
        upstream = ["header\n", "before\n", "LFM3 different\n", "after\n", "footer\n"]
        with self.assertRaisesRegex(RuntimeError, "missing anchor"):
            transplant(base, feature, upstream, PATH)

    def test_refuses_unapproved_path(self):
        with self.assertRaisesRegex(RuntimeError, "whitelisted"):
            transplant(["a\n"], ["a\n"], ["a\n"], "mistralrs-core/src/models/lfm3.rs")

    def test_rejects_destructive_source_change(self):
        base = ["before\n", "danger a\n", "danger b\n", "danger c\n", "danger d\n", "after\n"]
        feature = ["before\n", "after\n"]
        with self.assertRaisesRegex(RuntimeError, "destructive delta"):
            transplant(base, feature, base[:], PATH)

if __name__ == "__main__":
    unittest.main()
