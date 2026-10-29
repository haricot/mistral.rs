import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from resolve_dspark import transplant, ALLOWED
from campaign import migrate_lfm3_speculative_memory_test_api

PATH = "mistralrs-cli/src/args/mod.rs"

class DsparkReconcileTests(unittest.TestCase):
    def make_speculative_fixture(self, root_export=True, implementation="fn placeholder() {}"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        repo = Path(temporary.name)
        path = repo / "mistralrs-core/src/speculative/config.rs"
        path.parent.mkdir(parents=True)
        path.write_text(implementation + "\\n#[cfg(test)]\\nmod tests {\\n"
                        " let first = crate::MemoryGpuConfig::Utilization(0.9);\\n"
                        " let second = crate::MemoryGpuConfig::Utilization(0.8);\\n"
                        "}\\n")
        lib = repo / "mistralrs-core/src/lib.rs"
        lib.write_text("pub use memory_plan::{ExecutionMemoryLimits, MemoryGoal};\\n"
                       if root_export else "mod memory_plan;\\n")
        return repo, path

    def test_migrates_only_removed_memory_gpu_test_enum(self):
        repo, path = self.make_speculative_fixture()
        self.assertEqual(migrate_lfm3_speculative_memory_test_api(repo), 2)
        value = path.read_text()
        self.assertIn("fn placeholder() {}", value)
        self.assertEqual(value.count("crate::MemoryGoal::Utilization("), 2)
        self.assertNotIn("MemoryGpuConfig", value)
        self.assertEqual(migrate_lfm3_speculative_memory_test_api(repo), 0)

    def test_rejects_missing_lfm3_memorygoal_export(self):
        repo, path = self.make_speculative_fixture(root_export=False)
        value = path.read_text()
        with self.assertRaisesRegex(RuntimeError, "not exported"):
            migrate_lfm3_speculative_memory_test_api(repo)
        self.assertEqual(path.read_text(), value)

    def test_rejects_outdated_runtime_memory_api(self):
        repo, path = self.make_speculative_fixture(
            implementation="fn runtime() { crate::MemoryGpuConfig::Utilization(0.9); }"
        )
        value = path.read_text()
        with self.assertRaisesRegex(RuntimeError, "outside DSpark tests"):
            migrate_lfm3_speculative_memory_test_api(repo)
        self.assertEqual(path.read_text(), value)

    def test_rejects_unrecognized_old_memory_gpu_variant(self):
        repo, path = self.make_speculative_fixture()
        path.write_text(path.read_text().replace(
            "let second = crate::MemoryGpuConfig::Utilization(0.8);",
            "let second = crate::MemoryGpuConfig::Bytes(100);"
        ))
        with self.assertRaisesRegex(RuntimeError, "unrecognized historical"):
            migrate_lfm3_speculative_memory_test_api(repo)

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

    def test_duplicate_changes_are_applied_at_both_contexts(self):
        base = ["a\n","left\n","base\n","mid\n","right\n","base\n","z\n"]
        feature = ["a\n","left\n","extra\n","base\n","mid\n",
                   "right\n","extra\n","base\n","z\n"]
        merged, _ = transplant(base, feature, base[:], PATH)
        self.assertEqual(merged.count("extra\n"), 2)

    def test_bolt_gguf_module_is_not_overwritten_by_lfm3(self):
        base = ["mod base;\n","mod legacy;\n","mod early;\n",
                "mod other;\n","mod newer;\n","mod end;\n"]
        feature = ["mod base;\n","mod legacy;\n","mod gemma4_config;\n",
                   "mod early;\n","mod other;\n","mod newer;\n","mod end;\n"]
        upstream = ["mod base;\n","mod legacy;\n","mod lfm3_config;\n",
                    "mod early;\n","mod other;\n","mod newer;\n","mod end;\n"]
        merged, _ = transplant(base, feature, upstream,
                               "mistralrs-core/src/gguf/mod.rs")
        self.assertIn("mod gemma4_config;", merged)
        self.assertIn("mod lfm3_config;", merged)

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
