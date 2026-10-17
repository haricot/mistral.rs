import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "compose_lfm25_dspark.py"
SPEC = importlib.util.spec_from_file_location("compose_lfm25_dspark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CANDLE = "1" * 40


class CompositionTests(unittest.TestCase):
    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def cargo(self, pin, owner="haricot"):
        return "[workspace.dependencies]\n" + "".join(
            f'{crate} = {{ git = "https://github.com/{owner}/candle.git", version = "0.11.0", rev = "{pin}" }}\n'
            for crate in MODULE.CANDLE_CRATES
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-b", "master")
        self.git("config", "user.name", "Composition Test")
        self.git("config", "user.email", "composition@example.invalid")
        self.write("Cargo.toml", self.cargo("0" * 40, "huggingface"))
        self.write("Cargo.lock", "base lock\n")
        self.write("src/lfm.rs", "baseline\n")
        self.base = self.commit("base")
        self.git("switch", "-c", "source")
        self.write("Cargo.toml", self.cargo(CANDLE))
        self.write("Cargo.lock", "standalone lock\n")
        self.write("src/lfm.rs", "native dspark\n")
        self.write("src/proposer.rs", "native proposer\n")
        self.manifest = {
            "schema_version": 1, "kind": "lfm25-dspark-standalone",
            "base_master_sha": self.base, "candle_sha": CANDLE,
        }
        self.write(MODULE.MANIFEST_PATH, json.dumps(self.manifest))
        self.write("mistral-integration/README.md", "standalone source\n")
        self.source = self.commit("standalone")
        self.git("switch", "-c", "seed", self.base)
        self.write("Cargo.toml", self.cargo("0" * 40))
        self.write("Cargo.lock", "seed lock\n")
        self.write("src/seed.rs", "seed feature\n")
        self.write(MODULE.MANIFEST_PATH, json.dumps({"schema_version": 1, "seed_only": True}))
        self.write("mistral-integration/README.md", "seed documentation\n")
        self.seed = self.commit("seed")

    def update_source(self, path, text):
        self.git("switch", "source")
        self.write(path, text)
        self.source = self.commit("update standalone")
        self.git("switch", "seed")

    def test_merges_full_source_preserves_seed_and_binds_both_parents(self):
        component = MODULE.compose(self.repo, self.source, CANDLE)
        self.assertEqual((self.repo / "src/proposer.rs").read_text(), "native proposer\n")
        self.assertEqual((self.repo / "src/seed.rs").read_text(), "seed feature\n")
        self.assertEqual((self.repo / "Cargo.lock").read_text(), "seed lock\n")
        target = json.loads((self.repo / MODULE.MANIFEST_PATH).read_text())
        self.assertTrue(target["seed_only"])
        self.assertEqual(target["components"]["lfm25_dspark"]["source_sha"], self.source)
        self.assertEqual(component["base_master_sha"], self.base)
        self.assertEqual(self.git("diff", "--name-only", "--diff-filter=U"), "")
        self.commit("candidate")
        self.assertEqual(self.git("show", "-s", "--format=%P").split(), [self.seed, self.source])

    def test_rejects_wrong_candle_pin_before_mutation(self):
        with self.assertRaisesRegex(RuntimeError, "validated Candle"):
            MODULE.compose(self.repo, self.source, "2" * 40)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_rejects_wrong_source_kind(self):
        self.manifest["kind"] = "other-feature"
        self.update_source(MODULE.MANIFEST_PATH, json.dumps(self.manifest))
        with self.assertRaisesRegex(RuntimeError, "standalone contract"):
            MODULE.compose(self.repo, self.source, CANDLE)

    def test_rejects_unaccounted_dependency_change(self):
        self.update_source("Cargo.toml", self.cargo(CANDLE) + 'extra = "1"\n')
        with self.assertRaisesRegex(RuntimeError, "dependencies differ"):
            MODULE.compose(self.repo, self.source, CANDLE)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_rejects_functional_merge_conflict(self):
        self.write("src/lfm.rs", "independent seed implementation\n")
        self.commit("seed conflict")
        with self.assertRaisesRegex(RuntimeError, "functional merge conflicts"):
            MODULE.compose(self.repo, self.source, CANDLE)

    def test_rejects_dirty_seed(self):
        self.write("untracked.txt", "not part of candidate\n")
        with self.assertRaisesRegex(RuntimeError, "clean seed"):
            MODULE.compose(self.repo, self.source, CANDLE)

    def test_existing_source_can_be_recomposed(self):
        MODULE.compose(self.repo, self.source, CANDLE)
        self.commit("candidate")
        MODULE.compose(self.repo, self.source, CANDLE)
        self.assertEqual(self.git("status", "--porcelain"), "")


if __name__ == "__main__":
    unittest.main()
