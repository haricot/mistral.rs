"""CPU-only proof that master impact inspection never promotes branches."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import impact
import stage as st
from unittest.mock import patch


class MasterImpactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-b", "master")
        self.git("config", "user.name", "Impact Tester")
        self.git("config", "user.email", "impact@example.invalid")
        self.write("Cargo.toml", "base = 1\n")
        self.base = self.commit("base")
        self.git("switch", "-c", "fp8_mistralrs_standalone", self.base)
        self.write("Cargo.toml", "branch = 1\n")
        self.fp8 = self.commit("fp8")
        self.git("switch", "-c", "bf16_mistralrs_standalone", self.base)
        self.write("src/bf16.rs", "bf16")
        self.bf16 = self.commit("bf16")
        self.git("switch", "master")
        self.write("Cargo.toml", "upstream = 2\n")
        self.write("mistralrs-core/src/models/lfm3.rs", "pub struct Model;\n")
        self.master = self.commit("upstream")
        for name, sha in [
            ("master", self.master),
            ("mirror_orch", self.master),
            ("fp8_mistralrs_standalone", self.fp8),
            ("bf16_mistralrs_standalone", self.bf16),
        ]:
            self.git("update-ref", f"refs/remotes/origin/{name}", sha)
        self.cfg = {
            "schema_version": 1,
            "master_branch": "master",
            "orchestrator_branch": "mirror_orch",
            "tracked_standalones": [
                "fp8_mistralrs_standalone", "bf16_mistralrs_standalone"
            ],
            "high_impact_prefixes": ["Cargo.toml", "mistralrs-core/src/models/"],
        }

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def write(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-m", message)
        return self.git("rev-parse", "HEAD")

    def test_detects_conflict_and_nonconflicting_component_without_modification(self):
        report = impact.inspect(self.repo, self.cfg)
        self.assertTrue(report["mirror_contains_master"])
        self.assertFalse(report["gpu_tests_executed"])
        self.assertEqual(report["promotions"], [])
        result = {entry["branch"]: entry for entry in report["standalones"]}
        self.assertEqual(result["fp8_mistralrs_standalone"]["status"], "MERGE_CONFLICT")
        self.assertFalse(result["fp8_mistralrs_standalone"]["textual_merge_clean"])
        self.assertEqual(result["fp8_mistralrs_standalone"]["overlapping_paths"], ["Cargo.toml"])
        self.assertEqual(result["bf16_mistralrs_standalone"]["status"], "MERGE_CLEAN_UNTESTED")
        self.assertTrue(result["bf16_mistralrs_standalone"]["textual_merge_clean"])
        self.assertEqual(self.git("rev-parse", "refs/remotes/origin/fp8_mistralrs_standalone"), self.fp8)
        self.assertEqual(self.git("rev-parse", "refs/remotes/origin/bf16_mistralrs_standalone"), self.bf16)
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_missing_mirror_sync_is_visible(self):
        self.git("update-ref", "refs/remotes/origin/mirror_orch", self.base)
        report = impact.inspect(self.repo, self.cfg)
        self.assertFalse(report["mirror_contains_master"])

    def test_stages_only_clean_merge_candidates_without_promotion(self):
        report = impact.inspect(self.repo, self.cfg)
        refs = {
            "master": self.master,
            "mirror_orch": self.master,
            "fp8_mistralrs_standalone": self.fp8,
            "bf16_mistralrs_standalone": self.bf16,
        }
        with patch.object(st, "remote", side_effect=lambda _repo, branch: refs.get(branch)):
            result = st.stage(self.repo, report, dry_run=True)
        self.assertEqual([x["branch"] for x in result["candidates"]],
                         ["bf16_mistralrs_standalone"])
        self.assertEqual(result["promotions"], [])
        self.assertFalse(result["source_branches_modified"])
        self.assertEqual(result["candidates"][0]["qualification"], "NOT_EXECUTED")
        self.assertEqual(self.git("rev-parse", "refs/remotes/origin/fp8_mistralrs_standalone"), self.fp8)
        self.assertEqual(self.git("rev-parse", "refs/remotes/origin/bf16_mistralrs_standalone"), self.bf16)

    def test_ref_uses_both_master_and_source_to_avoid_stale_reuse(self):
        first = st.staged_ref(self.master, self.bf16, "bf16_mistralrs_standalone")
        second = st.staged_ref(self.master, self.fp8, "bf16_mistralrs_standalone")
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(RuntimeError, "invalid immutable"):
            st.staged_ref(self.master, self.bf16, "refs/heads/master")

    def test_source_lease_change_stops_candidate_staging(self):
        report = impact.inspect(self.repo, self.cfg)
        refs = {
            "master": self.master,
            "mirror_orch": self.master,
            "fp8_mistralrs_standalone": self.fp8,
            "bf16_mistralrs_standalone": self.base,
        }
        with patch.object(st, "remote", side_effect=lambda _repo, branch: refs.get(branch)):
            with self.assertRaisesRegex(RuntimeError, "source branch moved"):
                st.stage(self.repo, report, dry_run=True)

    def test_read_only_summary_never_claims_qualification(self):
        report = impact.inspect(self.repo, self.cfg)
        summary = impact.render_summary(report)
        self.assertIn("No standalone was updated", summary)
        self.assertIn("NOT", summary)
        self.assertIn("MERGE_CONFLICT", summary)


if __name__ == "__main__":
    unittest.main()
