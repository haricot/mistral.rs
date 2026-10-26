import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import campaign as c
import proof as p


class GitCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        c.git(self.repo, "init", "-b", "master")
        c.git(self.repo, "config", "user.name", "Test")
        c.git(self.repo, "config", "user.email", "test@example.invalid")
        self.write("Cargo.toml", "[workspace.dependencies]\n" + "".join(
            f'{name} = {{ git = "https://github.com/haricot/candle.git", version = "0.11.0", rev = "{ "0" * 40}" }}\n'
            for name in c.CANDLE_CRATES))
        self.write("Cargo.lock", "base lock\n")
        self.write("src/base.rs", "base\n")
        self.base = self.commit("base")
        for name in ("legacy", "a", "b"):
            c.git(self.repo, "switch", "-c", name, self.base)
            self.write(f"src/{name}.rs", name + "\n")
            self.commit(name)
        c.git(self.repo, "switch", "-c", "extended", self.base)
        c.git(self.repo, "switch", "master")
        self.write("src/upstream.rs", "upstream\n")
        self.master = self.commit("new master")
        self.remote = self.root / "remote.git"
        c.command(["git", "init", "--bare", self.remote])
        c.git(self.repo, "remote", "add", "origin", self.remote)
        c.git(self.repo, "push", "--all", "origin")
        self.config = {
            "toolchain": "1.99.0", "master_branch": "master", "features": ["cuda"],
            "components": {n: {"branch": n, "requires": [] if n == "legacy" else ["legacy"],
                                "candle_branch": "cuda_legacy" if n == "legacy" else "extended"}
                           for n in ("legacy", "a", "b")},
            "targets": {"extended": {"branch": "extended", "components": ["legacy", "a", "b"]}}
        }
        self.state = {
            "schema_version": 2, "config": self.config, "master": self.master,
            "target": "extended", "leases": {n: c.remote_head(self.repo, "origin", n)
                                              for n in ("master", "legacy", "a", "b", "extended")},
            "candle": {"cuda_legacy": {"sha": "1" * 40, "validation_ref": "legacy-proof"},
                       "extended": {"sha": "2" * 40, "validation_ref": "extended-proof"}},
            "image": "image@sha256:" + "3" * 64, "recipe": c.recipe(), "input_key": "key"}
        self.state_path = self.root / "snapshot.json"
        c.write_json(self.state_path, self.state)
        self.real_command = c.command
        self.cargo_calls = 0

    def write(self, path, value):
        dest = self.repo / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value)

    def commit(self, message):
        c.git(self.repo, "add", "-A")
        c.git(self.repo, "commit", "-m", message)
        return c.git(self.repo, "rev-parse", "HEAD")

    def fake_cargo(self, args, cwd=None, **kwargs):
        if args[0] == "cargo":
            self.cargo_calls += 1
            if "update" in args:
                (Path(cwd) / "Cargo.lock").write_text("locked " + c.file_hash(Path(cwd) / "Cargo.toml") + "\n")
            return subprocess.CompletedProcess(args, 0, "", "")
        return self.real_command(args, cwd=cwd, **kwargs)

    def prepare(self, output="first"):
        with patch.object(c, "command", side_effect=self.fake_cargo):
            return c.prepare(self.repo, self.state_path, self.root / output, publish=True)

    def test_standalones_receive_master_and_declared_prerequisite_without_peer(self):
        result = self.prepare()
        a = result["components"]["a"]
        tree = c.git(self.repo, "ls-tree", "-r", "--name-only", a).splitlines()
        self.assertIn("src/upstream.rs", tree)
        self.assertIn("src/legacy.rs", tree)
        self.assertIn("src/a.rs", tree)
        self.assertNotIn("src/b.rs", tree)
        self.assertTrue(c.ancestor(self.repo, self.master, a))
        self.assertTrue(c.ancestor(self.repo, result["components"]["legacy"], a))
        self.assertIn("src/b.rs", c.git(self.repo, "ls-tree", "-r", "--name-only", result["candidate"]))

    def test_unchanged_inputs_reuse_candidates_without_cargo_or_new_commits(self):
        first = self.prepare()
        calls = self.cargo_calls
        second = self.prepare("second")
        self.assertEqual(first["candidate"], second["candidate"])
        self.assertEqual(first["components"], second["components"])
        self.assertEqual(self.cargo_calls, calls)

    def test_preparation_does_not_publish_feature_branches(self):
        self.prepare()
        for branch, sha in self.state["leases"].items():
            self.assertEqual(c.remote_head(self.repo, "origin", branch), sha)

    def test_promoted_sources_reuse_the_same_candidates(self):
        first = self.prepare()
        for name, sha in first["components"].items():
            branch = "extended" if name == "extended" else self.config["components"][name]["branch"]
            c.git(self.repo, "push", "origin", f"{sha}:refs/heads/{branch}")
            self.state["leases"][branch] = sha
        c.write_json(self.state_path, self.state)
        calls = self.cargo_calls
        second = self.prepare("after-promotion")
        self.assertEqual(first["candidate"], second["candidate"])
        self.assertEqual(first["components"], second["components"])
        self.assertEqual(self.cargo_calls, calls)

    def proofs(self, plan):
        root = self.root / "proofs"
        archives = {}
        for node in plan["matrix"]:
            directory = root / node["component"]
            directory.mkdir(parents=True)
            (directory / "tests.tar.zst").write_bytes(node["candidate"].encode())
            archive = c.file_hash(directory / "tests.tar.zst")
            c.write_json(directory / "build.json", {
                **node, "status": "BUILD_AND_CPU_PASSED", "cpu_test_count": 3,
                "files": {"tests.tar.zst": archive}})
            archives[node["component"]] = archive
        c.write_json(root / "gpu/report.json", {
            "candidate": plan["candidate"], "run_id": "123", "status": "GPU_PASSED",
            "target": "sm_61", "archives": archives, "gpu_executed": True,
            "test_counts": {name: 1 for name in archives}})
        return root

    def remote_with_candle(self, repo, remote, branch):
        if remote != "origin":
            return self.state["candle"][branch]["sha"]
        return self.real_remote(repo, remote, branch)

    def test_promotion_updates_only_the_verified_standalones_and_composition(self):
        plan = self.prepare()
        proofs = self.proofs(plan)
        self.real_remote = c.remote_head
        with patch.object(c, "remote_head", side_effect=self.remote_with_candle):
            c.promote(self.repo, self.root / "first/plan.json", proofs, "123")
        for name, sha in plan["components"].items():
            branch = "extended" if name == "extended" else self.config["components"][name]["branch"]
            self.assertEqual(c.remote_head(self.repo, "origin", branch), sha)

    def test_changed_source_lease_blocks_all_publication(self):
        plan = self.prepare()
        proofs = self.proofs(plan)
        c.git(self.remote, "update-ref", "refs/heads/a", self.master)
        self.real_remote = c.remote_head
        with patch.object(c, "remote_head", side_effect=self.remote_with_candle):
            with self.assertRaisesRegex(RuntimeError, "source lease"):
                c.promote(self.repo, self.root / "first/plan.json", proofs, "123")
        self.assertEqual(c.remote_head(self.repo, "origin", "extended"), self.state["leases"]["extended"])
        self.assertEqual(c.remote_head(self.repo, "origin", "b"), self.state["leases"]["b"])

    def test_other_gpu_archive_blocks_publication(self):
        plan = self.prepare()
        proofs = self.proofs(plan)
        path = proofs / "gpu/report.json"
        report = json.loads(path.read_text())
        report["archives"]["a"] = "other"
        c.write_json(path, report)
        with self.assertRaisesRegex(RuntimeError, "hosted archive"):
            c.promote(self.repo, self.root / "first/plan.json", proofs, "123")

    def test_ancestor_check_rejects_unrelated_feature(self):
        self.assertFalse(c.ancestor(self.repo, self.state["leases"]["a"], self.state["leases"]["b"]))

    def test_gpu_report_without_execution_blocks_publication(self):
        plan = self.prepare()
        proofs = self.proofs(plan)
        path = proofs / "gpu/report.json"
        report = json.loads(path.read_text())
        report["gpu_executed"] = False
        c.write_json(path, report)
        with self.assertRaisesRegex(RuntimeError, "physical GPU proof"):
            c.promote(self.repo, self.root / "first/plan.json", proofs, "123")

    def test_functional_conflict_stops_preparation(self):
        c.git(self.repo, "switch", "a")
        self.write("src/base.rs", "feature a changed this\n")
        self.state["leases"]["a"] = self.commit("conflict on feature")
        c.git(self.repo, "push", "origin", "a")
        c.git(self.repo, "switch", "master")
        self.write("src/base.rs", "upstream changed this\n")
        self.state["master"] = self.commit("conflict on master")
        self.state["leases"]["master"] = self.state["master"]
        c.git(self.repo, "push", "origin", "master")
        c.write_json(self.state_path, self.state)
        with self.assertRaisesRegex(RuntimeError, "functional merge"):
            self.prepare()
        self.assertEqual(c.remote_head(self.repo, "origin", "a"), self.state["leases"]["a"])

    def test_pin_candle_rejects_missing_workspace_dependency(self):
        self.write("Cargo.toml", "[workspace.dependencies]\n")
        with self.assertRaisesRegex(RuntimeError, "workspace dependency"):
            c.pin_candle(self.repo, "1" * 40)

    def test_snapshot_rejects_mutable_image_before_accessing_remotes(self):
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            c.snapshot(self.repo, "extended", "nvidia/cuda:latest", self.root / "bad.json")


class ProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.expected = {"candidate": "1" * 40, "build_key": "k", "component": "a", "lock": "l"}
        (self.root / "tests.tar.zst").write_bytes(b"compiled tests")
        self.proof = {**self.expected, "status": "BUILD_AND_CPU_PASSED", "cpu_test_count": 3,
                      "files": {"tests.tar.zst": c.file_hash(self.root / "tests.tar.zst")}}
        c.write_json(self.root / "build.json", self.proof)

    def test_accepts_exact_build(self):
        self.assertEqual(c.verify_artifact(self.root, self.expected)["cpu_test_count"], 3)

    def test_rejects_archive_tampering(self):
        (self.root / "tests.tar.zst").write_bytes(b"different executable")
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            c.verify_artifact(self.root, self.expected)

    def test_rejects_other_candidate_or_build_environment(self):
        for key in ("candidate", "build_key", "component", "lock"):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "mismatch"):
                c.verify_artifact(self.root, {**self.expected, key: "wrong"})

    def test_rejects_path_outside_artifact(self):
        self.proof["files"] = {"../outside": "bad"}
        c.write_json(self.root / "build.json", self.proof)
        with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
            c.verify_artifact(self.root, self.expected)

    def test_cpu_report_requires_executed_backend_test(self):
        path = self.root / "junit.xml"
        for text in ('<testsuites/>', '<testsuite><testcase name="other"/></testsuite>',
                     '<testsuite><testcase name="campaign_cpu_backend"><skipped/></testcase></testsuite>',
                     '<testsuite><testcase name="campaign_cpu_backend"><failure/></testcase></testsuite>'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                p.test_count(path, "campaign_cpu_backend")

    def test_valid_cpu_execution_report(self):
        path = self.root / "junit.xml"
        path.write_text('<testsuite><testcase name="campaign_cpu_backend"/></testsuite>')
        self.assertEqual(p.test_count(path, "campaign_cpu_backend"), 1)

    def test_only_candle_source_conflicts_are_resolved_automatically(self):
        text = ('[workspace.dependencies]\n<<<<<<< HEAD\n'
                'candle-core = {git="old",version="0.11.0",rev="a"}\n=======\n'
                'candle-core = {git="new",version="0.11.0",rev="b"}\n>>>>>>> source\n')
        self.assertNotIn("<<<<<<<", c.resolve_candle_conflicts(text))
        self.assertIn("<<<<<<<", c.resolve_candle_conflicts(text.replace('version="0.11.0",rev="b"', 'version="0.12.0",rev="b"')))
        self.assertIn("<<<<<<<", c.resolve_candle_conflicts(text.replace("candle-core", "unrelated")))


if __name__ == "__main__":
    unittest.main()
