import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("pilot_campaign", Path(__file__).resolve().parents[1] / "campaign.py")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class PilotContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = c.identity("a" * 40, "b" * 40, "c" * 64, {"sources": ["source"]}, "d" * 40,
                                   "ghcr.io/haricot/compiler@sha256:" + "e" * 64,
                                   "sdk-inputs", "portable", {"build.sh": "f" * 64})
        self.plan = {"key": c.digest(self.identity), "identity": self.identity, "candidate": "a" * 40}

    def xml(self, names, suffix=""):
        return '<testsuites><testsuite>' + ''.join(f'<testcase name="{name}">{suffix}</testcase>' for name in names) + '</testsuite></testsuites>'

    def build(self):
        for name in ("tests.tar.zst", "mistralrs", "cargo-nextest", "compiler.txt", "nextest.toml"):
            (self.root / name).write_text("fixture " + name)
        (self.root / "Cargo.lock").write_text("locked")
        self.plan["identity"]["lock"] = c.sha(self.root / "Cargo.lock")
        self.plan["key"] = c.digest(self.plan["identity"])
        (self.root / "cpu-junit.xml").write_text(self.xml(c.CPU_REQUIRED))
        return c.build_proof(self.plan, self.root)

    def test_build_identity_excludes_run_attempt_and_changes_for_cpu_or_sdk(self):
        key = c.digest(self.identity)
        for name, value in [("rustflags", "-C target-cpu=x86-64-v3"), ("image", "ghcr.io/haricot/compiler@sha256:" + "1" * 64),
                            ("candidate", "2" * 40), ("recipe", {"build.sh": "3" * 64})]:
            altered = copy.deepcopy(self.identity)
            altered[name] = value
            self.assertNotEqual(key, c.digest(altered))
        self.assertNotIn("run_id", self.identity)
        self.assertNotIn("run_attempt", self.identity)

    def test_mutable_sdk_and_native_cpu_are_rejected(self):
        with self.assertRaises(ValueError):
            c.identity("a" * 40, "b" * 40, "c" * 64, {}, "d" * 40, "compiler:latest", "key", "portable", {})
        with self.assertRaises(ValueError):
            c.identity("a" * 40, "b" * 40, "c" * 64, {}, "d" * 40, self.identity["image"], "key", "native", {})

    def test_required_skipped_or_missing_tests_never_count(self):
        path = self.root / "tests.xml"
        for data in [self.xml(["unrelated"]), self.xml(["required"], "<skipped/>"), self.xml(["required"], "<failure/>")]:
            path.write_text(data)
            with self.assertRaises(ValueError):
                c.junit(path, ["required"])

    def test_complete_cpu_proof_does_not_claim_gpu(self):
        proof = self.build()
        self.assertFalse(proof["gpu_executed"])
        self.assertEqual(c.verify_build(self.plan, self.root), proof)

    def test_modified_binary_and_wrong_input_proof_are_rejected(self):
        self.build()
        (self.root / "mistralrs").write_text("changed")
        with self.assertRaises(ValueError):
            c.verify_build(self.plan, self.root)
        altered = copy.deepcopy(self.plan)
        altered["key"] = "different"
        with self.assertRaises(ValueError):
            c.verify_build(altered, self.root)

    def test_incomplete_build_file_manifest_is_rejected(self):
        proof = self.build()
        del proof["files"]["mistralrs"]
        c.write(self.root / "build.json", proof)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            c.verify_build(self.plan, self.root)

    def test_vl_and_gguf_numeric_failures_remain_informative(self):
        value = {"cpu_executed": True, "cuda_executed": True, "errors": {}, "status": "FAIL"}
        self.assertTrue(c.numeric_accepted("vl", value))
        self.assertTrue(c.numeric_accepted("gguf", value))
        self.assertFalse(c.numeric_accepted("text", value))
        self.assertEqual(value["status"], "FAIL")

    def test_execution_errors_are_blocking_even_for_vl(self):
        for mode in ("text", "vl", "gguf"):
            self.assertFalse(c.numeric_accepted(mode, {"cpu_executed": True, "cuda_executed": False, "errors": {}, "status": "PASS"}))
            self.assertFalse(c.numeric_accepted(mode, {"cpu_executed": True, "cuda_executed": True, "errors": {"cuda": "crashed"}, "status": "PASS"}))

    def test_numeric_parser_errors_preserve_observation_without_becoming_execution_failures(self):
        helper = type("Helper", (), {"measure": staticmethod(lambda *args: (_ for _ in ()).throw(ValueError("nonfinite logprob")))})
        result = c.measure_pair(helper, {}, {}, {})
        self.assertEqual(result, {"status": "FAIL", "numeric_error": "nonfinite logprob"})

    def test_archive_traversal_and_symlinks_are_rejected(self):
        for name, symlink in [("../escape", False), ("link", True)]:
            archive = self.root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as z:
                item = zipfile.ZipInfo(name)
                if symlink:
                    item.external_attr = 0o120777 << 16
                z.writestr(item, "outside")
            with self.assertRaises(ValueError):
                c.safe_extract(archive, self.root / "out")

    def test_gpu_proof_cannot_be_replaced_by_cpu_success(self):
        build = self.build()
        c.write(self.root / "gpu.json", {"status": "BUILD_AND_CPU_PASSED", "key": self.plan["key"]})
        with self.assertRaises(ValueError):
            c.verify_gpu(self.plan, self.root, build)

    def test_producer_must_match_workflow_branch_sha_and_successful_job(self):
        proof = {"origin": {"run_id": "12345678", "run_attempt": 1, "control_sha": "a" * 40}}
        entry = {"workflow_run": {"id": 12345678}}
        run = {"path": c.WORKFLOW, "head_branch": c.BRANCH, "head_sha": "a" * 40,
               "head_repository": {"full_name": c.REPOSITORY}}
        for altered in [{**run, "path": "other.yml"}, {**run, "head_sha": "b" * 40}]:
            with patch.object(c, "api", return_value=altered), self.assertRaises(ValueError):
                c.verify_producer(proof, "Hosted Legacy V3 CUDA and CPU", entry)
        with patch.object(c, "api", side_effect=[run, {"total_count": 1, "jobs": [{"name": "Hosted Legacy V3 CUDA and CPU", "conclusion": "failure"}]}]), self.assertRaises(ValueError):
            c.verify_producer(proof, "Hosted Legacy V3 CUDA and CPU", entry)

    def test_unknown_or_expired_exact_artifact_cannot_fall_back(self):
        with patch.object(c, "api", return_value={"name": "wrong", "expired": False}), self.assertRaises(ValueError):
            c.restore(self.plan, self.root / "restore", "build", 12345678)

    def test_plan_detects_changed_bundle_and_recipe(self):
        (self.root / "candidate.bundle").write_text("bundle")
        recipe = self.root / "orchestrator/build.sh"
        recipe.parent.mkdir()
        recipe.write_text("build")
        self.plan["identity"]["recipe"] = {"build.sh": c.sha(recipe)}
        self.plan["key"] = c.digest(self.plan["identity"])
        self.plan["bundle_sha256"] = c.sha(self.root / "candidate.bundle")
        c.write(self.root / "plan.json", self.plan)
        c.verify_plan(self.root)
        recipe.write_text("changed")
        with self.assertRaises(ValueError):
            c.verify_plan(self.root)


if __name__ == "__main__":
    unittest.main()
