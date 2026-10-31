import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qualification", ROOT / "qualification.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = q.read(ROOT / "models.json")

    def evidence(self, device):
        top = [{"token": str(n), "bytes": [48 + n], "logprob": -float(n + 1)} for n in range(5)]
        tokens = [{"token": "0", "bytes": [48], "logprob": -1.0, "top_logprobs": copy.deepcopy(top)} for _ in range(4)]
        return {"device": device, "request": {"model": "unit-test-only", "messages": ["test"]},
                "fixture_sha256": None, "response": {"usage": {"prompt_tokens": 8, "completion_tokens": 4},
                "choices": [{"logprobs": {"content": tokens}}]}}

    def report(self):
        context = {"policy": self.policy, "runtimes": {r: {"fingerprint": q.digest(r)} for r in q.RUNTIMES}}
        context["asd_profiles"] = {"current.json": "b" * 64, "current.asd": "c" * 64}
        context["fingerprint"] = q.digest({"runtimes": {r: context["runtimes"][r]["fingerprint"] for r in q.RUNTIMES},
                                              "asd_profiles": context["asd_profiles"]})
        report = {"schema": 1, "fingerprint": context["fingerprint"], "status": "PASS", "full_models_executed": True,
                  "sm": "6.1", "gpu_uuid": "unit-test-only", "driver": "unit-test-only",
                  "asd_profiles": context["asd_profiles"], "runtimes": {}, "files": {}}
        for runtime in q.RUNTIMES:
            report["runtimes"][runtime] = {**context["runtimes"][runtime], "binary_sha256": "a" * 64, "models": {}}
            for mode in q.MODES:
                pair = {d: self.evidence(d) for d in ("cpu", "cuda")}
                result = q.compare(pair["cpu"], pair["cuda"], self.policy)
                result["weights"] = self.policy["models"][mode]
                report["runtimes"][runtime]["models"][mode] = result
                for d, evidence in pair.items():
                    name = f"{runtime}-{mode}-{d}.json"
                    q.write(self.root / name, evidence)
                    report["files"][name] = q.sha(self.root / name)
        q.write(self.root / "report.json", report)
        return context, report

    def test_complete_references_are_verified(self):
        context, _ = self.report()
        self.assertEqual(q.check_report(self.root, context)["status"], "PASS")

    def test_missing_vl_model_rejects_promotion(self):
        context, report = self.report()
        del report["runtimes"]["extended"]["models"]["vl"]
        q.write(self.root / "report.json", report)
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_kernel_only_report_is_rejected(self):
        context, report = self.report()
        report["full_models_executed"] = False
        q.write(self.root / "report.json", report)
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_altered_numeric_artifact_is_rejected(self):
        context, _ = self.report()
        (self.root / "legacy-text-cpu.json").write_text("{}")
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_rehashed_false_counter_is_recomputed_from_raw_references(self):
        context, report = self.report()
        path = self.root / "legacy-text-cpu.json"
        data = q.read(path)
        data["response"]["usage"]["completion_tokens"] = 0
        q.write(path, data)
        report["files"][path.name] = q.sha(path)
        q.write(self.root / "report.json", report)
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_device_identity_must_be_cpu_then_cuda(self):
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), self.evidence("cpu"), self.policy)

    def test_prompt_or_fixture_change_is_rejected(self):
        cuda = self.evidence("cuda")
        cuda["request"]["messages"] = ["other prompt"]
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_nan_logprob_is_rejected(self):
        cuda = self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] = float("nan")
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_numeric_drift_exceeding_fixed_tolerance_is_rejected(self):
        cuda = self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] += 0.1
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_greedy_decode_divergence_is_rejected(self):
        cuda = self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["token"] = "other"
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_low_probability_fifth_token_boundary_swap_is_accepted(self):
        cpu, cuda = self.evidence("cpu"), self.evidence("cuda")
        for item in (cpu, cuda):
            item["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][-1]["logprob"] = -8.0
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][-1]["token"] = "boundary"
        self.assertEqual(q.compare(cpu, cuda, self.policy)["status"], "PASS")

    def test_significant_fifth_token_swap_is_rejected(self):
        cuda = self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][-1]["token"] = "other"
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_multiple_top_token_replacements_are_rejected(self):
        cuda = self.evidence("cuda")
        for token in cuda["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][-2:]:
            token["token"] = "different" + token["token"]
            token["logprob"] = -9.0
        with self.assertRaises(RuntimeError): q.compare(self.evidence("cpu"), cuda, self.policy)

    def test_stale_runtime_fingerprint_is_rejected(self):
        context, report = self.report()
        report["runtimes"]["legacy"]["fingerprint"] = "c" * 64
        q.write(self.root / "report.json", report)
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_modified_model_or_tolerance_policy_invalidates_evidence(self):
        context, _ = self.report()
        context["policy"] = copy.deepcopy(context["policy"])
        context["policy"]["logprob_abs_tolerance"] = 100
        with self.assertRaises(RuntimeError): q.check_report(self.root, context)

    def test_source_fingerprint_reuses_docs_but_rejects_code_lock_and_flags(self):
        repo = self.root / "repo"
        repo.mkdir()
        q.run("git", "-C", repo, "init", "-q")
        q.run("git", "-C", repo, "config", "user.name", "Test")
        q.run("git", "-C", repo, "config", "user.email", "test@example.invalid")
        def commit(name, content):
            (repo / name).write_text(content)
            q.run("git", "-C", repo, "add", ".")
            q.run("git", "-C", repo, "commit", "-qm", "test")
            return q.run("git", "-C", repo, "rev-parse", "HEAD")
        plan = {"image": self.policy["image"], "config": {"features": ["cuda"], "toolchain": "1.99.0"}}
        original = commit("model.rs", "fn main() {}")
        first = q.source_fingerprint(repo, original, plan, self.policy)
        doc = commit("README.md", "documentation")
        self.assertEqual(first, q.source_fingerprint(repo, doc, plan, self.policy))
        code = commit("model.rs", "fn main() { println!(\"changed\"); }")
        self.assertNotEqual(first, q.source_fingerprint(repo, code, plan, self.policy))
        lock = commit("Cargo.lock", "different dependency")
        self.assertNotEqual(q.source_fingerprint(repo, code, plan, self.policy), q.source_fingerprint(repo, lock, plan, self.policy))
        changed = copy.deepcopy(plan)
        changed["config"]["features"].append("cudnn")
        self.assertNotEqual(q.source_fingerprint(repo, lock, plan, self.policy), q.source_fingerprint(repo, lock, changed, self.policy))


if __name__ == "__main__":
    unittest.main()
