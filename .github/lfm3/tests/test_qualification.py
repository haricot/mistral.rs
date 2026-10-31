import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

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

    def collection(self, infer, collection_only=True, corrupt_binary=False):
        context, report = self.report()
        context["collection_only"] = collection_only
        output = self.root / "collection"
        output.mkdir()
        for runtime in q.RUNTIMES:
            context["runtimes"][runtime]["candidate"] = q.digest(runtime)[:40]
            directory = output / "binaries" / runtime
            directory.mkdir(parents=True)
            binary = directory / "mistralrs"
            binary.write_bytes(b"unit-test-only")
            q.write(directory / "build.json", {**context["runtimes"][runtime],
                    "binary_sha256": q.sha(binary), "cargo_check": "PASS"})
            if corrupt_binary:
                binary.write_bytes(b"tampered")
        report.update(status="COLLECTING", full_models_executed=False, runtimes={}, files={})
        with mock.patch.object(q, "infer", side_effect=infer) as observed, contextlib.redirect_stdout(io.StringIO()):
            complete, passed = q.collect_models(output, context, {}, report)
        return context, report, output / "model-proof", observed, complete, passed

    def test_collection_finishes_all_six_after_numeric_failure(self):
        def infer(binary, mode, device, *_):
            data = self.evidence(device)
            if binary.parent.name == "legacy" and mode == "text" and device == "cuda":
                data["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] += 0.1
            return data
        context, report, dest, observed, complete, passed = self.collection(infer)
        self.assertEqual(observed.call_count, 12)
        self.assertTrue(complete)
        self.assertFalse(passed)
        self.assertEqual(report["status"], "COLLECTED")
        self.assertEqual(report["numeric_status"], "FAIL")
        self.assertTrue(report["full_models_executed"])
        self.assertEqual(len(list(dest.glob("*-cpu.json"))), 6)
        self.assertEqual(len(list(dest.glob("*-cuda.json"))), 6)
        self.assertIn("legacy | text", (dest / "summary.md").read_text())
        self.assertIn("extended | gguf", (dest / "summary.md").read_text())
        with self.assertRaises(RuntimeError): q.check_report(dest, context)

    def test_collection_continues_after_model_server_error(self):
        def infer(binary, mode, device, *_):
            if binary.parent.name == "legacy" and mode == "vl" and device == "cpu":
                raise OSError("HTTP Error 500: Internal Server Error")
            return self.evidence(device)
        _, report, dest, observed, complete, passed = self.collection(infer)
        self.assertEqual(observed.call_count, 12)
        self.assertFalse(complete or passed)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["full_models_executed"])
        case = report["runtimes"]["legacy"]["models"]["vl"]
        self.assertFalse(case["cpu_executed"])
        self.assertTrue(case["cuda_executed"])
        self.assertIn("HTTP Error 500", case["errors"]["cpu"])
        self.assertEqual(q.read(dest / "report.json"), report)

    def test_invalid_numeric_reference_does_not_stop_remaining_models(self):
        def infer(binary, mode, device, *_):
            data = self.evidence(device)
            if binary.parent.name == "legacy" and mode == "vl" and device == "cuda":
                data["response"]["usage"]["completion_tokens"] = 0
            return data
        _, report, _, observed, complete, passed = self.collection(infer)
        self.assertEqual(observed.call_count, 12)
        self.assertFalse(complete or passed)
        self.assertIn("references", report["runtimes"]["legacy"]["models"]["vl"]["errors"])

    def test_greedy_divergence_marks_later_prefixes_as_incomparable(self):
        cpu, cuda = self.evidence("cpu"), self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][1]["token"] = "other"
        result = q.measure(cpu, cuda, self.policy)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["metrics"]["greedy_tokens_equal"])
        self.assertTrue(result["metrics"]["positions"][1]["same_prefix"])
        self.assertFalse(result["metrics"]["positions"][2]["same_prefix"])

    def test_generated_and_alternative_numeric_errors_are_separate(self):
        cpu, cuda = self.evidence("cpu"), self.evidence("cuda")
        cuda["response"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["logprob"] += 0.08
        result = q.measure(cpu, cuda, self.policy)
        self.assertEqual(result["metrics"]["max_generated_logprob_error"], 0)
        self.assertAlmostEqual(result["metrics"]["max_common_top_logprob_error"], 0.08)
        self.assertGreater(result["metrics"]["max_common_top_probability_error"], 0)
        self.assertEqual(result["status"], "FAIL")

    def test_successful_collection_only_never_authorizes_promotion(self):
        context, report, dest, _, complete, passed = self.collection(lambda _b, _m, d, *_: self.evidence(d))
        self.assertTrue(complete and passed)
        self.assertEqual(report["status"], "COLLECTED")
        self.assertFalse(report["promotion_eligible"])
        report["status"] = "PASS"
        q.write(dest / "report.json", report)
        with self.assertRaises(RuntimeError): q.check_report(dest, context)

    def test_strict_qualification_still_rejects_collected_numeric_failure(self):
        def infer(_b, _m, device, *_):
            data = self.evidence(device)
            if device == "cuda":
                data["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] += 0.1
            return data
        context, report, dest, observed, complete, passed = self.collection(infer, collection_only=False)
        self.assertEqual(observed.call_count, 12)
        self.assertTrue(complete)
        self.assertFalse(passed)
        self.assertEqual(report["status"], "FAIL")
        with self.assertRaises(RuntimeError): q.check_report(dest, context)

    def test_strict_success_rechecks_all_raw_references(self):
        context, report, dest, _, complete, passed = self.collection(
            lambda _b, _m, d, *_: self.evidence(d), collection_only=False)
        self.assertTrue(complete and passed)
        self.assertTrue(report["promotion_eligible"])
        self.assertEqual(q.check_report(dest, context)["status"], "PASS")

    def test_collection_keeps_binary_sha256_gate(self):
        with mock.patch.object(q, "infer") as observed:
            with self.assertRaisesRegex(RuntimeError, "compiled runtime verification failed"):
                self.collection(observed, corrupt_binary=True)
            observed.assert_not_called()

    def test_gguf_does_not_force_the_external_tokenizer(self):
        paths = {m: self.root / m for m in q.MODES}
        for path in paths.values(): path.mkdir()
        q.write(paths["text"] / "config.json", {"num_hidden_layers": 32})
        q.write(paths["vl"] / "config.json", {"text_config": {"num_hidden_layers": 32}})
        for mode in q.MODES:
            with self.subTest(mode=mode), mock.patch.object(q.subprocess, "Popen") as launch:
                launch.return_value.poll.return_value = None
                with mock.patch.object(q, "http", side_effect=[{"data": [{"id": "unit-test"}]}, {"choices": []}]):
                    q.infer(self.root / "unit-test-binary", mode, "cuda", paths, self.policy, self.root / (mode + ".log"))
                args = launch.call_args.args[0]
                if mode == "gguf":
                    self.assertNotIn("--tokenizer", args)
                    self.assertEqual(args[args.index("--tok-model-id") + 1], str(paths["text"]))
                    self.assertEqual(args[args.index("--format") + 1], "gguf")
                else:
                    self.assertEqual(args[args.index("--tokenizer") + 1], str(paths[mode] / "tokenizer.json"))
                launch.return_value.terminate.assert_called_once()

    def resumed_fixture(self):
        context, previous = self.report()
        output = self.root / "resumed"
        directory = output / "reused-measurements"
        directory.mkdir(parents=True)
        context.update(collection_only=True, resumed_measurements={"run_id": 123, "artifact": {"digest": "sha256:" + "d" * 64}})
        previous.update(run_id="123", run_attempt="1")
        for runtime in q.RUNTIMES:
            candidate = q.digest(runtime)[:40]
            context["runtimes"][runtime]["candidate"] = candidate
            binary = output / "binaries" / runtime / "mistralrs"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"unit-test-only")
            build = {**context["runtimes"][runtime], "candidate": candidate,
                     "binary_sha256": q.sha(binary), "cargo_check": "PASS"}
            q.write(binary.parent / "build.json", build)
            previous["runtimes"][runtime].update(build)
            for mode in ("text", "vl"):
                for device in ("cpu", "cuda"):
                    name = f"{runtime}-{mode}-{device}"
                    q.write(directory / (name + ".json"), self.evidence(device))
                    (directory / (name + ".log")).write_text("unit-test server log")
                    for suffix in (".json", ".log"):
                        previous["files"][name + suffix] = q.sha(directory / (name + suffix))
        q.write(directory / "report.json", previous)
        report = {**previous, "run_id": "456", "runtimes": {}, "files": {}, "status": "COLLECTING"}
        return output, context, report

    def test_resumed_collection_executes_only_four_gguf_inferences(self):
        output, context, report = self.resumed_fixture()
        with mock.patch.object(q, "infer", side_effect=lambda _b, _m, d, *_: self.evidence(d)) as infer:
            with contextlib.redirect_stdout(io.StringIO()):
                complete, passed = q.collect_models(output, context, {}, report)
        self.assertTrue(complete and passed)
        self.assertEqual(infer.call_count, 4)
        self.assertTrue(all(call.args[1] == "gguf" for call in infer.call_args_list))
        for runtime in q.RUNTIMES:
            models = report["runtimes"][runtime]["models"]
            for mode in ("text", "vl"):
                self.assertEqual(models[mode]["evidence_origin"]["run_id"], "123")
                self.assertTrue(models[mode]["evidence_origin"]["reused"])
            self.assertEqual(models["gguf"]["evidence_origin"]["run_id"], "456")
        with self.assertRaises(RuntimeError): q.check_report(output / "model-proof", context)

    def test_reused_measurement_requires_identical_binary_sha256(self):
        output, context, report = self.resumed_fixture()
        proof = q.read(output / "binaries/legacy/build.json")
        proof["binary_sha256"] = "f" * 64
        (output / "model-proof").mkdir()
        with self.assertRaisesRegex(RuntimeError, "exact executable"):
            q.reuse_case(output, context, "legacy", "text", proof)

    def test_reused_measurement_rejects_changed_raw_json(self):
        output, context, report = self.resumed_fixture()
        (output / "model-proof").mkdir()
        (output / "reused-measurements/legacy-text-cpu.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "raw observation"):
            q.reuse_case(output, context, "legacy", "text", q.read(output / "binaries/legacy/build.json"))

    def test_measurement_reuse_never_enables_strict_promotion(self):
        with mock.patch.object(q, "api") as api:
            with self.assertRaisesRegex(RuntimeError, "limited to collection-only"):
                q.prepare_measurement_reuse(self.root, {"collection_only": False}, 123)
            api.assert_not_called()

    def test_preparation_validates_and_restores_original_measurements(self):
        output, context, _ = self.resumed_fixture()
        source = output / "reused-measurements"
        previous = q.read(source / "report.json")
        previous["files"] = {name: value for name, value in previous["files"].items() if (source / name).is_file()}
        for runtime in q.RUNTIMES:
            previous["runtimes"][runtime]["image"] = self.policy["image"]
        q.write(source / "report.json", previous)
        destination = self.root / "new-preparation"
        destination.mkdir()
        prior = {"path": q.WORKFLOW, "event": "workflow_dispatch", "head_branch": "mirror_orch",
                 "status": "completed", "head_sha": q.MEASUREMENT_SOURCE_SHA, "run_attempt": 1}
        jobs = [{"name": n, "conclusion": "success"} for n in ("prepare", "cpu", "cuda_compile")]
        jobs.append({"name": "SM61 Mistral Extended runtime (candidate=" + context["runtimes"]["extended"]["candidate"] + ")",
                     "id": 1234, "status": "completed"})
        def download(artifact, path):
            if artifact["id"] == 1: q.write(path / "context.json", context)
            else: q.shutil.copytree(source, path)
        def artifact(_run, prefix): return {"id": 1 if prefix.startswith("mistral-asd-prep-") else 2}
        with mock.patch.object(q, "api", side_effect=[prior, {"jobs": jobs, "total_count": len(jobs)}]):
            with mock.patch.object(q, "artifact_for", side_effect=artifact), mock.patch.object(q, "download_artifact", side_effect=download):
                with contextlib.redirect_stdout(io.StringIO()): q.prepare_measurement_reuse(destination, context, 123)
        self.assertEqual(context["resumed_measurements"]["run_id"], 123)
        self.assertEqual(context["resumed_measurements"]["gpu_job_id"], 1234)
        self.assertFalse((destination / "previous-preparation").exists())


if __name__ == "__main__":
    unittest.main()


