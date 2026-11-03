"""Offline contracts for Extended-Release-R1, without building Rust or pulling images."""
import json
import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_r1 as r
from unittest import mock


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

    def image_fixture(self):
        manifest = self.record()
        image = self.root / "image"
        image.mkdir()
        for name in ("loader-help.txt", "loader-ldd.txt"):
            (image / name).write_text("unit-test loader evidence")
        (image / "image.tar.zst").write_bytes(b"unit-test archive")
        record = {"schema": r.IMAGE_SCHEMA, "release": manifest, "image_id": "sha256:" + "e" * 64,
                  "binary_sha256": manifest["binary_sha256"], "containerfile_sha256": r.hash_file(ROOT / "Containerfile.release-r1"),
                  "archive": "image.tar.zst", "archive_sha256": r.hash_file(image / "image.tar.zst"),
                  "loader_smoke": "PASS", "gpu_executed": False,
                  "loader_help_sha256": r.hash_file(image / "loader-help.txt"),
                  "loader_ldd_sha256": r.hash_file(image / "loader-ldd.txt")}
        (image / "oci.json").write_text(json.dumps(record))
        return image

    def model_fixture(self):
        q = r.model_module()
        policy = q.read(q.ROOT / "models.json")
        self.plan["image"] = policy["image"]
        self.plan_file.write_text(json.dumps(self.plan))
        image = self.image_fixture()
        proofs = self.root / "proofs"
        destination = proofs / "release-r1"
        destination.mkdir(parents=True)
        kernel = {"candidate": self.plan["candidate"], "run_id": "123", "status": "GPU_PASSED", "asd_required": True,
                  "gpu_executed": True, "target": "sm_61",
                  "asd_profiles": {"current.json": "d" * 64, "current.asd": "e" * 64}}
        q.write(proofs / "report.json", kernel)
        report = {"schema": r.GPU_SCHEMA, "release": r.inspect(self.plan_file, self.root), "run_id": "123",
                  "status": "PASS", "full_models_executed": True, "sm": "6.1", "gpu_uuid": "unit-test", "driver": "unit-test",
                  "kernel_report_sha256": r.hash_file(proofs / "report.json"),
                  "image_proof_sha256": r.hash_file(image / "oci.json"),
                  "harness_sha256": r.hash_file(q.ROOT / "qualification.py"),
                  "policy_sha256": r.hash_file(q.ROOT / "models.json"), "models": {}, "files": {}}
        for mode in q.MODES:
            pair = {}
            for device in ("cpu", "cuda"):
                tokens = [{"token": "a", "bytes": [97], "logprob": -1.0,
                           "top_logprobs": [{"token": str(n), "logprob": -float(n + 1)} for n in range(5)]} for _ in range(4)]
                pair[device] = {"device": device, "request": {"model": "unit-test"}, "fixture_sha256": None,
                    "response": {"usage": {"prompt_tokens": 8, "completion_tokens": 4}, "choices": [{"logprobs": {"content": tokens}}]}}
                q.write(destination / f"{mode}-{device}.json", pair[device])
            report["models"][mode] = {**q.compare(pair["cpu"], pair["cuda"], policy), "weights": policy["models"][mode], "errors": {}}
        report["files"] = {p.name: r.hash_file(p) for p in destination.iterdir()}
        q.write(destination / "report.json", report)
        return image, proofs, report

    def test_release_specific_raw_model_proofs_are_required(self):
        image, proofs, _ = self.model_fixture()
        self.assertEqual(r.verify_gpu(self.plan_file, self.root, proofs, image)["status"], "PASS")

    def test_campaign_profile_cannot_qualify_release_binary(self):
        image, proofs, report = self.model_fixture()
        report["release"]["profile"] = "campaign"
        (proofs / "release-r1/report.json").write_text(json.dumps(report))
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_release_features_must_match_the_tested_binary(self):
        image, proofs, report = self.model_fixture()
        report["release"]["features"].remove("code-execution")
        (proofs / "release-r1/report.json").write_text(json.dumps(report))
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_collection_without_numeric_acceptance_cannot_publish(self):
        image, proofs, report = self.model_fixture()
        report["status"] = "COLLECTED"
        (proofs / "release-r1/report.json").write_text(json.dumps(report))
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_image_changed_after_gpu_qualification_is_rejected(self):
        image, proofs, _ = self.model_fixture()
        record = json.loads((image / "oci.json").read_text())
        record["image_id"] = "sha256:" + "f" * 64
        (image / "oci.json").write_text(json.dumps(record))
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_changed_image_archive_is_rejected(self):
        image = self.image_fixture()
        (image / "image.tar.zst").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "archive SHA256"):
            r.image_record(self.plan_file, self.root, image, archive=True)

    def test_rehashed_numeric_drift_is_recomputed(self):
        image, proofs, report = self.model_fixture()
        path = proofs / "release-r1/gguf-cuda.json"
        data = json.loads(path.read_text())
        data["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] += 0.2
        path.write_text(json.dumps(data))
        report["files"][path.name] = r.hash_file(path)
        (proofs / "release-r1/report.json").write_text(json.dumps(report))
        with self.assertRaises(RuntimeError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def collect_fixture(self, execution_error=False, paged=False):
        image, proofs, _ = self.model_fixture()
        if paged:
            self.plan["config"]["fp8_kv_qualification"] = True
            self.plan_file.write_text(json.dumps(self.plan))
        q = r.model_module()
        reference = {device: q.read(proofs / f"release-r1/text-{device}.json") for device in ("cpu", "cuda")}
        actual_hash = r.hash_file
        def fingerprint(path):
            value = str(path)
            if value == "/run/haricot/asd/fallbacks/current.json": return "d" * 64
            if value == "/run/haricot/asd/profiles/current.asd": return "e" * 64
            return actual_hash(path)
        def infer(_binary, mode, device, *_, **options):
            if execution_error and mode == "gguf" and device == "cuda": raise OSError("unit-test model error")
            data = json.loads(json.dumps(reference[device]))
            if options.get("paged_cache_type") is not None:
                cache = options["paged_cache_type"]
                data["request"]["max_tokens"] = r.PAGED_BENCH_TOKENS
                data["paged_kv"] = {"cache_type":cache, "warmup":q.KV_BENCH_WARMUP,
                    "command":["--paged-attn","on","--pa-cache-type",cache,"--kv-cache","2048tokens",
                               "--prefix-cache-n","0","--max-num-batched-tokens","128","--max-prefill-chunk-tokens","128"],
                    "samples":[{"wall_seconds":1.0 if cache == "auto" else 2.0,"response":data["response"]}
                               for _ in range(q.KV_BENCH_REPEATS)]}
            if mode == "vl" and device == "cuda": data["response"]["choices"][0]["logprobs"]["content"][0]["logprob"] += 0.1
            return data
        env = {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1", "EXPECTED_RUN_ID": "123",
               "EXPECTED_CANDIDATE_SHA": self.plan["candidate"], "RUNNER_REPO": "haricot/mistral.rs", "RUNNER_NO_DEFAULT_LABELS": "true"}
        with mock.patch.dict(os.environ, env), mock.patch.object(r, "model_module", return_value=q), mock.patch.object(r, "hash_file", side_effect=fingerprint):
            with mock.patch.object(q, "model_paths", return_value={}), mock.patch.object(q, "run", return_value="GPU-unit-test, 6.1, unit-driver"):
                with mock.patch.object(q, "infer", side_effect=infer) as observed, contextlib.redirect_stdout(io.StringIO()):
                    if execution_error:
                        with self.assertRaisesRegex(ValueError, "all three models were attempted"):
                            r.collect_gpu(self.plan_file, self.root, proofs, image)
                        result = q.read(proofs / "release-r1/report.json")
                    else:
                        result = r.collect_gpu(self.plan_file, self.root, proofs, image)
        self.assertEqual(observed.call_count, 8 if paged else 6)
        self.assertTrue(all(call.args[0] == self.executable for call in observed.call_args_list))
        return result, image, proofs

    def test_actual_release_binary_collects_all_models_without_relaxing_policy(self):
        result, image, proofs = self.collect_fixture()
        self.assertEqual(result["release"]["profile"], "release")
        self.assertIn("code-execution", result["release"]["features"])
        self.assertTrue(result["full_models_executed"])
        self.assertEqual(result["status"], "COLLECTED")
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_release_paged_benchmark_keeps_actual_token_counts_without_relaxing_model_verdicts(self):
        result, image, proofs = self.collect_fixture(paged=True)
        self.assertTrue(result["full_models_executed"])
        self.assertEqual(result["status"], "COLLECTED")
        self.assertEqual(result["paged_fp8"]["status"], "PASS")
        self.assertEqual(result["paged_fp8"]["request_speed_ratio_fp16_over_fp8"], 0.5)
        self.assertIn("paged-kv-f8e4m3.json", result["files"])
        self.assertEqual(result["models"]["vl"]["status"], "FAIL")
        with self.assertRaises(ValueError): r.verify_gpu(self.plan_file, self.root, proofs, image)

    def test_release_collection_keeps_execution_errors_distinct_from_numeric_drift(self):
        result, _, _ = self.collect_fixture(execution_error=True)
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["full_models_executed"])
        self.assertIn("cuda", result["models"]["gguf"]["errors"])

    @unittest.skipUnless(shutil.which("cc"), "C compiler required for the real loader-stub regression")
    def test_actual_release_script_reaches_manifest_with_one_build(self):
        source = self.root / "source"
        source.mkdir()
        (source / "Cargo.lock").write_text("unit-test lock")
        self.plan["matrix"][0]["lock"] = r.hash_file(source / "Cargo.lock")
        self.plan_file.write_text(json.dumps(self.plan))
        wrappers = self.root / "bin"
        wrappers.mkdir()
        (wrappers / "git").write_text('#!/bin/sh\ncase "$*" in *"rev-parse HEAD"*) printf "%s\\n" ' + "a" * 40 + ';; *ls-tree*) echo unit-test-source;; esac\n')
        (wrappers / "cargo").write_text('''#!/usr/bin/env python3
import json,os,pathlib,shutil,sys
with open(os.environ['TEST_CARGO_LOG'],'a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='build':
    out=pathlib.Path(os.environ['CARGO_TARGET_DIR'])/'release/mistralrs'
    out.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2('/bin/true',out)
''')
        for path in wrappers.iterdir(): path.chmod(0o755)
        cuda = self.root / "cuda"
        stub = cuda / "lib64/stubs/libcuda.so"
        stub.parent.mkdir(parents=True)
        c = self.root / "stub.c"
        c.write_text("int cuInit(unsigned int ignored) { return 34; }\n")
        subprocess.run(["cc", "-shared", "-fPIC", str(c), "-o", str(stub)], check=True)
        log = self.root / "cargo.log"
        env = dict(os.environ, PATH=str(wrappers) + os.pathsep + os.environ["PATH"],
                   CAMPAIGN_CACHE=str(self.root / "cache"), CUDA_HOME=str(cuda), TEST_CARGO_LOG=str(log))
        out = self.root / "delivery"
        result = subprocess.run(["bash", str(ROOT / "release-r1.sh"), str(self.plan_file), str(out), str(source)],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(r.inspect(self.plan_file, out)["profile"], "release")
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([x[0] for x in calls], ["check", "build"])
        self.assertTrue(all("--release" in x and "mistralrs-cli" in x for x in calls))
        self.assertTrue((out / "loader-only/libcuda.so.1").is_file())


if __name__ == "__main__":
    unittest.main()

