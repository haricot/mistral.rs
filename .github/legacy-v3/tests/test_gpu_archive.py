import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu


@unittest.skipUnless(os.environ.get("LEGACY_V3_NEXTEST"), "set LEGACY_V3_NEXTEST to the pinned cargo-nextest executable")
class RealArchiveExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source"
        (cls.source / "src").mkdir(parents=True)
        (cls.source / "Cargo.toml").write_text('[package]\nname="archive-regression"\nversion="0.1.0"\nedition="2021"\n[workspace]\n')
        (cls.source / "src/lib.rs").write_text(r'''
#[test]
fn archive_runs_without_gpu() { assert_eq!(2 + 2, 4); }
#[test]
fn archive_failure() { panic!("synthetic archived test failure"); }
#[test]
fn archive_writes_bound_report() {
    let candidate = std::env::var("HARICOT_FP8_KV_CANDIDATE").unwrap();
    let component = std::env::var("HARICOT_FP8_KV_COMPONENT").unwrap();
    let run_id = std::env::var("GITHUB_RUN_ID").unwrap();
    let attempt = std::env::var("GITHUB_RUN_ATTEMPT").unwrap();
    assert_eq!(candidate, "a".repeat(40));
    assert_eq!(component, "legacy_v3");
    assert_eq!(run_id, "12345678");
    assert_eq!(attempt, "2");
    let report = format!(
        "{{\"candidate\":\"{candidate}\",\"component\":\"{component}\",\"run_id\":\"{run_id}\",\"run_attempt\":\"{attempt}\"}}"
    );
    std::fs::write(std::env::var("HARICOT_FP8_KV_REPORT").unwrap(), report).unwrap();
}
''')
        cls.builds = cls.root / "builds"
        cls.builds.mkdir()
        shutil.copy2(os.environ["LEGACY_V3_NEXTEST"], cls.builds / "cargo-nextest")
        result = subprocess.run([str(cls.builds / "cargo-nextest"), "nextest", "archive", "--manifest-path", str(cls.source / "Cargo.toml"),
                                 "--archive-file", str(cls.builds / "tests.tar.zst")], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env={**os.environ, "RUSTFLAGS": "-C target-cpu=x86-64"})
        if result.returncode:
            raise RuntimeError(result.stdout)
        cls.workspace = cls.root / "relocated-source"
        cls.source.rename(cls.workspace)

    def run_archive(self, name, test_filter=None):
        report = self.root / name / "evidence"
        self.assertFalse(report.exists())
        config_root = self.root / name / "recipe"
        config_root.mkdir(parents=True)
        expression = test_filter or "test(" + name + ")"
        (config_root / "nextest.toml").write_text('[profile.default]\nfail-fast = true\n[profile.gpu]\ndefault-filter = "' + expression + '"\njunit.path = "gpu-junit.xml"\n')
        output = io.StringIO()
        environment = {"GITHUB_RUN_ID": "12345678", "GITHUB_RUN_ATTEMPT": "2",
                       "HARICOT_FP8_KV_CANDIDATE": "untrusted inherited candidate",
                       "HARICOT_FP8_KV_COMPONENT": "unrelated", "HARICOT_FP8_KV_REPORT": "/does-not-exist/report"}
        with patch.object(gpu.c, "ROOT", config_root), patch.object(gpu.c, "GPU_REQUIRED", [name]), patch.dict(os.environ, environment), contextlib.redirect_stdout(output):
            try:
                return gpu.run_archived_tests(self.builds, report, self.workspace, {"candidate": "a" * 40}), report, output.getvalue()
            except subprocess.CalledProcessError as error:
                return error, report, output.getvalue()

    def test_archive_extracts_and_executes_from_a_fresh_relocated_workspace(self):
        archive = self.builds / "tests.tar.zst"
        before = hashlib.sha256(archive.read_bytes()).hexdigest()
        result, report, output = self.run_archive("archive_runs_without_gpu")
        self.assertIsInstance(result, dict, output)
        self.assertEqual(result["passed"], 1)
        self.assertIn("archive_runs_without_gpu", output)
        self.assertEqual((report / "gpu-tests.log").read_text(), output)
        self.assertTrue((report / "target").is_dir())
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), before)

    def test_archived_failure_keeps_the_exit_code_and_prints_the_real_log(self):
        error, report, output = self.run_archive("archive_failure")
        self.assertIsInstance(error, subprocess.CalledProcessError)
        self.assertNotEqual(error.returncode, 0)
        self.assertIn("synthetic archived test failure", output)
        self.assertEqual((report / "gpu-tests.log").read_text(), output)

    def test_archived_report_receives_the_exact_candidate_and_producer_context(self):
        result, report, output = self.run_archive("archive_writes_bound_report")
        self.assertIsInstance(result, dict, output)
        self.assertEqual(result["passed"], 1)
        self.assertEqual(json.loads((report / "fp8-kv.json").read_text()),
                         {"candidate": "a" * 40, "component": "legacy_v3", "run_id": "12345678", "run_attempt": "2"})

    def test_all_archived_tests_finish_even_when_one_fails(self):
        error, report, output = self.run_archive("complete_failed_suite", "all()")
        self.assertIsInstance(error, subprocess.CalledProcessError, output)
        cases = list(ET.parse(report / "gpu-junit.xml").getroot().iter("testcase"))
        self.assertEqual(len(cases), 3)
        self.assertEqual(sum(case.find("failure") is not None for case in cases), 1)
        self.assertTrue((report / "fp8-kv.json").is_file())
