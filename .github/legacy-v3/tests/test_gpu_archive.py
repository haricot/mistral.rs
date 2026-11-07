import contextlib
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        (cls.source / "src/lib.rs").write_text('#[test]\nfn archive_runs_without_gpu() { assert_eq!(2 + 2, 4); }\n#[test]\nfn archive_failure() { panic!("synthetic archived test failure"); }\n')
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

    def run_archive(self, name):
        report = self.root / name / "evidence"
        self.assertFalse(report.exists())
        config_root = self.root / name / "recipe"
        config_root.mkdir(parents=True)
        (config_root / "nextest.toml").write_text('[profile.gpu]\ndefault-filter = "test(' + name + ')"\njunit.path = "gpu-junit.xml"\n')
        output = io.StringIO()
        with patch.object(gpu.c, "ROOT", config_root), patch.object(gpu.c, "GPU_REQUIRED", [name]), contextlib.redirect_stdout(output):
            try:
                return gpu.run_archived_tests(self.builds, report, self.workspace), report, output.getvalue()
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
