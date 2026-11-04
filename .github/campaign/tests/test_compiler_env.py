import os
import json
import shutil
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QEMU = os.environ.get("CAMPAIGN_TEST_QEMU") or shutil.which("qemu-x86_64")
PROBE = '''
#[inline(never)]
fn magnitude(value: f32) -> f32 { value.abs() }

fn main() {
    let value: f32 = std::env::args().nth(1).unwrap().parse().unwrap();
    assert_eq!(magnitude(std::hint::black_box(value)), 3.0);
    println!("avx={} avx2={} avx512f={}",
        cfg!(target_feature = "avx"), cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f"));
}
'''


class CompilerEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "src").mkdir()
        (self.root / ".cargo").mkdir()
        (self.root / "Cargo.toml").write_text(
            '[package]\nname = "portable-probe"\nversion = "0.1.0"\nedition = "2021"\n'
            '[profile.dev]\nopt-level = 1\ndebug = 0\n')
        (self.root / ".cargo/config.toml").write_text(
            '[build]\nrustflags = ["-C", "target-cpu=native"]\n')
        (self.root / "src/main.rs").write_text(PROBE)

    def compile(self, portable, profile=None):
        target = self.root / (profile or ("portable" if portable else "avx512"))
        env = {**os.environ, "CARGO_TARGET_DIR": str(target),
               "RUSTFLAGS": "-C target-cpu=native",
               "CARGO_ENCODED_RUSTFLAGS": "-C\x1ftarget-cpu=x86-64-v4"}
        env.pop("RUSTC_WRAPPER", None)
        script = 'source "$1"; cargo build --offline -vv' if portable else 'cargo build --offline -vv'
        if profile:
            plan = self.root / "plan.json"
            plan.write_text(json.dumps({"config": {"cpu_profile": profile}}))
            script = 'source "$1" --plan "$2"; cargo check --offline; cargo build --offline -vv'
        result = subprocess.run(["bash", "-euo", "pipefail", "-c", script,
                                 "compiler-env-test", ROOT / "compiler-env.sh", self.root / "plan.json"],
                                cwd=self.root, env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return target / "debug/portable-probe", result.stderr

    def test_cargo_overrides_native_config_and_inherited_encoded_flags(self):
        binary, _ = self.compile(portable=True)
        result = subprocess.run([binary, "-3"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "avx=false avx2=false avx512f=false")

    def test_avx2_cargo_check_and_binary_override_native_and_encoded_flags(self):
        binary, log = self.compile(portable=True, profile="avx2")
        result = subprocess.run([binary, "-3"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "avx=true avx2=true avx512f=false")
        self.assertIn("target-cpu=x86-64-v3", log)

    @unittest.skipUnless(QEMU, "QEMU is required for the CPU instruction regression")
    def test_portable_archive_code_runs_on_cpus_without_avx512(self):
        avx512, _ = self.compile(portable=False)
        failed = subprocess.run([QEMU, "-cpu", "Skylake-Client", avx512, "-3"],
                                text=True, capture_output=True)
        self.assertIn(failed.returncode, (-signal.SIGILL, 128 + signal.SIGILL), failed.stderr)
        portable, _ = self.compile(portable=True)
        for cpu in ("Skylake-Client", "Nehalem"):
            with self.subTest(cpu=cpu):
                result = subprocess.run([QEMU, "-cpu", cpu, portable, "-3"],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "avx=false avx2=false avx512f=false")
