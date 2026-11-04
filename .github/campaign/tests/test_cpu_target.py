import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cpu_target as cpu


class CpuTargetTests(unittest.TestCase):
    def test_only_explicit_profiles_are_accepted(self):
        self.assertEqual(cpu.rustflags({}), "-C target-cpu=x86-64")
        self.assertEqual(cpu.rustflags({"cpu_profile": "avx2"}), "-C target-cpu=x86-64-v3")
        for value in ("native", "x86-64-v4", "avx2 -C target-feature=+avx512f"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cpu.rustflags({"cpu_profile": value})

    def test_avx2_cpu_accepts_linux_aliases_without_requiring_avx512(self):
        flags = cpu.AVX2_FEATURES - {"sse3", "lzcnt"}
        info = "flags : " + " ".join(sorted(flags | {"pni", "abm"}))
        with patch.object(cpu.platform, "machine", return_value="x86_64"):
            cpu.check_host({"cpu_profile": "avx2"}, cpuinfo=info)

    def test_every_visible_processor_must_support_the_target(self):
        info = "flags : " + " ".join(sorted(cpu.AVX2_FEATURES))
        info += "\nflags : " + " ".join(sorted(cpu.AVX2_FEATURES - {"avx2"}))
        with patch.object(cpu.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(ValueError, "missing avx2"):
                cpu.check_host({"cpu_profile": "avx2"}, cpuinfo=info)

    def test_unobservable_or_wrong_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            cpu.check_host({"cpu_profile": "avx2"}, cpuinfo="")
        with patch.object(cpu.platform, "machine", return_value="aarch64"):
            with self.assertRaisesRegex(ValueError, "x86_64"):
                cpu.check_host({"cpu_profile": "avx2"}, cpuinfo="")

    def test_invalid_plan_stops_shell_before_executing_a_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"config": {"cpu_profile": "native"}}))
            result = subprocess.run(["bash", "-euc", 'source "$1" --plan "$2"; echo UNREACHABLE',
                                     "cpu-profile-test", ROOT / "compiler-env.sh", plan],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("UNREACHABLE", result.stdout)
