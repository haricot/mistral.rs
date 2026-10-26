import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("cc"), "a C compiler is required for the ELF loader regression")
class CpuRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stub = self.library("stub", 34)

    def library(self, name, result):
        path = self.root / f"{name}.so"
        subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-",
                        "-Wl,-soname,libcuda.so.1", "-o", path],
                       input=f"int cuInit(unsigned int flags) {{ return {result}; }}\n",
                       text=True, check=True, capture_output=True)
        return path

    def launch(self, command, stub=None):
        return subprocess.run([sys.executable, ROOT / "cpu-runtime.py", "--stub", stub or self.stub,
                               "--", *command], text=True, capture_output=True)

    def test_cuda_linked_cpu_executable_loads_without_a_driver_installation(self):
        binary = self.root / "cpu-test"
        subprocess.run(["cc", "-x", "c", "-", "-x", "none", str(self.stub), "-o", binary],
                       input="int cuInit(unsigned int); int main(void) { return cuInit(0) != 34; }\n",
                       text=True, check=True, capture_output=True)
        env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
        missing = subprocess.run([binary], env=env, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 127)
        self.assertIn("libcuda.so.1", missing.stderr)
        result = self.launch([binary])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPU_CUDA_LINK_STUB=ENABLED", result.stdout)

    def test_loader_path_is_temporary_and_child_failure_is_preserved(self):
        before = os.environ.get("LD_LIBRARY_PATH")
        result = self.launch([sys.executable, "-c",
                              "import os,sys;print(os.environ['LD_LIBRARY_PATH'].split(':')[0]);sys.exit(7)"])
        self.assertEqual(result.returncode, 7)
        directory = result.stdout.splitlines()[-1]
        self.assertFalse(Path(directory).exists())
        self.assertEqual(os.environ.get("LD_LIBRARY_PATH"), before)

    def test_successful_driver_cannot_be_used_as_a_cpu_link_stub(self):
        real = self.library("successful-driver", 0)
        marker = self.root / "executed"
        result = self.launch([sys.executable, "-c", f"open({str(marker)!r},'w').close()"], stub=real)
        self.assertEqual(result.returncode, 1)
        self.assertIn("requires the CUDA link stub", result.stderr)
        self.assertFalse(marker.exists())
