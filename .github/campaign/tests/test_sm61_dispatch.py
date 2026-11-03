import fnmatch
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compat_sm61 as patch

OLD = '#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]'
NEW = '#[cfg(all(feature = "cuda", has_fp8_mma_kernels))]'


class Sm61DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("build.rs", '''let mut excluded_files = if cc_over_80 {
            vec!["dummy_*.cu", "*_dummy.cu"]
        } else {
            vec![
                "marlin_*.cu",
                "*_fp8_mma.cu",
                "*_wmma.cu",
                "moe_data.cu",
                "grouped_mm_*.cu",
            ]
        };
        if !cc_over_80 && !legacy_fp8 {
            excluded_files.extend(["*_fp8.cu", "*_fp8_gemm.cu"]);
        }
        builder = builder.exclude(&excluded_files);
''')
        self.write("src/moe/mod.rs", '#[cfg(has_blockwise_fp8_kernels)]\npub(crate) mod grouped_fp8;\npub mod cuda;\n')
        self.write("src/mxfp8/mod.rs", OLD+'\nfn grouped_gather() {}\nfn scalar_gather() {}\n')
        self.write("src/pertensor_fp8/mod.rs", '\n'.join(OLD+f'\nfn grouped_{i}() {{}}' for i in range(4))+
                   '\n#[cfg(all(feature = "cuda", not(has_blockwise_fp8_kernels)))]\nfn cpu_fallback() {}\n')

        self.write("src/fp8/quantize.rs", '    fn test_roundtrip_f8e4m3() -> Result<()> {\n        let dev = Device::cuda_if_available(0)?;\n        exercise(&dev)\n    }\n')

    def write(self, name, text):
        path = self.root / "mistralrs-quant" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def read(self, name):
        return (self.root / "mistralrs-quant" / name).read_text()

    def test_legacy_fp8_excludes_tensor_core_group_without_excluding_scalar_kernels(self):
        self.assertTrue(patch.install(self.root))
        import re
        text = self.read("build.rs")
        legacy = re.search(r'vec!\[(.*?)\]', text.split('} else {', 1)[1], re.S).group(1)
        excluded = re.findall(r'"([^"\n]+\.cu)"', legacy)
        for source in ("grouped_fp8.cu", "grouped_mm_2x.cu", "blockwise_fp8_mma.cu"):
            self.assertTrue(any(fnmatch.fnmatch(source, p) for p in excluded), source)
        for source in ("blockwise_fp8.cu", "blockwise_fp8_gemm.cu", "scalar_fp8.cu", "routed_gemv_fp8.cu"):
            self.assertFalse(any(fnmatch.fnmatch(source, p) for p in excluded), source)
        self.assertIn('if !cc_over_80 && !legacy_fp8', text)
        self.assertNotIn('grouped_fp8.cu', text.split('} else {', 1)[0])

    def test_mma_dispatch_is_not_enabled_by_scalar_fp8_and_cpu_fallback_is_preserved(self):
        patch.install(self.root)
        self.assertEqual(self.read("src/pertensor_fp8/mod.rs").count(NEW), 4)
        self.assertNotIn(OLD, self.read("src/mxfp8/mod.rs"))
        self.assertIn('not(has_blockwise_fp8_kernels)', self.read("src/pertensor_fp8/mod.rs"))
        self.assertIn('fn scalar_gather()', self.read("src/mxfp8/mod.rs"))
        self.assertIn('#[cfg(has_fp8_mma_kernels)]\npub(crate) mod grouped_fp8;', self.read("src/moe/mod.rs"))

    def test_repeated_install_does_not_change_or_rehash_sources(self):
        patch.install(self.root)
        before = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertFalse(patch.install(self.root))
        self.assertEqual(before, {p:p.read_bytes() for p in before})

    def test_unknown_upstream_dispatch_requires_an_audit(self):
        self.write("src/pertensor_fp8/mod.rs", OLD+'\nfn new_dispatch() {}\n')
        with self.assertRaisesRegex(RuntimeError, 'dispatch contract'):
            patch.install(self.root)

    def test_changed_cuda_exclusion_contract_fails_closed(self):
        self.write("build.rs", 'let excluded_files = vec![];\n')
        with self.assertRaisesRegex(RuntimeError, 'exclusion contract'):
            patch.install(self.root)


class Fp8TestRoutingTests(unittest.TestCase):
    SOURCE = '''#[derive(Debug)]
enum Device { Cpu, Cuda }
type Result<T> = std::result::Result<T, &'static str>;
impl Device {
    fn cuda_if_available(_: usize) -> Result<Self> { panic!("CUDA runtime called from CPU test") }
    fn new_cuda(_: usize) -> Result<Self> { Err("physical CUDA test must not run here") }
}
mod tests {
    use super::*;
    #[test]
    #[cfg(not(feature = "metal"))]
    fn test_roundtrip_f8e4m3() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        assert!(matches!(dev, Device::Cpu));
        Ok(())
    }
}
'''

    def test_common_roundtrip_body_is_preserved_and_split_is_idempotent(self):
        fixed, changed = patch.split_fp8_roundtrip(self.SOURCE)
        self.assertTrue(changed)
        self.assertIn('assert!(matches!(dev, Device::Cpu));', fixed)
        self.assertIn('roundtrip_f8e4m3_on_device(Device::Cpu)', fixed)
        self.assertIn('roundtrip_f8e4m3_on_device(Device::new_cuda(0)?)', fixed)
        self.assertEqual(patch.split_fp8_roundtrip(fixed), (fixed, False))
        filters = (ROOT/'nextest.toml').read_text()
        self.assertIn('!test(cuda)', filters.split('[profile.gpu]')[0])
        self.assertIn('test(sm61)', filters.split('[profile.gpu]')[1])

    def test_unknown_upstream_roundtrip_requires_an_audit(self):
        with self.assertRaisesRegex(RuntimeError, 'roundtrip test'):
            patch.split_fp8_roundtrip(self.SOURCE.replace('cuda_if_available(0)', 'cuda_if_available(1)'))

    def test_cpu_execution_does_not_initialize_cuda_with_cuda_feature_enabled(self):
        import shutil, subprocess
        if not shutil.which('rustc'):
            self.skipTest('rustc is unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'routing.rs'
            binary = root/'routing'
            for expected_success, text in ((False, self.SOURCE), (True, patch.split_fp8_roundtrip(self.SOURCE)[0])):
                source.write_text(text)
                subprocess.run(['rustc', '--edition=2021', '--test', '--cfg', 'feature="cuda"', str(source), '-o', str(binary)], check=True, capture_output=True, text=True)
                result = subprocess.run([binary, '--exact', 'tests::test_roundtrip_f8e4m3'], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, expected_success, result.stdout + result.stderr)
                if expected_success:
                    self.assertIn('1 passed', result.stdout)
                else:
                    self.assertIn('CUDA runtime called from CPU test', result.stdout)
