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
