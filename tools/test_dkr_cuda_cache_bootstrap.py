#!/usr/bin/env python3
"""CPU-only tests for the DKR CUDA cache bootstrapper (no nvcc/Cargo)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import dkr_cuda_cache_bootstrap as bootstrap


class CacheBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name) / "mistral.rs"
        self.repo.mkdir()
        self.target = self.repo / "target"
        self.root = self.target / "dkr-cuda-cache" / "sm61"
        for package, _, _, _ in bootstrap.PACKAGES:
            directory = self.repo / package
            directory.mkdir()
            (directory / "build.rs").write_text("// fixture\n")

    def add_cache(self, package, part, arch="sm_61", corrupt=False):
        crate = self.repo / package
        source = crate / "kernels" / "fixture.cu"
        source.parent.mkdir(exist_ok=True)
        source.write_text('extern "C" __global__ void fixture() {}\n')
        old_dir = self.target / "debug" / "build" / f"{package}-abc" / "out"
        if package == "mistralrs-quant":
            old_dir = old_dir / "kernels"
        old_dir.mkdir(parents=True)
        obj = old_dir / "fixture-abcd.o"
        obj.write_bytes(b"object-not-a-real-ELF")
        source_name = str(source.relative_to(crate))
        key = f"{source_name}:{obj}"
        record = {
            "content_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
            "watch_hash": "fixture-watch-hash",
            "modified_time": 1234,
            "object_path": str(obj),
            "gpu_arch": arch,
            "args_hash": "fixture-args-hash",
        }
        if corrupt:
            record["content_hash"] = "0" * 64
        (old_dir / bootstrap.CACHE_FILE).write_text(
            json.dumps({"entries": {key: record}, "version": 1})
        )
        return obj, record

    def test_dry_run_does_not_modify_files(self):
        self.add_cache("mistralrs-quant", "kernels")
        self.assertEqual(bootstrap.run(self.repo, self.target, self.root, "sm_61", False), 0)
        self.assertFalse(self.root.exists())

    def test_adopts_objects_with_new_cache_keys(self):
        original, metadata = self.add_cache("mistralrs-quant", "kernels")
        self.assertEqual(bootstrap.run(self.repo, self.target, self.root, "sm_61", True), 0)
        new_root = self.root / "quant" / "kernels"
        copied = new_root / original.name
        self.assertEqual(copied.read_bytes(), original.read_bytes())
        copied_meta = json.loads((new_root / bootstrap.CACHE_FILE).read_text())
        target_key = f"kernels/fixture.cu:{copied}"
        self.assertIn(target_key, copied_meta["entries"])
        self.assertEqual(copied_meta["entries"][target_key]["object_path"], str(copied))
        self.assertEqual(copied_meta["entries"][target_key]["args_hash"], metadata["args_hash"])
        self.assertTrue(original.is_file())  # Keep Cargo's original artifacts.

    def test_filters_mismatching_arch_and_corrupt_sources(self):
        self.add_cache("mistralrs-core", "", arch="sm_90")
        self.add_cache("mistralrs-paged-attn", "", corrupt=True)
        self.assertEqual(bootstrap.run(self.repo, self.target, self.root, "sm_61", True), 0)
        self.assertFalse(self.root.exists())

    def test_never_overwrites_existing_shared_cache(self):
        self.add_cache("mistralrs-quant", "kernels")
        cache = self.root / "quant" / "kernels" / bootstrap.CACHE_FILE
        cache.parent.mkdir(parents=True)
        cache.write_text('{"entries":{}, "version":1}\n')
        self.assertEqual(bootstrap.run(self.repo, self.target, self.root, "sm_61", True), 0)
        self.assertEqual(json.loads(cache.read_text())["entries"], {})


if __name__ == "__main__":
    unittest.main()
