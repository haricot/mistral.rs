#!/usr/bin/env python3
"""CPU-only unit fixtures for the R1.26 NVFP4 offline inventory preflight."""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from dkr_r126_nvfp4_preflight import (
    PreflightError, inspect_checkpoint, inspect_packages, PROFILES
)


class Nvfp4PreflightTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.packages = self.root / "sm61"
        self.packages.mkdir()
        (self.snapshot / "config.json").write_text(json.dumps({
            "model_type": "qwen3",
            "architectures": ["DFlashDraftModel"],
            "quantization_config": {
                "quant_algo": "W4A16_NVFP4", "group_size": 16
            }
        }))
        # Zero-filled weight payload, only the Safetensors header is read.
        header = json.dumps({
            "layers.0.self_attn.q_proj.weight": {
                "dtype": "U8", "shape": [4], "data_offsets": [0, 4]
            }
        }).encode()
        (self.snapshot / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + bytes(4))
        entries = [f"dkr_nvfp4_gemv_{dtype}_{mode}" for dtype, mode in PROFILES]
        ptx = (
            ".version 7.0\n.target sm_61\n.address_size 64\n"
            + "".join(f".visible .entry {entry} () {{ ret; }}\n" for entry in entries)
        ).encode()
        sha = hashlib.sha256(ptx).hexdigest()
        for dtype, mode in PROFILES:
            folder = self.packages / "nvfp4-gemv" / dtype / mode
            folder.mkdir(parents=True)
            (folder / "nvfp4_gemv.ptx").write_bytes(ptx)
            manifest = {
                "schema": "DKR-KERNEL-MODULE-V1",
                "abi": "dkr.typed.v1",
                "target_cc": 61,
                "operation": f"quant.nvfp4.gemv.{mode}.b128.v1",
                "entry": f"dkr_nvfp4_gemv_{dtype}_{mode}",
                "block": [128, 1, 1],
                "artifact": "nvfp4_gemv.ptx",
                "sha256": sha,
            }
            (folder / "manifest.json").write_text(json.dumps(manifest))

    def test_valid_header_and_shared_four_ptx_entries(self):
        ckpt = inspect_checkpoint(self.snapshot)
        self.assertEqual(ckpt["safetensors"]["tensor_count"], 1)
        self.assertTrue(ckpt["is_dflash_draft"])
        pkg = inspect_packages(self.packages)
        self.assertEqual(pkg["profiles"], 4)
        self.assertEqual(pkg["ptx_images"], 1)
        self.assertEqual(pkg["cubin_status"], "PTX_ONLY")

    def test_wrong_quantization_refused(self):
        path = self.snapshot / "config.json"
        c = json.loads(path.read_text())
        c["quantization_config"]["group_size"] = 32
        path.write_text(json.dumps(c))
        with self.assertRaisesRegex(PreflightError, "group_size=16"):
            inspect_checkpoint(self.snapshot)

    def test_safetensors_offset_bounds_refused(self):
        path = self.snapshot / "model.safetensors"
        header = json.dumps({
            "tensor": {"dtype": "U8", "shape": [4],
                       "data_offsets": [0, 99999]}
        }).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
        with self.assertRaisesRegex(PreflightError, "data offsets"):
            inspect_checkpoint(self.snapshot)

    def test_wrong_ptx_entry_refused(self):
        path = self.packages / "nvfp4-gemv" / "f16" / "plain" / "manifest.json"
        c = json.loads(path.read_text())
        c["entry"] = "dkr_nvfp4_gemv_f16_WRONG"
        path.write_text(json.dumps(c))
        with self.assertRaisesRegex(PreflightError, "invalid entry"):
            inspect_packages(self.packages)

    def test_invalid_partial_cubin_refused(self):
        path = self.packages / "nvfp4-gemv" / "f16" / "plain" / "manifest.json"
        c = json.loads(path.read_text())
        c["cubin_artifact"] = "image.cubin"
        path.write_text(json.dumps(c))
        with self.assertRaisesRegex(PreflightError, "partial CUBIN"):
            inspect_packages(self.packages)

    def test_wrong_sm_ptx_refused(self):
        path = self.packages / "nvfp4-gemv" / "f16" / "plain" / "nvfp4_gemv.ptx"
        path.write_text(path.read_text().replace("sm_61", "sm_80"))
        with self.assertRaisesRegex(PreflightError, "PTX digest mismatch"):
            inspect_packages(self.packages)


if __name__ == "__main__":
    unittest.main()
