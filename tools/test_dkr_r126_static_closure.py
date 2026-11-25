#!/usr/bin/env python3
"""CPU-only synthetic regression of the S9C catalogue and evidence gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dkr_r126_full_catalog_inventory import inspect, manifest_profiles
from dkr_r126_full_catalog_stage import snapshot, copy_profiles, validate_copy
from dkr_r126_static_closure_probe import component_id, inspect_build

def sha(data):
    return hashlib.sha256(data).hexdigest()

class ClosureTests(unittest.TestCase):
    def fixture(self, root):
        folder=root/"flashinfer-decode"/"safe2-image-test"
        folder.mkdir(parents=True)
        ptx=b".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry a() {ret;}\n.visible .entry b() {ret;}\n"
        (folder/"test.ptx").write_bytes(ptx)
        profiles=[dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
                       target_cc=61,min_cc=61,max_cc=61,entry=e,
                       operation=f"op.{e}",implementation_id=f"cuda.safe2.{e}",
                       artifact="test.ptx",sha256=sha(ptx)) for e in ("a","b")]
        m=dict(schema="DKR-KERNEL-SHARD-V1",profiles=profiles)
        (folder/"manifest.json").write_text(json.dumps(m))
        return folder,m

    def test_shard_copy_pair_and_metadata_rejection(self):
        family="flashinfer-decode"
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"original";dest=Path(tmp)/"assembly"
            folder,m=self.fixture(root)
            info=inspect(root,family)
            self.assertEqual((info["profiles"],info["physical_manifests"],info["paired_profiles"]),(2,1,0))
            prior=snapshot(root,family);plan={family:(root,info,prior)}
            copy_profiles(root,dest,prior)
            validate_copy(dest,plan,False)
            pair_folder=dest/folder.relative_to(root)
            # Synthetic ELF header: tests the inventory, NOT a real CUBIN.
            cubin=bytearray(64);cubin[:6]=b"\x7fELF\x02\x01"
            cubin[18:20]=(190).to_bytes(2,"little")
            cubin[48:52]=(61).to_bytes(4,"little")
            (pair_folder/"test.cubin").write_bytes(cubin)
            for row in m["profiles"]:
                row.update(cubin_artifact="test.cubin",cubin_sha256=sha(cubin))
            mf=pair_folder/"manifest.json"
            mf.write_text(json.dumps(m))
            validate_copy(dest,plan,True)
            self.assertEqual(snapshot(root,family),prior)
            m["profiles"][0]["operation"]="changed"
            mf.write_text(json.dumps(m))
            with self.assertRaises(ValueError):validate_copy(dest,plan,True)

    def test_invalid_shard_and_old_generation(self):
        with self.assertRaises(ValueError):
            manifest_profiles(dict(schema="DKR-KERNEL-SHARD-V1",profiles=[]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder,m=self.fixture(root)
            m["profiles"][0]["implementation_id"]="cuda.old.a"
            (folder/"manifest.json").write_text(json.dumps(m))
            with self.assertRaises(ValueError):inspect(root,"flashinfer-decode")

    def test_missing_producer_evidence_is_not_success(self):
        self.assertEqual(component_id("path+file:///home/np/mistralrs-core#0.10.0"),"mistralrs-core")
        self.assertEqual(component_id("mistralrs-quant 0.10.0 (path+file:///tmp)"),"mistralrs-quant")
        self.assertEqual(component_id("path+file:///tmp#other@0.10.0"),None)
        with self.assertRaises(ValueError):inspect_build([],{})

if __name__=="__main__":
    unittest.main()
