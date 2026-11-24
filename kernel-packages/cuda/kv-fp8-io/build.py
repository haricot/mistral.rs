#!/usr/bin/env python3
"""Stamp the Cudaforge-built FP8 KV IO image; no independent compiler."""
import argparse
import hashlib
import json
from pathlib import Path

def tensor(name, dtype, rank, access, representation=None):
    a = dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)
    if representation:
        a["representation"] = representation
    return a

def parameters(op, dt):
    ks = tensor("k_scale", "f32", 1, "read")
    vs = tensor("v_scale", "f32", 1, "read")
    access = "read_write" if op == "store" else "read"
    kc = tensor("key_cache", "fp8_e4m3", 5, access, "vllm_k_blocked_x.v1")
    vc = tensor("value_cache", "fp8_e4m3", 4, access, "vllm_v_token_strided.v1")
    if op == "store":
        args = [tensor("key", dt, 3, "read"), tensor("value", dt, 3, "read"),
                kc, vc, tensor("slots", "i64", 1, "read"), ks, vs]
        names = ("tokens", "heads", "head_size", "blocks", "block_size", "x")
    else:
        args = [kc, vc, tensor("table", "i32", 2, "read"),
                tensor("offsets", "i32", 1, "read"),
                tensor("key_output", dt, 3, "write"), tensor("value_output", dt, 3, "write"),
                ks, vs]
        names = ("tokens", "seqs", "block_size", "table_stride", "heads", "head_size", "x", "blocks")
    return args + [dict(kind="scalar", name=n, dtype="u32") for n in names]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ptx-file", required=True, type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported cc")
    data = a.ptx_file.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    root = a.output / f"sm{a.cc}" / "kv-fp8-io"
    for op in ("store", "gather"):
        for dt in ("f16", "bf16", "f32"):
            folder = root / op / dt
            folder.mkdir(parents=True, exist_ok=True)
            artifact = folder / "kv_fp8_io.ptx"
            manifest = folder / "manifest.json"
            # Preserve an already paired, identical profile on an idempotent build.
            m = dict(schema="DKR-KERNEL-MODULE-V1", abi="dkr.typed.v1",
                     operation=f"kv_cache.fp8_{op}.vllm.b256.v1",
                     implementation_id=f"cuda.kv_fp8_io.{op}.{dt}.b256.v1",
                     artifact=artifact.name, sha256=sha,
                     entry=f"dkr_kv_fp8_{op}_{dt}", target_cc=a.cc,
                     min_cc=a.cc, max_cc=120, block=[256,1,1], shared_mem_bytes=0,
                     arguments=parameters(op, dt))
            if manifest.is_file() and artifact.is_file():
                old = json.loads(manifest.read_text())
                base = {k:v for k,v in old.items() if k not in ("cubin_artifact", "cubin_sha256")}
                if base == m and artifact.read_bytes() == data:
                    continue
            artifact.write_bytes(data)
            manifest.write_text(json.dumps(m, indent=2) + "\n")
    print(f"DKR_KV_FP8_IO_PACKAGES=STAGED profiles=6 cc={a.cc}")

if __name__ == "__main__":
    main()
