#!/usr/bin/env python3
"""Typed MLA cache IO, using Cudaforge output only."""
import argparse
import hashlib
import json
from pathlib import Path

def tensor(name, dtype, rank, access, representation=None):
    a = dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)
    if representation is not None:
        a["representation"] = representation
    return a

def scalar(name, dtype="u32"):
    return dict(kind="scalar", name=name, dtype=dtype)

def args(op, dt):
    access = "read_write" if op == "store" else "read"
    cc = tensor("ckv_cache", dt, 3, access, "mla.ckv.paged.v1")
    pc = tensor("kpe_cache", dt, 3, access, "mla.kpe.paged.v1")
    if op == "store":
        return [tensor("ckv_span", dt, 1, "read", "mla.row_span.v1"),
                tensor("kpe_span", dt, 1, "read", "mla.row_span.v1"),
                cc, pc, tensor("slots", "i64", 1, "read"),
                *[scalar(n) for n in ("tokens","blocks","page","rank","rope")],
                scalar("ckv_stride","u64"), scalar("kpe_stride","u64")]
    return [cc,pc,tensor("table","i32",2,"read"),tensor("offsets","i32",1,"read"),
            tensor("token_to_seq","i32",1,"read"),tensor("ckv_out",dt,2,"write"),
            tensor("kpe_out",dt,2,"write"),
            *[scalar(n) for n in ("tokens","seqs","blocks","page","stride","rank","rope")]]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--ptx-file",type=Path,required=True)
    a=p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported cc")
    data=a.ptx_file.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    for op in ("store","gather"):
        for dt in ("f16","bf16","f32"):
            folder=a.output/f"sm{a.cc}"/"mla-cache-io"/op/dt
            folder.mkdir(parents=True,exist_ok=True)
            artifact=folder/"mla_cache_io.ptx"
            manifest=folder/"manifest.json"
            m=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
                   operation=f"kv_cache.mla.{op}.b256.v1",
                   implementation_id=f"cuda.mla_cache_io.{op}.{dt}.b256.v1",
                   artifact=artifact.name,sha256=sha,entry=f"dkr_mla_{op}_{dt}",
                   target_cc=a.cc,min_cc=a.cc,max_cc=120,block=[256,1,1],
                   shared_mem_bytes=0,arguments=args(op,dt))
            if manifest.is_file() and artifact.is_file():
                old=json.loads(manifest.read_text())
                if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")} == m and artifact.read_bytes()==data:
                    continue
            artifact.write_bytes(data)
            manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_MLA_CACHE_IO_PACKAGES=STAGED profiles=6 cc={a.cc}")

if __name__=="__main__":
    main()
