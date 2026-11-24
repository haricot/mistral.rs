#!/usr/bin/env python3
"""Stamp Cudaforge prefill sinks profiles, preserving head-specific tiling."""
import argparse
import hashlib
import json
from pathlib import Path

DIMS = (64,80,96,112,128,192,256)

def tensor(name, dtype, rank, access):
    return dict(kind="tensor",name=name,dtype=dtype,rank=rank,access=access)

def scalar(name, dtype="u32"):
    return dict(kind="scalar",name=name,dtype=dtype)

def profiles():
    for dt in ("f16","bf16","f32"):
        for d in DIMS:
            for mode in ("dense","varlen","packed"):
                for sink in ("plain","sink"):
                    qr = 3 if mode == "packed" else 4
                    kr = 4 if mode == "dense" else 3
                    args = [tensor("query",dt,qr,"read"),tensor("key",dt,kr,"read"),
                            tensor("value",dt,kr,"read"),tensor("output",dt,qr,"write")]
                    if sink == "sink":
                        args.append(tensor("sinks","f32",1,"read"))
                    if mode != "dense":
                        args.extend([tensor("cu_q","u32",1,"read"),tensor("cu_k","u32",1,"read")])
                    args.extend([scalar("scale","f32"),scalar("q_len")])
                    args.extend([scalar("kv_len")] if mode == "dense" else [scalar("total_q"),scalar("total_kv")])
                    args.extend([scalar("heads"),scalar("kv_heads"),scalar("window")])
                    yield dt,d,mode,sink,args

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
    for dt,d,mode,sink,args in profiles():
        folder=a.output/f"sm{a.cc}"/"prefill-sinks"/mode/sink/dt/f"d{d}"
        folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/"prefill_sinks.ptx"
        manifest=folder/"manifest.json"
        m=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
               operation=f"attention.prefill.sinks.{mode}.{sink}.d{d}.v1",
               implementation_id=f"cuda.prefill_sinks.{mode}.{sink}.{dt}.d{d}.b256.v1",
               artifact=artifact.name,sha256=sha,entry=f"dkr_prefill_sinks_{mode}_{sink}_{dt}_d{d}",
               target_cc=a.cc,min_cc=a.cc,max_cc=120,block=[256,1,1],shared_mem_bytes=0,arguments=args)
        if manifest.is_file() and artifact.is_file():
            old=json.loads(manifest.read_text())
            if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")} == m and artifact.read_bytes()==data:
                continue
        artifact.write_bytes(data)
        manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_PREFILL_SINKS_PACKAGES=STAGED profiles=126 cc={a.cc}")

if __name__=="__main__":
    main()
