#!/usr/bin/env python3
"""Stamp typed HND IO profiles from a single Cudaforge PTX image."""
import argparse
import hashlib
import json
from pathlib import Path

def tensor(name,dtype,rank,access,representation=None):
    a=dict(kind="tensor",name=name,dtype=dtype,rank=rank,access=access)
    if representation is not None:
        a["representation"]=representation
    return a

def scalar(name,dtype="u32"):
    return dict(kind="scalar",name=name,dtype=dtype)

def profiles():
    for dt in ("f16","bf16","f32"):
        for mode in ("plain","fp8"):
            cd=dt if mode=="plain" else "fp8_e4m3"
            def cache(access):
                return [tensor(n,cd,4,access,"flashinfer.hnd.v1") for n in ("key_cache","value_cache")]
            scales=[] if mode=="plain" else [scalar("k_scale","f32"),scalar("v_scale","f32")]
            a=[tensor(n,dt,1,"read","kv_cache.row_span.v1") for n in ("key_span","value_span")]
            a+=cache("read_write")+[tensor("slots","i64",1,"read")]
            a += [scalar(n) for n in ("tokens","blocks","heads","hd","page")]
            a += [scalar("key_stride","u64"),scalar("value_stride","u64")]+scales
            yield dt,f"store_{mode}",a
            for bt in ("i32","u32"):
                for ot in ("i32","u32"):
                    a=cache("read")+[tensor("table",bt,2,"read"),tensor("offsets",ot,1,"read"),
                        tensor("key_out",dt,3,"write"),tensor("value_out",dt,3,"write")]
                    a += [scalar(n) for n in ("tokens","seqs","blocks","heads","hd","page","stride")]+scales
                    yield dt,f"gather_{mode}_{bt}_{ot}",a

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--ptx-file",type=Path,required=True)
    a=p.parse_args()
    if not 50<=a.cc<=120:
        p.error("unsupported cc")
    data=a.ptx_file.read_bytes()
    sha=hashlib.sha256(data).hexdigest()
    for dt,mode,args in profiles():
        folder=a.output/f"sm{a.cc}"/"flashinfer-hnd-io"/mode/dt
        folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/"flashinfer_hnd_io.ptx"
        manifest=folder/"manifest.json"
        m=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
            operation=f"kv_cache.flashinfer.hnd.{mode}.b256.v1",
            implementation_id=f"cuda.flashinfer_hnd_io.{mode}.{dt}.b256.v1",
            artifact=artifact.name,sha256=sha,entry=f"dkr_hnd_{mode}_{dt}",
            target_cc=a.cc,min_cc=a.cc,max_cc=120,block=[256,1,1],
            shared_mem_bytes=0,arguments=args)
        if manifest.is_file() and artifact.is_file():
            old=json.loads(manifest.read_text())
            if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")}==m and artifact.read_bytes()==data:
                continue
        artifact.write_bytes(data)
        manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_FLASHINFER_HND_IO_PACKAGES=STAGED profiles=30 cc={a.cc}")

if __name__=="__main__":
    main()
