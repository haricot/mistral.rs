#!/usr/bin/env python3
"""Stamp the unchanged FlashInfer persistent split-KV merge device body."""
import argparse
import hashlib
import json
from pathlib import Path

def profiles():
    for dt,width in (("f16",2),("bf16",2),("f32",4)):
        for hd in (64,128,256,512):
            vec=max(16//width,hd//32)
            x=hd//vec
            y=128//x
            shared=4*y*hd*width+128*4
            args=[
                dict(kind="tensor",name="partial",dtype=dt,rank=3,access="read",representation="flashinfer.splitkv.partial.v1"),
                dict(kind="tensor",name="lse",dtype="f32",rank=2,access="read",representation="flashinfer.splitkv.log2_lse.v1"),
                dict(kind="tensor",name="o_indptr",dtype="i32",rank=1,access="read"),
                dict(kind="tensor",name="output",dtype=dt,rank=3,access="write"),
                dict(kind="scalar",name="batch",dtype="u32"),
                dict(kind="scalar",name="heads",dtype="u32"),
            ]
            yield dt,hd,[x,y,1],shared,args

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
    for dt,hd,block,shared,args in profiles():
        folder=a.output/f"sm{a.cc}"/"flashinfer-merge"/dt/f"d{hd}"
        folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/"flashinfer_merge.ptx"
        manifest=folder/"manifest.json"
        m=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
            operation=f"attention.flashinfer.merge.d{hd}.v1",
            implementation_id=f"cuda.flashinfer_merge.{dt}.d{hd}.v1",
            artifact=artifact.name,sha256=sha,entry=f"dkr_flashinfer_merge_{dt}_d{hd}",
            target_cc=a.cc,min_cc=a.cc,max_cc=a.cc,block=block,
            shared_mem_bytes=shared,arguments=args)
        if manifest.is_file() and artifact.is_file():
            old=json.loads(manifest.read_text())
            if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")}==m and artifact.read_bytes()==data:
                continue
        artifact.write_bytes(data)
        manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_FLASHINFER_MERGE_PACKAGES=STAGED profiles=12 cc={a.cc}")

if __name__=="__main__":
    main()
