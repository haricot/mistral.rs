#!/usr/bin/env python3
"""Stamp typed nonpartitioned MLA decode from the Cudaforge image."""
import argparse
import hashlib
import json
from pathlib import Path

def tensor(name,dtype,rank,access,representation=None):
    a=dict(kind="tensor",name=name,dtype=dtype,rank=rank,access=access)
    if representation is not None:
        a["representation"]=representation
    return a

def profiles(cc):
    stages=2 if cc>=80 else 1
    for dt,width in (("f16",2),("bf16",2),("f32",4)):
        args=[tensor("q_nope",dt,3,"read"),tensor("q_pe",dt,3,"read"),
              tensor("ckv",dt,3,"read","mla.ckv.paged.v1"),
              tensor("kpe",dt,3,"read","mla.kpe.paged.v1")]
        args += [tensor(n,"i32",1,"read") for n in
                 ("indptr","indices","last_page_len","request_indices","tile_indices","o_indptr","chunk_size")]
        args += [tensor("output",dt,3,"write")]
        args += [dict(kind="scalar",name=n,dtype="u32") for n in
                 ("batch","heads","page","magic","shift","add")]
        args += [dict(kind="scalar",name="scale",dtype="f32")]
        # Exact historical dispatch geometry: 32*8*1 threads, two
        # query heads per y lane; size_t offsets are 64-bit in CUDA.
        shared=stages*8*(512+64)*width+256*8*2
        yield dt,shared,args

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
    for dt,shared,args in profiles(a.cc):
        folder=a.output/f"sm{a.cc}"/"mla-decode"/dt
        folder.mkdir(parents=True,exist_ok=True)
        artifact=folder/"mla_decode.ptx"
        manifest=folder/"manifest.json"
        m=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
               operation="attention.mla.decode.ckv512.kpe64.v1",
               implementation_id=f"cuda.mla_decode.{dt}.b32x8.v1",
               artifact=artifact.name,sha256=sha,entry=f"dkr_mla_decode_{dt}",
               target_cc=a.cc,min_cc=a.cc,max_cc=a.cc,block=[32,8,1],
               shared_mem_bytes=shared,arguments=args)
        if manifest.is_file() and artifact.is_file():
            old=json.loads(manifest.read_text())
            if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")}==m and artifact.read_bytes()==data:
                continue
        artifact.write_bytes(data)
        manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_MLA_DECODE_PACKAGES=STAGED profiles=3 cc={a.cc}")

if __name__=="__main__":
    main()
