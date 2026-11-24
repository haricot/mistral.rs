#!/usr/bin/env python3
"""Explicit typed paged options profiles, packaged from Cudaforge PTX."""
import argparse
import hashlib
import json
from pathlib import Path

def tensor(name, dtype, rank, access, representation=None):
    a = dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)
    if representation:
        a["representation"] = representation
    return a

def scalar(name, dtype="u32"):
    return dict(kind="scalar", name=name, dtype=dtype)

def profiles(dt):
    for scaled in (False, True):
        for alibi in (False, True):
            mode = ("fp8" if scaled else "plain") + ("_alibi" if alibi else "_noalibi")
            cd = "fp8_e4m3" if scaled else dt
            args = [tensor("query",dt,3,"read"),
                    tensor("key",cd,5,"read","vllm_k_blocked_x.v1"),
                    tensor("value",cd,4,"read","vllm_v_token_strided.v1"),
                    tensor("table","u32",2,"read"),tensor("lens","u32",1,"read")]
            if scaled:
                args += [tensor("k_scale","f32",1,"read"),tensor("v_scale","f32",1,"read")]
            if alibi:
                args.append(tensor("slopes","f32",1,"read"))
            args += [tensor("partial","f32",4,"write"),tensor("lse","f32",3,"write")]
            args += [scalar(n) for n in ("seqs","heads","kv_heads","hd","page","x","stride","max_context","blocks")]
            args += [scalar("scale","f32"),scalar("window"),scalar("cap","f32")]
            yield mode,args
    yield "sink_merge", [tensor("partial","f32",4,"read"),tensor("lse","f32",3,"read"),
        tensor("sinks","f32",1,"read"),tensor("output",dt,3,"write"),
        *[scalar(n) for n in ("seqs","heads","hd","parts")]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--ptx-file",type=Path,required=True)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("unsupported cc")
    data = a.ptx_file.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    for dt in ("f16","bf16","f32"):
        for mode,args in profiles(dt):
            folder = a.output / f"sm{a.cc}" / "paged-options" / mode / dt
            folder.mkdir(parents=True,exist_ok=True)
            artifact = folder / "paged_options.ptx"
            manifest = folder / "manifest.json"
            m = dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
                operation=f"attention.paged.options.{mode}.p256.v1",
                implementation_id=f"cuda.paged_options.{mode}.{dt}.p256.b128.v1",
                artifact=artifact.name,sha256=sha,entry=f"dkr_paged_options_{mode}_{dt}",
                target_cc=a.cc,min_cc=a.cc,max_cc=120,block=[128,1,1],shared_mem_bytes=0,arguments=args)
            if manifest.is_file() and artifact.is_file():
                old=json.loads(manifest.read_text())
                if {k:v for k,v in old.items() if k not in ("cubin_artifact","cubin_sha256")} == m and artifact.read_bytes() == data:
                    continue
            artifact.write_bytes(data)
            manifest.write_text(json.dumps(m,indent=2)+"\n")
    print(f"DKR_PAGED_OPTIONS_PACKAGES=STAGED profiles=15 cc={a.cc}")

if __name__ == "__main__":
    main()
