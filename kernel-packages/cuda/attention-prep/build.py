#!/usr/bin/env python3
"""Bounded typed GGUF MMVQ image plan and PTX stamping; never invokes CUDA."""
import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

FORMATS = ("q4_0","q4_1","q5_0","q5_1","q8_0","q2_k","q3_k","q4_k","q5_k","q6_k")
DTYPES = ("f16","bf16","f32")
ARTIFACT = "gguf_mmvq.ptx"
MAX_MANIFEST = 256*1024
MAX_IMAGE = 32*1024*1024

def tensor(name, dtype, rank, access, representation=None):
    result=dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)
    if representation is not None:
        result["representation"]=representation
    return result

def scalar(name):
    return dict(kind="scalar", name=name, dtype="i32")

def profile(cc, op, impl, entry, block, args):
    return dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
                operation=op,implementation_id=impl,entry=entry,
                artifact=ARTIFACT,target_cc=cc,min_cc=cc,max_cc=cc,
                block=block,shared_mem_bytes=0,arguments=args)

def plan(cc):
    images={}
    for image,dt in enumerate(DTYPES,1):
        rows=[]
        def t(name,access="read",dtype=dt):return tensor(name,dtype,1,access)
        def s(name,dtype="i32"):return dict(kind="scalar",name=name,dtype=dtype)
        def add(op,entry,block,args):
            r=profile(cc,op,"cuda."+entry+".v1",entry,[block,1,1],args)
            r["artifact"]=f"attention_prep_{image}.ptx";rows.append(r)
        for b in (32,64,128,256,512,1024):
            for kind in ("qk","qk_pos","qkv_pos"):
                triple=kind=="qkv_pos";indexed=kind!="qk"
                names=("q","k","v") if triple else ("q","k")
                args=[t(n) for n in names]+[t(n+"_weight") for n in names]+[t("cos"),t("sin")]
                if indexed:args.append(t("positions",dtype="u32"))
                args += [t(n+"_out","write") for n in names]
                args += [s(n+"_"+d,"u64") for n in names for d in ("bh","sd")]
                args += [s(n) for n in ("batch","q_heads","k_heads","seq_len","head_dim","rot_dim")]
                if not indexed:args.append(s("cos_stride"))
                args += [s(n+"_eps","f32") for n in names]+[s("neox")]
                if not triple:args.append(s("token_major"))
                if indexed:args.append(s("cache_rows"))
                add(f"attention.prep.{kind}.b{b}.v1",f"dkr_ap_{kind}_{dt}_b{b}",b,args)
        add("attention.prep.sincos.v1",f"dkr_ap_sincos_{dt}",256,
            [t("positions",dtype="u32"),t("inv_freq",dtype="f32"),t("cos","write"),t("sin","write"),s("rows"),s("width")])
        for indexed in (False,True):
            args=[t("query","read_write"),t("cos"),t("sin")]
            if indexed:args.append(t("positions",dtype="u32"))
            args += [s("head"),s("rot"),s("heads"),s("stride","i64"),s("neox")]
            if indexed:args.append(s("cache_rows"))
            suffix="rotary_pos" if indexed else "rotary"
            add(f"attention.prep.{suffix}.v1",f"dkr_{suffix}_{dt}",256,args)
        assert len(rows)==21 and len({r["entry"] for r in rows})==21
        assert all(len(r["arguments"])<=32 for r in rows)
        images[image]=rows
    return images

def widths(row):
    return [64 if p["kind"]=="tensor" or p["dtype"] in ("i64","u64") else 32 for p in row["arguments"]]

def verify(data, rows, cc):
    if len(data)>MAX_IMAGE:
        raise ValueError("oversized image")
    src=data.decode()
    if not re.search(r"\.target\s+sm_"+str(cc)+r"(?:[\s,]|$)",src) or not re.search(r"\.address_size\s+64\b",src):
        raise ValueError("PTX target/address mismatch")
    entries={}
    for name,params in re.findall(r"\.entry\s+(\w+)\s*\(([^)]*)\)",src,re.S):
        if name in entries:
            raise ValueError("duplicate entry")
        found=[]
        for arg in params.split(",") if params.strip() else []:
            w=re.search(r"\.(?:u|s|b|f)(8|16|32|64)\b",arg)
            if not w:
                raise ValueError("invalid param")
            found.append(int(w.group(1)))
        entries[name]=found
    if entries!={r["entry"]:widths(r) for r in rows}:
        raise ValueError("PTX ABI/entry set differs")

def package(root,cc,image,rows,data):
    verify(data,rows,cc)
    sha=hashlib.sha256(data).hexdigest()
    shard=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[dict(r,sha256=sha) for r in rows])
    encoded=(json.dumps(shard,indent=2)+"\n").encode()
    if len(encoded)+256*len(rows)>MAX_MANIFEST:
        raise ValueError("manifest pairing budget")
    folder=root/f"sm{cc}"/"attention-prep"/f"image-{image:02}"
    folder.mkdir(parents=True,exist_ok=True)
    manifest,artifact=folder/"manifest.json",folder/rows[0]["artifact"]
    if manifest.exists() or artifact.exists():
        if not manifest.is_file() or not artifact.is_file():
            raise ValueError("partial image; use fresh staging root")
        old=json.loads(manifest.read_text())
        old["profiles"]=[{k:v for k,v in r.items() if k not in ("cubin_artifact","cubin_sha256")}
                         for r in old.get("profiles",[])]
        if old!=shard or artifact.read_bytes()!=data:
            raise ValueError("existing image differs; use fresh staging root")
        return
    artifact.write_bytes(data)
    manifest.write_bytes(encoded)

def self_test(cc,images):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for image,rows in images.items():
            src=f".version 8.0\n.target sm_{cc}\n.address_size 64\n"
            for r in rows:
                params=",".join(f".param .b{w} arg{i}" for i,w in enumerate(widths(r)))
                src+=f".visible .entry {r['entry']}({params}) {{ret;}}\n"
            data=src.encode()
            package(root,cc,image,rows,data)
            manifest=root/f"sm{cc}"/"attention-prep"/f"image-{image:02}"/"manifest.json"
            original=json.loads(manifest.read_text())
            for r in original["profiles"]:
                r.update(cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
            paired=json.dumps(original,indent=2)+"\n"
            manifest.write_text(paired)
            package(root,cc,image,rows,data)
            assert manifest.read_text()==paired
            for invalid in (data.replace(b".param .b64",b".param .b32",1),
                            data.replace(b".address_size 64",b".address_size 32")):
                try:
                    verify(invalid,rows,cc)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid PTX accepted")
        assert len(list(root.rglob("manifest.json")))==3
    print(f"DKR_R126_S9C2C_ATTENTION_PREP_SELF_TEST=PASS cc={cc} images=3 profiles=63 synthetic_ptx=true")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--json-out",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--image",type=int)
    p.add_argument("--ptx-file",type=Path)
    a=p.parse_args()
    if not 50<=a.cc<=120:
        p.error("unsupported target")
    images=plan(a.cc)
    if a.self_test:
        self_test(a.cc,images)
    elif a.json_out:
        # The SM61 base is the previously audited upper bound, not a new real inventory.
        report=dict(schema="dkr.r126.attention.prep.plan.v1",cc=a.cc,images=images,
                    profiles=63,physical_manifests=3,compilation=False,link_qualified=False)
        if a.cc==61:
            report.update(prospective_physical_upper=526,prospective_logical=3824)
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(report,indent=2)+"\n")
        print(f"DKR_R126_S9C2C_ATTENTION_PREP_PLAN=COMPLETE cc={a.cc} images=3 profiles=63 compilation=false")
    elif a.image in images and a.output is not None and a.ptx_file is not None:
        package(a.output,a.cc,a.image,images[a.image],a.ptx_file.read_bytes())
        print(f"DKR_GGUF_MMVQ_SHARD=STAGED cc={a.cc} image={a.image} profiles={len(images[a.image])}")
    else:
        p.error("specify --self-test, --json-out or --image --output --ptx-file")
if __name__=="__main__":
    main()
