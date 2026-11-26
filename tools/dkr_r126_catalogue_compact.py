#!/usr/bin/env python3
"""Compact a paired SM61 catalogue into a new root, preserving profile fields and image bytes."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from dkr_r126_full_catalog_inventory import (
    manifest_profiles, read_manifest, inspect, MAX_SHARD, MAX_MANIFEST, MAX_PHYSICAL, MAX_LOGICAL)
from dkr_r126_nvfp4_preflight import digest

def require(ok,msg):
    if not ok: raise ValueError(msg)

def key(p):
    return p["operation"],p["implementation_id"],p["target_cc"]

def encode(rows):
    return (json.dumps(dict(schema="DKR-KERNEL-SHARD-V1",profiles=rows),indent=2)+"\n").encode()

def collect(root):
    require(root.is_dir(),"missing source")
    rows={}
    manifests=[]
    for path in sorted(root.rglob("manifest.json")):
        require(not any(p.is_symlink() for p in [path,*path.parents] if p==root or p.is_relative_to(root)),
                "symlink in catalogue")
        rel=path.relative_to(root)
        require(len(rel.parts)>=2,"manifest outside family")
        family=rel.parts[0]
        m=read_manifest(path)
        manifests.append((family,path,digest(path)))
        for p in manifest_profiles(m):
            k=key(p)
            require(k not in rows,"duplicate identity")
            require(p.get("cubin_artifact") and p.get("cubin_sha256"),"only fully paired catalogues admitted")
            names={}
            for field,hashfield in (("artifact","sha256"),("cubin_artifact","cubin_sha256")):
                name=p[field]
                require(Path(name).name==name and name not in ("",".",".."),"unsafe image name")
                image=path.parent/name
                require(image.is_file() and not image.is_symlink(),"missing/unsafe image")
                require(digest(image)==p[hashfield],"image digest mismatch")
                require(name not in names,"PTX/CUBIN name alias")
                names[name]=(p[hashfield],image)
            rows[k]=(family,p,names)
    require(0<len(rows)<=MAX_LOGICAL,"logical count")
    for family in sorted({v[0] for v in rows.values()}):
        inspect(root, "logits" if family=="sparse-logits" else family)
    return rows,manifests

def plan(rows):
    # First-fit within family, preferring a bin already containing the same image.
    # Never rename images: the canonical typed profile digest is unchanged.
    bins=[]
    for _,(family,p,names) in sorted(rows.items(),key=lambda item:(item[1][0],item[1][1]["sha256"],item[0])):
        candidates=[]
        for b in bins:
            if b["family"]!=family or len(b["rows"])>=MAX_SHARD: continue
            if any(n in b["images"] and b["images"][n][0]!=v[0] for n,v in names.items()): continue
            if len(encode(b["rows"]+[p]))>MAX_MANIFEST: continue
            candidates.append(b)
        if candidates:
            b=max(candidates,key=lambda b:sum(n in b["images"] for n in names))
        else:
            b=dict(family=family,rows=[],images={})
            bins.append(b)
        require(len(encode([p]))<=MAX_MANIFEST,"profile too large")
        b["rows"].append(p)
        b["images"].update(names)
    return bins

def compact(source,output=None):
    source=source.resolve()
    rows,original=collect(source)
    bins=plan(rows)
    require(len(bins)<=MAX_PHYSICAL,"compacted catalogue still exceeds physical cap")
    report=dict(schema="dkr.r126.catalogue.compaction.v1",source=str(source),
        original_manifests=len(original),compacted_manifests=len(bins),
        saved_manifests=len(original)-len(bins),profiles=len(rows),
        image_pairs=len({(p["sha256"],p["cubin_sha256"]) for _,p,_ in rows.values()}),
        original_manifest_sha256={str(p.relative_to(source)):h for _,p,h in original},
        compilation=False,full_catalogue_qualified=False)
    if output is None:return report
    output=output.resolve()
    require(not output.exists() and not output.is_relative_to(source) and not source.is_relative_to(output),
            "destination must be new and disjoint")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dkr-compact-",dir=output.parent) as tmp:
        stage=Path(tmp)/"sm61"
        for i,b in enumerate(bins):
            folder=stage/b["family"]/f"compact-{i:04}"
            folder.mkdir(parents=True)
            for name,(sha,path) in b["images"].items():
                shutil.copyfile(path,folder/name)
                require(digest(folder/name)==sha,"copy digest mismatch")
            (folder/"manifest.json").write_bytes(encode(b["rows"]))
        now,_=collect(stage)
        require({k:(f,p) for k,(f,p,_) in now.items()}=={k:(f,p) for k,(f,p,_) in rows.items()},
                "profile metadata drift")
        require(all(digest(p)==h for _,p,h in original),"original manifest changed")
        stage.rename(output)
    report["output"]=str(output)
    return report

def self_test():
    with tempfile.TemporaryDirectory() as t:
        src=Path(t)/"original";out=Path(t)/"compacted"
        ptx=b".version 8.0\n.target sm_61\n.address_size 64\n.visible .entry e() {ret;}\n"
        elf=bytearray(64);elf[:6]=b"\x7fELF\x02\x01";elf[18:20]=(190).to_bytes(2,"little");elf[48]=61
        for i in range(65):
            folder=src/"demo"/str(i);folder.mkdir(parents=True)
            (folder/"x.ptx").write_bytes(ptx);(folder/"x.cubin").write_bytes(elf)
            p=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",operation=f"demo.{i}",
                implementation_id=f"cuda.demo.{i}",target_cc=61,entry="e",artifact="x.ptx",
                sha256=hashlib.sha256(ptx).hexdigest(),cubin_artifact="x.cubin",
                cubin_sha256=hashlib.sha256(elf).hexdigest(),block=[32,1,1],shared_mem_bytes=0,arguments=[])
            (folder/"manifest.json").write_text(json.dumps(p))
        r=compact(src,out)
        assert (r["original_manifests"],r["compacted_manifests"],r["profiles"])==(65,2,65)
        try:compact(src,src/"bad")
        except ValueError:pass
        else:raise AssertionError("overlap accepted")
        (src/"demo"/"0"/"x.ptx").write_bytes(b"corrupt")
        try:compact(src)
        except ValueError:pass
        else:raise AssertionError("corrupt image accepted")
    # Exercise over-cap admission: 9 families of 64 compatible manifests
    # are valid separately but must be compacted before runtime loading.
    with tempfile.TemporaryDirectory() as t:
        src=Path(t)/"original";out=Path(t)/"compacted"
        for family in range(9):
            for j in range(64):
                folder=src/f"family-{family}"/str(j);folder.mkdir(parents=True)
                (folder/"x.ptx").write_bytes(ptx);(folder/"x.cubin").write_bytes(elf)
                p=dict(schema="DKR-KERNEL-MODULE-V1",abi="dkr.typed.v1",
                    operation=f"demo.{family}.{j}",implementation_id=f"cuda.demo.{family}.{j}",
                    target_cc=61,entry="e",artifact="x.ptx",sha256=hashlib.sha256(ptx).hexdigest(),
                    cubin_artifact="x.cubin",cubin_sha256=hashlib.sha256(elf).hexdigest(),
                    block=[32,1,1],shared_mem_bytes=0,arguments=[])
                (folder/"manifest.json").write_text(json.dumps(p))
        r=compact(src,out)
        assert (r["original_manifests"],r["compacted_manifests"],r["profiles"])==(576,9,576)
    print("DKR_R126_S9C2B4_COMPACTION_SELF_TEST=PASS synthetic=true manifests=65->2 profiles=65")
    print("DKR_R126_S9C2B4E_OVER_CAP_COMPACTION=PASS synthetic=true manifests=576->9 profiles=576")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.source or not a.json_out:p.error("--source and --json-out required")
    require(not a.json_out.exists(),"report exists")
    require(not a.json_out.resolve().is_relative_to(a.source.resolve()),"report overlaps source")
    if a.output:require(not a.json_out.resolve().is_relative_to(a.output.resolve()),"report overlaps output")
    r=compact(a.source,a.output)
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    with a.json_out.open("x") as f:json.dump(r,f,indent=2);f.write("\n")
    print(f"DKR_R126_S9C2B4_COMPACTION=COMPLETE before={r['original_manifests']} after={r['compacted_manifests']} profiles={r['profiles']} compilation=false")
if __name__=="__main__":main()
