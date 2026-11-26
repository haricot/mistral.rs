#!/usr/bin/env python3
"""Offline, explicit model catalogue selection; never compiles or loads CUDA.

The master store is not a runtime catalogue. Only the bounded materialized view
may be supplied to DKR_KERNEL_PACKAGES or DKR_EMBED_TYPED_DIR. No model coverage
is inferred by this tool: the caller must supply a complete exact allowlist.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

MAX_MANIFEST=256*1024
MAX_IMAGE=32*1024*1024
MAX_ACTIVE_PROFILES=4096
MAX_ACTIVE_MANIFESTS=512
MAX_SHARD=64
MAX_MASTER_MANIFESTS=16384
MAX_MASTER_PROFILES=262144
MAX_SELECTION_BYTES=8*1024*1024
MODULE="DKR-KERNEL-MODULE-V1"
SHARD="DKR-KERNEL-SHARD-V1"
SELECTION="DKR-MODEL-SELECTION-V1"
IDENT=re.compile(r"[A-Za-z0-9_.-]+\Z")
HEX=re.compile(r"[A-Fa-f0-9]{64}\Z")

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def read(path,limit):
    require(path.is_file() and not path.is_symlink(),f"not a regular input: {path}")
    with path.open("rb") as f:
        data=f.read(limit+1)
    require(len(data)<=limit,f"bounded read exceeded: {path}")
    return data

def sha(data):
    return hashlib.sha256(data).hexdigest()

def unique_object(pairs):
    result={}
    for k,v in pairs:
        require(k not in result,"duplicate JSON key: "+k)
        result[k]=v
    return result

def decode(data):
    return json.loads(data,object_pairs_hook=unique_object)

def identity(row):
    op,impl,cc=(row[k] for k in ("operation","implementation_id","target_cc"))
    require(isinstance(op,str) and IDENT.fullmatch(op) and isinstance(impl,str) and IDENT.fullmatch(impl)
            and type(cc) is int and 50<=cc<=255,"invalid profile identity")
    return op,impl,cc

def artifact_name(row,key):
    name=row[key]
    require(isinstance(name,str) and name not in ("",".","..") and
            Path(name).name==name and "/" not in name and "\\" not in name,
            "artifact must be a local filename")
    return name

def inventory(root):
    root=root.resolve(strict=True)
    require(root.is_dir(),"master root is not a directory")
    records={}
    manifests={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        folder=Path(directory)
        require(len(folder.relative_to(root).parts)<=8,"master nesting limit exceeded")
        for n in dirs+files:
            require(not (folder/n).is_symlink(),"symlink in master store")
        dirs.sort()
        if "manifest.json" not in files:
            continue
        path=folder/"manifest.json"
        raw=read(path,MAX_MANIFEST)
        value=decode(raw)
        schema=value.get("schema")
        if schema==SHARD:
            require(set(value)=={"schema","profiles"},"unknown shard fields")
            rows=value["profiles"]
            require(isinstance(rows,list) and 1<=len(rows)<=MAX_SHARD,"invalid shard profile count")
        else:
            require(schema==MODULE,"unknown manifest schema")
            rows=[value]
        require(len(manifests)<MAX_MASTER_MANIFESTS,"master manifest bound")
        manifests[path]=(raw,schema,rows)
        for row in rows:
            require(row.get("schema")==MODULE and row.get("abi")=="dkr.typed.v1","invalid profile schema/ABI")
            key=identity(row)
            require(key not in records,"duplicate identity: "+str(key))
            require(len(records)<MAX_MASTER_PROFILES,"master profile bound")
            require(type(row.get("min_cc")) is int and type(row.get("max_cc")) is int
                    and key[2]<=row["min_cc"]<=row["max_cc"],"invalid CC bounds")
            require(isinstance(row.get("arguments"),list) and 1<=len(row["arguments"])<=32,"invalid ABI argument count")
            artifact_name(row,"artifact")
            require(isinstance(row.get("sha256"),str) and HEX.fullmatch(row["sha256"]),"invalid PTX digest")
            require((row.get("cubin_artifact") is None)==(row.get("cubin_sha256") is None),"incomplete paired image")
            if row.get("cubin_artifact") is not None:
                artifact_name(row,"cubin_artifact")
                require(isinstance(row["cubin_sha256"],str) and HEX.fullmatch(row["cubin_sha256"]),"invalid CUBIN digest")
            records[key]=(path,row)
    require(bool(records),"empty master store")
    return root,records,manifests

def selection(spec,records,manifests,limits=(MAX_ACTIVE_PROFILES,MAX_ACTIVE_MANIFESTS)):
    require(spec.get("schema")==SELECTION,"unsupported selection schema")
    require(isinstance(spec.get("model"),str) and spec["model"].strip(),"model label required")
    require(type(spec.get("device_cc")) is int and 50<=spec["device_cc"]<=255,"invalid device CC")
    require(isinstance(spec.get("profiles"),list) and bool(spec["profiles"]),"empty selection")
    require(len(spec["profiles"])<=limits[0],"active logical profile limit exceeded")
    chosen=set()
    grouped={}
    for row in spec["profiles"]:
        require(set(row)=={"operation","implementation_id","target_cc"},"selection requires exact identities")
        key=identity(row)
        require(key not in chosen,"duplicate selection identity")
        require(key in records,"missing selected profile: "+str(key))
        path,profile=records[key]
        require(profile["min_cc"]<=spec["device_cc"]<=profile["max_cc"],"profile incompatible with device")
        chosen.add(key)
        grouped.setdefault(path,set()).add(key)
    require(len(grouped)<=limits[1],"active physical manifest limit exceeded")
    return grouped

def selected_bytes(path,keys,manifests):
    raw,schema,rows=manifests[path]
    selected=[r for r in rows if identity(r) in keys]
    require(len(selected)==len(keys),"selection drift")
    if len(selected)==len(rows):
        return raw,selected
    require(schema==SHARD,"cannot rewrite single-profile identity")
    data=(json.dumps({"schema":SHARD,"profiles":selected},indent=2)+"\n").encode()
    require(len(data)<=MAX_MANIFEST,"active shard byte limit")
    return data,selected

def report(root,records,manifests,spec,grouped):
    images={}
    manifest_rows=[]
    for path,keys in sorted(grouped.items()):
        data,rows=selected_bytes(path,keys,manifests)
        manifest_rows.append(dict(path=str(path.relative_to(root)),profiles=len(rows),sha256=sha(data)))
        for r in rows:
            for a,h in (("artifact","sha256"),("cubin_artifact","cubin_sha256")):
                if r.get(a) is None:
                    continue
                image=path.parent/artifact_name(r,a)
                expected=r[h].lower()
                require(image not in images or images[image]==expected,"conflicting image hashes")
                images[image]=expected
    # Check every selected image, including a paired CUBIN. Never substitute PTX
    # for a corrupt CUBIN. Runtime still checks typed signatures and ELF target.
    total=0
    for image,expected in sorted(images.items()):
        data=read(image,MAX_IMAGE)
        require(sha(data)==expected,"selected artifact digest mismatch: "+str(image))
        total+=len(data)
    return dict(schema="DKR-MODEL-SCOPE-REPORT-V1",model=spec["model"],device_cc=spec["device_cc"],
                master_profiles=len(records),master_manifests=len(manifests),
                active_profiles=sum(len(keys) for keys in grouped.values()),active_manifests=len(grouped),
                active_images=len(images),active_image_bytes=total,manifests=manifest_rows,
                selection_sha256=sha(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()),
                selection_complete_for_model=False,verified_image_hashes=True,
                runtime_abi_qualified=False,compilation=False,link_qualified=False),images

def materialize(root,dest,grouped,manifests,images):
    dest=dest.resolve()
    require(dest!=root and root not in dest.parents and dest not in root.parents,
            "destination must be separate from master root")
    require(not dest.exists(),"destination must be fresh")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dkr-scope-",dir=dest.parent) as tmp:
        stage=Path(tmp)/"scope"
        stage.mkdir()
        for path,keys in sorted(grouped.items()):
            # Detect input mutation between inventory and publication.
            require(read(path,MAX_MANIFEST)==manifests[path][0],"master manifest changed")
            data,_=selected_bytes(path,keys,manifests)
            out=stage/path.relative_to(root)
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(data)
        for path,expected in sorted(images.items()):
            data=read(path,MAX_IMAGE)
            require(sha(data)==expected,"master image changed")
            (stage/path.relative_to(root)).write_bytes(data)
        # Keep the selected root clean: no unselected manifests, no symlinks.
        require(not dest.exists(),"destination appeared during staging")
        stage.rename(dest)

def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/"master"
        def row(n,cc=61):
            data=b"synthetic PTX only"
            return dict(schema=MODULE,abi="dkr.typed.v1",operation="test.op",implementation_id=f"impl.{n}",
                        target_cc=cc,min_cc=cc,max_cc=cc,artifact="image.ptx",sha256=sha(data),
                        arguments=[dict(kind="scalar",name="n",dtype="u32")])
        def put(name,rows,shard=True):
            folder=root/name;folder.mkdir(parents=True)
            (folder/"image.ptx").write_bytes(b"synthetic PTX only")
            value={"schema":SHARD,"profiles":rows} if shard else rows[0]
            (folder/"manifest.json").write_text(json.dumps(value))
        put("a",[row(1),row(2)])
        put("b",[row(3)],False)
        put("c",[row(4,80)])
        # Exercise the paired-image digest contract on a selected profile.
        paired=row(5)
        paired.update(cubin_artifact="image.cubin",cubin_sha256=sha(b"synthetic cubin"))
        put("d",[paired])
        (root/"d/image.cubin").write_bytes(b"synthetic cubin")
        base,rec,man=inventory(root)
        def spec(*keys):
            return dict(schema=SELECTION,model="synthetic",device_cc=61,
                        profiles=[dict(zip(("operation","implementation_id","target_cc"),key)) for key in keys])
        key1=identity(row(1));key3=identity(row(3));key4=identity(row(4,80))
        chosen=spec(key1,key3)
        group=selection(chosen,rec,man)
        result,images=report(base,rec,man,chosen,group)
        assert result["master_profiles"]==5 and result["active_profiles"]==2
        dest=Path(tmp)/"active"
        materialize(base,dest,group,man,images)
        _,selected,selected_manifests=inventory(dest)
        assert set(selected)=={key1,key3}
        assert (dest/"b/manifest.json").read_bytes()==(root/"b/manifest.json").read_bytes()
        assert len(selected_manifests)==2 and not (dest/"c").exists()
        failures=0
        def rejects(fn):
            nonlocal failures
            try:
                fn()
            except ValueError:
                failures+=1
            else:
                raise AssertionError("invalid scope accepted")
        rejects(lambda:selection(spec(key1,key1),rec,man))
        rejects(lambda:selection(spec(("test.op","missing",61)),rec,man))
        rejects(lambda:selection(spec(key4),rec,man))
        rejects(lambda:selection(chosen,rec,man,(1,512)))
        rejects(lambda:selection(chosen,rec,man,(4096,1)))
        rejects(lambda:materialize(base,dest,group,man,images))
        rejects(lambda:materialize(base,root/"nested",group,man,images))
        paired_spec=spec(identity(paired))
        paired_group=selection(paired_spec,rec,man)
        paired_report,_=report(base,rec,man,paired_spec,paired_group)
        assert paired_report["active_images"]==2
        (root/"d/image.cubin").write_bytes(b"corrupt")
        rejects(lambda:report(base,rec,man,paired_spec,paired_group))
        (root/"d/image.cubin").write_bytes(b"synthetic cubin")
        (root/"a/image.ptx").write_bytes(b"corrupt")
        rejects(lambda:report(base,rec,man,chosen,group))
        (root/"a/image.ptx").write_bytes(b"synthetic PTX only")
        (root/"symlink").symlink_to(root/"a",target_is_directory=True)
        rejects(lambda:inventory(root))
        (root/"symlink").unlink()
        put("duplicate",[row(1)])
        rejects(lambda:inventory(root))
    print(f"DKR_R126_MODEL_SCOPE_SELF_TEST=PASS negative_cases={failures} synthetic=true")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--master-root",type=Path)
    p.add_argument("--inventory-out",type=Path)
    p.add_argument("--selection",type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--materialize",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test();return
    if a.master_root is None or not (a.inventory_out or a.selection):
        p.error("specify --master-root and --inventory-out and/or --selection")
    if (a.materialize or a.json_out) and not a.selection:
        p.error("--materialize/--json-out require --selection")
    root,records,manifests=inventory(a.master_root)
    if a.inventory_out:
        rows=[dict(operation=k[0],implementation_id=k[1],target_cc=k[2],
                   min_cc=v[1]["min_cc"],max_cc=v[1]["max_cc"],
                   manifest=str(v[0].relative_to(root)),arguments=v[1]["arguments"])
              for k,v in sorted(records.items())]
        with a.inventory_out.open("x") as f:
            json.dump(dict(schema="DKR-MASTER-INVENTORY-V1",profiles=rows,
                           manifest_count=len(manifests),runtime_verified=False),f,indent=2)
            f.write("\n")
        print(f"DKR_R126_MASTER_INVENTORY=COMPLETE profiles={len(records)} manifests={len(manifests)} runtime_verified=false")
    if a.selection:
        spec=decode(read(a.selection,MAX_SELECTION_BYTES))
        grouped=selection(spec,records,manifests)
        result,images=report(root,records,manifests,spec,grouped)
        if a.materialize:
            materialize(root,a.materialize,grouped,manifests,images)
        if a.json_out:
            with a.json_out.open("x") as f:
                json.dump(result,f,indent=2);f.write("\n")
        print(f"DKR_R126_MODEL_SCOPE=COMPLETE active_profiles={result['active_profiles']} "
              f"active_manifests={result['active_manifests']} master_profiles={result['master_profiles']} "
              f"materialized={str(bool(a.materialize)).lower()} model_coverage=UNPROVEN compilation=false")

if __name__=="__main__":
    main()
