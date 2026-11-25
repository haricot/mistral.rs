#!/usr/bin/env python3
"""S9C read-only closure inventory and optional exact-build evidence inspection.

Never executes Cargo, nvcc or ptxas. Existing cache objects are NOT evidence.
A source-only result is not a claim of link closure.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from dkr_r126_static_ffi_inventory import COMPONENTS, source_inventory, git, digest

ARCHIVES = ("mistralrscuda", "mistralrsquant", "mistralrspagedattention",
            "mistralrsflashinfergdn", "mistralrsdeepgemm", "mistralrsnvfp4",
            "mistralrsfa3paged")
GUARDS = {
    "mistralrs-core": '#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]',
    "mistralrs-quant": '#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]',
    "mistralrs-paged-attn": 'if cfg!(feature = "dkr-paged-catalog-only")',
}
def require(ok, why):
    if not ok:
        raise ValueError(why)

def source(root):
    head = git(root,"rev-parse","HEAD").decode().strip()
    paths = [p for c in COMPONENTS for p in
             (c+"/src",c+"/kernels",c+"/build.rs",c+"/Cargo.toml")]
    require(not git(root,"status","--porcelain","--untracked-files=no","--",*paths).strip(),
            "tracked production source is dirty")
    tracked = sorted(p.decode() for p in git(root,"ls-files","-z","--",*paths).split(b"\0") if p)
    items = {}
    for c in COMPONENTS:
        b=(root/c/"build.rs").read_text()
        require(GUARDS[c] in b and f"DKR_STATIC_PRODUCER_DISABLED component={c}" in b,
                f"{c}: missing source guard or evidence marker")
        item=source_inventory(root,tracked,c)
        items[c]=item
        print(f"DKR_R126_S9C_COMPONENT component={c} cuda_sources={len(item['cuda_translation_units'])} "
              f"explicit_ffi={len(item['explicit_ffi_symbols'])} macro_blocks={len(item['macro_extern_blocks'])} "
              "source_guard=true link_qualified=false")
    require(head==git(root,"rev-parse","HEAD").decode().strip() and not
            git(root,"status","--porcelain","--untracked-files=no","--",*paths).strip(),
            "production source changed during audit")
    return head,items

def parse_messages(path):
    records=[]
    for line in path.read_text().splitlines():
        try:
            obj=json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj,dict):
            records.append(obj)
    return records

def component_id(package_id):
    for c in COMPONENTS:
        if re.search(r"(?:#|/|^)" + re.escape(c) + r"(?:@|#| |$)",package_id):
            return c
    return None

def nm_symbols(path):
    # GNU nm on rlibs may report metadata members without symbols; keep
    # the successful symbol listing, reject an actual nonzero tool status.
    run=subprocess.run(["nm","--format=posix",str(path)],capture_output=True,text=True,timeout=120)
    require(run.returncode==0,f"nm failed for {path}: {run.stderr[:400]}")
    defined,undefined=set(),set()
    for line in run.stdout.splitlines():
        cols=line.split()
        if len(cols)>=2:
            if cols[1]=="U": undefined.add(cols[0])
            elif cols[1] in ("T","W","I","D","B","R"): defined.add(cols[0])
    return defined,undefined

def inspect_build(messages,items):
    build_records={}
    rlibs={}
    executables=[]
    complete=False
    for obj in messages:
        why=obj.get("reason")
        if why=="build-finished":
            complete=obj.get("success") is True
        c=component_id(obj.get("package_id",""))
        if c and why=="build-script-executed":
            require(c not in build_records,f"ambiguous build-script records for {c}")
            build_records[c]=obj
        if c and why=="compiler-artifact":
            target=obj.get("target",{})
            if "lib" in target.get("kind",[]) and not obj.get("profile",{}).get("test",False):
                for f in obj.get("filenames",[]):
                    if f.endswith(".rlib"):
                        rlibs.setdefault(c,set()).add(f)
        if why=="compiler-artifact" and obj.get("executable") and obj.get("profile",{}).get("test"):
            if c: executables.append(obj["executable"])
    require(set(build_records)==set(COMPONENTS),"missing or incomplete three-crate build-script evidence")
    for c,obj in build_records.items():
        libs=obj.get("linked_libs",[])
        require(not any(any(a in lib for a in ARCHIVES) for lib in libs),
                f"{c}: static archive link directive present: {libs}")
        out=Path(obj["out_dir"])
        log=out.parent/"output"
        require(log.is_file() and f"DKR_STATIC_PRODUCER_DISABLED component={c}" in log.read_text(),
                f"{c}: exact build script disable marker absent")
        require(not any(p.is_file() and any(a in p.name for a in ARCHIVES)
                        for p in out.rglob("*")),f"{c}: stale or generated static archive in OUT_DIR")
    declared=set().union(*(set(v["explicit_ffi_symbols"]) for v in items.values()))
    all_defined=set()
    undefined={}
    for c,files in rlibs.items():
        require(len(files)==1,f"{c}: ambiguous rlib")
        path=Path(next(iter(files)))
        d,u=nm_symbols(path)
        all_defined.update(d)
        undefined[c]=u
    blockers={c:sorted((u-all_defined)&declared) for c,u in undefined.items()}
    print("DKR_R126_S9C_BUILD_EVIDENCE=INSPECTED producers=3 "
          f"rlibs={len(rlibs)} known_undefined={sum(map(len,blockers.values()))} "
          f"build_success={str(complete).lower()} test_executables={len(executables)}")
    return dict(build_success=complete,build_records=build_records,
                rlibs={c:sorted(v) for c,v in rlibs.items()},
                known_undefined=blockers,test_executables=executables,
                full_link_qualified=False,
                limitations="known literal externs only; no macro expansion, full model or dynamic dependency proof")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--cargo-messages",type=Path,help="optional Cargo JSON from dedicated closure-probe build")
    p.add_argument("--json-out",type=Path,required=True)
    a=p.parse_args()
    try:
        head,items=source(a.source_root.resolve(strict=True))
        report=dict(schema="dkr.r126.static-closure.probe.v1",source_head=head,
                    status="SOURCE_GUARDS_PRESENT",components=items,
                    static_producer_elimination_qualified=False,
                    static_fallback_elimination_qualified=False,
                    compilation_invoked=False)
        if a.cargo_messages:
            report["build_evidence"]=inspect_build(parse_messages(a.cargo_messages),items)
            report["cargo_messages_sha256"]=digest(a.cargo_messages)
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
        print(f"DKR_R126_S9C_SOURCE=COMPLETE components=3 closure_qualified=false report={a.json_out}")
    except (OSError,ValueError,TypeError,KeyError,subprocess.SubprocessError) as exc:
        raise SystemExit(f"DKR_R126_S9C_SOURCE=BLOCKED reason={exc}") from exc

if __name__=="__main__":
    main()
