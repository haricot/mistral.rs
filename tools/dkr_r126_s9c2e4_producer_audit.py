#!/usr/bin/env python3
"""Producer-only audit; must not claim FFI, link or GPU closure."""
import argparse
import copy
import importlib.util
import json
import re
from pathlib import Path

PKG=Path("kernel-packages/cuda/sampling")
CORE=Path("mistralrs-core/src/cuda")
TYPES={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","u32":"uint32_t","i32":"int32_t"}

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def load(root):
    spec=importlib.util.spec_from_file_location("sampling_build",root/PKG/"build.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def normalize(s):
    return re.sub(r"\s*\*\s*","* "," ".join(s.split()))

def check(source,rows):
    entries=re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{(.*?)\}',source,re.S)
    found={name:(params,body) for name,params,body in entries}
    require(len(entries)==len(found)==len(rows),"entry count")
    require(set(found)=={r["entry"] for r in rows},"entry set")
    for row in rows:
        params,body=found[row["entry"]]
        expected=[]
        for a in row["arguments"]:
            ty=TYPES[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+a["name"])
        require([normalize(p) for p in params.split(",")]==expected,"ABI "+row["entry"])
        require(row["implementation_id"]=="cuda."+row["entry"]+".v1","identity")
        require(len(row["arguments"])<=16,"argument budget")
        require(all(re.search(r"\b"+a["name"]+r"\b",body) for a in row["arguments"]),"unused argument")

def audit(root,self_test=False):
    m=load(root)
    source=(root/PKG/"sampling.cu").read_text()
    groups={int(i):body for i,body in re.findall(r'#(?:if|elif) DKR_SAMPLING_IMAGE == (\d+)\n(.*?)(?=#elif DKR_SAMPLING_IMAGE|#else\n#error)',source,re.S)}
    require(set(groups)=={1,2,3},"image guards")
    for cc in (61,80):
        seen=set()
        for i,count in ((1,14),(2,16),(3,2)):
            rows=m.plan(cc,i)
            require(len(rows)==count and all(r["target_cc"]==r["min_cc"]==r["max_cc"]==cc for r in rows),"target/count")
            check(groups[i],rows)
            for r in rows:
                require(r["entry"] not in seen,"duplicate entry")
                seen.add(r["entry"])
            if self_test:
                m.self_test(cc,rows)
        require(len(seen)==32,"logical inventory")
    for legacy,header,count in (("sort.cu","sampling_device.cuh",4),("dflash_selector.cu","dflash_selector_device.cuh",2),("speculative_rejection.cu","speculative_rejection_device.cuh",2)):
        a=(root/CORE/legacy).read_text()
        b=(root/CORE/header).read_text()
        require(f'#include "{header}"' in a and "#ifdef DKR_SAMPLING_EXTERNAL" in b,"shared bodies")
        require(len(re.findall(r"DKR_\w+_ENTRY void",b))==count,"body inventory")
    if self_test:
        rows=m.plan(61,1)
        bad=copy.deepcopy(rows)
        bad[0]["arguments"][0]["dtype"]="u32"
        try:
            check(groups[1],bad)
        except ValueError:
            pass
        else:
            raise AssertionError("incorrect ABI accepted")
    return dict(status="PRODUCER_SOURCE_COMPLETE",profiles=32,images=3,max_args=16,
        projected_master_logical=4245,projected_master_physical_upper=551,
        residual_ffi=28,ffi_closed=False,compilation=False,link_qualified=False,
        numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    result=audit(a.source_root.resolve(),a.self_test)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(result,indent=2)+"\n")
    print("DKR_R126_S9C2E4_PRODUCER_"+("SELF_TEST=PASS" if a.self_test else "SOURCE=COMPLETE")+
          " profiles=32 images=3 max_args=16 residual_ffi=28 ffi_closed=false compilation=false link_qualified=false")
if __name__=="__main__":
    main()
