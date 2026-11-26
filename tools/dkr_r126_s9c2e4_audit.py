#!/usr/bin/env python3
"""S9C.2E.4 source-only sampling closure; no CUDA compiler or device work."""
import argparse
import json
import re
from pathlib import Path
from dkr_r126_s9c2e3_audit import audit as previous
from dkr_r126_s9c2e4_producer_audit import audit as producer, load
from dkr_r126_s9c2e0_audit import SAMPLING_GROUPS
from dkr_r126_s9c2e1_audit import check_declarations, checked_refs
from dkr_r126_gguf_residual_audit import mask, guarded_ranges
from dkr_r126_s9c2d3c_audit import strip_comments

CORE=Path("mistralrs-core/src")
NAMES=set().union(*(SAMPLING_GROUPS[k] for k in ("sample","dflash","rejection","top1","categorical")))

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def audit(root):
    p=previous(root)
    require(p["residual_declarations"]==16 and set(p["closed_sampling_declarations"])==NAMES,"cumulative closure")
    result=producer(root)
    check_declarations((root/CORE/"cuda/ffi.rs").read_text(),NAMES)
    found=set()
    pattern=re.compile(r"\bffi::("+"|".join(sorted(NAMES))+r")\b")
    for path in (root/CORE).rglob("*.rs"):
        if path==root/CORE/"cuda/ffi.rs":
            continue
        source=mask(strip_comments(path.read_text()))
        spans=guarded_ranges(source)
        for match in pattern.finditer(source):
            name=match.group(1)
            require(path in (root/CORE/"ops.rs",root/CORE/"cuda/speculative_rejection.rs"),"unexpected call site")
            require(any(a<=match.start()<b for a,b in spans),"unguarded call "+name)
            found.add(name)
    require(found==NAMES,"legacy call inventory")
    adapter=(root/CORE/"cuda/sampling_dynamic.rs").read_text()
    m=load(root)
    for i in m.IMAGES:
        for row in m.plan(61,i):
            require(f'entry: "{row["entry"]}", operation: "{row["operation"]}", implementation: "{row["implementation_id"]}", block: {row["block"][0]}, shared: {row["shared_mem_bytes"]}' in adapter,"adapter profile identity")
    for name in ("launch_top1","launch_categorical","launch_ranked","launch_dflash"):
        require("sampling_dynamic::"+name in (root/CORE/"ops.rs").read_text(),"missing strict route "+name)
    require("sampling_dynamic::launch_rejection" in (root/CORE/"cuda/speculative_rejection.rs").read_text(),"missing rejection route")
    for marker in ("LaunchPlanCache<Key,Plan>","prevalidate_launch_with_default_shared","first:first.prepare(device)?",
                   "second:second.as_ref().map(|s|s.prepare(device)).transpose()?",
                   "pub fn dkr_sampling_requirements","sampling stream/context mismatch"):
        require(marker in adapter,"adapter contract "+marker)
    require("pub use cuda::sampling_dynamic::{dkr_sampling_requirements, DkrSamplingMode};" in (root/CORE/"lib.rs").read_text(),"capability export")
    require(p["projected_master_logical"]+32==4245 and p["projected_master_physical_upper"]+3==551,"budget")
    return dict(result,status="SOURCE_COMPLETE",guarded_ffi=12,residual_ffi=16,
                ffi_closed=True,model_scope_fit_proven=False,compilation=False,link_qualified=False,
                numeric_qualified=False,full_closure=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path.cwd())
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--json-out",type=Path)
    a=parser.parse_args()
    root=a.source_root.resolve()
    if a.self_test:
        producer(root,True)
        print("DKR_R126_S9C2E4_AUDIT_SELF_TEST=PASS synthetic_ptx=true")
        return
    r=audit(root)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print("DKR_R126_S9C2E4_SOURCE=COMPLETE guarded_ffi=12 residual=16 profiles=32 images=3 max_args=16 projected_master_logical=4245 projected_master_physical=551 model_scope_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")
if __name__=="__main__":
    main()
