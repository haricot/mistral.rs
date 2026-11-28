#!/usr/bin/env python3
"""Source closure of the final two legacy Graph Copy FFI; no compilation."""
import argparse
import json
import re
from pathlib import Path
from dkr_r126_s9c2e5_audit import audit as previous
from dkr_r126_s9c2e0_audit import GRAPH, self_test as partition_self_test
from dkr_r126_s9c2e1_audit import check_declarations
from dkr_r126_gguf_residual_audit import guarded_ranges, mask
from dkr_r126_s9c2d3c_audit import strip_comments

CORE=Path("mistralrs-core/src")
RUNTIME=Path("mistralrs-kernel-runtime/src")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
PATTERN=re.compile(r"\bffi::("+"|".join(sorted(GRAPH))+r")\b")

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def checked_calls(source):
    clean=strip_comments(source)
    spans=guarded_ranges(clean)
    found=[]
    for match in PATTERN.finditer(mask(clean)):
        require(any(a<=match.start()<b for a,b in spans),"unguarded graph copy "+match.group(1))
        found.append(match.group(1))
    return found

def self_test():
    sample="\n".join(GUARD+"\nfn old_"+name+"(){ crate::cuda::ffi::"+name+"(); }" for name in sorted(GRAPH))
    require(set(checked_calls(sample))==GRAPH,"guard fixture")
    failures=0
    for bad in (sample.replace(GUARD,"",1),sample.replace(GUARD,"// "+GUARD,1),
                sample+"\nfn leak(){ffi::cuda_graph_copy_bytes();}"):
        try:
            checked_calls(bad)
        except ValueError:
            failures+=1
        else:
            raise AssertionError("invalid graph copy guard accepted")
    partition_self_test()
    print(f"DKR_R126_S9C2E6_SELF_TEST=PASS negative_cases={failures} compilation=false")

def audit(root):
    prior=previous(root)
    require(prior["residual_ffi"]==0 and set(prior["closed_graph_declarations"])==GRAPH,"cumulative graph closure")
    ffi=(root/CORE/"cuda/ffi.rs").read_text()
    check_declarations(ffi,GRAPH)
    clean=strip_comments(ffi)
    spans=guarded_ranges(clean)
    residual=[m.group(1) for m in re.finditer(r'\bpub\(crate\)\s+fn\s+(\w+)\s*\(',mask(clean))
              if not any(a<=m.start()<b for a,b in spans)]
    require(not residual,"remaining core FFI declaration inventory "+str(residual))
    found=[]
    for path in (root/CORE).rglob("*.rs"):
        if path==root/CORE/"cuda/ffi.rs":
            continue
        calls=checked_calls(path.read_text())
        if calls:
            require(path==root/CORE/"cuda/graph.rs","unexpected graph call site")
        found.extend(calls)
    require(len(found)==2 and set(found)==GRAPH,"legacy graph call inventory")
    graph=(root/CORE/"cuda/graph.rs").read_text()
    strict=graph.split('#[cfg(feature = "dkr-static-closure-probe")]\npub fn copy_tensor',1)[1]
    for marker in ("CopyGeometry::new(","src_layout.start_offset(), dst_layout.start_offset()",
                   "s.len(), d.len()","geometry.bind(src_ptr, dst_ptr)",
                   "stream.context().bind_to_thread()","std::sync::Arc::ptr_eq",
                   "s.stream().cu_stream() != stream.cu_stream()",
                   "d.stream().cu_stream() != stream.cu_stream()",
                   "for (src_addr, dst_addr, bytes) in bound.rows()",
                   "result::memcpy_dtod_async(dst_addr, src_addr, bytes, stream.cu_stream())",
                   "drop(dst_guard)","drop(src_guard)"):
        require(marker in strict,"driver binding contract "+marker)
    require("ffi::" not in strict and "synchronize(" not in strict and "cuMemcpy2DAsync(" not in strict,
            "no legacy call or synchronous/retry route")
    require(strict.count("result::memcpy_dtod_async(")==1,"one submission site")
    geometry=(root/RUNTIME/"graph_copy_geometry.rs").read_text()
    for marker in ("checked_mul","checked_add","src_span <= mul(src_len-src_offset,elem_size)?",
                   "dst_span <= mul(dst_len-dst_offset,elem_size)?","se <= dst || de <= src",
                   "contiguous(dims,dst_strides)?","pub fn is_empty","fn dkr_graph_copy_geometry_rejects_invalid",
                   "fn dkr_graph_copy_geometry_contiguous_and_pitched"):
        require(marker in geometry,"geometry contract "+marker)
    require("pub mod graph_copy_geometry;" in (root/RUNTIME/"lib.rs").read_text(),"CPU geometry export")
    mod=(root/CORE/"cuda/mod.rs").read_text()
    require('any(test, feature = "dkr-static-closure-probe", all(feature = "flash-attn", target_family = "unix"))' in mod,
            "strict graph module must compile")
    producer=(root/RUNTIME/"bin/dkr_package_builder.rs").read_text()
    require('name: "graph-copy"' not in producer and 'name: "graph-copies"' not in producer,"no copy producer")
    pipeline=(root/CORE/"pipeline/cuda_graph.rs").read_text()
    require("fn graph_copy_supports_dense_row_source()" in pipeline and
            "fn dkr_graph_copy_contiguous_offset_capture()" in pipeline,"GPU replay fixtures")
    require(prior["projected_master_logical"]==4268 and prior["projected_master_physical_upper"]==552,"master budget")
    return dict(schema="dkr.r126.s9c2e6.source.v1",status="SOURCE_COMPLETE",
                guarded_ffi=2,residual_ffi=0,closed_inventory=60,profiles_added=0,images_added=0,
                driver_route="dtod_async_rows",source_inventory_closed=True,
                projected_master_logical=4268,projected_master_physical_upper=552,
                active_limits_unchanged=True,model_scope_fit_proven=False,
                compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print("DKR_R126_S9C2E6_SOURCE=COMPLETE guarded_ffi=2 residual=0 closed_inventory=60 profiles_added=0 images_added=0 driver_route=dtod_async_rows projected_master_logical=4268 projected_master_physical=552 model_scope_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
