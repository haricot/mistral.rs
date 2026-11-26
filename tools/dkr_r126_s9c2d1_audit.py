#!/usr/bin/env python3
"""S9C.2D1 partial source audit; deliberately NOT a full closure or link proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges, mask

BASE_PHYSICAL=526
BASE_LOGICAL=3849
MAX_PHYSICAL=512
MAX_LOGICAL=4096
MAX_ARGS=32
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,message):
    if not ok:
        raise ValueError(message)

def load_plan(root,name):
    path=root/"kernel-packages/cuda"/name/"build.py"
    spec=importlib.util.spec_from_file_location(name.replace("-","_"),path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan(61)

def audit(root):
    ssm=load_plan(root,"ssm")
    conv=load_plan(root,"dynamic-conv")
    require(len(ssm)==4 and len(conv)==6,"unexpected profile count")
    rows=ssm+conv
    require(len({r["implementation_id"] for r in rows})==len(rows),"duplicate implementation")
    require(max(len(r["arguments"]) for r in rows)<=MAX_ARGS,"argument budget")
    for group in (ssm,conv):
        require(len({r["artifact"] for r in group})==1,"multiple images in shard")
        require(all(r["target_cc"]==61 for r in group),"target mismatch")
        paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
            dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,
                 cubin_sha256="0"*64) for r in group])
        require(len(json.dumps(paired,indent=2).encode())<256*1024,"paired manifest budget")
    require([r["entry"] for r in ssm]==[f"dkr_ssm_scan_cf{n}" for n in (1,2,4,8)],"SSM entry identity")
    require(all(len(r["arguments"])==15 for r in ssm),"SSM ABI")
    require(all(len(r["arguments"])==18 for r in conv),"dynamic conv ABI")
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    require('name: "ssm"' in builder and 'name: "dynamic-conv"' in builder and
            'root.join("mistralrs-core/src/cuda/dynamic_conv_device.cuh")' in builder,
            "builder registration or dependency missing")
    source=(root/"kernel-packages/cuda/dynamic-conv/dynamic_conv.cu").read_text()
    require({r["entry"] for r in conv}==set(re.findall(r'void (dkr_dynamic_conv_\w+)\(',source)),
            "dynamic conv source entry mismatch")
    core=root/"mistralrs-core/src/cuda"
    ffi=(core/"ffi.rs").read_text()
    for name in ("dynamic_conv_f16","dynamic_conv_bf16","dynamic_conv_f32"):
        require(GUARD+"\n    pub(crate) fn "+name+"(" in ffi,"missing FFI fence: "+name)
    require('#[cfg(not(feature = "dkr-ssm"))]\n    pub(crate) fn selective_scan_cuda(' in ffi,
            "SSM FFI declaration not excluded")
    ssm_dispatch=(core/"ssm.rs").read_text()
    require('#[cfg(all(feature = "cuda", not(feature = "dkr-ssm")))]' in ssm_dispatch and
            'super::ssm_dynamic::selective_scan_cuda(' in ssm_dispatch,"SSM typed dispatch missing")
    features=(root/"mistralrs-core/Cargo.toml").read_text()
    closure=features.split("dkr-static-closure-probe = [",1)[1].split("]",1)[0]
    typed=features.split("dkr-typed-operations = [",1)[1].split("]",1)[0]
    require('"dkr-typed-operations"' in closure and '"dkr-ssm"' in typed,
            "closure probe must select SSM catalogue")
    dispatch=(core/"dynamic_conv.rs").read_text()
    matches=list(re.finditer(r'crate::cuda::ffi::\$\w+\s*\(',mask(dispatch)))
    spans=guarded_ranges(dispatch)
    require(len(matches)==1 and all(any(a<=m.start()<b for a,b in spans) for m in matches),
            "unguarded dynamic conv macro FFI")
    require('typed::launch(self,' in dispatch and 'with_prevalidated_launch' in dispatch,
            "typed launch missing")
    # Capture function-pointer selections too: no opening parenthesis required.
    gdn=(core/"gdn.rs").read_text()
    gdn_spans=guarded_ranges(gdn)
    refs={}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(gdn)):
        name=m.group(1)
        if not any(a<=m.start()<b for a,b in gdn_spans):
            refs.setdefault(name,[]).append(gdn.count("\n",0,m.start())+1)
    arch_only={n:refs.pop(n) for n in tuple(refs)
               if n.startswith(("gdn_mma_","mistralrs_flashinfer_gdn_sm90_"))}
    lfm={}
    for path in ("mistralrs-quant/src/lfm3_conv/cuda.rs","mistralrs-quant/src/lfm3_spatial_conv.rs"):
        source=(root/path).read_text()
        lfm[path]=sorted(set(re.findall(r'\bfn (lfm3_(?:spatial_)?conv_(?:f16|bf16|f32))\(',mask(source))))
    remaining=["gdn_recurrence_and_state","gdn_quantized_norm","lfm3_causal_conv","lfm3_spatial_conv",
               "architecture_specific_gdn_compile_time_exclusion"]
    # Prior upper bound contains four SSM manifests, replaced by one.
    physical=BASE_PHYSICAL-3+1
    logical=BASE_LOGICAL+len(conv)
    require(logical<=MAX_LOGICAL,"logical budget exceeded")
    return dict(schema="dkr.r126.s9c2d1.source.v1",status="PARTIAL_SOURCE",
        ssm_reused_profiles=4,ssm_manifest_savings=3,dynamic_conv_profiles=6,
        dynamic_conv_images=1,max_arguments=max(len(r["arguments"]) for r in rows),
        projected_physical_upper=physical,physical_savings_required=max(0,physical-MAX_PHYSICAL),
        projected_logical=logical,logical_headroom=MAX_LOGICAL-logical,
        budget_is_projection=True,gdn_residual_ffi=refs,gdn_arch_provider_refs=arch_only,
        lfm3_residual_declarations=lfm,remaining=remaining,
        compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    p.add_argument("--require-complete",action="store_true")
    a=p.parse_args()
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2D1_SOURCE=PARTIAL ssm_reused=4 physical_savings=3 "
          f"dynamic_conv_profiles=6 max_args=18 projected_physical={r['projected_physical_upper']} "
          f"projected_logical={r['projected_logical']} logical_headroom={r['logical_headroom']} "
          f"gdn_residual={len(r['gdn_residual_ffi'])} arch_provider_refs={len(r['gdn_arch_provider_refs'])} "
          "compilation=false link_qualified=false full_closure=false")
    if a.require_complete and not r["full_closure"]:
        raise SystemExit(2)

if __name__=="__main__":
    main()
