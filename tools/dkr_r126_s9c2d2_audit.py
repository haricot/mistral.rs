#!/usr/bin/env python3
"""S9C.2D.2 source/ABI/resource audit; never compilation or numeric qualification."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

BASE_PHYSICAL=524
BASE_LOGICAL=3855
MAX_PHYSICAL=512
MAX_LOGICAL=4096
SYMBOLS=("gated_delta_rule_recurrence","warp_gated_delta_rule_recurrence",
         "vmajor_warp_gated_delta_rule_recurrence","vmajor_grouped_warp_gated_delta_rule_recurrence",
         "chunked_gated_delta_rule_recurrence","vmajor_chunked_gated_delta_rule_recurrence",
         "causal_conv1d_update","causal_conv1d_full")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,message):
    if not ok:
        raise ValueError(message)

def plan(root,cc):
    path=root/"kernel-packages/cuda/gdn-recurrence/build.py"
    spec=importlib.util.spec_from_file_location("gdn_plan",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan(cc)

def guarded_refs(source):
    spans=guarded_ranges(source)
    refs={}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(source)):
        n=m.group(1)
        if n in SYMBOLS:
            require(any(a<=m.start()<b for a,b in spans),f"unguarded {n}")
            refs[n]=refs.get(n,0)+1
    require(set(refs)==set(SYMBOLS),"incomplete migrated FFI inventory")
    return refs

def audit(root):
    images=plan(root,61)
    rows=[p for group in images.values() for p in group]
    modern=[p for group in plan(root,80).values() for p in group]
    require(len(images)==3 and len(rows)==49 and len(modern)==55,"unexpected profile count")
    require(len({r["implementation_id"] for r in rows})==49,"duplicate implementation")
    require(all(r["target_cc"]==61 and len(r["arguments"])<=32 and r["shared_mem_bytes"]<=49152 for r in rows),
            "SM61 ABI or resource limit exceeded")
    for group in images.values():
        require(1<=len(group)<=64 and len({r["artifact"] for r in group})==1,"image grouping")
        paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
            dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
            for r in group])
        require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,"paired manifest bytes")
    require(sum(r["shared_mem_bytes"]==50176 for r in modern)==6,"BT64 K128 variants lost")
    require(sum(r["shared_mem_bytes"]==21248 for r in rows)==6,"BT32 K128 variants lost")
    require(sum(r["shared_mem_bytes"]==33536 for r in rows)==3,"BT64 K64 variants lost")
    require(sum("gdn.conv." in r["operation"] for r in rows)==10,"convolution variants lost")
    core=root/"mistralrs-core/src/cuda"
    dispatch=(core/"gdn.rs").read_text()
    ffi=(core/"ffi.rs").read_text()
    migrated=guarded_refs(dispatch)
    for name in SYMBOLS:
        require(GUARD+"\n    pub(crate) fn "+name+"(" in ffi,"missing declaration guard "+name)
    require("recurrence_dynamic::recurrence(" in dispatch and "recurrence_dynamic::convolution(" in dispatch,
            "typed dispatch missing")
    header=(core/"gdn_recurrence_conv_device.cuh").read_text()
    legacy=(core/"gdn.cu").read_text()
    package=(root/"kernel-packages/cuda/gdn-recurrence/gdn_recurrence.cu").read_text()
    require('gdn_recurrence_conv_device.cuh' in legacy and 'gdn_recurrence_conv_device.cuh' in package,"device bodies not shared")
    # Check the three barrier-using kernels individually, not the warp-only kernels.
    for name in ("gated_delta_rule_recurrence_kernel_tiled","gated_delta_rule_recurrence_kernel_fallback","chunked_gated_delta_rule_kernel"):
        start=header.index(name+"(")
        end=header.index("\n}\n",start)
        body=header[start:end]
        require("const bool active = v_idx < v_dim;" in body and "__syncthreads()" in body and
                "if (v_idx >= v_dim)" not in body and "if (active) out_bh[" in body and
                "if (active) state_bh[" in body,
                "partial tile synchronization fence missing: "+name)
    require("CHUNK(chunk32k128,32,128,false)" in package and "CHUNK(chunk64k128,64,128,false)" in package and
            "__CUDA_ARCH__ >= 700" in package,"chunk entries missing")
    adapter=(core/"gdn_recurrence_dynamic.rs").read_text()
    require("MAX_SHARED_MEMORY_PER_BLOCK_OPTIN" in adapter and "prevalidate_launch_with_optin" in adapter and
            "with_prevalidated_launch" in adapter and "LaunchPlanCache" in adapter,"resource-safe adapter missing")
    require("Tensor::from_vec(vec![0u32],1,device)" in adapter and "slot_tensor" in adapter,"typed optional-slot contract")
    runtime=(root/"mistralrs-kernel-runtime/src/universal_cuda.rs").read_text()
    require("CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES" in runtime and
            "CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES" in runtime,"runtime opt-in missing")
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    require('name: "gdn-recurrence"' in builder and "DKR_GDN_DTYPE={id}" in builder and
            "gdn_recurrence_conv_device.cuh" in builder,"Cudaforge registration missing")
    require("dispatch_gdn_chunk_resource" in legacy,"legacy resource gate missing")
    residual={}
    arch={}
    spans=guarded_ranges(dispatch)
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(dispatch)):
        n=m.group(1)
        if any(a<=m.start()<b for a,b in spans):
            continue
        group=arch if n.startswith(("gdn_mma_","mistralrs_flashinfer_gdn_sm90_")) else residual
        group.setdefault(n,[]).append(dispatch.count("\n",0,m.start())+1)
    physical=BASE_PHYSICAL+len(images)
    logical=BASE_LOGICAL+len(rows)
    require(logical<=MAX_LOGICAL,"logical budget exceeded")
    return dict(schema="dkr.r126.s9c2d2.source.v1",status="PARTIAL_SOURCE",
                images=len(images),profiles=len(rows),modern_profiles=len(modern),
                recurrence_profiles=39,convolution_profiles=10,max_arguments=max(len(r["arguments"]) for r in rows),
                migrated_ffi=migrated,gdn_residual_ffi=residual,arch_provider_refs=arch,
                projected_physical_upper=physical,physical_savings_required=max(0,physical-MAX_PHYSICAL),
                projected_logical=logical,logical_headroom=MAX_LOGICAL-logical,budget_is_projection=True,
                compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2D2_SOURCE=PARTIAL profiles={r['profiles']} images={r['images']} "
          f"guarded_ffi={len(r['migrated_ffi'])} gdn_residual={len(r['gdn_residual_ffi'])} "
          f"arch_provider_refs={len(r['arch_provider_refs'])} max_args={r['max_arguments']} "
          f"projected_physical={r['projected_physical_upper']} projected_logical={r['projected_logical']} "
          f"logical_headroom={r['logical_headroom']} compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
