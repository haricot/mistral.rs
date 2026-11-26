#!/usr/bin/env python3
"""Source-only audit of specialized GDN recurrence and deferred-state closure."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

BASE_LOGICAL=3973
BASE_PHYSICAL=531
MIGRATED=("gdn_speculative_recurrence_checkpoints",
          "gdn_deferred_recurrence_rmsnorm_gate_value_major_128",
          "gdn_flush_deferred_state_value_major_128")
ARCH=("gdn_mma_prefill_init","gdn_mma_prefill","gdn_mma_fused_prefill",
      "mistralrs_flashinfer_gdn_sm90_launch","mistralrs_flashinfer_gdn_sm90_workspace_size")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
CTYPE={"f16":"__half","bf16":"__nv_bfloat16","f32":"float","fp8_e4m3":"gdn_fp8_e4m3",
       "u32":"uint32_t","i32":"int","i64":"int64_t","u64":"uint64_t"}

def require(ok,message):
    if not ok:
        raise ValueError(message)

def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def check_entries(producer,profiles,image):
    sections=re.split(r"^#(?:if|elif) DKR_GDN_SPEC_IMAGE_ID == ([123])\s*$",producer,flags=re.M)
    blocks={int(sections[i]):sections[i+1] for i in range(1,len(sections),2)}
    require(set(blocks)=={1,2,3},"image assignment")
    matches=re.findall(r'extern "C" __global__ (?:__launch_bounds__\([^)]*\)\s*)?\s*void (\w+)\((.*?)\)\s*\{',blocks[image],re.S)
    entries=dict(matches)
    require(len(matches)==len(entries)==len(profiles),"explicit CUDA entry count")
    require(set(entries)=={r["entry"] for r in profiles},"CUDA/manifest symbol inventory")
    for r in profiles:
        actual=[" ".join(v.strip().split()) for v in entries[r["entry"]].split(",")]
        expected=[]
        for arg in r["arguments"]:
            ty=CTYPE[arg["dtype"]]
            if arg["kind"]=="tensor":
                ty=("const " if arg["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+arg["name"])
        require(actual==expected,"CUDA/manifest argument order/type: "+r["entry"])

def audit(root):
    pkg=root/"kernel-packages/cuda/gdn-speculative"
    module=module_at(pkg/"build.py","gdn_speculative_plan")
    producer=(pkg/"gdn_speculative.cu").read_text()
    require(module.images(61)==(1,2) and module.images(80)==(1,2,3),"modern-only deferred image")
    try:
        module.plan(61,3)
    except ValueError:
        pass
    else:
        raise ValueError("SM61 admitted deferred image")
    manifests={}
    all_profiles=[]
    for image,expected,maxargs in ((1,18,23),(2,36,31),(3,5,30)):
        cc=80 if image==3 else 61
        profiles=module.plan(cc,image)
        require(len(profiles)==expected and max(len(r["arguments"]) for r in profiles)==maxargs,"profile/ABI budget")
        require(all(r["target_cc"]==cc and len(r["arguments"])<=32 for r in profiles),"target/ABI limit")
        require(len({r["artifact"] for r in profiles})==1,"one image per shard")
        check_entries(producer,profiles,image)
        paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
            dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
            for r in profiles])
        size=len((json.dumps(paired,indent=2)+"\n").encode())
        require(size<=256*1024,"paired manifest size")
        manifests[str(image)]=dict(profiles=expected,max_arguments=maxargs,paired_manifest_bytes=size)
        if image!=3:
            all_profiles.extend(profiles)
        if image==2:
            for r in profiles:
                args={a["name"]:a for a in r["arguments"]}
                op=r["operation"]
                require(("out8" in args)==(".fp8." in op) and ("scales" in args)==(".fp8." in op),"specialized output")
                require(("tk" in args)==(".batch." in op) and ("pk" in args)==(".direct." in op),"specialized transition journal")
                require(not ("tk" in args and "pk" in args),"mixed transition signature")
                for n in ("b01","b2a0","a12","g01","g23","batch_seq","kh_vh","lanes_tiled"):
                    require(args[n]["kind"]=="scalar" and args[n]["dtype"]=="u64","packed scalar ABI "+n)
                require(not any("table" in a["name"] for a in r["arguments"]),"opaque pointer table")
    require(len(all_profiles)==54,"SM61 profile count")
    # Recheck the immediately preceding checkpoint package plan as a cumulative gate.
    cp=module_at(root/"kernel-packages/cuda/gdn-checkpoint/build.py","gdn_checkpoint_plan")
    require(len(cp.plan(61))==22 and max(len(r["arguments"]) for r in cp.plan(61))==22,"prior checkpoint inventory")
    core=root/"mistralrs-core/src/cuda"
    source=(core/"gdn.rs").read_text()
    ffi=(core/"ffi.rs").read_text()
    spans=guarded_ranges(source)
    guarded,residual,arch={},{},{}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(source)):
        name=m.group(1)
        group=guarded if any(a<=m.start()<b for a,b in spans) else arch if name in ARCH else residual
        group.setdefault(name,[]).append(source.count("\n",0,m.start())+1)
    require(not residual,"ordinary GDN residual FFI: "+str(sorted(residual)))
    # Historical B.2B permitted these five references. C.3C closes all five.
    # Accept either complete state, never a partial/mixed migration.
    closed_arch={name:guarded[name] for name in ARCH if name in guarded}
    require((set(arch)==set(ARCH) and not closed_arch) or
            (not arch and set(closed_arch)==set(ARCH)), "architecture-provider inventory")
    for name in MIGRATED:
        require(name in guarded and GUARD+"\n    pub(crate) fn "+name+"(" in ffi,"missing FFI guard "+name)
    adapter=(core/"gdn_speculative_dynamic.rs").read_text()
    common=(core/"gdn_preparation_dynamic.rs").read_text()
    header=(core/"gdn_speculative_device.cuh").read_text()
    legacy=(core/"gdn.cu").read_text()
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    require(all("gdn_speculative_device.cuh" in s for s in (producer,legacy,builder)),"shared body/cache dependency")
    require('name: "gdn-speculative"' in builder and 'pkg.name == "gdn-speculative"' in builder and
            "let ids=if cc>=80{&[1,2,3][..]}else{&[1,2][..]}" in builder,"Cudaforge architecture selection")
    require("pointers.len()+scalars.len()>32" in common,"total ABI bound")
    require("u32::try_from(a)" in adapter and "u32::try_from(b)" in adapter and
            "u64::from(a)|(u64::from(b)<<32)" in adapter,"checked scalar pair encoding")
    require(header.count("active_slot >= (uint32_t)pool_capacity")==3,"pool capacity fences")
    require("invalid_cursor ? NAN : inverse_quant_scale" in header and
            "if (cursor >= GDN_DEFERRED_STATE_DEPTH) return;" in header and
            "deferred_cursor[active_slot] < GDN_DEFERRED_STATE_DEPTH" in header,
            "corrupt cursor preservation")
    for name,finalizer in (("launch_deferred","advance"),("launch_flush","clear")):
        start=adapter.index("pub(super) fn "+name+"(")
        end=adapter.index("\n}\n",start)
        sub=adapter[start:end]
        require(sub.count("p::preflight(")==2 and sub.count("p::launch_on_stream(")==2 and
                sub.index('p::preflight(device,"gdn.spec.deferred.'+finalizer)<sub.index("p::launch_on_stream("),
                "deferred preflight order "+name)
    require("reject_storage_aliases" in adapter and source.count("speculative_dynamic::reject_storage_aliases(")==2,
            "deferred lock/alias validation")
    require("d::launch_recurrence(" in source and "d::launch_deferred(" in source and "d::launch_flush(" in source,
            "typed dispatch sites")
    runtime=(root/"mistralrs-kernel-runtime/src/universal.rs").read_text()
    require("MAX_ARGUMENTS: usize = 32" in runtime and "MAX_CATALOG_PROFILES: usize = 4096" in runtime and
            "MAX_DISCOVERED_PACKAGES: usize = 512" in runtime,"runtime limits unchanged")
    logical=BASE_LOGICAL+len(all_profiles)
    physical=BASE_PHYSICAL+2
    require(logical<=4096,"logical budget")
    return dict(schema="dkr.r126.s9c2d3b2b.source.v1",status="ORDINARY_GDN_SOURCE_COMPLETE",
        profiles=len(all_profiles),images=2,modern_profiles=59,modern_images=3,max_arguments=31,
        image_contracts=manifests,migrated_ffi={n:guarded[n] for n in MIGRATED},
        gdn_residual_ffi=residual,arch_provider_refs=arch,arch_provider_guarded=closed_arch,projected_physical_upper=physical,
        physical_savings_required=physical-512,projected_logical=logical,logical_headroom=4096-logical,
        budget_is_projection=True,compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False,
        outstanding_contracts=["unique_slots_or_checkpoint_groups","same_stream_order",
            "epoch_wrap_collision","device_failure_after_first_launch","modern_deferred_gpu_qualification"])

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
    print(f"DKR_R126_S9C2D3B2B_SOURCE=COMPLETE_ORDINARY profiles={r['profiles']} images={r['images']} "
          f"guarded_ffi={len(r['migrated_ffi'])} gdn_residual={len(r['gdn_residual_ffi'])} "
          f"arch_provider_refs={len(r['arch_provider_refs'])} max_args={r['max_arguments']} "
          f"projected_physical={r['projected_physical_upper']} projected_logical={r['projected_logical']} "
          f"logical_headroom={r['logical_headroom']} compilation=false link_qualified=false full_closure=false")
if __name__=="__main__":
    main()
