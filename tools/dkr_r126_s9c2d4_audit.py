#!/usr/bin/env python3
"""LFM3 source closure and residual budget inventory; never compiles CUDA."""
import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from dkr_r126_s9c2d3c_audit import audit as prior_audit, strip_comments
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
CTYPE={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","u64":"uint64_t","u32":"uint32_t"}

def require(ok,message):
    if not ok:
        raise ValueError(message)

def module_at(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def entries(source,rows):
    matches=re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{',source,re.S)
    found=dict(matches)
    require(len(matches)==len(found)==len(rows),"CUDA entry count")
    require(set(found)=={r["entry"] for r in rows},"CUDA entry inventory")
    for r in rows:
        actual=[" ".join(x.strip().split()) for x in found[r["entry"]].split(",")]
        expected=[]
        for a in r["arguments"]:
            ty=CTYPE[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+a["name"])
        require(actual==expected,"CUDA/manifest ABI mismatch: "+r["entry"])
        args={a["name"]:a for a in r["arguments"]}
        if ".causal." in r["operation"]:
            bias=r["operation"].endswith(".bias.v1")
            require(("bias" in args)==bias and ("bs" in args)==bias,"bias specialization")

def check_legacy(source,prefix):
    code=strip_comments(source)
    spans=guarded_ranges(code)
    for dt in ("f32","f16","bf16"):
        name=prefix+dt
        matches=list(re.finditer(r'\b'+name+r'\b',mask(code)))
        require(len(matches)==2 and all(any(a<=m.start()<b for a,b in spans) for m in matches),
                "legacy FFI declaration/invocation guard: "+name)

def dflash_dtype_inventory(cuda):
    """Read the actual definition, not an unrelated or orphaned header."""
    legacy=strip_comments((cuda/"dflash_selector.cu").read_text())
    include=re.findall(r'^\s*#\s*include\s+"dflash_selector_device\.cuh"\s*$',legacy,re.M)
    header=cuda/"dflash_selector_device.cuh"
    if include:
        require(len(include)==1 and header.is_file(),"DFlash selector shared header")
        require(not re.search(r'\benum\s+DFlashSelectorDType\b',legacy),"duplicate DFlash selector dtype definition")
        source=strip_comments(header.read_text())
    else:
        require(not header.exists(),"DFlash selector header not included")
        source=legacy
    enums=re.findall(r'\benum\s+DFlashSelectorDType\s*:\s*int\s*\{([^}]+)\}\s*;',source,re.S)
    require(len(enums)==1,"DFlash selector dtype definition")
    values=re.findall(r'\b(DFLASH_SELECTOR_\w+)\s*=\s*(\d+)\b',enums[0])
    require(values==[("DFLASH_SELECTOR_F32","0"),("DFLASH_SELECTOR_BF16","1")],
            "DFlash selector dtype inventory")
    return "shared_header" if include else "legacy_inline"

def dflash_dtype_inventory_self_test():
    enum="enum DFlashSelectorDType : int { DFLASH_SELECTOR_F32 = 0, DFLASH_SELECTOR_BF16 = 1, };"
    include='#include "dflash_selector_device.cuh"\n'
    with tempfile.TemporaryDirectory() as tmp:
        cuda=Path(tmp)
        legacy=cuda/"dflash_selector.cu"
        header=cuda/"dflash_selector_device.cuh"
        legacy.write_text(enum)
        require(dflash_dtype_inventory(cuda)=="legacy_inline","inline fixture")
        legacy.write_text(include)
        header.write_text(enum)
        require(dflash_dtype_inventory(cuda)=="shared_header","shared fixture")
        failures=0
        for legacy_text,header_text in (
            (include,None),
            ("// "+include,enum),
            (include+enum,enum),
            (include,enum.replace("BF16 = 1","BF16 = 2")),
            (include,"// "+enum),
        ):
            legacy.write_text(legacy_text)
            if header_text is None:
                header.unlink(missing_ok=True)
            else:
                header.write_text(header_text)
            try:
                dflash_dtype_inventory(cuda)
            except ValueError:
                failures+=1
            else:
                raise AssertionError("invalid DFlash dtype source accepted")
        require(failures==5,"DFlash inventory negative fixtures")
    return failures

def budget_inventory(root,prior):
    core=root/"mistralrs-core/src"
    ffi=(core/"cuda/ffi.rs").read_text()
    spans=guarded_ranges(strip_comments(ffi))
    declarations=[m.group(1) for m in re.finditer(r'\bpub\(crate\)\s+fn\s+(\w+)\s*\(',mask(ffi))
                  if not any(a<=m.start()<b for a,b in spans)]
    require(declarations==prior["remaining_core_ffi_declarations"],"unexpected core declaration inventory drift")
    # Scan exact symbol tokens, including macro invocations. This is an
    # inventory of lexical candidates, not a claim about Rust cfg expansion.
    mentions={n:[] for n in declarations}
    pattern=re.compile(r'\b(?:'+"|".join(map(re.escape,declarations))+r')\b') if declarations else None
    for path in sorted(core.rglob("*.rs")):
        if path==core/"cuda/ffi.rs":
            continue
        source=path.read_text()
        code=mask(source)
        guarded=guarded_ranges(strip_comments(source))
        for m in (pattern.finditer(code) if pattern is not None else ()):
            mentions[m.group()].append(dict(path=str(path.relative_to(root)),line=source.count("\n",0,m.start())+1,
                explicit_negative_closure_guard=any(a<=m.start()<b for a,b in guarded)))
    sort=(core/"cuda/sort.cu").read_text()
    experts=[int(x) for x in re.findall(r'LAUNCH_MOE_ROUTER_TOPK\((\d+)\)',sort)]
    require(experts==[1,2,4,8,16,32,64,128,256,512,576],"MoE router specialization inventory drift")
    combos=re.findall(r'launch_moe_router_topk<T,\s*(true|false),\s*(true|false)>',sort)
    require(len(set(combos))==4,"MoE bias/scale combination inventory")
    dflash_dtype_source=dflash_dtype_inventory(core/"cuda")
    runtime=(root/"mistralrs-kernel-runtime/src/universal.rs").read_text()
    require("MAX_CATALOG_PROFILES: usize = 4096" in runtime and "MAX_DISCOVERED_PACKAGES: usize = 512" in runtime
            and "MAX_ARGUMENTS: usize = 32" in runtime,"catalogue limits")
    element_types=runtime.split("pub enum ElementType {",1)[1].split("}",1)[0]
    ssm=(core/"cuda/ssm.rs").read_text()
    cargo=(root/"mistralrs-core/Cargo.toml").read_text()
    require("super::ssm_dynamic::selective_scan_cuda(" in ssm and
            'not(feature = "dkr-ssm")' in ssm and
            '"dkr-ssm"' in cargo.split("dkr-typed-operations = [",1)[1].split("]",1)[0],
            "SSM typed reuse contract")
    return dict(remaining_core_ffi_declarations=declarations,lexical_mentions=mentions,
        inventory_is_not_cfg_or_link_proof=True,ssm_typed_route_present=True,
        moe_router=dict(expert_sizes=experts,activation_dtypes=3,bias_scale_combinations=4,
                        naive_template_profiles=len(experts)*3*4,
                        recommendation="dispatch expert-count variants within a typed device entry; preserve existing templates"),
        dflash=dict(dtype_source=dflash_dtype_source,input_dtypes=["f32","bf16"],independent_tensors=3,selector_modes=2,
                    naive_typed_profiles=2**3*2,recommendation="verify reachable dtype combinations before allocating profiles"),
        argsort=dict(legacy_f64_present="ASORT_OP(double," in sort,
                     typed_f64_supported=bool(re.search(r'\bF64\b',element_types)),
                     recommendation="audit live uses and semantics before deciding F64 admission"),
        graph_copy=dict(recommendation="inspect driver DtoD and 2D copy primitives; do not spend profiles for pure copies"),
        final_remaining_profile_requirement=None,budget_fit_proven=False)

def audit(root):
    prior=prior_audit(root)
    pkg=root/"kernel-packages/cuda/lfm3-conv"
    module=module_at(pkg/"build.py","lfm3_conv_plan")
    rows=module.plan(61)
    require(len(rows)==15 and max(len(r["arguments"]) for r in rows)==20,"LFM3 ABI/profile budget")
    require(len({r["artifact"] for r in rows})==1,"LFM3 physical shard")
    require(all(r["block"]==[256,1,1] and r["shared_mem_bytes"]==0 for r in rows),"LFM3 launch geometry")
    entries((pkg/"lfm3_conv.cu").read_text(),rows)
    entries((pkg/"lfm3_conv.cu").read_text(),module.plan(80))
    paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
        dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
        for r in rows])
    size=len((json.dumps(paired,indent=2)+"\n").encode())
    require(size<=256*1024,"paired manifest size")
    quant=root/"mistralrs-quant"
    temporal=(quant/"src/lfm3_conv/cuda.rs").read_text()
    spatial=(quant/"src/lfm3_spatial_conv.rs").read_text()
    check_legacy(temporal,"lfm3_conv_")
    check_legacy(spatial,"lfm3_spatial_conv_")
    require("dkr::causal(key,device," in temporal and "dkr::spatial(key,device," in spatial,"typed dispatch")
    require("dkr::span(projected_layout,x.len())" in temporal and "dkr::span(pl,x.len())" in spatial,
            "strided span checks")
    adapter=(quant/"src/lfm3_dynamic.rs").read_text()
    require("LaunchPlanCache<Key,PrevalidatedLaunchPlan>" in adapter and
            "layout.start_offset().checked_add(len)" in adapter,"cached typed plans and storage bounds")
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    require('name: "lfm3-conv", source: "lfm3_conv.cu", fast_math: true' in builder and
            'pkg.name == "lfm3-conv"' in builder,"Cudaforge registration")
    for name in ("lfm3_conv","lfm3_spatial_conv"):
        header=name+"_device.cuh"
        require(header in builder and header in (pkg/"lfm3_conv.cu").read_text() and
                header in (quant/"kernels/ops"/(name+".cu")).read_text() and
                (quant/"kernels/ops"/header).is_file(),"shared body and cache dependency "+header)
    budget=budget_inventory(root,prior)
    logical=prior["projected_logical"]+len(rows)
    physical=prior["projected_physical_upper"]+1
    require(logical==4042 and physical==534,"projection baseline")
    return dict(schema="dkr.r126.s9c2d4.source.v1",status="LFM3_SOURCE_COMPLETE",
        profiles=15,images=1,max_arguments=20,guarded_legacy_ffi=6,paired_manifest_bytes=size,
        projected_logical=logical,logical_headroom=4096-logical,projected_physical_upper=physical,
        physical_savings_required=physical-512,residual_budget=budget,
        compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False,budget_is_projection=True)

def self_test():
    source="\n".join(GUARD+"\nfn old_"+dt+"(){ lfm3_conv_"+dt+"(); lfm3_conv_"+dt+"(); }" for dt in ("f32","f16","bf16"))
    check_legacy(source,"lfm3_conv_")
    failures=0
    for bad in (source.replace(GUARD,"",1),source.replace(GUARD,"// "+GUARD,1),
                source+"\nfn leak(){ lfm3_conv_f32(); }"):
        try:
            check_legacy(bad,"lfm3_conv_")
        except ValueError:
            failures+=1
        else:
            raise AssertionError("legacy FFI leak accepted")
    dflash_failures=dflash_dtype_inventory_self_test()
    print(f"DKR_R126_S9C2D4_AUDIT_SELF_TEST=PASS negative_cases={failures} dflash_negative_cases={dflash_failures}")

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
    print("DKR_R126_S9C2D4_SOURCE=COMPLETE profiles=15 images=1 guarded_ffi=6 max_args=20 "
          "projected_physical=534 projected_logical=4042 logical_headroom=54 "
          "remaining_budget_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
