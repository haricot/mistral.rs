#!/usr/bin/env python3
"""S9C.3A.1A.1: existing typed routes / explicit unsupported FP8 boundary.

Re-run the workspace inventory and verify exact declaration exclusions and
qualified legacy reference fences. No compiler, new package or link claim.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from dkr_r126_s9c3a_audit import audit as inventory, Rust

CORE="mistralrs-core/src/cuda/"
QUANT="mistralrs-quant/src/"
TYPES=("f16","bf16","f32")
MOE={"moe_gemm","moe_gemm_wmma","moe_gemm_hfma2","moe_gemv"}
MXFP4={"launch_mxfp4_matmul_"+d for d in TYPES[:2]} | {
    "launch_mxfp4_matmul_wmma_"+d for d in TYPES[:2]} | {
    "launch_mxfp4_moe_grouped_gemm_"+d for d in TYPES[:2]} | {
    "launch_mxfp4_moe_grouped_gemm_wmma_"+d for d in TYPES[:2]} | {"mxfp4_get_max_smem_optin"}
SCALAR={"launch_fp8_to_"+d+"_kernel" for d in TYPES} | {"launch_"+d+"_to_fp8_kernel" for d in TYPES}
BLOCK_MATMUL={"launch_fp8_"+op+"_"+d for op in ("matmul","indexed_moe_gemm") for d in TYPES[:2]}
BLOCK_GAP={"launch_"+op+"_fp8_blockwise_kernel_"+d for op in ("quant","dequant") for d in TYPES}
SOFTMAX={"softmax_with_sinks_"+d for d in TYPES}
CLOSED={
    CORE+"ffi.rs":MOE,
    QUANT+"mxfp4/ffi.rs":MXFP4,
    QUANT+"scalar_fp8/ffi.rs":SCALAR,
    QUANT+"blockwise_fp8/ffi.rs":BLOCK_MATMUL|BLOCK_GAP,
    QUANT+"utils/ffi.rs":SOFTMAX,
}
UTILITY_REMAIN={
    "nonzero_count":18, "bitwise_binary":12, "leftshift":4,
    "gptoss_swiglu":6, "glu":6, "glu_quantized":2, "softcap":3,
}
EXCLUDED={"local_cfg_excluded","module_cfg_excluded"}

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def check_closed(rows,qualified):
    evidence={}
    for path,names in CLOSED.items():
        by_name={name:[x for x in rows if x["path"]==path and x["name"]==name]
                 for name in sorted(names)}
        require(all(len(v)==1 for v in by_name.values()),"missing/duplicate declaration "+path)
        selected=[v[0] for v in by_name.values()]
        require(all(x["classification"] in EXCLUDED for x in selected),"unexcluded declaration "+path)
        if path==CORE+"ffi.rs":
            require(by_name["moe_gemm_hfma2"][0]["link_name"]=="mistralrs_moe_gemm_hfma2","MoE alias")
        component=path.split("/")[0]
        hits=[q for q in qualified if q["site"].startswith(component+"/")
              and q["module"]=="ffi" and q["name"] in names and q["scope"]=="library_source"]
        bad=[q["site"] for q in hits if not q["guard"] and not q["module_excluded"]]
        require(not bad,"unexcluded qualified legacy call "+str(bad))
        evidence[path]=dict(declarations=len(selected),qualified_legacy_sites=len(hits),
                            symbols=sorted(names))
    require(sum(v["declarations"] for v in evidence.values())==32,"32 declaration contract")
    return evidence

def remaining_kind(row):
    name=row["name"] or ""
    if row["path"]==QUANT+"moe/cuda.rs": return "quant_moe"
    if row["path"]!=QUANT+"utils/ffi.rs": return None
    if name.startswith(("count_nonzero_","nonzero_")): return "nonzero_count"
    if name.startswith("bitwise_"): return "bitwise_binary"
    if name.startswith("leftshift_"): return "leftshift"
    if name.startswith("gptoss_swiglu_"): return "gptoss_swiglu"
    if name.startswith(("fused_glu_quantize_","fused_split_glu_quantize_")): return "glu_quantized"
    if name.startswith(("fused_glu_","fused_split_glu_")): return "glu"
    if name.startswith("softcap_"): return "softcap"
    raise ValueError("unclassified utility "+name)

def check_routes(root):
    # These are positive source-route markers, not model/ABI qualification.
    cases={
        CORE+"moe.rs": ('#[cfg(feature = "dkr-moe")]', '#[path = "moe_dynamic.rs"]',
                       '#[cfg(not(feature = "dkr-moe"))]', '#[path = "moe_legacy.rs"]'),
        CORE+"moe_dynamic.rs": ('"moe.grouped_gemm.v1"',),
        QUANT+"mxfp4/ops.rs": ("quantized_gemv_dynamic::try_mxfp4(",
                               "quantized_gemm_dynamic::try_mxfp4(",
                               "DKR_STATIC_CLOSURE: MXFP4 dense signature not admitted"),
        QUANT+"mxfp4/mod.rs": ("crate::try_routed_gather(x, indices, &routed)",
                               "DKR_STATIC_CLOSURE: MXFP4 routed signature not admitted",
                               'return cfg!(feature = "dkr-static-closure-probe") || ffi::HAVE_MXFP4_GEMM_KERNELS;'),
        QUANT+"scalar_fp8/ops.rs": ("super::dynamic::try_cast(",
                                    "DKR_STATIC_CLOSURE: FP8 dequant cast signature not admitted",
                                    "DKR_STATIC_CLOSURE: FP8 quant cast signature not admitted"),
        QUANT+"scalar_fp8/dynamic.rs": ('quant.fp8_e4m3.{direction}.v1',),
        QUANT+"blockwise_fp8/ops.rs": ("quantized_gemv_dynamic::try_fp8(",
                                       "quantized_gemm_dynamic::try_fp8(",
                                       "DKR_STATIC_CLOSURE: blockwise FP8 dequant requires migration",
                                       "DKR_STATIC_CLOSURE: blockwise FP8 quant requires migration",
                                       "DKR_STATIC_CLOSURE: blockwise FP8 weight quant requires migration",
                                       "DKR_STATIC_CLOSURE: blockwise FP8 indexed signature requires typed routed dispatch"),
        QUANT+"blockwise_fp8/mod.rs": ("crate::try_routed_gather(x, indices, &routed)",
                                       "DKR_STATIC_CLOSURE: FP8 dense signature not admitted (no dense materialization)",
                                       "DKR_STATIC_CLOSURE: FP8 routed signature not admitted (no dense materialization)"),
        QUANT+"utils/ops.rs": ('#[cfg(all(feature = "cuda", feature = "dkr-reduce-norm"))]',
                               "crate::softmax_sinks_dynamic::cuda_fwd(",
                               '#[cfg(all(feature = "cuda", not(feature = "dkr-reduce-norm")))]'),
        QUANT+"softmax_sinks_dynamic.rs": ("attention.softmax_sinks.b64.v1",
                                            "attention.softmax_sinks.b1024.v1"),
        "kernel-packages/cuda/reduce-norm/build.py": ("attention.softmax_sinks.b{block}.v1",),
        "kernel-packages/cuda/fp8-cast/build.py": ("quant.fp8_e4m3.{direction}.v1",),
    }
    for path,markers in cases.items():
        source=(root/path).read_text()
        for marker in markers:
            require(marker in source,path+": missing "+marker)
    return list(cases)

def self_test():
    rows=[]
    for path,names in CLOSED.items():
        for name in names:
            rows.append(dict(path=path,name=name,classification="local_cfg_excluded",
                             link_name="mistralrs_moe_gemm_hfma2" if name=="moe_gemm_hfma2" else name))
    q=dict(site=CORE+"moe_legacy.rs:1",module="ffi",name="moe_gemm",
           scope="library_source",guard=None,module_excluded=True)
    check_closed(rows,[q])
    failures=0
    bad_rows=[dict(x) for x in rows]
    bad_rows[0]["classification"]="review_legacy_candidate"
    bad_alias=[dict(x) for x in rows]
    next(x for x in bad_alias if x["name"]=="moe_gemm_hfma2")["link_name"]="moe_gemm_hfma2"
    for r,refs in ((bad_rows,[q]),(rows[:-1],[q]),(rows+[rows[0]],[q]),
                   (bad_alias,[q]),(rows,[dict(q,module_excluded=False)])):
        try: check_closed(r,refs)
        except ValueError: failures+=1
        else: raise AssertionError("invalid legacy fence admitted")
    # Exercise the actual lexer on whole-block and declaration-level gates.
    for source in ('#[cfg(not(feature="strict"))] extern "C" { pub fn f(); }',
                   'extern "C" { #[cfg(not(feature="strict"))] pub fn f(); }'):
        r=Rust("fixture.rs",source,{"strict"},{"strict"})
        require(len(r.externs())==1 and r.externs()[0]["guard"] is not None,"lexer exclusion fixture")
    require(remaining_kind(dict(path=QUANT+"utils/ffi.rs",name="fused_split_glu_quantize_bf16"))=="glu_quantized",
            "quantized GLU separated from ordinary GLU")
    require(len(BLOCK_GAP)==6 and sum(UTILITY_REMAIN.values())==51,"gap inventory")
    print(f"DKR_R126_S9C3A1A1_SELF_TEST=PASS negative_cases={failures} fenced=32 typed_route_candidates=26 explicit_unsupported=6")

def audit(root):
    r=inventory(root)
    require(r["previous_inventory_closed"]==60 and r["previous_inventory_residual"]==0,"prior closure")
    require(r["projected_master_logical"]==4268 and r["projected_master_physical_upper"]==552,"master budget")
    require(sum(r["extern_counts"].values())==491,"baseline total extern inventory drift")
    evidence=check_closed(r["extern_inventory"],r["qualified_ffi_inventory"])
    routes=check_routes(root)
    active=r["source_feature_intent"]
    require("dkr-moe" in active["mistralrs-core"] and
            {"dkr-reduce-norm","dkr-fp8-cast","dkr-quant-gemv","dkr-quant-gemm","dkr-routed-quant",
             "dkr-static-closure-probe"}.issubset(active["mistralrs-quant"]),"typed feature intent")
    remaining=[x for x in r["extern_inventory"] if x["classification"].startswith("review_")]
    group=Counter(k for x in remaining if (k:=remaining_kind(x)) is not None)
    require(group==Counter({**UTILITY_REMAIN,"quant_moe":5}),"remaining group-A family drift "+str(group))
    require(len(remaining)==r["review_candidates"]==126,"global review inventory drift")
    return dict(schema="dkr.r126.s9c3a1a1.source.v1",source_head=r["source_head"],
                status="SOURCE_TRANCHE_COMPLETE",baseline_review=158,baseline_group_a=88,
                guarded_legacy_declarations=32,typed_route_candidates=26,explicit_unsupported=6,
                unsupported_symbols=sorted(BLOCK_GAP),remaining_group_a=56,remaining_global=126,
                remaining_group_a_families=dict(group),guard_evidence=evidence,positive_route_files=routes,
                profiles_added=0,images_added=0,projected_master_logical=4268,
                projected_master_physical_upper=552,active_limits_unchanged=True,
                cargo_lock_dirty=r["cargo_lock_dirty"],source_inventory=r,
                compilation=False,link_qualified=False,numeric_qualified=False,
                model_scope_fit_proven=False,group_a_complete=False,full_closure=False)

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
    for family,n in sorted(r["remaining_group_a_families"].items()):
        print(f"DKR_R126_S9C3A1A1_REMAIN family={family} records={n}")
    print("DKR_R126_S9C3A1A1_SOURCE=TRANCHE_COMPLETE guarded=32 typed_route_candidates=26 explicit_unsupported=6 group_a_remaining=56 review_remaining=126 profiles_added=0 images_added=0 projected_master_logical=4268 projected_master_physical=552 compilation=false link_qualified=false group_a_complete=false full_closure=false")

if __name__=="__main__":
    main()
