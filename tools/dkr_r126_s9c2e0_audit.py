#!/usr/bin/env python3
"""S9C.2E.0: source reachability candidates, specialization and launch budget."""
import argparse
import json
import re
from pathlib import Path
from dkr_r126_s9c2d4_audit import audit as previous
from dkr_r126_gguf_residual_audit import mask

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def text(root,path):
    return (root/path).read_text()

CORE="mistralrs-core/src/"
RUNTIME="mistralrs-kernel-runtime/"
REUSE={
    "dkr-indexed-copy": ["indexed_row_copy_"+d for d in ("bf16","f16","f32")],
    "dkr-input-pad": ["pad_decode_input_u32"],
    "dkr-logits": ["apply_sparse_penalties_f32","apply_sparse_logits_bias_f32"],
    "dkr-causal-mask": ["apply_causal_mask_f32"],
    "dkr-rms-variants": [p+d for p in ("rms_norm_residual_","rms_norm_residual_then_rms_norm_") for d in ("bf16","f16","f32")],
    "dkr-add-rms-norm": ["add_rms_norm_"+d for d in ("bf16","f16","f32")],
    "dkr-ssm": ["selective_scan_cuda"],
}
GRAPH={"cuda_graph_copy_bytes","cuda_graph_copy_2d_bytes"}
ARG_SORT={"asort_"+direction+"_"+dtype for direction in ("asc","desc") for dtype in ("f32","f16","bf16","f64","u8","u32","i64")}
SAMPLING_GROUPS={
    "small_topk":{"topk_"+d for d in ("f32","f16","bf16")},
    "router":{"moe_router_topk_"+d for d in ("f32","f16","bf16")},
    "large_topk":{"topk_large_f32","topk_large_f32_packed"}|
        {"topk_large_"+d+"_packed_batched" for d in ("f32","f16","bf16")},
    "ranked":{"topk_large_ranked_"+d+"_packed_batched" for d in ("f32","f16","bf16")},
    "workspace_query":{"topk_large_ranked_state_words_per_row"},
    "sample":{"sample_ranked_topk"},
    "dflash":{"dflash_greedy_select","dflash_sample_select"},
    "rejection":{"sparse_rejection_categorical_f32","sparse_rejection_topk_f32"},
    "top1":{"top1_large_"+d+"_packed"+suffix for d in ("f32","f16","bf16") for suffix in ("","_batched")},
    "categorical":{"categorical_large_f32_packed_batched"},
}

def partition(declarations,closed_reuse=(),closed_router=(),closed_large=()):
    groups={"already_typed":set().union(*REUSE.values()),"argsort":ARG_SORT,"graph":GRAPH,**SAMPLING_GROUPS}
    require(sum(len(v) for v in groups.values())==len(set().union(*groups.values())),"overlapping symbol partition")
    closed=set(closed_reuse)
    require(closed in (set(),groups["already_typed"]),"partial reuse closure")
    router=set(closed_router)
    require(router in (set(),groups["router"]|groups["small_topk"]),"partial router/top-k closure")
    require(not router or closed==groups["already_typed"],"router closure before reuse closure")
    large=set(closed_large)
    all_large=groups["large_topk"]|groups["ranked"]|groups["workspace_query"]
    require(large in (set(),all_large),"partial large top-k closure")
    require(not large or router==groups["router"]|groups["small_topk"],"large top-k closure before router closure")
    require(set(declarations)==set().union(*groups.values())-closed-router-large and len(declarations)==60-len(closed)-len(router)-len(large),"residual inventory drift")
    return {k:sorted(v) for k,v in groups.items()}

def evidence(source,markers,label):
    for marker in markers:
        require(marker in source,label+": missing "+marker)
    return list(markers)

def reuse_evidence(root):
    cargo=text(root,"mistralrs-core/Cargo.toml")
    strict=cargo.split("dkr-static-closure-probe = [",1)[1].split("]",1)[0]
    typed=cargo.split("dkr-typed-operations = [",1)[1].split("]",1)[0]
    require('"dkr-typed-operations"' in strict,"strict profile no longer includes typed ops")
    ops=text(root,CORE+"ops.rs")
    checks={
        "dkr-indexed-copy":("cuda/indexed_copy.rs",['#[cfg(feature = "dkr-indexed-copy")]','#[path = "indexed_copy_dynamic.rs"]','#[cfg(not(feature = "dkr-indexed-copy"))]','#[path = "indexed_copy_legacy.rs"]']),
        "dkr-input-pad":("cuda/input_packing.rs",['#[cfg(feature = "dkr-input-pad")]','with_pad_plan(','#[cfg(not(feature = "dkr-input-pad"))]','super::ffi::pad_decode_input_u32(']),
        "dkr-logits":("ops.rs",['#[cfg(feature = "dkr-logits")]','#[cfg(not(feature = "dkr-logits"))]','ffi::apply_sparse_penalties_f32(','ffi::apply_sparse_logits_bias_f32(']),
        "dkr-causal-mask":("ops.rs",['#[cfg(all(feature = "cuda", feature = "dkr-causal-mask"))]','pub(crate) use crate::cuda::causal_mask_dynamic::','#[cfg(all(feature = "cuda", not(feature = "dkr-causal-mask")))]']),
        "dkr-rms-variants":("ops.rs",['#[cfg(feature = "dkr-rms-variants")]','rms_variants_dynamic::with_rms_plan(','#[cfg(not(feature = "dkr-rms-variants"))]','ffi::$ffi_fn(']),
        "dkr-add-rms-norm":("ops.rs",['#[cfg(all(feature = "cuda", feature = "dkr-add-rms-norm"))]','pub use crate::cuda::add_rms_norm_dynamic::','#[cfg(all(feature = "cuda", not(feature = "dkr-add-rms-norm")))]']),
        "dkr-ssm":("cuda/ssm.rs",['#[cfg(all(feature = "cuda", feature = "dkr-ssm"))]','super::ssm_dynamic::selective_scan_cuda(','#[cfg(all(feature = "cuda", not(feature = "dkr-ssm")))]']),
    }
    out={}
    for feature,(path,markers) in checks.items():
        require('"'+feature+'"' in typed,"typed feature missing: "+feature)
        out[feature]=dict(symbols=REUSE[feature],path=CORE+path,
            evidence=evidence(text(root,CORE+path),markers,feature),
            explicit_closure_guard_pending=True,additional_profiles_expected=0)
    # Check both RMS macro bodies independently rather than treating a marker
    # anywhere in ops.rs as sufficient evidence.
    for name in ("cuda_rms_norm_residual","cuda_rms_norm_residual_then_rms_norm"):
        body=ops.split("pub fn "+name+"(",1)[1].split("\n}\n",1)[0]
        evidence(body,['#[cfg(feature = "dkr-rms-variants")]','rms_variants_dynamic::with_rms_plan(',
                       '#[cfg(not(feature = "dkr-rms-variants"))]','ffi::$ffi_fn('],name)
    return out

def usage_candidates(root):
    # Lexical call candidates, not an exhaustive Rust call graph. Trait methods
    # with a shared name (notably sort) are intentionally not called proven.
    needles=re.compile(r'\b(?:cuda_moe_router_topk(?:_if_supported)?|moe_router_topk|'
        r'cuda_topk_logits_f32(?:_packed(?:_batched)?)?|cuda_topk_logits_packed_batched(?:_with_workspace)?|'
        r'cuda_topk_ranked_packed_batched(?:_with_workspace)?|cuda_top1_logits_[A-Za-z_]\w*|'
        r'cuda_categorical_logits_f32_packed_batched|cuda_dflash_(?:greedy|sample)_select|'
        r'cuda_topk_sampling_[A-Za-z_]\w*|topk_unsorted|arg_sort|ArgSortOp)\b')
    result={}
    ambiguous_sort=[]
    for path in sorted((root/CORE).rglob("*.rs")):
        source=path.read_text()
        code=mask(source)
        rel=str(path.relative_to(root))
        for m in needles.finditer(code):
            line=source.count("\n",0,m.start())+1
            snippet=source.splitlines()[line-1].strip()
            result.setdefault(m.group(),[]).append(dict(path=rel,line=line,source=snippet))
        if re.search(r'\bArgSortOp\b',code):
            for m in re.finditer(r'\.\s*sort\s*\(',code):
                line=source.count("\n",0,m.start())+1
                ambiguous_sort.append(dict(path=rel,line=line,source=source.splitlines()[line-1].strip()))
    return dict(exact_token_candidates=result,ambiguous_sort_candidates=ambiguous_sort,
                includes_definitions_and_tests=True,reachability_proven=False)

def sampling_budget(root):
    ops=text(root,CORE+"ops.rs")
    sort=text(root,CORE+"cuda/sort.cu")
    radix=text(root,CORE+"cuda/radix_topk.cuh")
    dflash=text(root,CORE+"speculative/dflash.rs")
    pipeline=text(root,CORE+"pipeline/sampling.rs")
    graph=text(root,CORE+"cuda/graph.cu")
    runtime=text(root,RUNTIME+"src/universal_cuda.rs")
    universal=text(root,RUNTIME+"src/universal.rs")
    embedded=text(root,RUNTIME+"build.rs")
    evidence(ops,['indices.arg_sort(true)?','CudaStorageSlice::F64(inp)','ffi::asort_asc_f64','ffi::asort_desc_f64'],"argsort")
    evidence(dflash,['predecessor_codebook: vb.get_unchecked("predecessor_codebook")?',
        'successor_codebook: vb.get_unchecked("successor_codebook")?',
        'let projection_dtype = self.hidden_projection.dtype();','MtpDraftSamplingMethod::Auto',
        'MtpDraftSamplingMethod::Probabilistic','DType::BF16 | DType::F32'],"DFlash")
    evidence(pipeline,['cuda_categorical_logits_f32_packed_batched(','cuda_topk_ranked_packed_batched(',
        'cuda_topk_logits_f32_packed_batched('],"sampling request branches")
    experts=[int(x) for x in re.findall(r'LAUNCH_MOE_ROUTER_TOPK\((\d+)\)',sort)]
    combos=set(re.findall(r'launch_moe_router_topk<T,\s*(true|false),\s*(true|false)>',sort))
    require(len(experts)==11 and len(combos)==4,"router specialization drift")
    evidence(sort,["topk_large_stage1<float, false, true>","topk_large_stage1<T, true, true>",
                   "topk_large_stage1<T, true, false>","topk_large_stage2_f32_packed<false, true>",
                   "topk_large_stage2_f32_packed<true, true>","topk_large_stage2_f32_packed<true, false>",
                   "topk_large_ranked_stage1_radix<T>","topk_large_ranked_stage2_radix",
                   "top1_large_stage1<T, BATCHED, false>","top1_large_stage1<float, true, true>",
                   "top1_large_stage2_f32_packed<BATCHED>"],"large selection stages")
    evidence(radix,["cudaLaunchCooperativeKernel","cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                    "cudaDevAttrCooperativeLaunch","resident_ctas / ctas_per_group",
                    "ranked_radix_sort_pack<<<nrows, 32, 0, stream>>>",
                    "case 1:","case 2:","case 4:","case 8:","if constexpr (sizeof(T) == 2)"],"cooperative radix")
    evidence(graph,["cudaMemcpyAsync(","cudaMemcpy2DAsync("],"graph copy")
    evidence(universal,["MAX_DISCOVERED_PACKAGES: usize = 512","MAX_CATALOG_PROFILES: usize = 4096",
                        "let catalog = Self::load(&paths)?;"],"active discovery")
    evidence(embedded,["KernelCatalog::discover(&root)?","DKR_EMBED_TYPED_DIR"],"embedded discovery")
    evidence(runtime,["KernelCatalog::discover_shared(root)?","crate::embedded::catalog()?","DKR_KERNEL_PACKAGES"],"runtime discovery")
    evidence(text(root,CORE+"cuda/kernel_runtime.rs"),["static KERNEL_CACHE: OnceLock","KernelExecutionCache::discover(&root)"],"core cache")
    evidence(text(root,"mistralrs-quant/src/dkr_cuda.rs"),["static CATALOG: OnceLock","KernelExecutionCache::discover(&root)"],"quant cache")
    # Count distinct device instantiations with no cross-mode fusion. This is
    # intentionally a scenario, not the exact number of future ABI profiles.
    counts={"argsort_bitonic_only":14,"small_topk":3,"router":len(experts)*3*len(combos),
            "large_topk_stage1":1+3+3,"large_topk_stage2":1+3,
            "ranked_two_stage_radix":3+1,"cooperative_radix":3+4+4+1,
            "top1_and_categorical":6+1+2+1,"sample_ranked":1,"dflash_selectors":2*2**3,
            "sparse_rejection":2}
    return dict(uncompressed_entry_scenario=counts,scenario_total=sum(counts.values()),
        scenario_is_not_final_profile_requirement=True,
        argsort=dict(public_generic_api=True,u32_reachable_via_topk_unsorted=True,
            f64_legacy_supported=True,f64_typed_supported=bool(re.search(r'\bF64\b',universal.split("pub enum ElementType {",1)[1].split("}",1)[0])),
            f64_unreachable_proven=False,additional_preparation_profiles_not_counted=True,
            preserve_tie_and_padding_semantics=True),
        router=dict(expert_sizes=experts,dtypes=["f32","f16","bf16"],optional_combinations=len(combos),
            preserve_compile_time_expert_specialization=True,profiles_per_fixed_expert_dtype_options=1,
            scope_needs_union_of_all_loaded_layers=True),
        dflash=dict(all_three_dtypes_independent_at_api=True,global_dtype_tuples=8,global_mode_dtype_profiles=16,
            per_resolved_dtype_tuple_profiles_if_both_modes=2,auto_may_choose_probabilistic_or_greedy=True,
            projection_dtype_from_actual_weight=True,codebook_dtypes_from_actual_weights=True),
        cooperative_radix=dict(vec_widths={"f32":[1,2,4],"f16":[1,2,4,8],"bf16":[1,2,4,8]},
            occupancy_and_residency_required=True,ordinary_launch_is_not_valid=True,
            cooperative_contract_to_qualify=True,
            typed_runtime_cooperative_marker_present="launch_cooperative" in runtime or "cuLaunchCooperativeKernel" in runtime,
            workspace_query_is_host_only=True,workspace_words_per_row_candidate=1284,
            workspace_constant_requires_shared_layout_proof=True),
        graph=dict(expected_new_profiles=0,driver_copy_migration_required=True),
        catalogue_scope=dict(selection_before_discovery_required=True,master_not_runtime_catalogue=True,
            active_limits_unchanged=True,existing_core_quant_once_lock_global=True,
            initial_prototype="offline_exact_identity_materialized_view",automatic_model_coverage_proven=False,
            hot_model_scope_switch_supported=False,multiple_model_scope_requires_union_or_isolated_contexts=True,
            runtime_request_modes_must_be_included=True),
        total_future_profiles=None,full_master_fit_proven=False,model_scope_fit_proven=False)

def audit(root):
    prior=previous(root)
    declarations=prior["residual_budget"]["remaining_core_ffi_declarations"]
    reused=set().union(*REUSE.values())
    closed=reused-set(declarations)
    closed_router=(SAMPLING_GROUPS["router"]|SAMPLING_GROUPS["small_topk"])-set(declarations)
    closed_large=(SAMPLING_GROUPS["large_topk"]|SAMPLING_GROUPS["ranked"]|SAMPLING_GROUPS["workspace_query"])-set(declarations)
    groups=partition(declarations,closed,closed_router,closed_large)
    reuse=reuse_evidence(root)
    for item in reuse.values():
        item["explicit_closure_guard_pending"]=not bool(closed)
    budget=sampling_budget(root)
    return dict(schema="dkr.r126.s9c2e0.source.v1",status="AUDIT_COMPLETE",
        residual_groups=groups,reuse=reuse,usage=usage_candidates(root),budget=budget,
        remaining_core_ffi_count=len(declarations),closed_reuse_declarations=sorted(closed),
        closed_router_declarations=sorted(closed_router),
        closed_large_topk_declarations=sorted(closed_large),
        current_projected_logical=prior["projected_logical"],
        current_projected_physical_upper=prior["projected_physical_upper"],
        current_logical_headroom=prior["logical_headroom"],profiles_added=0,images_added=0,
        compilation=False,link_qualified=False,full_closure=False,model_coverage_proven=False)

def self_test():
    declarations=sorted(set().union(*REUSE.values(),GRAPH,ARG_SORT,*SAMPLING_GROUPS.values()))
    groups=partition(declarations)
    assert len(groups["already_typed"])==17 and len(groups["argsort"])==14 and len(groups["graph"])==2
    failures=0
    for bad in (declarations[:-1],declarations+["unexpected"],declarations+[declarations[0]]):
        try:
            partition(bad)
        except ValueError:
            failures+=1
        else:
            raise AssertionError("inventory drift accepted")
    print(f"DKR_R126_S9C2E0_SELF_TEST=PASS negative_cases={failures}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test();return
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2E0_SOURCE=COMPLETE residual={r['remaining_core_ffi_count']} reuse_candidates=17 argsort=14 sampling=27 graph=2 "
          f"uncompressed_entry_scenario={r['budget']['scenario_total']} profiles_added=0 images_added=0 "
          f"projected_logical={r['current_projected_logical']} projected_physical={r['current_projected_physical_upper']} "
          "model_scope_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
