#!/usr/bin/env python3
"""S9C.2E.3: large top-k source/ABI, cooperative admission and workspace audit."""
import argparse
import copy
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from dkr_r126_s9c2e2_audit import audit as previous
from dkr_r126_s9c2e1_audit import check_declarations, checked_refs
from dkr_r126_s9c2e0_audit import SAMPLING_GROUPS
from dkr_r126_gguf_residual_audit import mask, guarded_ranges
from dkr_r126_s9c2d3c_audit import strip_comments
import dkr_r126_model_catalog_scope as scope

CORE="mistralrs-core/src/"
RUNTIME="mistralrs-kernel-runtime/"
PKG="kernel-packages/cuda/large-topk/"
NAMES=SAMPLING_GROUPS["large_topk"]|SAMPLING_GROUPS["ranked"]|SAMPLING_GROUPS["workspace_query"]
CTYPE={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","i32":"int32_t","u32":"uint32_t"}
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def module_at(root):
    spec=importlib.util.spec_from_file_location("large_topk_build",root/PKG/"build.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def normalize(s):
    return re.sub(r'\s*\*\s*','* ', " ".join(s.strip().split()))

def validate_entries(source,rows,image):
    matches=re.findall(r'extern "C" __global__ (?:__launch_bounds__\((\d+)\) )?void (\w+)\((.*?)\)\s*\{(.*?)\}',source,re.S)
    found={m[1]:m for m in matches}
    require(len(matches)==len(found)==len(rows),"CUDA entry count")
    require(set(found)=={r["entry"] for r in rows},"CUDA entry identity")
    for row in rows:
        bound,entry,params,body=found[row["entry"]]
        actual=[normalize(x) for x in params.split(",")]
        expected=[]
        names=[a["name"] for a in row["arguments"]]
        for a in row["arguments"]:
            ty=CTYPE[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+a["name"])
        require(actual==expected,"CUDA/manifest ABI mismatch: "+entry)
        require(row["implementation_id"]=="cuda."+entry+".v1","implementation identity")
        cooperative=entry.startswith("dkr_ranked_cooperative_") and entry!="dkr_ranked_cooperative_sort"
        require(row["operation"].startswith("cuda.cooperative.")==cooperative,"cooperative namespace mismatch")
        if entry.startswith("dkr_large_topk_stage1_"):
            m=re.fullmatch(r'dkr_large_topk_stage1_(single_softmax|batch_softmax|batch_ranked)_(f32|f16|bf16)',entry)
            require(image==1 and m is not None,"stage1 identity")
            mode,dt=m.groups()
            batch=mode!="single_softmax";soft=mode!="batch_ranked"
            require(batch or dt=="f32","single dtype")
            require(row["operation"]==f"sampling.large_topk.stage1.{mode}.v1" and
                    row["block"]==[256,1,1] and row["shared_mem_bytes"]==2048 and not bound,"stage1 contract")
            require(("block_maxes" in names)==soft and ("block_sums" in names)==soft and
                    ("inv_temperatures" in names)==(soft and batch) and
                    ("inv_temperature" in names)==(soft and not batch),"stage1 optional ABI")
            call=(f"topk_large_stage1<{CTYPE[dt]}, {str(batch).lower()}, {str(soft).lower()}>"
                  f"(input, block_values, block_indices, {'block_maxes, block_sums' if soft else 'nullptr, nullptr'}, "
                  f"ncols, k, chunk_size, {'inv_temperatures' if soft and batch else 'nullptr'}, "
                  f"{'inv_temperature' if soft and not batch else '0.0f'});")
        elif entry.startswith("dkr_large_topk_stage2_"):
            mode=entry.removeprefix("dkr_large_topk_stage2_")
            require(image==1 and mode in ("single_split","single_packed","batch_packed","batch_ranked"),"stage2 identity")
            soft=mode!="batch_ranked";batch=mode.startswith("batch")
            require(row["operation"]==f"sampling.large_topk.stage2.{mode}.v1" and
                    row["block"]==[256,1,1] and row["shared_mem_bytes"]==47*1024 and not bound,"stage2 contract")
            require(("block_maxes" in names)==soft and ("block_sums" in names)==soft,"stage2 softmax ABI")
            if mode=="single_split":
                call="topk_large_stage2_f32(block_values, block_indices, block_maxes, block_sums, values_out, indices_out, softmax_info_out, nblocks, k);"
            else:
                call=(f"topk_large_stage2_f32_packed<{str(batch).lower()}, {str(soft).lower()}>"
                      f"(block_values, block_indices, {'block_maxes, block_sums' if soft else 'nullptr, nullptr'}, packed_out, nblocks, k);")
        elif entry.startswith("dkr_ranked_radix_stage1_"):
            dt=entry.rsplit("_",1)[1]
            require(image==1 and dt in ("f32","f16","bf16") and not bound and row["block"]==[256,1,1] and
                    row["shared_mem_bytes"]==0 and row["operation"]=="sampling.ranked_radix.stage1.v1","CUB first stage")
            call=f"topk_large_ranked_stage1_radix<{CTYPE[dt]}>(input, block_values, block_indices, ncols, k);"
        elif entry=="dkr_ranked_radix_stage2":
            require(image==1 and not bound and row["block"]==[256,1,1] and row["shared_mem_bytes"]==0 and
                    row["operation"]=="sampling.ranked_radix.stage2.v1","CUB second stage")
            call="topk_large_ranked_stage2_radix(block_values, block_indices, packed_out, nblocks, k);"
        elif cooperative:
            m=re.fullmatch(r'dkr_ranked_cooperative_(f32|f16|bf16)_v([1248])',entry)
            require(image==2 and m is not None,"cooperative entry identity")
            dt,v=m.groups()
            require(not (dt=="f32" and v=="8") and bound=="1024" and row["block"]==[1024,1,1] and
                    row["shared_mem_bytes"]==96*1024 and row["operation"]==f"cuda.cooperative.ranked_radix.vec{v}.v1",
                    "cooperative launch contract")
            state=next(a for a in row["arguments"] if a["name"]=="row_states")
            require(state==dict(kind="tensor",name="row_states",dtype="u32",rank=1,access="read_write"),
                    "cooperative workspace ABI")
            call=(f"mistralrs_radix_topk::ranked_radix_select<{CTYPE[dt]}, {v}>(input, packed_out, nrows, ncols, k, "
                  "reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);")
        else:
            require(entry=="dkr_ranked_cooperative_sort" and image==2 and not bound and row["block"]==[32,1,1] and
                    row["shared_mem_bytes"]==0 and row["operation"]=="sampling.ranked_radix.sort_pack.v1",
                    "ordinary final pack sort")
            call="mistralrs_radix_topk::ranked_radix_sort_pack(packed_out, k);"
        require(" ".join(body.split())==call,"template dispatch/argument order: "+entry)

def plans(root):
    m=module_at(root)
    require(m.IMAGES==(1,2),"image inventory")
    source=(root/PKG/"large_topk.cu").read_text()
    groups={int(i):body for i,body in re.findall(r'#(?:if|elif) DKR_LARGE_TOPK_IMAGE == (\d+)\n(.*?)(?=#elif DKR_LARGE_TOPK_IMAGE|#else\n#error)',source,re.S)}
    require(set(groups)=={1,2} and source.count('extern "C" __global__')==27,"image guards")
    require("#define DKR_ROUTER_TOPK_EXTERNAL" in groups[1] and
            "#define DKR_LARGE_TOPK_EXTERNAL" in groups[1] and
            "#define DKR_COOPERATIVE_RADIX_EXTERNAL" in groups[2],"device-only header contract")
    for cc in (61,80):
        seen=set()
        for image,count in ((1,15),(2,12)):
            rows=m.plan(cc,image)
            require(len(rows)==count and len({r["artifact"] for r in rows})==1,"physical shard")
            validate_entries(groups[image],rows,image)
            for r in rows:
                require(r["implementation_id"] not in seen and r["target_cc"]==r["min_cc"]==r["max_cc"]==cc,
                        "duplicate or wrong target")
                require(len(r["arguments"])<=9,"argument budget")
                seen.add(r["implementation_id"])
            paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
                dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
                for r in rows])
            require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,"manifest byte limit")
        require(len(seen)==27,"logical count")
    return m,groups

def ranked_ids(dt):
    return {f"cuda.dkr_large_topk_stage1_batch_ranked_{dt}.v1",
            "cuda.dkr_large_topk_stage2_batch_ranked.v1",
            f"cuda.dkr_ranked_radix_stage1_{dt}.v1",
            "cuda.dkr_ranked_radix_stage2.v1","cuda.dkr_ranked_cooperative_sort.v1"} | {
                f"cuda.dkr_ranked_cooperative_{dt}_v{v}.v1" for v in ((1,2,4) if dt=="f32" else (1,2,4,8))}

def scope_example(m):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/"master"
        for image in m.IMAGES:
            rows=m.plan(61,image)
            ptx=".version 8.0\n.target sm_61\n.address_size 64\n"
            for row in rows:
                params=",".join(f".param .b{64 if a['kind']=='tensor' else 32} a{j}"
                                for j,a in enumerate(row["arguments"]))
                ptx+=f".visible .entry {row['entry']}({params}) {{ret;}}\n"
            m.stage(root/f"image-{image:02}",61,rows,ptx.encode())
        root,records,manifests=scope.inventory(root)
        union=ranked_ids("f32")|ranked_ids("bf16")
        keys=[key for key in records if key[1] in union]
        require(len(ranked_ids("f32"))==8 and len(ranked_ids("bf16"))==9 and len(union)==len(keys)==14,
                "ranked conservative scope inventory")
        spec=dict(schema=scope.SELECTION,model="synthetic-ranked-target-draft",device_cc=61,
                  profiles=[dict(zip(("operation","implementation_id","target_cc"),key)) for key in keys])
        group=scope.selection(spec,records,manifests)
        report,images=scope.report(root,records,manifests,spec,group)
        dest=Path(tmp)/"active"
        scope.materialize(root,dest,group,manifests,images,61)
        _,selected,selected_manifests=scope.inventory(dest)
        require(set(selected)==set(keys) and len(selected_manifests)==2,"ranked scope materialization")
        require(report["scope_receipt"]==scope.decode((dest/scope.SCOPE_FILENAME).read_bytes()),"scope receipt")
        return dict(master_profiles=27,master_manifests=2,f32_ranked=8,bf16_ranked=9,
                    union_profiles=14,union_manifests=2,synthetic_only=True)

def guard_audit(root):
    core=root/CORE
    check_declarations((core/"cuda/ffi.rs").read_text(),NAMES)
    ops=(core/"ops.rs").read_text()
    for name in sorted(NAMES):
        checked_refs(ops,r'\bffi::'+name+r'\b',1,name)
    pattern=re.compile(r'\bffi::('+"|".join(sorted(NAMES))+r')\b')
    direct=[]
    for path in sorted(core.rglob("*.rs")):
        if path==core/"cuda/ffi.rs":
            continue
        code=mask(strip_comments(path.read_text()))
        for match in pattern.finditer(code):
            require(path==core/"ops.rs","unexpected residual large top-k use")
            direct.append(match.group(1))
    require(len(direct)==9 and set(direct)==NAMES,"legacy call inventory")
    require(ops.count("dkr::launch(\n            &candle_core::Device::Cuda(dev.clone()), &stream,")==4,
            "typed launch adapter route inventory")
    for mode in ("SingleSplit","SinglePacked","BatchedPacked","Ranked"):
        require("DkrLargeTopKMode::"+mode in ops,"typed mode "+mode)
    require("crate::cuda::large_topk_dynamic::RADIX_STATE_WORDS_PER_ROW" in ops and
            ops.count("cuda_ranked_state_words_per_row()")==4,"host workspace query closure")
    require("offset.checked_add(count).is_none_or(|end| end > allocation_elems)" in ops and
            "base.checked_add(bytes)" in ops,"checked storage binding")
    return dict(guarded_ffi=9,device_launch_ffi=8,host_workspace_query=1,typed_modes=4)

def contract(root):
    markers={
        CORE+"cuda/large_topk_device.cuh":[
            "#define DKR_LARGE_TOPK_ENTRY __device__ __forceinline__",
            "topk_large_stage1(","topk_large_stage2_f32(","topk_large_stage2_f32_packed(",
            "topk_large_ranked_stage1_radix(","topk_large_ranked_stage2_radix(",
            "cub::BlockRadixSort","RANKED_TOPK_RADIX_THREADS = 256",
            "RANKED_TOPK_RADIX_ITEMS_PER_THREAD = 8"],
        CORE+"cuda/radix_topk.cuh":[
            "static_assert(alignof(RowState) == 16)",
            "static_assert(offsetof(RowState, histogram) == 0)",
            "static_assert(offsetof(RowState, arrival_counter) == 3072)",
            "static_assert(offsetof(RowState, gt_count) == 3076)",
            "static_assert(offsetof(RowState, eq_count) == 4100)",
            "static_assert(sizeof(RowState) == 5136)",
            "#ifndef DKR_COOPERATIVE_RADIX_EXTERNAL","__device__ __forceinline__ void ranked_radix_select(",
            "cudaLaunchCooperativeKernel","store_release(&state->arrival_counter, 0)"],
        RUNTIME+"src/cooperative.rs":[
            'COOPERATIVE_PREFIX: &str = "cuda.cooperative."','resident / u64::from(ctas_per_group)',
            'groups * u64::from(ctas_per_group)','fn dkr_cooperative_grid_residency()',
            'fn dkr_cooperative_namespace_is_not_ordinary()'],
        RUNTIME+"src/universal_cuda.rs":[
            "pub struct PrevalidatedCooperativeLaunchPlan",
            'ensure!(!requires_cooperative_launch(&self.manifest.operation)',
            'self.launch_config_internal(request, device)',
            'CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH',
            'CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN',
            'CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES',
            'let ceiling = available.min(self.manifest.shared_mem_bytes)',
            'occupancy_max_active_blocks_per_multiprocessor(',
            'resident_group_grid(rows, ctas_per_group, active, limits.sms)',
            'pub unsafe fn prevalidate_cooperative_grouped(',
            'args.launch_cooperative(config)',
            'expected.cu_stream() == stream.cu_stream()'],
        CORE+"cuda/large_topk_dynamic.rs":[
            "RADIX_STATE_WORDS_PER_ROW: usize = 1284",
            "COOP_FIXED_BYTES: u32 = 1184","COOP_HEADROOM: u32 = 2048",
            "COOP_MAX_GROUP_CTAS: u32 = 256","COOP_THREADS: u32 = 1024",
            "gcd(gcd(max,cols),align)","validate_ptr(state,16)?",
            "LaunchPlanCache<Key,Plan>","pub fn dkr_large_topk_requirements(",
            "out.extend([cub1(dt),cub2(),sort_pack()])",
            "if let Some(plan)=cooperative_plan(key,device)?{return Ok(plan);}",
            "return cub_plan(key,device);","iterative_plan(key,device)",
            "p.prevalidate_cooperative_grouped(&request,device,geo.shared,g.rows as u32,geo.ctas)",
            "plan.launch_cooperative(stream,|b|",
            "b.arg(&work.input).arg(&packed).arg(&rows).arg(&cols).arg(&k).arg(&state).arg(chunk).arg(ctas)",
            "fn dkr_large_topk_geometry_and_requirements()",
            "fn dkr_large_topk_cooperative_contract_gpu()",
            "this qualification requires actual cooperative selection, not numerical fallback",
            "kernel.prevalidate_launch(&request,&device).is_err()",
            "p.prevalidate_launch_with_default_shared(&request,device,shared)"],
        RUNTIME+"src/bin/dkr_package_builder.rs":[
            'name: "large-topk", source: "large_topk.cu", fast_math: true',
            'pkg.name == "large-topk"','"large_topk_device.cuh","router_topk_device.cuh","radix_topk.cuh"',
            '-DDKR_LARGE_TOPK_IMAGE={id}'],
        CORE+"lib.rs":[
            "pub use cuda::large_topk_dynamic::{dkr_large_topk_requirements, DkrLargeTopKMode};"]
    }
    count=0
    for path,needles in markers.items():
        source=(root/path).read_text()
        for n in needles:
            require(n in source,"source contract: "+path+" "+n)
            count+=1
    sort=(root/CORE/"cuda/sort.cu").read_text()
    require('#include "large_topk_device.cuh"' in sort and 'void topk_large_stage1(' not in sort and
            'void topk_large_stage2_f32(' not in sort,"shared historical device bodies")
    return dict(markers=count,cooperative_namespace=True,ordinary_rejection=True,
                residency_prevalidated=True,layout_static_assertions=True,workspace_words_per_row=1284,
                no_post_launch_fallback=True,model_capability_producer=True,source_only=True)

def audit(root):
    prev=previous(root)
    require(prev["residual_declarations"] in (28,16,2,0) and set(prev["closed_large_topk_declarations"])==NAMES,
            "cumulative residual closure")
    m,_=plans(root)
    guards=guard_audit(root)
    contracts=contract(root)
    scoped=scope_example(m)
    logical=prev["projected_master_logical"]+27
    physical=prev["projected_master_physical_upper"]+2
    require(logical==4213 and physical==548,"master budget projection")
    return dict(schema="dkr.r126.s9c2e3.source.v1",status="SOURCE_COMPLETE",guards=guards,
                contract=contracts,ordinary_profiles=15,cooperative_select_profiles=11,sort_pack_profiles=1,
                profiles_added=27,images_added=2,max_args=9,residual_declarations=prev["residual_declarations"],closed_sampling_declarations=prev["closed_sampling_declarations"],closed_argsort_declarations=prev["closed_argsort_declarations"],closed_graph_declarations=prev["closed_graph_declarations"],
                projected_master_logical=logical,projected_master_physical_upper=physical,
                scope_example=scoped,active_limits_unchanged=True,actual_model_scope_fit_proven=False,
                compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def self_test(root):
    m,groups=plans(root)
    rows=m.plan(61,2);source=groups[2]
    bad_mode=copy.deepcopy(rows);bad_mode[0]["operation"]="sampling.ordinary.invalid.v1"
    bad_bound=copy.deepcopy(rows);bad_bound[0]["block"]=[512,1,1]
    bad_workspace=copy.deepcopy(rows);bad_workspace[0]["arguments"][5]["access"]="write"
    failures=0
    for text,manifest in ((source.replace("__launch_bounds__(1024)","__launch_bounds__(512)",1),rows),
                          (source.replace("ranked_radix_select<float, 1>","ranked_radix_select<float, 2>",1),rows),
                          (source,bad_mode),(source,bad_bound),(source,bad_workspace)):
        try:
            validate_entries(text,manifest,2)
        except ValueError:
            failures+=1
        else:
            raise AssertionError("incorrect cooperative contract accepted")
    scope_example(m)
    print(f"DKR_R126_S9C2E3_AUDIT_SELF_TEST=PASS negative_cases={failures} scoped_fixture=true")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    root=a.source_root.resolve()
    if a.self_test:
        self_test(root);return
    r=audit(root)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print("DKR_R126_S9C2E3_SOURCE=COMPLETE guarded_ffi=9 residual=28 profiles=27 images=2 max_args=9 "
          "projected_master_logical=4213 projected_master_physical=548 workspace_words=1284 "
          "cooperative_contract=SOURCE scoped_fixture=PASS model_scope_fit=UNPROVEN "
          "compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
