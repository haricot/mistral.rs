#!/usr/bin/env python3
"""S9C.2E.2: shared device bodies, exact typed entry ABI and model-scope budget."""
import argparse
import importlib.util
import json
import re
import tempfile
from pathlib import Path
from dkr_r126_s9c2e1_audit import audit as previous, check_declarations, checked_refs
from dkr_r126_s9c2e0_audit import SAMPLING_GROUPS
from dkr_r126_gguf_residual_audit import mask, guarded_ranges
from dkr_r126_s9c2d3c_audit import strip_comments
import dkr_r126_model_catalog_scope as scope

CORE="mistralrs-core/src/"
RUNTIME="mistralrs-kernel-runtime/"
PKG="kernel-packages/cuda/router-topk/"
EXPERTS=(1,2,4,8,16,32,64,128,256,512,576)
DTYPES={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","i32":"int32_t","u32":"uint32_t"}
BLOCKS=(64,128,256,512)
NAMES=SAMPLING_GROUPS["router"]|SAMPLING_GROUPS["small_topk"]
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def module_at(root):
    spec=importlib.util.spec_from_file_location("router_topk_build",root/PKG/"build.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def normalize(s):
    return re.sub(r'\s*\*\s*','* ', " ".join(s.strip().split()))

def validate_entries(source,rows,image):
    matches=re.findall(r'extern "C" __global__ (?:__launch_bounds__\((\d+),\s*(\d+)\) )?void (\w+)\((.*?)\)\s*\{(.*?)\}',source,re.S)
    found={m[2]:m for m in matches}
    require(len(matches)==len(found)==len(rows),"CUDA entry count")
    require(set(found)=={r["entry"] for r in rows},"CUDA entry identity")
    for row in rows:
        bound,min_blocks,entry,params,body=found[row["entry"]]
        actual=[normalize(x) for x in params.split(",")]
        expected=[]
        for a in row["arguments"]:
            ty=DTYPES[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+a["name"])
        require(actual==expected,"CUDA/manifest ABI mismatch: "+entry)
        names=[a["name"] for a in row["arguments"]]
        if image==0:
            dt=entry.split("_")[3]
            block=int(entry.rsplit("_b",1)[1])
            require(not bound and not min_blocks and block in BLOCKS,"small top-k launch contract")
            require(row["block"]==[block,1,1],"small top-k block")
            require(row["shared_mem_bytes"]==(block*(5 if dt=="f32" else 3) if block<512 else 48*1024),
                    "small top-k shared memory ceiling")
            expected_call=f"topk_kernel<{DTYPES[dt]}>(input, values, indices, rows, cols, k);"
        else:
            m=re.fullmatch(r'dkr_moe_router_e(\d+)_(f32|f16|bf16)_b([01])_s([01])',entry)
            require(m is not None,"router specialization identity")
            e,dt,b,s=m.groups()
            require(int(e)==image and bound=="128" and min_blocks=="1" and row["block"]==[32,4,1]
                    and row["shared_mem_bytes"]==0,"router launch bounds")
            require(("bias" in names)==(b=="1") and ("scale" in names)==(s=="1"),"optional typed ABI")
            expected_call=(f"moe_router_topk_kernel<{DTYPES[dt]}, {e}, {'true' if b=='1' else 'false'}, {'true' if s=='1' else 'false'}>"
                           f"(logits, weights, ids, {'bias' if b=='1' else 'nullptr'}, {'scale' if s=='1' else 'nullptr'}, "
                           "rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);")
        require(" ".join(body.split())==expected_call,"template dispatch/argument order: "+entry)

def plans(root):
    module=module_at(root)
    require(module.IMAGES==(0,)+EXPERTS,"image assignment")
    source=(root/PKG/"router_topk.cu").read_text()
    groups=dict((int(i),body) for i,body in re.findall(r'#if DKR_ROUTER_TOPK_IMAGE == (\d+)\n(.*?)#endif',source,re.S))
    require(set(groups)==set(module.IMAGES),"CUDA image guards")
    require(source.count('extern "C" __global__')==144,"entry outside image guard")
    for cc in (61,80):
        seen=set()
        for i in module.IMAGES:
            rows=module.plan(cc,i)
            require(len(rows)==12 and len({r["artifact"] for r in rows})==1,"physical shard")
            validate_entries(groups[i],rows,i)
            for r in rows:
                require(r["implementation_id"] not in seen and r["target_cc"]==r["min_cc"]==r["max_cc"]==cc,
                        "duplicate/incompatible profile")
                seen.add(r["implementation_id"])
            paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
                dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
                for r in rows])
            require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,"manifest byte limit")
        require(len(seen)==144,"profile count")
    return module,groups

def scope_example(module):
    # Synthetic package content; this is a family-scope demonstration only,
    # not a full model allowlist or actual CUDA image qualification.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/"master"
        for i in module.IMAGES:
            rows=module.plan(61,i)
            ptx=".version 8.0\n.target sm_61\n.address_size 64\n"
            for row in rows:
                params=",".join(f".param .b{64 if a['kind']=='tensor' else 32} a{j}"
                                for j,a in enumerate(row["arguments"]))
                ptx+=f".visible .entry {row['entry']}({params}) {{ret;}}\n"
            module.stage(root/f"image-{i:03}",61,rows,ptx.encode())
        root,records,manifests=scope.inventory(root)
        keys=[key for key in records if key[1]=="cuda.dkr_moe_router_e256_bf16_b0_s0.v1"]
        require(len(keys)==1,"exact router selection fixture")
        spec=dict(schema=scope.SELECTION,model="synthetic-router-only",device_cc=61,
                  profiles=[dict(zip(("operation","implementation_id","target_cc"),key)) for key in keys])
        group=scope.selection(spec,records,manifests)
        report,images=scope.report(root,records,manifests,spec,group)
        dest=Path(tmp)/"active"
        scope.materialize(root,dest,group,manifests,images,61)
        _,selected,selected_manifests=scope.inventory(dest)
        require(len(records)==144 and len(manifests)==12 and len(selected)==len(selected_manifests)==1,
                "model-scoped router fixture")
        require(report["scope_receipt"]==scope.decode((dest/scope.SCOPE_FILENAME).read_bytes()),"scope receipt")
        return dict(master_profiles=144,master_manifests=12,selected_router_profiles=1,selected_manifests=1,
                    image_contains_other_entries=True,synthetic_only=True)

def guard_audit(root):
    core=root/CORE
    check_declarations((core/"cuda/ffi.rs").read_text(),NAMES)
    ops=(core/"ops.rs").read_text()
    for dt in ("f32","f16","bf16"):
        checked_refs(ops,r'\bffi::topk_'+dt+r'\s*\(',1,"small top-k "+dt)
    marker=GUARD+'\n#[cfg(feature = "cuda")]\n#[allow(clippy::cast_possible_truncation)]\npub fn cuda_moe_router_topk('
    require(ops.count(marker)==1,"legacy router guard")
    part=ops[ops.index(marker):].split("\n}\n",1)[0]+"\n}\n"
    checked_refs(part,r'\bffi::\$ffi_fn\s*\(',1,"legacy router macro")
    for name in SAMPLING_GROUPS["router"]:
        require(len(re.findall(r'\b'+name+r'\b',mask(part)))==1,"router invocation "+name)
    require('crate::cuda::router_topk_dynamic::small_topk(&final_logits_row(input)?, k)' in ops
            and 'crate::cuda::router_topk_dynamic::moe_router_topk(logits, config, selection_bias, expert_scale)' in ops,
            "typed route missing or final row semantics changed")
    # Reject direct calls outside the audited file and macro argument drift.
    pattern=re.compile(r'\bffi::('+"|".join(sorted(NAMES))+r')\b')
    direct=[]
    for path in sorted(core.rglob("*.rs")):
        if path==core/"cuda/ffi.rs":
            continue
        source=strip_comments(path.read_text())
        for m in pattern.finditer(mask(source)):
            require(path==core/"ops.rs" and m.group(1) in SAMPLING_GROUPS["small_topk"],
                    "unexpected direct router/top-k FFI")
            direct.append(m.group(1))
    require(set(direct)==SAMPLING_GROUPS["small_topk"] and len(direct)==3,"small top-k direct inventory")
    return dict(guarded_ffi=6,macro_bodies=1,direct_calls=3)

def audit(root):
    prev=previous(root)
    require(prev["residual_declarations"] in (37,28,16,2,0) and set(prev["closed_router_declarations"])==NAMES,
            "cumulative residual closure")
    module,groups=plans(root)
    guards=guard_audit(root)
    header=(root/CORE/"cuda/router_topk_device.cuh").read_text()
    sort=(root/CORE/"cuda/sort.cu").read_text()
    require('#include "router_topk_device.cuh"' in sort and 'void topk_kernel(' not in sort
            and 'void moe_router_topk_kernel(' not in sort,"legacy shared body")
    for marker in ('#define DKR_ROUTER_TOPK_ENTRY __device__ __forceinline__',
                   '#define DKR_ROUTER_MOE_ENTRY __device__ __forceinline__',
                   'extern __shared__ char smem[];', 'const int row = blockIdx.x * blockDim.y + threadIdx.y;',
                   'selection[i] = expert < n_experts ? score[i] : -INFINITY;',
                   'constexpr int values_per_thread = n_experts > 32 ? n_experts / 32 : 1;'):
        require(marker in header,"shared kernel contract: "+marker)
    adapter=(root/CORE/"cuda/router_topk_dynamic.rs").read_text()
    for marker in ('pub fn dkr_moe_router_requirement(', 'pub fn dkr_small_topk_requirement(',
                   'LaunchPlanCache<Key, PrevalidatedLaunchPlan>', 'p.manifest.implementation_id!=implementation',
                   'prevalidate_launch_with_default_shared(&request,device,shared)',
                   'span(layout.start_offset(),input.elem_count(),src.len())',
                   'span(layout.start_offset(),logits.elem_count(),src.len())',
                   'span(layout.start_offset(),cols,src.len())',
                   '!v.device().same_device(logits.device())','if let Some((ptr,_))=&bias_ptr {b.arg(ptr);}',
                   'if let Some((ptr,_))=&scale_ptr {b.arg(ptr);}',
                   'fn dkr_router_topk_capabilities()', 'fn dkr_router_topk_gpu()'):
        require(marker in adapter,"adapter contract: "+marker)
    builder=(root/RUNTIME/"src/bin/dkr_package_builder.rs").read_text()
    require('name: "router-topk", source: "router_topk.cu", fast_math: true' in builder
            and 'pkg.name == "router-topk"' in builder and "router_topk_device.cuh" in builder
            and '-DDKR_ROUTER_TOPK_IMAGE={id}' in builder,"Cudaforge registration")
    runtime=(root/RUNTIME/"src/universal_cuda.rs").read_text()
    require("prevalidate_launch_with_default_shared" in runtime
            and 'static_bytes >= 0 && total <= limit' in runtime,"default shared memory admission")
    lib=(root/CORE/"lib.rs").read_text()
    require('pub use cuda::router_topk_dynamic::{dkr_moe_router_requirement, dkr_small_topk_requirement};' in lib,
            "capability producer exports")
    scoped=scope_example(module)
    logical=prev["projected_logical"]+144
    physical=prev["projected_physical_upper"]+12
    require(logical==4186 and physical==546,"projected master budget")
    return dict(schema="dkr.r126.s9c2e2.source.v1",status="SOURCE_COMPLETE",guards=guards,
                router_profiles=132,small_topk_profiles=12,profiles_added=144,images_added=12,max_args=15,
                residual_declarations=prev["residual_declarations"],closed_large_topk_declarations=prev["closed_large_topk_declarations"],closed_sampling_declarations=prev["closed_sampling_declarations"],closed_argsort_declarations=prev["closed_argsort_declarations"],closed_graph_declarations=prev["closed_graph_declarations"],
                projected_master_logical=logical,projected_master_physical_upper=physical,
                active_limits_unchanged=True,master_is_not_runtime_catalogue=True,scope_example=scoped,
                shared_body_preserved_except_padded_lane_safety=True,
                actual_model_scope_fit_proven=False,automatic_loader_capabilities=False,
                compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def self_test(root):
    module,groups=plans(root)
    rows=module.plan(61,1)
    source=groups[1]
    failures=0
    for bad in (source.replace("uint32_t renormalize","int32_t renormalize",1),
                source.replace("__launch_bounds__(128, 1)","__launch_bounds__(64, 1)",1),
                source.replace("nullptr, nullptr","nullptr, ids",1),
                source.replace("float *weights","const float *weights",1)):
        try:
            validate_entries(bad,rows,1)
        except ValueError:
            failures+=1
        else:
            raise AssertionError("incorrect CUDA ABI accepted")
    scope_example(module)
    print(f"DKR_R126_S9C2E2_AUDIT_SELF_TEST=PASS negative_cases={failures} scoped_fixture=true")

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
    print(f"DKR_R126_S9C2E2_SOURCE=COMPLETE guarded_ffi=6 residual={r['residual_declarations']} router_profiles=132 small_topk_profiles=12 images=12 max_args=15 "
          "projected_master_logical=4186 projected_master_physical=546 scoped_fixture=PASS "
          "model_scope_fit=UNPROVEN compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
