#!/usr/bin/env python3
"""Producer-only audit; must not claim FFI, link or GPU closure."""
import argparse
import copy
import importlib.util
import json
import re
import tempfile
from pathlib import Path
import dkr_r126_model_catalog_scope as scope

PKG=Path("kernel-packages/cuda/sampling")
CORE=Path("mistralrs-core/src/cuda")
TYPES={"f32":"float","f16":"__half","bf16":"__nv_bfloat16","u32":"uint32_t","i32":"int32_t"}

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def load(root):
    spec=importlib.util.spec_from_file_location("sampling_build",root/PKG/"build.py")
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def normalize(s):
    return re.sub(r"\s*\*\s*","* "," ".join(s.split()))

def call_contract(row):
    entry=row["entry"]
    names=[a["name"] for a in row["arguments"]]
    block,shared=256,0
    if entry.startswith("dkr_top1_stage1_"):
        mode,dt=entry.removeprefix("dkr_top1_stage1_").split("_")
        call=f"top1_large_stage1<{TYPES[dt]}, {str(mode=='batch').lower()}, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr)"
    elif entry.startswith("dkr_top1_stage2_"):
        mode,out=entry.removeprefix("dkr_top1_stage2_").split("_")
        call=(f"top1_large_stage2_f32_packed<{str(mode=='batch').lower()}>(block_values, block_indices, "
              f"{'packed_out' if out!='tokens' else 'nullptr'}, {'token_ids_out' if out!='packed' else 'nullptr'}, nblocks)")
    elif entry=="dkr_categorical_stage1":
        call="top1_large_stage1<float, true, true>(input, block_values, nullptr, block_sums, ncols, chunk_size, inv_temperatures)"
    elif entry=="dkr_categorical_stage2":
        shared=16384
        call="categorical_large_stage2_f32_packed(input, inv_temperatures, uniforms, block_values, block_sums, packed_out, ncols, chunk_size, nblocks)"
    elif entry=="dkr_sample_ranked_topk":
        block=128
        call="sample_ranked_topk_kernel(packed, params, tokens, nrows, packed_k)"
    elif entry.startswith("dkr_dflash_"):
        mode,h,p,s=entry.removeprefix("dkr_dflash_").split("_")
        types=[{"f32":"DFLASH_SELECTOR_F32","bf16":"DFLASH_SELECTOR_BF16"}[dt] for dt in (h,p,s)]
        require([a["dtype"] for a in row["arguments"][1:4]]==[h,p,s],"DFlash dtype identity")
        call=f"dflash_{mode}_select_kernel("+", ".join(names+types)+")"
    elif entry.startswith("dkr_sparse_rejection_"):
        if entry=="dkr_sparse_rejection_topk_f32":
            block=1
        call=entry.removeprefix("dkr_")+"_kernel("+", ".join(names)+")"
    else:
        raise ValueError("unknown sampling entry")
    require(row["block"]==[block,1,1] and row["shared_mem_bytes"]==shared,"launch geometry "+entry)
    return call+";"

def check(source,rows):
    entries=re.findall(r'extern "C" __global__ void (\w+)\((.*?)\)\s*\{(.*?)\}',source,re.S)
    found={name:(params,body) for name,params,body in entries}
    require(len(entries)==len(found)==len(rows),"entry count")
    require(set(found)=={r["entry"] for r in rows},"entry set")
    for row in rows:
        params,body=found[row["entry"]]
        expected=[]
        for a in row["arguments"]:
            ty=TYPES[a["dtype"]]
            if a["kind"]=="tensor":
                ty=("const " if a["access"]=="read" else "")+ty+"*"
            expected.append(ty+" "+a["name"])
        require([normalize(p) for p in params.split(",")]==expected,"ABI "+row["entry"])
        require(row["implementation_id"]=="cuda."+row["entry"]+".v1","identity")
        require(len(row["arguments"])<=16,"argument budget")
        require(" ".join(body.split())==call_contract(row),"dispatch/argument order "+row["entry"])

def scope_example(m):
    # Synthetic capability union, not an actual model admission receipt.
    chosen={"dkr_top1_stage1_batch_f32","dkr_top1_stage2_batch_both",
            "dkr_top1_stage1_batch_bf16","dkr_top1_stage2_batch_tokens",
            "dkr_categorical_stage1","dkr_categorical_stage2","dkr_sample_ranked_topk",
            "dkr_dflash_greedy_f32_bf16_bf16","dkr_dflash_sample_f32_bf16_bf16",
            "dkr_sparse_rejection_categorical_f32","dkr_sparse_rejection_topk_f32"}
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
        keys=[key for key in records if key[1].removeprefix("cuda.").removesuffix(".v1") in chosen]
        require(len(keys)==len(chosen)==11,"sampling scope inventory")
        spec=dict(schema=scope.SELECTION,model="synthetic-sampling-target-draft",device_cc=61,
                  profiles=[dict(zip(("operation","implementation_id","target_cc"),key)) for key in keys])
        group=scope.selection(spec,records,manifests)
        report,images=scope.report(root,records,manifests,spec,group)
        dest=Path(tmp)/"active"
        scope.materialize(root,dest,group,manifests,images,61)
        _,selected,selected_manifests=scope.inventory(dest)
        require(set(selected)==set(keys) and len(selected_manifests)==3,"sampling scope materialization")
        require(report["scope_receipt"]==scope.decode((dest/scope.SCOPE_FILENAME).read_bytes()),"scope receipt")
        return dict(master_profiles=32,master_manifests=3,union_profiles=11,union_manifests=3,synthetic_only=True)

def audit(root,self_test=False):
    m=load(root)
    source=(root/PKG/"sampling.cu").read_text()
    groups={int(i):body for i,body in re.findall(r'#(?:if|elif) DKR_SAMPLING_IMAGE == (\d+)\n(.*?)(?=#elif DKR_SAMPLING_IMAGE|#else\n#error)',source,re.S)}
    require(set(groups)=={1,2,3},"image guards")
    for cc in (61,80):
        seen=set()
        for i,count in ((1,14),(2,16),(3,2)):
            rows=m.plan(cc,i)
            require(len(rows)==count and all(r["target_cc"]==r["min_cc"]==r["max_cc"]==cc for r in rows),"target/count")
            check(groups[i],rows)
            for r in rows:
                require(r["entry"] not in seen,"duplicate entry")
                seen.add(r["entry"])
            if self_test:
                m.self_test(cc,rows)
        require(len(seen)==32,"logical inventory")
    for legacy,header,count in (("sort.cu","sampling_device.cuh",4),("dflash_selector.cu","dflash_selector_device.cuh",2),("speculative_rejection.cu","speculative_rejection_device.cuh",2)):
        a=(root/CORE/legacy).read_text()
        b=(root/CORE/header).read_text()
        require(f'#include "{header}"' in a and "#ifdef DKR_SAMPLING_EXTERNAL" in b,"shared bodies")
        require(len(re.findall(r"DKR_\w+_ENTRY void",b))==count,"body inventory")
    if self_test:
        rows=m.plan(61,1)
        bad=copy.deepcopy(rows)
        bad[0]["arguments"][0]["dtype"]="u32"
        bad_block=copy.deepcopy(rows)
        bad_block[0]["block"]=[128,1,1]
        bad_shared=copy.deepcopy(rows)
        bad_shared[12]["shared_mem_bytes"]=0
        for source,manifest in ((groups[1],bad),(groups[1],bad_block),(groups[1],bad_shared),
                               (groups[1].replace("block_values, block_indices, nullptr","block_indices, block_values, nullptr",1),rows),
                               (groups[2].replace("DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32",
                                                  "DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32",1),m.plan(61,2))):
            try:
                check(source,manifest)
            except ValueError:
                pass
            else:
                raise AssertionError("incorrect sampling contract accepted")
    scoped=scope_example(m)
    return dict(status="PRODUCER_SOURCE_COMPLETE",profiles=32,images=3,max_args=16,scope_example=scoped,
        projected_master_logical=4245,projected_master_physical_upper=551,
        producer_only=True,compilation=False,link_qualified=False,
        numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    result=audit(a.source_root.resolve(),a.self_test)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(result,indent=2)+"\n")
    print("DKR_R126_S9C2E4_PRODUCER_"+("SELF_TEST=PASS" if a.self_test else "SOURCE=COMPLETE")+
          " profiles=32 images=3 max_args=16 producer_only=true compilation=false link_qualified=false")
if __name__=="__main__":
    main()
