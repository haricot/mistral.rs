#!/usr/bin/env python3
"""Stage exact typed large top-k images; Cudaforge owns compilation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import profile, tensor, scalar, stage, self_test

IMAGES=(1,2)
DTYPES=("f32","f16","bf16")

def plan(cc,image):
    if image not in IMAGES:
        raise ValueError("invalid large top-k image")
    rows=[]
    def t(n,d,a="read"):
        return tensor(n,d,1,a)
    def s(n,d="i32"):
        return scalar(n,d)
    def add(tag,op,args,block=(256,1,1),shared=0):
        entry="dkr_"+tag
        row=profile(cc,op,f"cuda.{entry}.v1",entry,list(block),args,f"large_topk_{image}.ptx")
        row["shared_mem_bytes"]=shared
        rows.append(row)
    if image==1:
        for dt in DTYPES:
            for mode in ("single_softmax","batch_softmax","batch_ranked"):
                batch=mode!="single_softmax"
                soft=mode!="batch_ranked"
                if not batch and dt!="f32":
                    continue
                args=[t("input",dt),t("block_values","f32","write"),t("block_indices","u32","write")]
                if soft:
                    args += [t("block_maxes","f32","write"),t("block_sums","f32","write")]
                args += [s("ncols"),s("k"),s("chunk_size")]
                if soft:
                    args += [t("inv_temperatures","f32") if batch else s("inv_temperature","f32")]
                add(f"large_topk_stage1_{mode}_{dt}",f"sampling.large_topk.stage1.{mode}.v1",args,shared=2048)
        for mode in ("single_split","single_packed","batch_packed","batch_ranked"):
            args=[t("block_values","f32"),t("block_indices","u32")]
            if mode!="batch_ranked":
                args += [t("block_maxes","f32"),t("block_sums","f32")]
            if mode=="single_split":
                args += [t("values_out","f32","write"),t("indices_out","u32","write"),t("softmax_info_out","f32","write")]
            else:
                args += [t("packed_out","f32","write")]
            args += [s("nblocks"),s("k")]
            add(f"large_topk_stage2_{mode}",f"sampling.large_topk.stage2.{mode}.v1",args,shared=47*1024)
        for dt in DTYPES:
            add(f"ranked_radix_stage1_{dt}","sampling.ranked_radix.stage1.v1",
                [t("input",dt),t("block_values","f32","write"),t("block_indices","u32","write"),s("ncols"),s("k")])
        add("ranked_radix_stage2","sampling.ranked_radix.stage2.v1",
            [t("block_values","f32"),t("block_indices","u32"),t("packed_out","f32","write"),s("nblocks"),s("k")])
    else:
        for dt in DTYPES:
            for v in ((1,2,4) if dt=="f32" else (1,2,4,8)):
                add(f"ranked_cooperative_{dt}_v{v}",f"cuda.cooperative.ranked_radix.vec{v}.v1",
                    [t("input",dt),t("packed_out","f32","write"),s("nrows"),s("ncols"),s("k"),
                     t("row_states","u32","read_write"),s("chunk_size","u32"),s("ctas_per_group","u32")],
                    (1024,1,1),96*1024)
        add("ranked_cooperative_sort","sampling.ranked_radix.sort_pack.v1",
            [t("packed_out","f32","read_write"),s("k")],(32,1,1))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--image",type=int,choices=IMAGES)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--json-out",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--ptx-file",type=Path)
    a=p.parse_args()
    if not 50<=a.cc<=255:
        p.error("invalid compute capability")
    ids=IMAGES if a.image is None else (a.image,)
    groups=[plan(a.cc,i) for i in ids]
    count=sum(map(len,groups))
    max_args=max(len(r["arguments"]) for g in groups for r in g)
    if a.self_test:
        for g in groups:
            self_test(a.cc,g)
        print(f"DKR_R126_S9C2E3_SELF_TEST=PASS cc={a.cc} images={len(ids)} profiles={count} max_args={max_args} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=[r for g in groups for r in g],
            physical_manifests=len(ids),compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2E3_PLAN=COMPLETE cc={a.cc} images={len(ids)} profiles={count} compilation=false")
    elif a.image is not None and a.output and a.ptx_file:
        stage(a.output/f"sm{a.cc}"/"large-topk"/f"image-{a.image:02}",a.cc,groups[0],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out or --image --output --ptx-file")

if __name__=="__main__":
    main()
