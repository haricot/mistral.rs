#!/usr/bin/env python3
"""Cudaforge-owned compilation; package exact typed router/top-k entries."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import profile, tensor, scalar, stage, self_test

EXPERTS=(1,2,4,8,16,32,64,128,256,512,576)
DTYPES=("f32","f16","bf16")
BLOCKS=(64,128,256,512)
IMAGES=(0,)+EXPERTS

def plan(cc,image):
    if image not in IMAGES:
        raise ValueError("invalid router/top-k image")
    rows=[]
    def t(n,d,a="read"):
        return tensor(n,d,1,a)
    def s(n,d="i32"):
        return scalar(n,d)
    artifact=f"router_topk_{image}.ptx"
    if image==0:
        for dt in DTYPES:
            for block in BLOCKS:
                entry=f"dkr_small_topk_{dt}_b{block}"
                args=[t("input",dt),t("values",dt,"write"),t("indices","u32","write"),
                      s("rows"),s("cols"),s("k")]
                row=profile(cc,f"sampling.small_topk.b{block}.v1",f"cuda.{entry}.v1",
                            entry,[block,1,1],args,artifact)
                row["shared_mem_bytes"]=block*(5 if dt=="f32" else 3) if block<512 else 48*1024
                rows.append(row)
    else:
        for dt in DTYPES:
            for bias in (0,1):
                for scale in (0,1):
                    entry=f"dkr_moe_router_e{image}_{dt}_b{bias}_s{scale}"
                    args=[t("logits",dt),t("weights","f32","write"),t("ids","u32","write")]
                    if bias: args.append(t("bias","f32"))
                    if scale: args.append(t("scale","f32"))
                    args += [s(n) for n in ("rows","k","score_mode","weight_mode")]
                    args += [s(n,"u32") for n in ("renormalize","clamp_logits")]
                    args += [s(n,"f32") for n in ("clip_min","clip_max","norm_min","output_scale")]
                    rows.append(profile(cc,f"sampling.moe_router.e{image}.b{bias}.s{scale}.v1",
                                        f"cuda.{entry}.v1",entry,[32,4,1],args,artifact))
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
        print(f"DKR_R126_S9C2E2_SELF_TEST=PASS cc={a.cc} images={len(ids)} profiles={count} max_args={max_args} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=[r for g in groups for r in g],
            physical_manifests=len(ids),compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2E2_PLAN=COMPLETE cc={a.cc} images={len(ids)} profiles={count} compilation=false")
    elif a.image is not None and a.output and a.ptx_file:
        stage(a.output/f"sm{a.cc}"/"router-topk"/f"image-{a.image:03}",a.cc,groups[0],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out or --image --output --ptx-file")

if __name__=="__main__":
    main()
