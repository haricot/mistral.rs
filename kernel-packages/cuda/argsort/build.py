#!/usr/bin/env python3
"""Stage exact typed Argsort images; Cudaforge owns compilation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import profile, tensor, scalar, stage, self_test

IMAGES=(1,)
DTYPES=("f32","f16","bf16","f64","u8","u32","i64")

def plan(cc,image):
    if not 50<=cc<=255 or image not in IMAGES:
        raise ValueError("invalid argsort target/image")
    def t(n,d,a="read"):
        return tensor(n,d,1,a)
    def s(n):
        return scalar(n,"u32")
    rows=[]
    def add(tag,op,args):
        entry="dkr_argsort_"+tag
        rows.append(profile(cc,"tensor.argsort."+op+".v1","cuda."+entry+".v1",entry,[256,1,1],args,"argsort.ptx"))
    add("init","init",[t("work","u32","write"),s("padded")])
    for dt in DTYPES:
        for direction in ("asc","desc"):
            add(direction+"_"+dt,"compare."+direction,[t("input",dt),t("work","u32","read_write"),
                s("cols"),s("padded"),s("j"),s("k")])
    add("finish","finish",[t("work","u32"),t("indices","u32","write"),s("cols")])
    for dt in DTYPES:
        add("gather_"+dt,"gather",[t("input",dt),t("indices","u32"),t("output",dt,"write"),s("cols"),s("elements")])
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
        print(f"DKR_R126_S9C2E5_SELF_TEST=PASS cc={a.cc} images={len(ids)} profiles={count} max_args={max_args} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=[r for g in groups for r in g],
            physical_manifests=len(ids),compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2E5_PLAN=COMPLETE cc={a.cc} images={len(ids)} profiles={count} compilation=false")
    elif a.image is not None and a.output and a.ptx_file:
        stage(a.output/f"sm{a.cc}"/"argsort"/f"image-{a.image:02}",a.cc,groups[0],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out or --image --output --ptx-file")

if __name__=="__main__":
    main()
