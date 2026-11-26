#!/usr/bin/env python3
"""Speculative commit and convolution checkpoints. Cudaforge compiles the image."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    rows=[]
    def t(n,d,a="read"):
        return tensor(n,d,1,a)
    def ints(*names):
        return [scalar(n,"i32") for n in names]
    def add(op,entry,args,block=(256,1,1)):
        rows.append(profile(cc,"gdn."+op+".v1","cuda."+entry+".v1",entry,list(block),args,"gdn_checkpoint.ptx"))
    indices=[t("keep","u32"),t("slots","u32")]
    dims=ints("batch","seq","channels","width","lanes","write","capacity")
    strides=[scalar(n,"i64") for n in ("stride_b","stride_s","stride_c")]
    for dt in ("f16","bf16"):
        add("commit.conv","dkr_gdn_commit_conv_"+dt,
            [t("x",dt),t("initial",dt),t("pool",dt,"read_write")]+indices+
            ints("batch","seq","channels","width","capacity"))
        for sd in ("f16","bf16","f32"):
            for layout in ("km","vm"):
                add("commit.rec."+layout,"dkr_gdn_commit_rec_"+dt+"_"+sd+"_"+layout,
                    [t(n,dt) for n in ("qkv","b","a")]+[t("initial",sd),t("a_log","f32"),t("dt_bias","f32"),
                    t("pool",sd,"read_write")]+indices+ints("batch","seq","kh","vh","kd","vd","tiled","capacity"),(32,4,1))
        base=[t("x",dt),t("weight",dt),t("pool",dt,"read_write"),t("output",dt,"write"),t("slots","u32")]
        pending=[t("pending_conv",dt,"read_write"),t("keep","u32"),t("epoch","u32"),t("applied","u32","read_write")]
        for suffix in ("","4"):
            add("checkpoint.conv"+suffix,"dkr_gdn_cp_conv"+suffix+"_"+dt,base+dims+strides)
            add("checkpoint.pending"+suffix,"dkr_gdn_cp_pending"+suffix+"_"+dt,base+pending+dims+
                ints("max_rows","pending_capacity","conv_blocks")+strides)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--ptx-file",type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    rows=plan(a.cc)
    if a.self_test:
        self_test(a.cc,rows)
        assert len(rows)==22 and max(len(r["arguments"]) for r in rows)==22
        print(f"DKR_R126_S9C2D3B2A_CP_SELF_TEST=PASS cc={a.cc} images=1 profiles=22 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D3B2A_CP_PLAN=COMPLETE cc={a.cc} images=1 profiles=22 compilation=false")
    elif a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-checkpoint"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --output --ptx-file")
if __name__=="__main__":
    main()
