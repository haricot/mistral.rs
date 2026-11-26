#!/usr/bin/env python3
"""Cudaforge-owned LFM3 temporal/spatial fused convolution shard."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    rows=[]
    for dt in ("f32","f16","bf16"):
        def t(n,a="read"):
            return tensor(n,dt,1,a)
        def add(op,entry,args):
            rows.append(profile(cc,"lfm3."+op+".v1","cuda."+entry+".v1",entry,[256,1,1],
                                args,"lfm3_conv.ptx"))
        for mode in ("generic","width7"):
            for bias in ("nobias","bias"):
                args=[t("x"),t("weight"),t("state")]
                if bias=="bias": args.append(t("bias"))
                args += [t("output","write"),t("next_state","write")]
                args += [scalar(n,"u64") for n in "batch seq hidden width xs0 xs1 xs2 ss0 ss1 ss2 ws0 ws2".split()]
                if bias=="bias": args.append(scalar("bs","u64"))
                args.append(scalar("use_state","u64"))
                add(f"causal.{mode}.{bias}",f"dkr_lfm3_causal_{dt}_{mode}_{bias}",args)
        add("spatial",f"dkr_lfm3_spatial_{dt}",[t("x"),t("weight"),t("output","write"),scalar("elements","u64")]+
            [scalar(n,"u32") for n in "height width merge channels kernel".split()])
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
        assert len(rows)==15 and max(len(r["arguments"]) for r in rows)==20
        print(f"DKR_R126_S9C2D4_LFM3_SELF_TEST=PASS cc={a.cc} images=1 profiles=15 max_args=20 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D4_LFM3_PLAN=COMPLETE cc={a.cc} images=1 profiles=15 compilation=false")
    elif a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"lfm3-conv"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --output --ptx-file")

if __name__=="__main__":
    main()
