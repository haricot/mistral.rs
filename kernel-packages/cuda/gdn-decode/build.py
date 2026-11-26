#!/usr/bin/env python3
"""Typed GDN decode: identical historical kernel selection, target-specific catalogue."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    rows=[]
    def t(n,d,access="read"):
        return tensor(n,d,1,access)
    for dt in ("f16","bf16"):
        for sd in ("f16","bf16","f32"):
            args=[t(n,dt) for n in ("qkv","b","a")]+[t(n,"f32") for n in ("a_log","dt_bias")]+[
                t("state",sd,"read_write"),t("output",dt,"write"),t("slots","u32")]
            args += [scalar(n,"i32") for n in ("batch","key_heads","value_heads","key_dim","value_dim","tiled")]
            args += [scalar(n,"i64") for n in ("b_batch","b_head","a_batch","a_head")]+[scalar("has_slots","i32")]
            variants=[("k64",64,0),("k128",64,0),("fallback",64,2560)]
            if cc>=80:
                variants += [("cooperative",128,0),("pipelined",256,0)]
                if dt=="bf16":
                    variants += [("vm4",32,0),("vm32",32,0)]
            for v,block,shared in variants:
                entry=f"dkr_gdn_decode_{dt}_{sd}_{v}"
                row=profile(cc,f"gdn.decode.{v}.v1","cuda."+entry+".v1",entry,[block,1,1],args,"gdn_decode.ptx")
                row["shared_mem_bytes"]=shared
                rows.append(row)
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
    count=len(rows)
    if a.self_test:
        self_test(a.cc,rows)
        assert len(plan(61))==18 and len(plan(80))==36
        assert all(len(r["arguments"])==19 for r in rows)
        print(f"DKR_R126_S9C2D3A_DECODE_SELF_TEST=PASS cc={a.cc} images=1 profiles={count} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D3A_DECODE_PLAN=COMPLETE cc={a.cc} images=1 profiles={count} compilation=false")
    elif a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-decode"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --output --ptx-file")

if __name__=="__main__":
    main()
