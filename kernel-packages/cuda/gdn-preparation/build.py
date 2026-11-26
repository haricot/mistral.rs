#!/usr/bin/env python3
"""GDN preparation and FP8 typed shard; Cudaforge owns compilation."""
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
    def s(n,d="i32"):
        return scalar(n,d)
    def add(op,entry,args,block=(256,1,1)):
        rows.append(profile(cc,"gdn."+op+".v1","cuda."+entry+".v1",entry,list(block),args,"gdn_preparation.ptx"))
    for dt in ("bf16","f32"):
        common=[t("source",dt),t("output",dt,"write"),t("cu_seqlens","u32")]
        add("ragged.pack","dkr_gdn_pack_"+dt,common+[s("total","u64"),s("padded"),s("width"),s("stride","i64"),s("pad","f32")])
        add("ragged.unpack","dkr_gdn_unpack_"+dt,common+[s("width")]+[s(n,"i64") for n in ("batch_stride","token_stride","feature_stride")]+[s("inner_width")])
        add("ragged.extract","dkr_gdn_extract_"+dt,[t("source",dt),t("initial",dt),t("output",dt,"write"),t("cu_seqlens","u32"),
            s("total","u64"),s("padded"),s("channels"),s("width"),s("batch_stride","i64"),s("token_stride","i64")])
    for dt in ("f16","bf16"):
        for out in ("f32","bf16"):
            add("prepare","dkr_gdn_prepare_"+dt+"_"+out,[t(n,dt) for n in ("mixed_qkv","b","a")]+
                [t(n,"f32") for n in ("a_log","dt_bias")]+[t(n,out,"write") for n in ("q","k","v")]+
                [t(n,"f32","write") for n in ("g","beta")]+[s(n) for n in ("batch","seq","stride","key_heads","value_heads","key_dim","value_dim","tiled")])
        add("gating","dkr_gdn_gating_"+dt,[t("b",dt),t("a",dt),t("a_log","f32"),t("dt_bias","f32"),
            t("beta",dt,"write"),t("g",dt,"write"),s("total"),s("heads")])
    fp8=[t(n,"bf16") for n in ("x","gate","weight")]+[t("output","fp8_e4m3","write"),t("scales","f32","write")]
    meta=[s(n) for n in ("rows","groups","scale_stride","scale_layout")]
    tail=[s("eps","f32"),s("sigmoid")]
    add("fp8.warp","dkr_gdn_fp8_warp",fp8+meta+[s("outer1"),s("outer2")]+
        [s(f"{n}{i}","i64") for n in ("x","gate") for i in range(4)]+tail,(32,8,1))
    add("fp8.rows4","dkr_gdn_fp8_rows4",fp8+meta+tail,(32,1,1))
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
        assert len(rows)==14 and max(len(r["arguments"]) for r in rows)==21
        print(f"DKR_R126_S9C2D3A_GDN_SELF_TEST=PASS cc={a.cc} images=1 profiles=14 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D3A_GDN_PLAN=COMPLETE cc={a.cc} images=1 profiles=14 compilation=false")
    elif a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-preparation"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --output --ptx-file")

if __name__=="__main__":
    main()
