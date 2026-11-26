#!/usr/bin/env python3
"""Batched GDN state transactions: one typed shard, compilation by Cudaforge."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    rows=[]
    def t(n,d):
        return tensor(n,d,1,"read")
    def add(op,tag,args,block=256):
        entry="dkr_gdn_tx_"+tag
        rows.append(profile(cc,"gdn.tx."+op+".v1","cuda."+entry+".v1",entry,[block,1,1],
                            args,"gdn_transaction.ptx"))
    table=[t("pointer_table","i64")]
    indices=table+[t("keep_rows","u32"),t("slots","u32")]
    apply=table+[t("slots","u32")]
    def ints(*names):
        return [scalar(n,"i32") for n in names]
    for dt in ("f16","bf16"):
        for sd in ("f16","bf16","f32"):
            suffix=dt+"_"+sd
            add("commit."+suffix,"commit_"+suffix,indices+ints(
                "layers","batch","seq","key_heads","value_heads","key_dim","value_dim",
                "conv_dim","conv_width","tiled","value_major","capacity"))
            add("apply."+suffix,"apply_"+suffix,apply+ints(
                "layers","batch","max_rows","capacity","key_heads","value_heads",
                "key_dim","value_dim","conv_dim","conv_width","tiled","conv_blocks"))
        add("stage."+dt,"stage_"+dt,indices+ints(
            "layers","batch","seq","max_rows","capacity","key_heads","value_heads",
            "key_dim","value_dim","conv_dim"))
    add("publish","publish",indices+ints("layers","batch","max_rows","capacity"),1)
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
        assert len(rows)==15 and max(len(r["arguments"]) for r in rows)==15
        print(f"DKR_R126_S9C2D3B1_TX_SELF_TEST=PASS cc={a.cc} images=1 profiles=15 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D3B1_TX_PLAN=COMPLETE cc={a.cc} images=1 profiles=15 compilation=false")
    elif a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-transaction"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --output --ptx-file")
if __name__=="__main__":
    main()
