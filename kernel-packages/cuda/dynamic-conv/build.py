#!/usr/bin/env python3
"""Typed strided dynamic convolution plan; no device compilation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    rows=[]
    for dtype in ("f16","bf16","f32"):
        for variant in ("k2","generic"):
            args=[tensor(n,dtype,1,"write" if n=="output" else "read")
                  for n in ("hidden","dynamic","base","output")]
            args += [scalar(n,"i32") for n in ("batch","sequence_length","hidden_size","group_size","kernel_size")]
            args += [scalar(n,"i64") for n in ("hidden_stride_b","hidden_stride_s","hidden_stride_h",
                 "dynamic_stride_b","dynamic_stride_s","dynamic_stride_k","dynamic_stride_g",
                 "base_stride_k","base_stride_h")]
            entry=f"dkr_dynamic_conv_{dtype}_{variant}"
            rows.append(profile(cc,f"convolution.dynamic.{variant}.v1","cuda."+entry+".v1",
                                entry,[256,1,1],args,"dynamic_conv.ptx"))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path,default=Path("kernel-packages/build/cuda"))
    m=p.add_mutually_exclusive_group(required=True)
    m.add_argument("--ptx-file",type=Path)
    m.add_argument("--json-out",type=Path)
    m.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if not 50<=a.cc<=120:
        p.error("unsupported CUDA compute capability")
    rows=plan(a.cc)
    if a.self_test:
        self_test(a.cc,rows)
        print(f"DKR_R126_S9C2D_DYNAMIC_CONV_SELF_TEST=PASS cc={a.cc} images=1 profiles=6 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,
            compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D_DYNAMIC_CONV_PLAN=COMPLETE cc={a.cc} images=1 profiles=6 compilation=false")
    else:
        stage(a.output/f"sm{a.cc}"/"dynamic-conv"/"image-01",a.cc,rows,a.ptx_file.read_bytes())
        print(f"DKR_DYNAMIC_CONV_PACKAGE=STAGED cc={a.cc} images=1 profiles=6")

if __name__=="__main__":
    main()
