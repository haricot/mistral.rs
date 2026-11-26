#!/usr/bin/env python3
"""Single-image F32 selective scan packaging; no device compilation."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

def plan(cc):
    args=[tensor("x","f32",4,"read"),tensor("dt","f32",3,"read"),
          tensor("a","f32",1,"read"),tensor("b","f32",4,"read"),
          tensor("c","f32",4,"read"),tensor("d","f32",1,"read"),
          tensor("dt_bias","f32",1,"read"),tensor("state","f32",4,"read_write"),
          tensor("output","f32",4,"write")]
    args += [scalar(n,"i32") for n in ("heads","head_dim","d_state","seq_len")]
    args += [scalar(n,"f32") for n in ("dt_min","dt_max")]
    return [profile(cc,f"state_space.ssm_scan_f32.cf{cf}.v1",
                    f"cuda.ssm_scan.f32.cf{cf}.v1",f"dkr_ssm_scan_cf{cf}",
                    [32,1,1],args,"ssm.ptx",max_cc=120) for cf in (1,2,4,8)]

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
        print(f"DKR_R126_S9C2D_SSM_SELF_TEST=PASS cc={a.cc} images=1 profiles=4 synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=rows,physical_manifests=1,
            compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D_SSM_PLAN=COMPLETE cc={a.cc} images=1 profiles=4 compilation=false")
    else:
        root=a.output/f"sm{a.cc}"/"ssm"
        # Never silently coexist with old cf1/cf2/cf4/cf8 manifests.
        if any((root/f"cf{cf}").exists() for cf in (1,2,4,8)):
            raise SystemExit("legacy SSM images present; use fresh catalogue staging root")
        stage(root/"image-01",a.cc,rows,a.ptx_file.read_bytes())
        print(f"DKR_SSM_PACKAGES=PASS target_cc={a.cc} variants=4 images=1")

if __name__=="__main__":
    main()
