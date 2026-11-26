#!/usr/bin/env python3
"""GDN recurrence and causal convolution: source plan and verified PTX staging."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import tensor,scalar,profile,stage,self_test

DTYPES=("f16","bf16","f32")
def shared(bt,bk):
    return (bt*bk+bt*bt+2*bt+bk)*4

def plan(cc):
    result={}
    for image,dt in enumerate(DTYPES,1):
        rows=[]
        def t(n,dtype=dt,access="read"):
            return tensor(n,dtype,1,access)
        def add(kind,variant,block,args,smem=0):
            entry=f"dkr_gdn_{kind}_{dt}_{variant}"
            row=profile(cc,f"gdn.{kind}.{variant}.v1","cuda."+entry+".v1",
                        entry,block,args,f"gdn_recurrence_{image}.ptx")
            row["shared_mem_bytes"]=smem
            rows.append(row)
        rec=[t(n,"f32") for n in ("q","k","v","g","beta")]
        rec += [t("state",access="read_write"),t("output","f32","write"),t("slots","u32")]
        rec += [scalar(n,"i32") for n in ("seq","k_dim","v_dim","heads","has_slots")]
        for name,block,smem in [
            ("scalar64",[64,1,1],0),("scalar128",[64,1,1],0),
            ("fallback",[64,1,1],2048),
            ("warp64",[32,4,1],0),("warp128",[32,4,1],0),
            ("vm1",[32,4,1],0),("vm2",[32,4,1],0),
            ("vm4",[32,4,1],0),("vm8",[32,4,1],0),
            ("chunk32k64",[64,1,1],shared(32,64)),
            ("chunk64k64",[64,1,1],shared(64,64)),
            ("chunk32k128",[64,1,1],shared(32,128)),
            ("vmchunk32",[64,1,1],shared(32,128)),
        ]:
            add("recurrence",name,block,rec,smem)
        # Extended shared-memory profiles are not admitted in a Pascal catalogue.
        if cc>=70:
            add("recurrence","chunk64k128",[64,1,1],rec,shared(64,128))
            add("recurrence","vmchunk64",[64,1,1],rec,shared(64,128))
        if dt!="f32":
            for mode in ("update","update4","full","full4","save"):
                args=([t("x"),t("state_in"),t("state_out",access="write"),t("slots","u32")]
                      if mode=="save" else
                      [t("x"),t("weight"),t("state",access="read_write" if mode.startswith("update") else "read"),
                       t("output",access="write"),t("slots","u32")])
                dims=("batch","channels","width") if mode.startswith("update") else ("batch","channels","seq","width")
                args += [scalar(n,"i32") for n in dims]
                args += [scalar(n,"i64") for n in ("stride_b","stride_s","stride_c")]
                args += [scalar("has_slots","i32")]
                add("conv",mode,[128 if mode=="full4" else 256,1,1],args)
        result[image]=rows
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--image",type=int)
    p.add_argument("--ptx-file",type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if not 50<=a.cc<=120:
        p.error("unsupported target")
    images=plan(a.cc)
    count=sum(map(len,images.values()))
    if a.self_test:
        for rows in images.values():
            self_test(a.cc,rows)
        assert shared(32,128)==21248 and shared(64,128)==50176
        assert all(r["shared_mem_bytes"]<=49152 for rows in plan(61).values() for r in rows)
        assert sum(map(len,plan(61).values()))==49
        assert sum(map(len,plan(80).values()))==55
        print(f"DKR_R126_S9C2D2_GDN_SELF_TEST=PASS cc={a.cc} images=3 profiles={count} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(images=images,profiles=count,physical_manifests=3,
            compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D2_GDN_PLAN=COMPLETE cc={a.cc} images=3 profiles={count} compilation=false")
    elif a.image in images and a.output is not None and a.ptx_file is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-recurrence"/f"image-{a.image:02}",a.cc,
              images[a.image],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --image --output --ptx-file")

if __name__=="__main__":
    main()
