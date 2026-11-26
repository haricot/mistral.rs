#!/usr/bin/env python3
"""GDN speculative images, with specialized output and transition signatures.
The compact scalar pair ABI carries two checked U32 values per U64; no pointers
are packed and no parameter buffer is allocated on the device.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dkr_s9c2d_shard import profile,stage,self_test

# Compact recipe mirrors the explicit wrappers; the source audit checks every
# entry, parameter name, type, order and image assignment.
from dkr_s9c2d_shard import tensor,scalar
def rows():
    def t(n,d,a="read"):
        return tensor(n,d,1,a)
    def s(n,d="i32"):
        return scalar(n,d)
    def ints(names):
        return [s(n) for n in names.split()]
    def typed(names,d):
        return [s(n,d) for n in names.split()]
    def base(dt,sd):
        return [t(n,dt) for n in ("qkv","b","a")]+[t("al","f32"),t("db","f32"),t("state",sd,"read_write"),t("slots","u32")]
    def row(image,op,tag,args,block):
        return dict(image=image,op=op,entry="dkr_gdn_spec_"+tag,args=args,block=block)
    for dt in ("f16","bf16"):
        for sd in ("f16","bf16","f32"):
            for mode in ("km","vm","vm128"):
                yield row(1,"plain."+mode,f"plain_{dt}_{sd}_{mode}",
                    base(dt,sd)+[t("out",dt,"write")]+typed("bs0 bs1 bs2 as0 as1 as2","i64")+
                    ints("batch seq kh vh kd vd lanes tiled cap"),[32,4 if mode=="vm128" else 1,1])
    pairs=typed("b01 b2a0 a12 g01 g23 batch_seq kh_vh lanes_tiled","u64")
    for dt in ("f16","bf16"):
        for sd in ("f16","bf16","f32"):
            for reduction in (("paired","serial") if sd=="f32" else ("paired",)):
                for mode in ("checkpoint","batch","direct"):
                    for output in (("native","fp8") if dt=="bf16" else ("native",)):
                        fp8=output=="fp8"
                        args=base(dt,sd)+[t("gate",dt),t("weight",dt)]
                        args += [t("out8","fp8_e4m3","write"),t("scales","f32","write")] if fp8 else [t("out",dt,"write")]
                        if mode=="batch":
                            args += [t(n,"f32","write") for n in ("tk","td","tr")]
                        if mode=="direct":
                            args += [t("pk","f32","read_write"),t("bank","u32"),t("pd","f32","read_write"),
                                     t("pr","f32","read_write"),t("keep","u32"),t("epoch","u32"),t("applied","u32","read_write")]
                        args += pairs+[s("cap"),s("eps","f32"),s("sigmoid")]
                        if fp8: args += [s("scale","u64")]
                        if mode=="direct": args += [s("pending","u64")]
                        yield row(2,f"fused.{mode}.{output}.{reduction}",f"fused_{dt}_{sd}_{mode}_{output}_{reduction}",args,[256,1,1])
    ddims=ints("batch cap kh vh tiled")
    logs=[t(n,"f32","read_write") for n in ("key","delta","decay")]+[t("cursor","u32")]
    for output in ("native","fp8"):
        fp8=output=="fp8"
        args=base("bf16","f32")+[t("gate","bf16"),t("weight","bf16")]+logs
        args += [t("out8","fp8_e4m3","write"),t("scales","f32","write")] if fp8 else [t("out","bf16","write")]
        args += typed("bs0 bs1 as0 as1 gs0 gs1 gs2","i64")+ddims+[s("eps","f32"),s("sigmoid")]
        if fp8: args += [s("scale","u64")]
        yield row(3,"deferred."+output,"deferred_"+output,args,[256,1,1])
    for mode in ("advance","clear"):
        yield row(3,"deferred."+mode,"deferred_"+mode,[t("cursor","u32","read_write"),t("slots","u32")]+ints("batch cap"),[256,1,1])
    yield row(3,"deferred.flush","deferred_flush",[t("state","f32","read_write"),t("slots","u32")]+
        [t(n,"f32") for n in ("key","delta","decay")]+[t("cursor","u32")]+ddims,[256,1,1])

ROWS=list(rows())

def images(cc):
    return (1,2,3) if cc>=80 else (1,2)

def plan(cc,image):
    if image not in images(cc):
        raise ValueError("speculative image not admitted for target")
    return [profile(cc,"gdn.spec."+r["op"]+".v1","cuda."+r["entry"]+".v1",
                    r["entry"],r["block"],r["args"],f"gdn_speculative_{image}.ptx")
            for r in ROWS if r["image"]==image]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc",type=int,required=True)
    p.add_argument("--image",type=int)
    p.add_argument("--list-images",action="store_true")
    p.add_argument("--output",type=Path)
    p.add_argument("--ptx-file",type=Path)
    p.add_argument("--json-out",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    ids=images(a.cc)
    if a.list_images:
        print(*ids,sep="\n")
        return
    if a.image is not None and a.image not in ids:
        p.error("image is not admitted for target")
    selected=(a.image,) if a.image is not None else ids
    rows=[plan(a.cc,i) for i in selected]
    count=sum(map(len,rows))
    max_args=max(len(r["arguments"]) for group in rows for r in group)
    if a.self_test:
        for group in rows:
            self_test(a.cc,group)
        assert max_args<=32
        print(f"DKR_R126_S9C2D3B2B_SPEC_SELF_TEST=PASS cc={a.cc} images={len(rows)} profiles={count} max_args={max_args} synthetic_ptx=true")
    elif a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(dict(profiles=[r for group in rows for r in group],physical_manifests=len(rows),compilation=False,link_qualified=False),indent=2)+"\n")
        print(f"DKR_R126_S9C2D3B2B_SPEC_PLAN=COMPLETE cc={a.cc} images={len(rows)} profiles={count} max_args={max_args} compilation=false")
    elif a.output is not None and a.ptx_file is not None and a.image is not None:
        stage(a.output/f"sm{a.cc}"/"gdn-speculative"/f"image-{a.image:02}",a.cc,rows[0],a.ptx_file.read_bytes())
    else:
        p.error("specify --self-test, --json-out, or --image --output --ptx-file")

if __name__=="__main__":
    main()
