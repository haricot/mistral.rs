#!/usr/bin/env python3
"""S9C.2B2: SM61 MMQ historical geometry and Q8_1 layout budget (no compilation)."""
import argparse
import json
from pathlib import Path

FORMATS={
    # Historical mmq_get_dp4a_tile_x_sizes at mmq_y=64, plus activation layout.
    "q4_0": (64*33,64*32//4+64//4,0,"DS4"),
    "q4_1": (64*33,64*32//4+64//4,0,"DS4"),
    "q5_0": (64*65,64*64//8+64//4,0,"D4"),
    "q5_1": (64*65,64*64//8+64//4,0,"DS4"),
    "q8_0": (64*65,64*64//8+64//4,0,"D4"),
    "q2_k": (64*65,64*33,0,"D2S6"),
    "q3_k": (64*65,64,64*32//8+64//8,"D4"),
    "q4_k": (64*33,64*32//32,64*32//8+64//8,"DS4"),
    "q5_k": (64*65,64*32//32+64//32,64*32//8+64//8,"DS4"),
    "q6_k": (64*65,64*32//32+64//32,64*32//8+64//8,"D4"),
}
SHARED_LIMIT=48*1024

def shared(fmt,tile):
    qs,dm,sc,_=FORMATS[fmt]
    y_bytes=tile*144
    return 4*tile+4*(qs+dm+sc)+(y_bytes+1023)//1024*1024

def plan(root):
    # Check audited source assumptions; these markers are not a parser.
    kernels=root/"mistralrs-quant"/"kernels"/"mmq_gguf"
    header=(kernels/"mmq_gguf.cuh").read_text()
    rust=(root/"mistralrs-quant/src/gguf/fast_mmq.rs").read_text()
    for marker in ("#define MMQ_TILE_NE_K 32","static size_t mmq_get_nbytes_shared",
                   "return nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int));",
                   "sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2)",
                   "__CUDA_ARCH__ < GGML_CUDA_CC_VOLTA"):
        if marker not in header:
            raise ValueError("MMQ source contract drift: "+marker)
    for fmt in FORMATS:
        src=(kernels/f"mmq_instance_{fmt}.cu").read_text()
        if "mmq_x += 8" not in src or "DEFINE_MMQ_MOE_LAUNCHER" not in src:
            raise ValueError("MMQ instance policy drift: "+fmt)
    if not all("DsLayout::"+name in rust for name in ("D4","DS4","D2S6")):
        raise ValueError("missing historical Q8_1 layouts")
    rows=[dict(format=fmt,mmq_x=x,mmq_y=64,block=[32,8,1],
               q8_1_layout=values[3],shared_bytes=shared(fmt,x),
               admitted=shared(fmt,x)<=SHARED_LIMIT)
          for fmt,values in FORMATS.items() for x in range(8,65,8)]
    eligible=sum(r["admitted"] for r in rows)
    # Dense 3 outputs x 2 row bounds, indexed MoE F32 x 2 row bounds.
    # Separate plain and indexed Q8_1 quantization; GLU/fixup are not counted.
    projected_profiles=eligible*8+3*3*2
    projected_images=11 # 10 format images (64 profiles max) + one quantizer image
    return dict(schema="dkr.r126.gguf.mmq.geometry.v1",cc=61,geometries=rows,
                eligible=eligible,quarantined=len(rows)-eligible,
                max_shared=max(r["shared_bytes"] for r in rows),
                projected_profiles=projected_profiles,projected_images=projected_images,
                combined_physical_upper=482+projected_images,
                combined_logical=2410+projected_profiles,
                compilation=False,device_abi_implemented=False,link_qualified=False,
                remaining=["typed_device_entry","scalar_fastdiv_contract","indexed_scheduler_metadata",
                           "mmq_q8_1_dense_and_indexed_quantize","fused_glu","stream_k_non_sm61"])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path,required=True)
    a=p.parse_args()
    r=plan(a.source_root)
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    a.json_out.write_text(json.dumps(r,indent=2)+"\n")
    print(f"DKR_R126_S9C2B_MMQ_PLAN=COMPLETE cc=61 eligible={r['eligible']} "
          f"quarantined={r['quarantined']} max_shared={r['max_shared']} "
          f"projected_profiles={r['projected_profiles']} projected_images={r['projected_images']} "
          f"physical_upper={r['combined_physical_upper']} logical_total={r['combined_logical']} "
          "compilation=false device_abi_implemented=false")
if __name__=="__main__":
    main()
