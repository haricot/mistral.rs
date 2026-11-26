#!/usr/bin/env python3
"""S9C.2C source ABI/budget/guard audit. No CUDA, Cargo or link proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges, mask

BASE_PHYSICAL=523
BASE_LOGICAL=3761
MAX_PHYSICAL=512
MAX_LOGICAL=4096
MAX_ARGS=32
MAX_MANIFEST=256*1024
REUSED=("rounded-rope","add-rms-norm","rms-variants","reduce-norm","gated-add","indexed-copy","input-pad")
# Explicit migrated declarations; GDN FP8 output is intentionally reserved for S9C.2D.
SYMBOLS=("qk_rms_norm_rope","qk_rms_norm_rope_positions","qkv_rms_norm_rope_positions","rope_sincos_positions",
         "gdn_rmsnorm_gated","pack_completion_input_u32","rms_norm_strided_4d_f16",
         "rms_norm_strided_4d_bf16","rms_norm_strided_4d_f32")

def audit(root):
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    registered=set(re.findall(r'Package\s*\{\s*name:\s*"([^"]+)"',builder))
    assert set(REUSED)|{"attention-prep"}<=registered
    path=root/"kernel-packages/cuda/attention-prep/build.py"
    spec=importlib.util.spec_from_file_location("attention_prep_plan",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    images=module.plan(61)
    rows=[r for image in images.values() for r in image]
    assert len(images)==3 and len(rows)==88
    assert len({r["implementation_id"] for r in rows})==88
    assert all(len(r["arguments"])<=MAX_ARGS for r in rows)
    sizes=[]
    for profiles in images.values():
        assert 1<=len(profiles)<=64
        paired=[dict(r,sha256="0"*64,cubin_artifact="attention_prep_1.cubin",cubin_sha256="0"*64) for r in profiles]
        size=len((json.dumps(dict(schema="DKR-KERNEL-SHARD-V1",profiles=paired),indent=2)+"\n").encode())
        assert size<=MAX_MANIFEST
        sizes.append(size)
    ffi=(root/"mistralrs-core/src/cuda/ffi.rs").read_text()
    guard='#[cfg(not(feature = "dkr-static-closure-probe"))]'
    for name in SYMBOLS:
        assert guard+"\n    pub(crate) fn "+name+"(" in ffi, name
    guarded=0
    paths=("mistralrs-core/src/ops.rs","mistralrs-core/src/cuda/input_packing.rs",
           "mistralrs-core/src/cuda/gdn.rs","mistralrs-quant/src/rotary/mod.rs")
    names=SYMBOLS+("rotary_embedding","rotary_embedding_positions")
    pattern=re.compile(r'\bffi::('+"|".join(names)+r')\s*\(')
    for path in paths:
        source=(root/path).read_text()
        spans=guarded_ranges(source)
        for match in pattern.finditer(mask(source)):
            if not any(a<=match.start()<b for a,b in spans):
                raise ValueError(f"{path}: unguarded {match.group(1)} at line {source.count(chr(10),0,match.start())+1}")
            guarded+=1
    # The strided RMSNorm launcher is a macro, so check its guarded call explicitly.
    ops=(root/paths[0]).read_text()
    norm=ops.split("pub(crate) fn try_cuda_rms_norm_strided_4d(",1)[1].split("pub fn cuda_rms_norm_residual(",1)[0]
    match=re.search(r'ffi::\$ffi_fn\s*\(',mask(norm))
    assert match and any(a<=match.start()<b for a,b in guarded_ranges(norm))
    rotary=(root/paths[3]).read_text()
    assert '#[cfg(all(feature = "cuda",not(feature = "dkr-static-closure-probe")))]\nmod ffi;' in rotary
    physical=BASE_PHYSICAL+len(images);logical=BASE_LOGICAL+len(rows)
    assert logical<=MAX_LOGICAL
    return dict(schema="dkr.r126.s9c2c.source.v1",new_profiles=len(rows),new_images=len(images),
        reused_families=list(REUSED),guarded_direct_calls=guarded,guarded_macro_calls=1,
        excluded_declarations=len(SYMBOLS),max_arguments=max(len(r["arguments"]) for r in rows),
        paired_manifest_bytes=sizes,projected_physical_upper=physical,projected_logical=logical,
        physical_savings_required=max(0,physical-MAX_PHYSICAL),logical_headroom=MAX_LOGICAL-logical,
        budget_is_projection=True,remaining=["gdn_quantized_rmsnorm","generic_candle_normalization_link_qualification",
            "other_core_and_quant_ffi","actual_cuda_numeric_and_perf_qualification"],
        compilation=False,link_qualified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:json.dump(r,f,indent=2);f.write("\n")
    print(f"DKR_R126_S9C2C_SOURCE=PASS profiles={r['new_profiles']} images={r['new_images']} reused={len(r['reused_families'])} guarded_direct_calls={r['guarded_direct_calls']} guarded_macro_calls=1 max_args={r['max_arguments']} projected_logical={r['projected_logical']} logical_headroom={r['logical_headroom']} compilation=false link_qualified=false")
if __name__=="__main__":main()
