#!/usr/bin/env python3
"""Read-only S9C.2 source dispatch contract. Not a Rust compiler or link proof."""
import argparse
import json
from pathlib import Path

# Each check includes the positive typed route and/or explicit negative gate.
# A lexical marker is not sufficient evidence for compiled coverage.
CASES = {
    "mxfp4_dense": ("mxfp4/mod.rs", 'cfg!(feature = "dkr-static-closure-probe") || ffi::HAVE_MXFP4_GEMM_KERNELS'),
    "mxfp4_no_static": ("mxfp4/ops.rs", "DKR_STATIC_CLOSURE: MXFP4 dense signature not admitted"),
    "scalar_fp8_encode": ("scalar_fp8/ops.rs", "DKR_STATIC_CLOSURE: FP8 quant cast signature not admitted"),
    "scalar_fp8_decode": ("scalar_fp8/ops.rs", "DKR_STATIC_CLOSURE: FP8 dequant cast signature not admitted"),
    "blockwise_fp8_dense": ("blockwise_fp8/mod.rs", "FP8 dense signature not admitted (no dense materialization)"),
    "blockwise_fp8_routed": ("blockwise_fp8/mod.rs", "FP8 routed signature not admitted (no dense materialization)"),
    "blockwise_fp8_quant_gap": ("blockwise_fp8/ops.rs", "blockwise FP8 weight quant requires migration"),
    "blockwise_fp8_dequant_gap": ("blockwise_fp8/ops.rs", "blockwise FP8 dequant requires migration"),
    "nvfp4_dense": ("nvfp4/mod.rs", "DKR_STATIC_CLOSURE: NVFP4 dense signature not admitted"),
    "nvfp4_routed": ("nvfp4/mod.rs", "DKR_STATIC_CLOSURE: NVFP4 routed signature not admitted"),
    "nvfp4_no_portable_ffi": ("nvfp4/portable.rs", '#[cfg(not(feature = "dkr-static-closure-probe"))]\nunsafe extern "C"'),
    "routed_bias": ("routed_gemv/mod.rs", "unbiased.bias = None;"),
    "routed_no_static_module": ("routed_gemv/mod.rs", '#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]\nmod cuda;'),
    "gguf_dense_reference": ("gguf/mod.rs", "let x = self.w.forward(&a_f32)?;"),
    "gguf_moe_typed": ("gguf/cuda.rs", "super::fast_mmq::moe_dynamic::indexed(qtensor,x,ids)"),
    "gguf_moe_grouped": ("gguf/fast_mmq.rs", "moe_dynamic::grouped(&[gate,up]"),
    "gguf_moe_worklist": ("gguf/moe_dynamic.rs", 'run!(p.work,bp,tp,cp,ei,ai,ci);'),
    "gguf_moe_dispatch": ("gguf/cuda.rs", "super::fast_mmq::moe_dynamic::dispatch("),
    "gguf_group_disabled": ("gguf/group.rs", '#[cfg(all(feature = "cuda", not(feature = "dkr-static-closure-probe")))]\nmod cuda {'),
    "gguf_gated_mix_optional": ("gguf/gated_mix.rs", "Optional fusion: caller must execute the unfused operation."),
}
FORCED = ("quantized_gemv_dynamic.rs", "quantized_gemm_dynamic.rs",
          "nvfp4/gemv_dynamic.rs", "routed_gemv/dynamic.rs")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    root=a.source_root/"mistralrs-quant"/"src"
    checks={}
    for key,(path,marker) in CASES.items():
        checks[key]=marker in (root/path).read_text()
    for path in FORCED:
        checks["forced:"+path]='if cfg!(feature = "dkr-static-closure-probe") { return true; }' in (root/path).read_text()
    failed=[name for name,ok in checks.items() if not ok]
    if failed:
        raise SystemExit("DKR_R126_S9C2_SOURCE=FAIL missing="+",".join(failed))
    report=dict(schema="dkr.r126.s9c2.dispatch.source.v1",checks=checks,
                status="SOURCE_MARKERS_PRESENT",compilation=False,
                full_link_qualified=False,full_model_qualified=False,
                unresolved=["blockwise_fp8_weight_quant_dequant","gguf_indexed_mmvq_optimized",
                            "gguf_direct_mmq_mmvq","higher_level_routed_moe_glu",
                            "other_core_and_quant_ffi"])
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(report,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2_SOURCE=COMPLETE checks={len(checks)} compilation=false link_qualified=false")
if __name__=="__main__":
    main()
