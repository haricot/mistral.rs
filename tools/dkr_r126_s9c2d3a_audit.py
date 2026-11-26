#!/usr/bin/env python3
"""Source/ABI audit of S9C.2D.3A, not speculative-state or link qualification."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

BASE_PHYSICAL=527
BASE_LOGICAL=3904
MAX_PHYSICAL=512
MAX_LOGICAL=4096
MIGRATED=("gdn_packed_to_padded","gdn_padded_to_packed","gdn_extract_ragged_conv_state",
          "gdn_prepare_recurrence","fused_gdn_gating","gdn_rmsnorm_gated_quantized_bf16",
          "gdn_decode_recurrence")
SPECULATIVE=("gdn_speculative_state_commit","gdn_speculative_conv_checkpoints",
             "gdn_speculative_recurrence_checkpoints","gdn_deferred_recurrence_rmsnorm_gate_value_major_128",
             "gdn_flush_deferred_state_value_major_128","gdn_speculative_transition_commit_batched",
             "gdn_speculative_transition_stage_batched","gdn_pending_transition_publish_batched",
             "gdn_pending_transition_apply_batched")
ARCH=("gdn_mma_prefill_init","gdn_mma_prefill","gdn_mma_fused_prefill",
      "mistralrs_flashinfer_gdn_sm90_launch","mistralrs_flashinfer_gdn_sm90_workspace_size")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,message):
    if not ok:
        raise ValueError(message)

def plan(root,family,cc):
    path=root/"kernel-packages/cuda"/family/"build.py"
    spec=importlib.util.spec_from_file_location(family.replace("-","_"),path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=module.plan(cc)
    require(1<=len(rows)<=64,f"{family}: profile count")
    require(len({r["entry"] for r in rows})==len(rows),f"{family}: duplicate entry")
    require(len({r["artifact"] for r in rows})==1,f"{family}: image count")
    require(all(r["target_cc"]==cc and len(r["arguments"])<=32 for r in rows),f"{family}: ABI budget")
    paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
        dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64) for r in rows])
    require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,f"{family}: manifest size")
    return rows

def inventory(source):
    spans=guarded_ranges(source)
    guarded={}
    residual={}
    arch={}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(source)):
        name=m.group(1)
        group=guarded if any(a<=m.start()<b for a,b in spans) else arch if name in ARCH else residual
        group.setdefault(name,[]).append(source.count("\n",0,m.start())+1)
    require(all(name in guarded for name in MIGRATED),"missing guarded migrated references")
    require(set(residual)==set(SPECULATIVE),"unexpected residual GDN FFI inventory: "+str(sorted(residual)))
    require(set(arch)==set(ARCH),"unexpected architecture provider inventory")
    return guarded,residual,arch

def audit(root):
    prep=plan(root,"gdn-preparation",61)
    decode=plan(root,"gdn-decode",61)
    prep80=plan(root,"gdn-preparation",80)
    decode80=plan(root,"gdn-decode",80)
    require((len(prep),len(decode),len(prep80),len(decode80))==(14,18,14,36),"profile totals")
    require(sum("ragged" in r["operation"] for r in prep)==6,"ragged variants")
    require(sum(r["operation"]=="gdn.prepare.v1" for r in prep)==4,"prepare variants")
    require(sum(r["operation"]=="gdn.gating.v1" for r in prep)==2,"gating variants")
    require(sum("gdn.fp8." in r["operation"] for r in prep)==2,"FP8 variants")
    require(all(r["shared_mem_bytes"]<=2560 for r in decode),"decode dynamic shared-memory ceiling")
    require(all(r["operation"] in {"gdn.decode.k64.v1","gdn.decode.k128.v1","gdn.decode.fallback.v1"} for r in decode),
            "architecture-specialized decode admitted to SM61")
    require(sum(r["operation"]=="gdn.decode.fallback.v1" and r["shared_mem_bytes"]==2560 for r in decode)==6,"fallback ceiling")
    require(sum(r["operation"]=="gdn.decode.cooperative.v1" for r in decode80)==6 and
            sum(r["operation"]=="gdn.decode.pipelined.v1" for r in decode80)==6 and
            sum(r["operation"] in {"gdn.decode.vm4.v1","gdn.decode.vm32.v1"} for r in decode80)==6,"modern decode variants")
    core=root/"mistralrs-core/src/cuda"
    source=(core/"gdn.rs").read_text()
    ffi=(core/"ffi.rs").read_text()
    guarded,residual,arch=inventory(source)
    for name in MIGRATED:
        require(GUARD+"\n    pub(crate) fn "+name+"(" in ffi,"missing declaration guard "+name)
    legacy=(core/"gdn.cu").read_text()
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    for family in ("gdn-preparation","gdn-decode"):
        stem=family.replace("-","_")
        header=stem+"_device.cuh"
        producer=(root/"kernel-packages/cuda"/family/(stem+".cu")).read_text()
        require(header in legacy and header in producer and header in builder,"shared source/cache dependency "+family)
        require(f'name: "{family}"' in builder and f'pkg.name == "{family}"' in builder,"builder registration "+family)
    header=(core/"gdn_preparation_device.cuh").read_text()
    require("0xffffu << ((threadIdx.x / 16) * 16)" in header and "__shfl_xor_sync(mask" in header,"FP8 partial half-warp mask")
    adapter=(core/"gdn_preparation_dynamic.rs").read_text()
    require("LaunchPlanCache" in adapter and "with_prevalidated_launch" in adapter and "prevalidate_launch_with_shared" in adapter,
            "typed cached preparation plans")
    adapter=(core/"gdn_decode_dynamic.rs").read_text()
    require("select_decode_kernel(policy,requested_kernel)" in adapter and "cuda_recurrent_state_inplace_ptr_with_guard" in adapter and
            "recurrence_dynamic::slot_tensor" in adapter and "launch_shared" in adapter,"decode policy/guard route")
    require("decode_dynamic::decode(launch)" in source and "p::launch(" in source,"typed dispatch missing")
    physical=BASE_PHYSICAL+2
    logical=BASE_LOGICAL+len(prep)+len(decode)
    require(logical<=MAX_LOGICAL,"logical budget exceeded")
    return dict(schema="dkr.r126.s9c2d3a.source.v1",status="PARTIAL_SOURCE",
        images=2,profiles=len(prep)+len(decode),modern_profiles=len(prep80)+len(decode80),
        preparation_profiles=len(prep),decode_profiles=len(decode),max_arguments=max(len(r["arguments"]) for r in prep+decode),
        migrated_ffi={name:guarded[name] for name in MIGRATED},gdn_residual_ffi=residual,arch_provider_refs=arch,
        projected_physical_upper=physical,physical_savings_required=max(0,physical-MAX_PHYSICAL),
        projected_logical=logical,logical_headroom=MAX_LOGICAL-logical,budget_is_projection=True,
        compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print(f"DKR_R126_S9C2D3A_SOURCE=PARTIAL profiles={r['profiles']} images={r['images']} "
          f"guarded_ffi={len(r['migrated_ffi'])} gdn_residual={len(r['gdn_residual_ffi'])} "
          f"arch_provider_refs={len(r['arch_provider_refs'])} max_args={r['max_arguments']} "
          f"projected_physical={r['projected_physical_upper']} projected_logical={r['projected_logical']} "
          f"logical_headroom={r['logical_headroom']} compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
