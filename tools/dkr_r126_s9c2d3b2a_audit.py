#!/usr/bin/env python3
"""S9C.2D.3B.2A source-only audit: snapshot commits and convolution checkpoints."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask
BASE_LOGICAL=3951
BASE_PHYSICAL=530
MIGRATED=("gdn_speculative_state_commit","gdn_speculative_conv_checkpoints")
RESIDUAL=("gdn_speculative_recurrence_checkpoints","gdn_deferred_recurrence_rmsnorm_gate_value_major_128","gdn_flush_deferred_state_value_major_128")
ARCH=("gdn_mma_prefill_init","gdn_mma_prefill","gdn_mma_fused_prefill","mistralrs_flashinfer_gdn_sm90_launch","mistralrs_flashinfer_gdn_sm90_workspace_size")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,message):
    if not ok:
        raise ValueError(message)

def audit(root):
    path=root/"kernel-packages/cuda/gdn-checkpoint/build.py"
    spec=importlib.util.spec_from_file_location("checkpoint_plan",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows=module.plan(61)
    for cc,profiles in ((61,rows),(80,module.plan(80))):
        require(len(profiles)==22 and len({r["entry"] for r in profiles})==22,"profile inventory")
        require(len({r["artifact"] for r in profiles})==1,"one physical image")
        require(all(r["target_cc"]==cc and r["shared_mem_bytes"]==0 and len(r["arguments"])<=32 for r in profiles),"ABI/resource budget")
        require(sum(r["operation"]=="gdn.commit.conv.v1" for r in profiles)==2,"conv commit profiles")
        require(sum(r["operation"].startswith("gdn.commit.rec.") for r in profiles)==12,"recurrent commit profiles")
        require(sum(r["operation"].startswith("gdn.checkpoint.") for r in profiles)==8,"checkpoint profiles")
        for r in profiles:
            args=r["arguments"]
            if "checkpoint.pending" in r["operation"]:
                require(len(args)==22 and all(a["kind"]=="tensor" for a in args[:9]),"pending ABI")
            elif "checkpoint.conv" in r["operation"]:
                require(len(args)==15 and all(a["kind"]=="tensor" for a in args[:5]),"plain ABI")
            require(any(a["name"]=="capacity" and a["dtype"]=="i32" for a in args),"pool capacity fence")
        paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64) for r in profiles])
        require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,"paired manifest budget")
    core=root/"mistralrs-core/src/cuda"
    src=(core/"gdn.rs").read_text()
    ffi=(core/"ffi.rs").read_text()
    spans=guarded_ranges(src)
    guarded,residual,arch={},{},{}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(src)):
        n=m.group(1)
        group=guarded if any(a<=m.start()<b for a,b in spans) else arch if n in ARCH else residual
        group.setdefault(n,[]).append(src.count("\n",0,m.start())+1)
    require(set(residual)==set(RESIDUAL),"unexpected residual FFI: "+str(sorted(residual)))
    require(set(arch)==set(ARCH),"unexpected architecture providers")
    for n in MIGRATED:
        require(n in guarded and GUARD+"\n    pub(crate) fn "+n+"(" in ffi,"missing FFI fence "+n)
    h=(core/"gdn_checkpoint_device.cuh").read_text()
    require(h.count("rows > seq_len")==2 and h.count("slot_indices[batch_idx] >= (uint32_t)capacity")==2,"snapshot commit prefix/slot guards")
    require(h.count("active_slot >= (uint32_t)capacity")==2,"checkpoint slot guards")
    require(h.count("if (!channel_valid)")==2 and h.count("__syncthreads();")>=2,"partial-channel barrier paths")
    a=(core/"gdn_checkpoint_dynamic.rs").read_text()
    require("p::preflight(device,\"gdn.commit.conv.v1\"" in a and "p::preflight(device,rop" in a and
            a.index("p::preflight(device,rop")<a.index("p::launch(device,\"gdn.commit.conv.v1\""),"commit preflight order")
    require("disjoint(output,input)" in a and "cuda_inplace_ptr_with_guard" in a and "vd%4!=0" in a,"snapshot/geometry guards")
    require("checkpoint_dynamic::commit::<T>" in src and "checkpoint_dynamic::conv::<T>" in src,"typed dispatch")
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    legacy=(core/"gdn.cu").read_text()
    producer=(path.parent/"gdn_checkpoint.cu").read_text()
    require(all("gdn_checkpoint_device.cuh" in s for s in (legacy,producer,builder)),"shared device bodies")
    require('name: "gdn-checkpoint"' in builder and 'pkg.name == "gdn-checkpoint"' in builder,"Cudaforge registration")
    # Count the actual residual fused device signature, not the host FFI.
    # The ABI limit must not be silently increased to migrate this operation.
    fused=re.search(r'void gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel\s*\((.*?)\)\s*\{',legacy,re.S)
    require(fused is not None,"residual fused kernel missing")
    fused_args=len(fused.group(1).split(","))
    require(fused_args>32,"residual fused ABI changed: re-evaluate packing plan")
    require("distinct slots. Pending metadata" in a and "for input in inputs.into_iter().chain(reads)" in a,
            "checkpoint-group and alias contract")
    logical=BASE_LOGICAL+len(rows)
    require(logical<=4096,"logical budget")
    return dict(schema="dkr.r126.s9c2d3b2a.source.v1",status="PARTIAL_SOURCE",profiles=len(rows),images=1,
        max_arguments=max(len(r["arguments"]) for r in rows),migrated_ffi={n:guarded[n] for n in MIGRATED},
        gdn_residual_ffi=residual,arch_provider_refs=arch,projected_physical_upper=BASE_PHYSICAL+1,
        physical_savings_required=BASE_PHYSICAL+1-512,projected_logical=logical,logical_headroom=4096-logical,
        residual_fused_arguments=fused_args,abi_limit=32,budget_is_projection=True,compilation=False,link_qualified=False,numeric_qualified=False,full_closure=False,
        outstanding_contracts=["unique_destination_slots_or_checkpoint_groups","same_stream","epoch_wrap_collision",
            "no_rollback_after_device_execution_failure"])

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
    print(f"DKR_R126_S9C2D3B2A_SOURCE=PARTIAL profiles={r['profiles']} images={r['images']} "
          f"guarded_ffi={len(r['migrated_ffi'])} gdn_residual={len(r['gdn_residual_ffi'])} "
          f"arch_provider_refs={len(r['arch_provider_refs'])} max_args={r['max_arguments']} "
          f"projected_physical={r['projected_physical_upper']} projected_logical={r['projected_logical']} "
          f"logical_headroom={r['logical_headroom']} compilation=false link_qualified=false full_closure=false")
if __name__=="__main__":
    main()
