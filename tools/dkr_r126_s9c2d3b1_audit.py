#!/usr/bin/env python3
"""Source audit for batched GDN transaction closure; remaining checkpoint FFI is explicit."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

BASE_PHYSICAL=529
BASE_LOGICAL=3936
MIGRATED=("gdn_speculative_transition_commit_batched","gdn_speculative_transition_stage_batched",
          "gdn_pending_transition_publish_batched","gdn_pending_transition_apply_batched")
RESIDUAL=("gdn_speculative_state_commit","gdn_speculative_conv_checkpoints",
          "gdn_speculative_recurrence_checkpoints","gdn_deferred_recurrence_rmsnorm_gate_value_major_128",
          "gdn_flush_deferred_state_value_major_128")
ARCH=("gdn_mma_prefill_init","gdn_mma_prefill","gdn_mma_fused_prefill",
      "mistralrs_flashinfer_gdn_sm90_launch","mistralrs_flashinfer_gdn_sm90_workspace_size")
GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'

def require(ok,message):
    if not ok:
        raise ValueError(message)

def inventory(source):
    ranges=guarded_ranges(source)
    guarded={}
    residual={}
    arch={}
    for m in re.finditer(r'crate::cuda::ffi::([A-Za-z_]\w*)',mask(source)):
        name=m.group(1)
        group=guarded if any(a<=m.start()<b for a,b in ranges) else arch if name in ARCH else residual
        group.setdefault(name,[]).append(source.count("\n",0,m.start())+1)
    require(all(name in guarded for name in MIGRATED),"missing guarded transaction references")
    require(set(residual)==set(RESIDUAL),"unexpected remaining GDN FFI: "+str(sorted(residual)))
    require(set(arch)==set(ARCH),"unexpected architecture provider inventory")
    return guarded,residual,arch

def plan(root,cc):
    path=root/"kernel-packages/cuda/gdn-transaction/build.py"
    spec=importlib.util.spec_from_file_location("gdn_transaction_plan",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plan(cc)

def check_plan(rows,cc):
    require(len(rows)==15 and len({r["entry"] for r in rows})==15,"transaction profile count")
    require(len({r["operation"] for r in rows})==15,"table pointee dtype pairs must have distinct operation IDs")
    require(len({r["artifact"] for r in rows})==1,"one physical transaction image")
    for r in rows:
        require(r["target_cc"]==cc and r["shared_mem_bytes"]==0,"transaction target/dynamic shared budget")
        args=r["arguments"]
        require(args[0]["dtype"]=="i64" and args[0]["name"]=="pointer_table" and args[0]["access"]=="read","internal pointer-table ABI")
        kind=r["operation"].split(".")[2]
        count,block={"commit":(15,256),"stage":(13,256),"publish":(7,1),"apply":(14,256)}[kind]
        require(len(args)==count and r["block"]==[block,1,1],"transaction launch contract "+kind)
        require(args[1]["dtype"]=="u32" and args[1]["kind"]=="tensor","transaction index type")
        if kind!="apply":
            require(args[2]["dtype"]=="u32" and args[2]["kind"]=="tensor","transaction slots type")
        require(all(a["kind"]=="scalar" and a["dtype"]=="i32" for a in args[2 if kind=="apply" else 3:]),"scalar ABI")
    paired=dict(schema="DKR-KERNEL-SHARD-V1",profiles=[
        dict(r,sha256="0"*64,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,cubin_sha256="0"*64)
        for r in rows])
    require(len((json.dumps(paired,indent=2)+"\n").encode())<=256*1024,"paired manifest size")
    require({kind:sum(r["operation"].split(".")[2]==kind for r in rows) for kind in ("commit","stage","publish","apply")}
            ==dict(commit=6,stage=2,publish=1,apply=6),"transaction variants")

def audit(root):
    rows=plan(root,61)
    check_plan(rows,61)
    check_plan(plan(root,80),80)
    core=root/"mistralrs-core/src/cuda"
    dispatch=(core/"gdn.rs").read_text()
    guarded,residual,arch=inventory(dispatch)
    ffi=(core/"ffi.rs").read_text()
    for n in MIGRATED:
        require(GUARD+"\n    pub(crate) fn "+n+"(" in ffi,"missing FFI declaration guard "+n)
    header=(core/"gdn_transition_device.cuh").read_text()
    legacy=(core/"gdn.cu").read_text()
    producer=(root/"kernel-packages/cuda/gdn-transaction/gdn_transaction.cu").read_text()
    builder=(root/"mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs").read_text()
    require(all("gdn_transition_device.cuh" in s for s in (legacy,producer,builder)),"shared source and Cudaforge cache dependency")
    require('name: "gdn-transaction"' in builder and 'pkg.name == "gdn-transaction"' in builder,"Cudaforge registration")
    require("rows > seq_len || rows > GDN_TX_MAX_ROWS" in header and "slot >= (uint32_t)capacity" in header,"commit invalid-prefix/slot fence")
    require(header.count("if (applied_epochs[applied_offset] == epoch)")==2 and
            header.count("applied_epochs[applied_offset] = epoch;")==2,"separate convolution/recurrent epoch guards")
    require("pending_key_bank[slot] ^ GDN_TX_BANK_MASK" in header and
            "next_epoch == 0 ? 1 : next_epoch" in header,"bank flip/epoch rollover")
    stage=header[header.index("gdn_speculative_transition_stage_batched_kernel("):header.index("gdn_pending_transition_publish_batched_kernel(")]
    require(stage.count("const uint32_t next_epoch = pending_epoch[slot] + 1;")==2 and
            stage.count("blockIdx.x == 0 && threadIdx.x == 0")==2,"single-writer stage epoch fence")
    adapter=(core/"gdn_transaction_dynamic.rs").read_text()
    require("p::launch(device,operation(kind,activation,state)?" in adapter and
            "table_segments(kind)" in adapter and "table.dtype()!=DType::I64" in adapter and
            "kind!=Kind::Apply" in adapter,"typed pointer table route")
    require(all("tx::Kind::"+kind in dispatch for kind in ("Commit","Stage","Publish","Apply")),"typed call sites")
    start=dispatch.index("pub fn speculative_transition_commit_batched_cuda(")
    end=dispatch.index('#[cfg(not(feature = "cuda"))]',start)
    commit=dispatch[start:end]
    require("minimum_capacity=minimum_capacity.min(conv_capacity)" in commit and
            "cuda_recurrent_state_inplace_ptr_with_guard" in commit and
            "drop(pointer_guards)" in commit,"commit capacity/pointee lifetime")
    physical=BASE_PHYSICAL+1
    logical=BASE_LOGICAL+len(rows)
    require(logical<=4096,"logical budget exceeded")
    return dict(schema="dkr.r126.s9c2d3b1.source.v1",status="PARTIAL_SOURCE",
                profiles=len(rows),images=1,max_arguments=max(len(r["arguments"]) for r in rows),
                migrated_ffi={n:guarded[n] for n in MIGRATED},gdn_residual_ffi=residual,arch_provider_refs=arch,
                projected_physical_upper=physical,physical_savings_required=max(0,physical-512),
                projected_logical=logical,logical_headroom=4096-logical,budget_is_projection=True,
                pointer_table_contract="internal_guarded_pointees_unique_slots_same_stream",
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
    print(f"DKR_R126_S9C2D3B1_SOURCE=PARTIAL profiles={r['profiles']} images={r['images']} "
          f"guarded_ffi={len(r['migrated_ffi'])} gdn_residual={len(r['gdn_residual_ffi'])} "
          f"arch_provider_refs={len(r['arch_provider_refs'])} max_args={r['max_arguments']} "
          f"projected_physical={r['projected_physical_upper']} projected_logical={r['projected_logical']} "
          f"logical_headroom={r['logical_headroom']} compilation=false link_qualified=false full_closure=false")
if __name__=="__main__":
    main()
