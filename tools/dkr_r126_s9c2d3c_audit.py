#!/usr/bin/env python3
"""GDN static provider source closure; no CUDA compilation or link proof."""
import argparse
import importlib.util
import json
import re
from pathlib import Path
from dkr_r126_gguf_residual_audit import guarded_ranges,mask

GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
ARCH=("gdn_mma_prefill_init","gdn_mma_prefill","gdn_mma_fused_prefill",
      "mistralrs_flashinfer_gdn_sm90_launch","mistralrs_flashinfer_gdn_sm90_workspace_size")
REF=r'(?<!std::)(?<!core::)\b(?:crate::cuda::)?ffi::(?!c_void\b)([A-Za-z_]\w*)'
MMA_ITEMS=("GdnMmaParams","GdnMmaFusedParams","GdnMmaInputKind","GdnMmaLaunch",
           "mma_prefill_device_supported","mma_prefill_supported","mma_prefill_recurrence",
           "mma_prefill","mma_fused_prefill")
FLASH_ITEMS=("FlashInferGdnSm90Params","FlashInferGdnSm90Launch",
             "flashinfer_sm90_prefill_supported","flashinfer_sm90_prefill",
             "flashinfer_sm90_prefill_dispatch")
CUTILE_ITEMS=("describe_prefill_launch","cutile_prefill_supported","cutile_prefill")

def require(ok,message):
    if not ok:
        raise ValueError(message)

def strip_comments(source):
    # Retain strings because cfg guards and build directives are inspected.
    return re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
        lambda m: m.group() if m.group().startswith('"') else re.sub(r'[^\n]',' ',m.group()),source)

def fenced(source,pattern,expected=None):
    code=strip_comments(source)
    spans=guarded_ranges(code)
    matches=list(re.finditer(pattern,mask(code)))
    if expected is not None:
        require(len(matches)==expected,"unexpected occurrence count: "+pattern)
    require(bool(matches),"missing occurrence: "+pattern)
    require(all(any(a<=m.start()<b for a,b in spans) for m in matches),
            "unguarded occurrence: "+pattern)
    return [source.count("\n",0,m.start())+1 for m in matches]

def provider_calls(source,ffi):
    calls={}
    for name in ARCH:
        calls[name]=dict(calls=fenced(source,r'\b(?:crate::cuda::)?ffi::'+name+r'\s*\(',1),
                         declarations=fenced(ffi,r'\bpub\(crate\)\s+fn\s+'+name+r'\s*\(',1))
    return calls

def check_build(build):
    code=strip_comments(build)
    spans=guarded_ranges(code)
    directives=("cargo:rustc-link-lib=mistralrscuda",
                "cargo:rustc-link-lib=mistralrsflashinfergdn",
                "cargo:rustc-cfg=has_flashinfer_gdn_sm90_kernel",
                "third_party/flashinfer_gdn_sm90/mistralrs_flashinfer_gdn_sm90.cu")
    for d in directives:
        positions=[m.start() for m in re.finditer(re.escape('"'+d+'"'),code)]
        require(len(positions)==1 and all(any(a<=p<b for a,b in spans) for p in positions),
                "unguarded build directive: "+d)
    require('DKR_STATIC_PRODUCER_DISABLED component=mistralrs-core mode=closure-probe' in code,
            "missing closure producer diagnostic")
    require('feature = "cutile", feature = "flash-attn", feature = "flash-attn-v3"' in code,
            "closure feature conflict fence")
    require('if compute_cap == 90' in code and '.compute_cap_arch("90a")' in code and
            'version >= FLASHINFER_GDN_MIN_CUDA' in code,"legacy FlashInfer provider constraints")
    return list(directives)

def check_policy(source):
    code=strip_comments(source)
    for name in MMA_ITEMS+FLASH_ITEMS+CUTILE_ITEMS:
        fenced(code,r'\b(?:fn|struct|enum)\s+'+name+r'\b',1)
    for name in ("GdnMmaLaunch","FlashInferGdnSm90Launch"):
        fenced(code,r'\bimpl\s+(?:candle_core::InplaceOp1 for )?'+name+r'\b')
    for name in ("mma_prefill_supported","mma_prefill","mma_fused_prefill",
                 "flashinfer_sm90_prefill_supported","flashinfer_sm90_prefill_dispatch",
                 "cutile_prefill_supported","cutile_prefill"):
        # Every invocation, apart from its definition, must be lexically
        # inside a negative guard (including auto dispatch).
        fenced(code,r'(?<!fn )\b'+name+r'\s*\(')
    start=code.index("fn prefill_kernel_override()")
    end=code.index("\n}\n",start)
    require('#[cfg(feature = "dkr-static-closure-probe")]\n            validate_closure_prefill_request(*kernel)?;' in code[start:end],
            "forced override rejection must be centralized")
    start=code.index("pub fn try_fused_vmajor_prefill_recurrence_cuda(")
    end=code.index("\n}\n",start)
    dispatch=code[start:end]
    require('#[cfg(feature = "dkr-static-closure-probe")]' in dispatch and
            'return Ok(None);' in dispatch and GUARD in dispatch and
            dispatch.index("return Ok(None);")<dispatch.index(GUARD),
            "strict auto route must decline static providers")
    require('fn dkr_gdn_provider_closure_policy()' in code and
            '["mma", "mma-fused", "flashinfer-sm90", "cutile"]' in code,
            "forced provider policy test")
    return dict(forced_excluded=["mma","mma-fused","flashinfer-sm90","cutile"],
                ordinary_auto="delegate_to_typed_recurrence",legacy_provider_paths_preserved=True)

def self_test():
    source="\n".join(GUARD+"\nfn call_"+str(i)+"(){ crate::cuda::ffi::"+n+"(); }"
                     for i,n in enumerate(ARCH))
    ffi="\n".join(GUARD+"\npub(crate) fn "+n+"();" for n in ARCH)
    provider_calls(source,ffi)
    failures=0
    cases=[(source.replace(GUARD,"",1),ffi),
           (source,ffi.replace(GUARD,"",1)),
           (source.replace(GUARD,"// "+GUARD,1),ffi),
           (source+"\nfn leak(){ ffi::"+ARCH[0]+"(); }",ffi),
           (source,ffi+"\npub(crate) fn "+ARCH[0]+"();")]
    for s,f in cases:
        try:
            provider_calls(s,f)
        except ValueError:
            failures+=1
        else:
            raise AssertionError("provider leak accepted")
    print(f"DKR_R126_S9C2D3C_SELF_TEST=PASS negative_cases={failures}")

def audit(root):
    core=root/"mistralrs-core"
    source=(core/"src/cuda/gdn.rs").read_text()
    ffi=(core/"src/cuda/ffi.rs").read_text()
    providers=provider_calls(source,ffi)
    policy=check_policy(source)
    directives=check_build((core/"build.rs").read_text())
    spans=guarded_ranges(strip_comments(source))
    matches=list(re.finditer(REF,mask(source)))
    residual=[m.group(1) for m in matches if not any(a<=m.start()<b for a,b in spans)]
    require(not residual,"residual GDN FFI: "+str(residual))
    # Count declarations only: a residual declaration is not proof of a
    # live call or of a new catalogue profile requirement.
    ffi_spans=guarded_ranges(strip_comments(ffi))
    remaining_declarations=[m.group(1) for m in re.finditer(r'\bpub\(crate\)\s+fn\s+(\w+)\s*\(',mask(ffi))
                            if not any(a<=m.start()<b for a,b in ffi_spans)]
    # Re-run the preceding cumulative gate, now accepting the fully closed
    # provider inventory as well as its original pre-C.3C state.
    spec=importlib.util.spec_from_file_location("previous_gate",root/"tools/dkr_r126_s9c2d3b2b_audit.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prev=module.audit(root)
    require(not prev["gdn_residual_ffi"] and not prev["arch_provider_refs"] and
            set(prev["arch_provider_guarded"])==set(ARCH),"cumulative provider inventory")
    require(prev["projected_logical"]==4027 and prev["projected_physical_upper"]==533,
            "baseline budget changed")
    return dict(schema="dkr.r126.s9c2d3c.source.v1",status="GDN_SOURCE_COMPLETE",
                provider_guards=providers,policy=policy,guarded_build_directives=directives,
                guarded_gdn_ffi=len(matches),gdn_residual_ffi=residual,arch_provider_refs=[],
                remaining_core_ffi_declarations=remaining_declarations,remaining_core_ffi_declarations_are_not_profile_counts=True,
                profiles_added=0,images_added=0,projected_logical=4027,logical_headroom=69,
                projected_physical_upper=533,physical_savings_required=21,
                compilation=False,link_qualified=False,numeric_qualified=False,
                full_gdn_source_closure=True,full_closure=False,budget_is_projection=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print("DKR_R126_S9C2D3C_SOURCE=COMPLETE providers_guarded=5 gdn_residual=0 arch_provider_refs=0 "
          "profiles_added=0 images_added=0 projected_physical=533 projected_logical=4027 "
          "logical_headroom=69 compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
