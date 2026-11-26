#!/usr/bin/env python3
"""S9C.2E.1 source gate: residual guards and immutable model scope contract."""
import argparse
import json
import re
from pathlib import Path
from dkr_r126_s9c2e0_audit import audit as previous, REUSE
from dkr_r126_s9c2d3c_audit import strip_comments
from dkr_r126_gguf_residual_audit import guarded_ranges, mask

GUARD='#[cfg(not(feature = "dkr-static-closure-probe"))]'
CORE="mistralrs-core/src/"
RUNTIME="mistralrs-kernel-runtime/"
NAMES=set().union(*REUSE.values())

def require(ok,msg):
    if not ok:
        raise ValueError(msg)

def checked_refs(source,pattern,count,label):
    clean=strip_comments(source)
    spans=guarded_ranges(clean)
    refs=list(re.finditer(pattern,mask(clean)))
    require(len(refs)==count,label+": unexpected reference count")
    require(all(any(a<=m.start()<b for a,b in spans) for m in refs),label+": unguarded reference")
    return len(refs)

def check_declarations(source,names=NAMES):
    for name in sorted(names):
        checked_refs(source,r'\bpub\(crate\)\s+fn\s+'+re.escape(name)+r'\s*\(',1,name)

def function_body(source,name):
    marker="pub fn "+name+"("
    require(source.count(marker)==1,"missing/duplicate function "+name)
    # Include the preceding attrs when checking a whole guarded function.
    start=source.index(marker)
    tail=source[start:].split("\n}\n",1)[0]+"\n}\n"
    return tail

def check_indexed_module(source):
    clean=strip_comments(source)
    marker=GUARD+'\n#[cfg(not(feature = "dkr-indexed-copy"))]\n#[path = "indexed_copy_legacy.rs"]\nmod provider;'
    require(clean.count(marker)==1 and clean.count('"indexed_copy_legacy.rs"')==1,
            "indexed legacy module not uniquely excluded")

def guard_audit(root):
    core=root/CORE
    ffi=(core/"cuda/ffi.rs").read_text()
    check_declarations(ffi)
    ops=(core/"ops.rs").read_text()
    exact={
        "apply_sparse_penalties_f32":"ops.rs",
        "apply_sparse_logits_bias_f32":"ops.rs",
        "apply_causal_mask_f32":"ops.rs",
        "pad_decode_input_u32":"cuda/input_packing.rs",
        "selective_scan_cuda":"cuda/ssm.rs",
    }
    for name,path in exact.items():
        checked_refs((core/path).read_text(),r'\bffi::'+name+r'\s*\(',1,name)
    check_indexed_module((core/"cuda/indexed_copy.rs").read_text())
    indexed=(core/"cuda/indexed_copy_legacy.rs").read_text()
    require(len(re.findall(r'\bffi::\$ffi\s*\(',mask(indexed)))==1,"indexed FFI macro body drift")
    for name in REUSE["dkr-indexed-copy"]:
        require(len(re.findall(r'\b'+name+r'\b',mask(indexed)))==1,"indexed macro argument drift")
    rms_names=("cuda_rms_norm_residual","cuda_rms_norm_residual_then_rms_norm")
    for fn in rms_names:
        body=function_body(ops,fn)
        checked_refs(body,r'\bffi::\$ffi_fn\s*\(',1,fn+" macro body")
        for dt in ("f32","f16","bf16"):
            symbol=fn.removeprefix("cuda_")+"_"+dt
            require(len(re.findall(r'\b'+symbol+r'\b',mask(body)))==1,"RMS macro invocation drift: "+symbol)
    add_marker=GUARD+'\n#[cfg(all(feature = "cuda", not(feature = "dkr-add-rms-norm")))]'
    require(ops.count(add_marker)==1,"add RMS legacy function guard")
    add_section=ops[ops.index(add_marker):]
    add_section=add_section.split("\n}\n",1)[0]+"\n}\n"
    checked_refs(add_section,r'\bffi::\$ffi_fn\s*\(',1,"add RMS macro body")
    for name in REUSE["dkr-add-rms-norm"]:
        require(len(re.findall(r'\b'+name+r'\b',mask(add_section)))==1,"add RMS macro invocation drift")
    # Explicit direct legacy symbol references must be confined to these
    # audited paths. Macro arguments in RMS and indexed providers are checked
    # separately above; they are not direct references after cfg expansion.
    pattern=re.compile(r'\bffi::('+"|".join(sorted(NAMES))+r')\b')
    direct=[]
    for path in sorted(core.rglob("*.rs")):
        if path==core/"cuda/ffi.rs":
            continue
        clean=strip_comments(path.read_text())
        for m in pattern.finditer(mask(clean)):
            rel=str(path.relative_to(core))
            require(exact.get(m.group(1))==rel,"unexpected direct residual call: "+rel+" "+m.group(1))
            direct.append(m.group(1))
    require(sorted(direct)==sorted(exact),"residual direct-call inventory drift")
    return dict(guarded_declarations=sorted(NAMES),direct_calls=len(direct),
                macro_bodies=4,indexed_legacy_module_excluded=True,
                cfg_and_link_proof=False)

def capability_audit(root):
    def source(path):
        return (root/path).read_text()
    paths={
        RUNTIME+"src/model_scope.rs":[
            'pub struct ModelCapabilityDescriptor','pub struct CapabilityComponent',
            'pub fn resolve_model_scope(', 'pub struct ProfileDependency',
            'pub fn materialization_selection(', 'selected.len() <= MAX_CATALOG_PROFILES',
            'required implementation unavailable','pub fn verified_scope_id(',
            'scope content identity mismatch','pub struct ScopeLatch',
            'DKR active scope differs from process snapshot',
            'dkr_model_scope_resolves_union_and_dependency_cycle',
            'dkr_model_scope_canonical_receipt_and_tamper',
            'dkr_model_scope_latch_rejects_live_switch'],
        RUNTIME+"src/universal_cuda.rs":[
            'read_scope_receipt(root)?','crate::embedded::SCOPE_RECEIPT',
            'receipt.verify(&catalog)?','DKR_REQUIRE_MODEL_SCOPE',
            'PROCESS_SCOPE.get_or_init(ScopeLatch::default).admit(scope.as_ref())?',
            'request.cc == scope.device_cc','pub fn scope_id(&self)'],
        RUNTIME+"build.rs":[
            'model_scope::read_scope_receipt(&root)?','receipt.verify(&catalog)?',
            'pub const SCOPE_RECEIPT: Option<&str>'],
        RUNTIME+"src/bin/dkr_model_scope_resolve.rs":[
            'resolve_model_scope(&descriptor,&inventory.profiles,&dependencies)?',
            'plan.materialization_selection()?','create_new(true)'],
        CORE+"pipeline/loaders/mod.rs":[
            'fn dkr_config_capabilities(','self.model_config(config)?',
            'model.layer_has_paged_kv_cache(i)','model.kv_cache_layout_for_layer(i)',
            'normalized_model_config_only','requirements: Vec::new()',
            'layer_operation_and_quantization_inventory'],
        "tools/dkr_r126_model_catalog_scope.py":[
            'def scope_receipt(','def manifest_row(','stage/SCOPE_FILENAME',
            'canonical_scope=true','MAX_ACTIVE_PROFILES=4096','MAX_ACTIVE_MANIFESTS=512'],
    }
    count=0
    for path,markers in paths.items():
        s=source(path)
        for marker in markers:
            require(marker in s,"missing model scope contract: "+path+" "+marker)
            count+=1
    known="084d8d0b35304955c8f6dc69461ef12beee7266500b06bbe97a1561d94747bcc"
    for path in (RUNTIME+"src/model_scope.rs","tools/dkr_r126_model_catalog_scope.py"):
        require(known in source(path),"cross-language canonical identity fixture drift")
    return dict(source_markers=count,descriptor_native_rust=True,
                loader_config_adapter_partial=True,automatic_loader_invocation=False,
                explicit_requirements_and_dependencies=True,verified_receipt_admission=True,
                process_scope_latch=True,embedded_scope_receipt=True,
                canonical_cross_language_fixture_present=True,
                native_tests_executed=False,model_coverage_proven=False,
                candle_and_paged_integration_unqualified=True)

def audit(root):
    prior=previous(root)
    require(set(prior["closed_reuse_declarations"])==NAMES and prior["remaining_core_ffi_count"]==43,
            "historical guard inventory not closed")
    guards=guard_audit(root)
    capabilities=capability_audit(root)
    return dict(schema="dkr.r126.s9c2e1.source.v1",status="SOURCE_COMPLETE",
                guards=guards,capabilities=capabilities,residual_declarations=43,
                profiles_added=0,images_added=0,projected_logical=prior["current_projected_logical"],
                projected_physical_upper=prior["current_projected_physical_upper"],
                compilation=False,link_qualified=False,full_closure=False,model_scope_fit_proven=False)

def self_test():
    sample=GUARD+"\npub(crate) fn example();\n"
    check_declarations(sample,{"example"})
    checked_refs(GUARD+"\nunsafe { ffi::example(); }",r'\bffi::example\s*\(',1,"sample")
    failures=0
    for bad in (sample.replace(GUARD,""),sample.replace(GUARD,"// "+GUARD),sample+"pub(crate) fn example();\n"):
        try:
            check_declarations(bad,{"example"})
        except ValueError:
            failures+=1
        else:
            raise AssertionError("invalid declaration guard accepted")
    for bad in ("unsafe { ffi::example(); }",GUARD+"\nunsafe { ffi::example(); }\nffi::example();"):
        try:
            checked_refs(bad,r'\bffi::example\s*\(',1,"sample")
        except ValueError:
            failures+=1
        else:
            raise AssertionError("invalid call guard accepted")
    print(f"DKR_R126_S9C2E1_SELF_TEST=PASS negative_cases={failures}")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--source-root",type=Path,default=Path.cwd())
    p.add_argument("--json-out",type=Path)
    a=p.parse_args()
    if a.self_test:
        self_test();return
    r=audit(a.source_root.resolve())
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(r,f,indent=2,sort_keys=True);f.write("\n")
    print("DKR_R126_S9C2E1_SOURCE=COMPLETE guarded_ffi=17 residual=43 profiles_added=0 images_added=0 "
          f"projected_logical={r['projected_logical']} projected_physical={r['projected_physical_upper']} "
          "descriptor=native loader_metadata=partial scope_receipt=source model_scope_fit=UNPROVEN "
          "compilation=false link_qualified=false full_closure=false")

if __name__=="__main__":
    main()
