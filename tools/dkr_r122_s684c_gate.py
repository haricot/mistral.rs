#!/usr/bin/env python3
"""Re-audit final model evidence in place; CPU feature/test gate."""
import argparse,json,re,subprocess
from pathlib import Path
from dkr_r122_s684b_gate import prior,audit_run,typed_index,features
from dkr_r122_s6_evidence import native_audit
from dkr_r122_s683_audit import need,sha

REV="556dca9fa4e31ec80b77c642ded8c03a76d690e1"
TESTS=(
"validates_package_without_cuda",
"rejects_artifact_changes_and_missing_symbol_or_wrong_target",
"rejects_oversized_artifact_before_reading",
"rejects_incompatible_contracts_and_budgets",
"rejects_offset_strides_dtype_and_rank",
"rejects_unknown_manifest_fields_and_invalid_entry",
"cache_identity_follows_content_even_for_same_provider_name",
"rejects_artifact_symlink_outside_package",
"independent_oracle_matches_candle_cpu",
"reuses_prevalidated_plan_for_same_signature_and_device",
"caps_number_of_resident_plans",
"missing_default_sender_is_model_not_found",
"reloading_sender_preserves_model_state_error",
"fallible_file_helpers_preserve_poisoned_engine_error",
"accepts_matching_external_tokenizer_and_preserves_special_tokens",
"rejects_external_tokenizer_with_remapped_ids",
"rejects_external_tokenizer_with_different_vocab_size",
"rejects_invalid_gguf_special_token_metadata",
"cpu_prefill_decode_and_cached_chunks_match_reference",
)
def read(p):return json.loads(p.read_text())
def model_gate(run,original,native):
    r=read(run/"qualification-s684b.json")
    need((run/"revision.txt").read_text().strip()==REV,"wrong replay revision")
    need(r["status"]=="FINAL_MODEL_GATE_PASS_SCOPED" and not r["failures"],"final model report not PASS")
    required={"binaries.sha256","model.sha256","revision.txt","static-features.txt"}|{f"{leg}.{ext}" for leg in ("static-contiguous","external-contiguous","static-paged","external-paged") for ext in ("exit","json","log")}
    need(required<=set(r["artifact_sha256"]),"missing sealed model artifacts")
    for name,digest in r["artifact_sha256"].items():
        need(Path(name).name==name and sha(run/name)==digest,"artifact changed: "+name)
    for name in ("binaries.sha256","model.sha256"):
        subprocess.run(["sha256sum","-c",str(run/name)],check=True)
    native_audit(native)
    prior(original,native)
    index=typed_index(run/"typed/sm61")
    cases,ev,failures,_,static,kv,attention=audit_run(run,index,native)
    need(not failures and static and kv and attention,"model re-audit failed: "+str(failures))
    need(cases["contiguous"]==r["contiguous"] and cases["paged"]==r["paged"] and ev==r["telemetry"],"recomputed model evidence differs")
    for rows in cases.values():
        for c in rows.values():
            need(all(x["max_abs"]==0 for x in c["raw_ab"]) and all(x["max_logprob_delta"]==0 for x in c["decode_ab"]),"model not exact")
    return {"status":"FINAL_MODEL_SEAL_PASS","model_report_sha256":sha(run/"qualification-s684b.json"),
            "typed_manifest_sha256":{str(p.relative_to(run/"typed/sm61")):sha(p) for p in sorted((run/"typed/sm61").rglob("manifest.json"))},
            "static_feature_inventory":features(run/"static-features.txt"),"contiguous_exact":True,"paged_exact":True}

def cpu_features(path):
    entries=[(s.split("|",1)[0],s.split("|",1)[1].split(",")) for s in path.read_text().splitlines() if "|" in s]
    need(any(p.startswith("mistralrs v") for p,_ in entries),"SDK absent from CPU feature graph")
    forbidden=[(p,f) for p,fs in entries for f in fs if f=="cuda" or f.startswith(("cuda-","dkr-")) or f in ("shared-registry","metal","cutile","cudnn","flash-attn","nccl")]
    need(not forbidden,"non-CPU feature enabled: "+str(forbidden))
    return len(entries)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--filter",action="store_true")
    p.add_argument("--cpu-features",type=Path)
    p.add_argument("--model-run",type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--cpu-log",type=Path)
    a=p.parse_args()
    if a.filter:
        print(" or ".join(f"test({t})" for t in TESTS));return
    if a.cpu_features and not a.cpu_log:
        print("DKR_S684C_CPU_FEATURES=PASS",cpu_features(a.cpu_features));return
    if a.cpu_log:
        count=cpu_features(a.cpu_features)
        log=a.cpu_log.read_text()
        missing=[t for t in TESTS if not re.search(r"(?m)^\s*PASS\s+\[[^\]]+\].*::"+re.escape(t)+r"\s*$",log)]
        need(not missing,"missing PASS cases: "+str(missing))
        summary=re.search(r"Summary\s+\[[^\]]+\]\s+(\d+) tests? run: (\d+) passed",log)
        need(summary and int(summary[1])==int(summary[2])==len(TESTS),"unexpected CPU test count")
        result={"status":"CPU_DEFAULT_SELECTED_REGRESSION_PASS","tests":len(TESTS),"feature_entries":count,"log_sha256":sha(a.cpu_log),
                "cpu_feature_sha256":sha(a.cpu_features),"full_cpu_suite_verified":False,"production_promotion_approved":False}
    else:result=model_gate(a.model_run,a.prior,a.native_root)
    with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2));print("DKR_S684C="+result["status"])
if __name__=="__main__":main()
