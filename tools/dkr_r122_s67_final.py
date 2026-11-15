#!/usr/bin/env python3
"""Final S6.7 scoped A/B, CPU reference, telemetry and provenance closure."""
import argparse
import json
import re
from pathlib import Path
from dkr_r122_s6_external_recovery import oracle
from dkr_r122_s6_evidence import native_audit, compare
from dkr_r122_s5_evidence import merge
from dkr_r122_s67_promotion import coverage
from dkr_r122_candle_integration import FAMILIES

NON_GPU = {
    "mistralrs-kernel-runtime launch_plan_cache::tests::reuses_prevalidated_plan_for_same_signature_and_device",
    "mistralrs-paged-attn cuda::backend::cache::tests::dkr_kv_copy_independence_contract",
    "mistralrs-quant routed_gemv::dynamic::tests::dkr_routed_quant_refuses_biased_or_affine_weights",
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--native-root",required=True,type=Path)
    a=p.parse_args()
    root=a.report_dir
    native_audit(a.native_root)
    legacy=(root/"legacy.log").read_text()
    external=(root/"external.log").read_text()
    la=json.loads((root/"native-legacy.json").read_text())
    ex=json.loads((root/"native-external.json").read_text())
    ab,ab_bad=compare(la,ex)
    lo,eo=oracle(la),oracle(ex)
    cpu_bad={name:[k for k,v in x.items() if v.get("pass") is False]
             for name,x in (("legacy",lo),("external",eo))}
    le,ee=merge(legacy.splitlines()),merge(external.splitlines())
    cov=coverage(external)
    rows=[r for r in ee["rows"] if r["abi"]=="candle.native.ptx.v1" and r["submitted"]]
    observed={r["operation"] for r in rows}
    provenance_bad=[]
    for r in rows:
        if r["operation"] not in FAMILIES:
            provenance_bad.append(r["operation"])
            continue
        m=json.loads((a.native_root/r["operation"]/"manifest.json").read_text())
        if r["sha256"]!=m["sha256"] or r["target_cc"]!=61:
            provenance_bad.append(r["operation"])
    counter_bad=[r for r in ee["rows"] if any(r[k] for k in ("adapter_error","prepare_error","rejected","fallback_selected"))]
    # Intentional invalid geometry regression fails before a launch, but legacy
    # selection is expected in A; no external fallback or adapter errors in B.
    legacy_external=le["native_external_submission_rows"]
    legacy_mmq = [r for r in le["rows"] if r["operation"]=="fast_mmq" and r["fallback_selected"]]
    static_mmq = sum(r["fallback_selected"] for r in legacy_mmq if r["entry"]=="static_ffi")
    failures=[]
    if static_mmq != 50 or any(r["entry"]!="static_ffi" for r in legacy_mmq):
        failures.append("legacy static MMQ route coverage")
    if not re.search(r"Summary\s+\[[^\]]+\]\s+1 test run: 1 passed",legacy):
        failures.append("legacy test not successful")
    if not re.search(r"Summary\s+\[[^\]]+\]\s+34 tests run: 34 passed, 2428 skipped",external):
        failures.append("external suite incomplete")
    if ab_bad or any(cpu_bad.values()): failures.append("numeric mismatch")
    if observed!=FAMILIES or provenance_bad: failures.append("native coverage/provenance")
    if counter_bad or legacy_external: failures.append("unexpected counters")
    if cov["started"]!=34 or cov["mapped_snapshots"]!=31 or ee["processes"]!=31 or set(cov["without_snapshot"])!=NON_GPU:
        failures.append("GPU telemetry coverage")
    if not ee["typed_adapter_success_rows"]: failures.append("typed evidence missing")
    report={"status":"SCOPED_AB_AND_TELEMETRY_PASS" if not failures else "REVIEW_REQUIRED",
            "failures":failures,"ab_cases":len(ab),"ab_failures":ab_bad,
            "cpu_failures":cpu_bad,"native_missing":sorted(FAMILIES-observed),
            "native_submissions":sum(r["submitted"] for r in rows),
            "typed_adapter_success_rows":ee["typed_adapter_success_rows"],
            "telemetry":cov,"counter_failures":counter_bad,"provenance_failures":provenance_bad,
            "legacy_external_submissions":legacy_external,"legacy_static_mmq_calls":static_mmq,
            "legacy_static_mmq_qualified":not failures,
            "ab_parity_qualified_for_smoke_scope":not failures,
            "full_numerical_qualification":False,"production_promotion_approved":False}
    (root/"qualification-s67.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S67_FINAL="+report["status"])
    return 4 if failures else 0

if __name__=="__main__":
    raise SystemExit(main())
