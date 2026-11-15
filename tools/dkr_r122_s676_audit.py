#!/usr/bin/env python3
"""Read-only external qualification audit and separate diagnostic A/B."""
import argparse, json, re
from pathlib import Path
from dkr_r122_s6_external_recovery import oracle
from dkr_r122_s6_evidence import native_audit, compare
from dkr_r122_s5_evidence import merge
from dkr_r122_s67_promotion import coverage
from dkr_r122_candle_integration import FAMILIES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external",type=Path,required=True)
    p.add_argument("--native-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--probe",type=Path)
    a=p.parse_args()
    native_audit(a.native_root)
    log=(a.external/"external.log").read_text()
    data=json.loads((a.external/"native-external.json").read_text())
    ev=merge(log.splitlines())
    cov=coverage(log)
    results=oracle(data)
    oracle_bad=[k for k,v in results.items() if v.get("pass") is False]
    rows=[r for r in ev["rows"] if r["abi"]=="candle.native.ptx.v1" and r["submitted"]]
    seen={r["operation"] for r in rows}
    provenance_bad=[]
    for r in rows:
        if r["operation"] not in FAMILIES:
            provenance_bad.append(r["operation"])
            continue
        m=json.loads((a.native_root/r["operation"]/"manifest.json").read_text())
        if r["sha256"]!=m["sha256"] or r["target_cc"]!=61:
            provenance_bad.append(r["operation"])
    errors=[r for r in ev["rows"] if any(r[k] for k in ("adapter_error","prepare_error","rejected","fallback_selected"))]
    ok=(re.search(r"Summary\s+\[[^\]]+\]\s+34 tests run: 34 passed, 2428 skipped",log)
        and seen==FAMILIES and not provenance_bad and not errors and not oracle_bad
        and ev["typed_adapter_success_rows"]>0)
    report={"external_scope_ok":bool(ok),"native_families":len(seen),"native_submissions":sum(r["submitted"] for r in rows),
            "typed_adapter_success_rows":ev["typed_adapter_success_rows"],"snapshots":ev["processes"],"telemetry":cov,
            "oracle_checks":sum("pass" in v for v in results.values()),"oracle_failures":oracle_bad,
            "provenance_failures":provenance_bad,"counter_failures":errors,"promotion_ready":False}
    if a.probe:
        text=(a.probe/"probe.log").read_text()
        history=(a.probe/"history.log").read_text()
        d={"history_errors":[l for l in history.splitlines() if l.startswith("DKR_S676_HISTORY")],
           "probe_markers":[l for l in text.splitlines() if l.startswith("DKR_S676_")],
           "probe_completed":bool(re.search(r"test result: ok\. 1 passed; 0 failed",text))}
        file=a.probe/"native-legacy-probe.json"
        if d["probe_completed"] and file.is_file():
            legacy=json.loads(file.read_text())
            ab,bad=compare(legacy,data)
            d.update(ab_cases=len(ab),ab_failures=bad,
                     oracle_failures=[k for k,v in oracle(legacy).items() if v.get("pass") is False])
        report["diagnostic_only"]=d
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S676_EXTERNAL_AUDIT="+("PASS" if ok else "REVIEW_REQUIRED"))
    return 0 if ok else 4

if __name__=="__main__":
    raise SystemExit(main())
