#!/usr/bin/env python3
"""S6 A/B parity and explicit native execution coverage; no GPU work."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from dkr_r122_candle_integration import FAMILIES
from dkr_r122_s5_evidence import merge


def native_audit(root):
    actual = {p.parent.name for p in root.glob("*/manifest.json")}
    if actual != FAMILIES:
        raise ValueError(f"native inventory mismatch {sorted(actual ^ FAMILIES)}")
    for family in sorted(FAMILIES):
        d=root/family
        m=json.loads((d/"manifest.json").read_text())
        b=(d/f"{family}.ptx").read_bytes()
        text=b.decode()
        if (m["family"] != family or m["target_cc"] != 61
                or m["schema"] != "CANDLE-DKR-S31-NATIVE-MODULE-V1"
                or m["sha256"] != hashlib.sha256(b).hexdigest()
                or re.findall(r"(?m)^\s*\.target\s+sm_(\d+)",text) != ["61"]
                or not re.search(r"(?m)^\s*\.address_size\s+64\b",text)
                or not re.search(r"\.entry\s+\S+",text)):
            raise ValueError(f"invalid native package {family}")
    print(f"DKR_S6_NATIVE_ARTIFACTS=PASS families={len(actual)}")


def compare(a,b):
    if not a or a.keys()!=b.keys():
        raise ValueError("A/B result keys mismatch")
    errors={}
    failures=[]
    for k,x in a.items():
        y=b[k]
        if not x or len(x)!=len(y) or not all(math.isfinite(v) for v in x+y):
            raise ValueError(f"invalid A/B values: {k}")
        atol,rtol=(0.,0.) if k.endswith("/sort") else (0.025,0.025) if "/mat/" in k or k.startswith("moe_") else (0.015,0.015)
        err=max(abs(u-v) for u,v in zip(x,y))
        errors[k]={"max_abs":err,"atol":atol,"rtol":rtol,"elements":len(x)}
        if any(abs(u-v)>atol+rtol*max(abs(u),abs(v)) for u,v in zip(x,y)):
            failures.append(k)
    return errors,failures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--report-dir",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        assert compare({"f/sort":[1.]},{"f/sort":[1.]})[1]==[]
        assert compare({"f/sort":[1.]},{"f/sort":[2.]})[1]==["f/sort"]
        print("DKR_S6_AB_SELF_TEST=PASS")
        return
    if a.native_root:
        native_audit(a.native_root)
        return
    root=a.report_dir
    errors,failures=compare(json.loads((root/"native-legacy.json").read_text()),json.loads((root/"native-external.json").read_text()))
    legacy=merge((root/"legacy.log").read_text().splitlines())
    external=merge((root/"external.log").read_text().splitlines())
    n="candle.native.ptx.v1"
    observed={r["operation"] for r in external["rows"] if r["abi"]==n and r["sha256"] and r["submitted"]>0}
    fallback={r["operation"] for r in external["rows"] if r["abi"]==n and r["fallback_selected"]>0}
    legacy_external=any(r["abi"]==n and r["sha256"] and r["submitted"]>0 for r in legacy["rows"])
    legacy_selected=any(r["abi"]==n and r["fallback_selected"]>0 for r in legacy["rows"])
    missing=sorted(FAMILIES-observed)
    report={"status":"REVIEW_REQUIRED" if failures or missing or legacy_external or not legacy_selected or not external["typed_adapter_success_rows"] else "SMOKE_SCOPE_PASS",
            "ab_cases":len(errors),"ab_failures":failures,"ab_errors":errors,
            "native_observed":sorted(observed),"native_missing":missing,"native_fallbacks":sorted(fallback),
            "legacy_external_submission":legacy_external,"legacy_fallback_selected":legacy_selected,
            "typed_adapter_success_rows":external["typed_adapter_success_rows"],
            "legacy_evidence":legacy,"external_evidence":external,
            "full_numerical_qualification":False}
    (root/"qualification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"DKR_S6_EVIDENCE={report['status']} ab_cases={len(errors)} native_observed={len(observed)} missing={missing} failures={failures}")
    if report["status"]!="SMOKE_SCOPE_PASS":
        raise SystemExit(4)


if __name__=="__main__":
    main()
