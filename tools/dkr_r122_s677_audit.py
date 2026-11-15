#!/usr/bin/env python3
"""Audit MMQ attribute counterfactual without declaring production parity."""
import argparse
import json
import re
from pathlib import Path
from dkr_r122_s6_evidence import compare
from dkr_r122_s6_external_recovery import oracle

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--external",required=True,type=Path)
    a=p.parse_args()
    root=a.report_dir
    b=json.loads(a.external.read_text())
    report={"promotion_ready":False,"diagnostic_only":True}
    for mode in ("observe","omit"):
        s=(root/f"{mode}.log").read_text()
        attr=re.findall(r"DKR_S677_RETURN name=cudaFuncSetAttribute status=(\d+) attr=(\d+) bytes=(\d+)",s)
        launches=re.findall(r"DKR_S677_RETURN name=cudaLaunchKernel(?:_ptsz)? status=(\d+) ",s)
        skipped=re.findall(r"DKR_S677_OMIT_ATTRIBUTE attr=(\d+) bytes=(\d+)",s)
        completed=bool(re.search(r"test result: ok\. 1 passed; 0 failed",s))
        row={"attribute_calls":len(attr),"attribute_errors":[list(map(int,x)) for x in attr if int(x[0])],
             "launch_returns":len(launches),"launch_errors":[int(x) for x in launches if int(x)],
             "omitted_attributes":len(skipped),"test_completed":completed,
             "debugger_error":bool(re.search(r"Python Exception|Error in sourced command file",s))}
        file=root/f"native-legacy-{mode}.json"
        if completed and file.is_file():
            data=json.loads(file.read_text())
            ab,bad=compare(data,b)
            row.update(ab_cases=len(ab),ab_failures=bad,
                       oracle_failures=[k for k,v in oracle(data).items() if v.get("pass") is False])
        report[mode]=row
    x,y=report["observe"],report["omit"]
    confirmed=(x["attribute_calls"]>0 and x["launch_returns"]>0 and x["launch_errors"]
        and y["omitted_attributes"]>0 and y["launch_returns"]>0 and not y["launch_errors"]
        and y["test_completed"] and y.get("ab_cases",0)==105
        and not y["ab_failures"] and not y["oracle_failures"]
        and not x["debugger_error"] and not y["debugger_error"])
    report["status"]="ATTRIBUTE_OMISSION_COUNTERFACTUAL_PASS" if confirmed else "REVIEW_REQUIRED"
    (root/"mmq-attribute-audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S677="+report["status"])
    return 0

if __name__=="__main__":
    raise SystemExit(main())
