#!/usr/bin/env python3
"""Merge explicit nextest process snapshots; never infer launches from loads."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PREFIX = "DKR_RUNTIME_EVIDENCE "
EVENTS = ("resolved","function_hit","function_miss","module_loaded","plan_hit","plan_miss",
          "plan_evicted","prepare_error","submitted","adapter_ok","adapter_error","fallback_selected","rejected")
KEYS = ("abi","operation","entry","sha256","target_cc")


def merge(lines):
    rows = defaultdict(lambda: dict.fromkeys(EVENTS,0))
    processes = set()
    for line in lines:
        if PREFIX not in line:
            continue
        s = json.loads(line.split(PREFIX,1)[1])
        if s.get("schema") != "DKR-RUNTIME-EVIDENCE-V1" or s.get("error") or s.get("dropped_rows") != 0:
            raise ValueError("incomplete or invalid runtime evidence")
        pid = s["pid"]
        if pid in processes:
            raise ValueError(f"duplicate cumulative snapshot for pid {pid}; use nextest one process per test")
        processes.add(pid)
        for row in s["rows"]:
            key = tuple(row[k] for k in KEYS)
            for event in EVENTS:
                n = row[event]
                if type(n) is not int or n < 0:
                    raise ValueError(f"invalid event counter: {event}")
                rows[key][event] += n
    if not processes:
        raise ValueError("no runtime evidence snapshots found")
    result = [{**dict(zip(KEYS,key)),**events} for key,events in sorted(rows.items())]
    return {"schema":"DKR-RUNTIME-EVIDENCE-SUMMARY-V1","processes":len(processes),
            "native_external_submission_rows":sum(r["abi"]=="candle.native.ptx.v1" and bool(r["sha256"]) and r["submitted"]>0 for r in result),
            "typed_adapter_success_rows":sum(r["abi"]=="dkr.typed.v1" and r["adapter_ok"]>0 for r in result),
            "gpu_completion_proven":False,"rows":result}


def self_test():
    row = dict(zip(KEYS,("candle.native.ptx.v1","unary","exp","abc",61)))
    row.update(dict.fromkeys(EVENTS,0))
    row["module_loaded"] = 1
    s = {"schema":"DKR-RUNTIME-EVIDENCE-V1","pid":1,"dropped_rows":0,"rows":[row]}
    line = PREFIX+json.dumps(s)
    assert merge([line])["native_external_submission_rows"] == 0
    row["submitted"] = 1
    assert merge([PREFIX+json.dumps(s)])["native_external_submission_rows"] == 1
    try:
        merge([line,line])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate snapshot accepted")
    print("DKR_S5_EVIDENCE_SELF_TEST=PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--log",type=Path)
    p.add_argument("--output",type=Path,default=Path("/tmp/dkr-s5-runtime-evidence.json"))
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.log:
        p.error("--log required")
    result=merge(a.log.read_text().splitlines())
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"DKR_S5_EVIDENCE_SUMMARY=PASS processes={result['processes']} output={a.output}")


if __name__=="__main__":
    main()
