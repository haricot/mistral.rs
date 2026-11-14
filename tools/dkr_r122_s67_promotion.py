#!/usr/bin/env python3
"""S6.7 diagnostic, not a promotion approval. No GPU or compilation."""
import argparse
import json
import re
from pathlib import Path
from dkr_r122_s5_evidence import merge

def classify(log):
    blocks=log.split("DKR_S67_CPP_THROW\n")[1:]
    # Only classify frames from the throw site, not arbitrary text or abort stack.
    stacks=[b.split("\nDKR_RUNTIME_EVIDENCE",1)[0].split("\nThread ",1)[0] for b in blocks]
    matching=[s for s in stacks if re.search(r"(?im)^#\d+.*(?:thrust::|cub::|calculate_expert_offsets)",s)
              and re.search(r"(?im)^#\d+.*(?:moe_gemm_simt_f16|calculate_expert_offsets)",s)]
    aborted=bool(re.search(r"(?i)(received signal SIGABRT|terminated with signal SIGABRT)",log))
    if matching and aborted:
        status="LEGACY_SIMT_OFFSET_SCAN_THROW_REPRODUCED"
    elif blocks and aborted:
        status="LEGACY_CPP_THROW_OTHER_OR_UNSYMBOLIZED"
    elif aborted:
        status="LEGACY_ABORT_WITHOUT_CAUGHT_THROW"
    else:
        status="LEGACY_DIAGNOSTIC_INCONCLUSIVE"
    return {"status":status,"caught_throws":len(blocks),"sigabrt":aborted,
            "scan_stack_matches":len(matching)}

def coverage(log):
    # no-capture nextest runs one test at a time. Separate missing snapshots
    # from positive submissions; some tests are CPU/contract-only.
    started=[]
    with_snapshot=set()
    current=None
    for line in log.splitlines():
        m=re.search(r"\bSTART\s+\[[^\]]*\]\s+\(\s*\d+/\d+\)\s+(.+)$",line)
        if m:
            current=m.group(1).strip()
            started.append(current)
        elif "DKR_RUNTIME_EVIDENCE " in line and current:
            with_snapshot.add(current)
    return {"started":len(started),"mapped_snapshots":len(with_snapshot),
            "without_snapshot":[s for s in started if s not in with_snapshot]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",type=Path)
    p.add_argument("--gdb-log",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        assert classify("DKR_S67_CPP_THROW\n#0 __cxa_throw\n#1 thrust::inclusive_scan\n#2 moe_gemm_simt_f16\nThread 2 received signal SIGABRT")["status"]=="LEGACY_SIMT_OFFSET_SCAN_THROW_REPRODUCED"
        assert classify("DKR_S67_CPP_THROW\n#0 __cxa_throw\nThread 2 received signal SIGABRT")["status"]=="LEGACY_CPP_THROW_OTHER_OR_UNSYMBOLIZED"
        assert classify("no error")["status"]=="LEGACY_DIAGNOSTIC_INCONCLUSIVE"
        assert coverage("START [ ] ( 1/2) a\nDKR_RUNTIME_EVIDENCE {}\nSTART [ ] ( 2/2) b")["without_snapshot"]==["b"]
        print("DKR_S67_SELF_TEST=PASS")
        return 0
    if not a.report_dir or not a.gdb_log:
        p.error("report directory and GDB log required")
    root=a.report_dir
    recovery=json.loads((root/"qualification-external-recovery.json").read_text())
    log=(root/"external.log").read_text()
    ev=merge(log.splitlines())
    cov=coverage(log)
    diagnostic=classify(a.gdb_log.read_text())
    external_ok=(recovery["status"]=="EXTERNAL_SMOKE_AND_CPU_SCOPE_PASS_LEGACY_OPEN"
                 and not recovery["oracle_failures"] and not recovery["missing_families"]
                 and not recovery["counter_failures"] and ev["processes"]==20
                 and cov["started"]==34 and cov["mapped_snapshots"]==20)
    report={"status":"EXTERNAL_CANDIDATE_VALIDATED_PROMOTION_BLOCKED" if external_ok else "REVIEW_REQUIRED",
            "legacy_diagnostic":diagnostic,"external_scope_ok":external_ok,
            "native_families":len(recovery["native_families"]),
            "native_submissions":recovery["native_submissions"],
            "oracle_scope_counts":recovery["oracle_scope_counts"],
            "telemetry_coverage":cov,"promotion_ready":False,"ab_parity_qualified":False,
            "open_gates":["legacy fallback failure and missing A/B reference",
                          "telemetry scope review of tests without snapshots",
                          "exhaustive numerical coverage and inference performance not established"]}
    out=root/"s67"/"promotion-readiness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S67_DIAGNOSTIC="+diagnostic["status"])
    print("DKR_S67_PROMOTION="+report["status"])
    # Exit 4 is a blocked promotion, not a failed external qualification.
    return 4

if __name__=="__main__":
    raise SystemExit(main())
