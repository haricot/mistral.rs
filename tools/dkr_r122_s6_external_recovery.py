#!/usr/bin/env python3
"""S6 external evidence and CPU reference check, using the original archived test output.

No compilation, no GPU. A legacy abort remains an explicit open gate.
"""
import argparse
import collections
import json
import math
import re
import struct
from pathlib import Path
from dkr_r122_candle_integration import FAMILIES
from dkr_r122_s5_evidence import merge
from dkr_r122_s6_evidence import native_audit

CODECS = ("Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q2K","Q3K","Q4K","Q5K","Q6K")
W = [(i % 17 - 8)/16 for i in range(5*256)]
X = [1.,-2.,3.,0.5,-1.,4.,2.,-3.]

def half(v):
    return struct.unpack("<e",struct.pack("<e",v))[0]

def q8_reference(weights):
    out=[]
    for i in range(0,len(weights),32):
        block=weights[i:i+32]
        scale=max(map(abs,block))/127
        out.extend(half(scale)*math.copysign(math.floor(abs(v/scale)+0.5),v) if scale else 0. for v in block)
    return out

def expected_cases(data):
    """(prediction, elementwise tolerance, scope). Quantized dots use observed weights,
    and an activation quantization error envelope, not a bit-exact codec oracle."""
    def item(ref,tol,scope="cpu_independent"):
        return (ref,[tol]*len(ref) if isinstance(tol,(int,float)) else tol,scope)
    cases={}
    for dtype in ("F32","F16","BF16"):
        probs=[]
        for row in (X[:4],X[4:]):
            ex=[math.exp(v-max(row)) for v in row]
            probs.extend(v/sum(ex) for v in ex)
        values={"affine":[v*.5+1 for v in X],
                "binary":[v+X[i%4] for i,v in enumerate(X)],
                "unary":[v*v for v in X],"reduce":[sum(X[:4]),sum(X[4:])],
                "cast":X,"fill":[1.]*8,"indexing":X[4:]+X[:4],
                "ternary":list(map(abs,X)),
                "sort":sorted(range(4),key=lambda i:X[i])+sorted(range(4),key=lambda i:X[4+i])}
        for op,ref in values.items():
            cases[f"{dtype}/{op}"]=item(ref,0.)
        cases[f"{dtype}/softmax"]=item(probs,{"F32":2e-6,"F16":0.0015,"BF16":0.008}[dtype])
    cases["conv2d"]=item([12.,16.,24.,28.],0.)
    for codec in CODECS:
        key=f"{codec}/dequant"
        q=data[key]
        if len(q)!=1280 or not all(math.isfinite(v) for v in q):
            raise ValueError(f"invalid dequantized operands {key}")
        # Only Q8_0 has an independent dequantization oracle here.
        if codec=="Q8_0":
            # CPU/GPU may choose different tie rounding at half steps;
            # one quantization step is reported separately as tolerance.
            cases[key]=item(q8_reference(W),0.0041)
        for batch in (1,8,9):
            ref=[];bounds=[]
            for m in range(batch):
                x=[((m*256+i)%11-5)/16 for i in range(256)]
                for n in range(5):
                    w=q[n*256:(n+1)*256]
                    ref.append(math.fsum(a*b for a,b in zip(w,x)))
                    # Q8_1 approximation: half-step plus FP16 scale rounding.
                    # Additional 0.01 allows affine sum compensation/accumulation.
                    bounds.append(math.fsum(abs(v) for v in w)*(0.5*0.3125/127+0.0002)+0.01)
            for repeat in (0,1):
                cases[f"{codec}/mat/{batch}/{repeat}"]=item(ref,bounds,"conditional_dequant_operand")
    simt=[sum(W[i:i+256]) for i in range(0,len(W),256)]*2
    for prefill in ("false","true"):
        cases[f"moe_simt/{prefill}"]=item(simt,0.001)
        # Q8_0 reference from independently reconstructed weights; all-one activation.
        # Account for half-scale rounding; CPU/GPU ties may differ.
        q=data["Q8_0/dequant"]
        ref=[math.fsum(q[i:i+256]) for i in range(0,len(q),256)]*2
        cases[f"moe_gguf/{prefill}"]=item(ref,0.015,"conditional_dequant_operand")
    return cases

def oracle(data):
    # The other nine dequantizations are diagnostics, not certified references.
    cases=expected_cases(data)
    diagnostics={f"{c}/dequant" for c in CODECS if c!="Q8_0"}
    if set(data)!=set(cases)|diagnostics:
        raise ValueError(f"unexpected output inventory: {sorted(set(data)^(set(cases)|diagnostics))}")
    results={}
    for key,(ref,tol,scope) in sorted(cases.items()):
        got=data[key]
        if len(got)!=len(ref) or not all(math.isfinite(v) for v in got):
            raise ValueError(f"bad output {key}")
        errors=[abs(a-b) for a,b in zip(got,ref)]
        results[key]={"scope":scope,"max_abs":max(errors),"max_allowed":max(tol),
                      "pass":all(e<=t for e,t in zip(errors,tol))}
    for key in sorted(diagnostics):
        got=data[key]
        results[key]={"scope":"diagnostic_only","max_abs_vs_unquantized":max(abs(a-b) for a,b in zip(got,W))}
    for c in CODECS:
        for b in (1,8,9):
            a,bv=(data[f"{c}/mat/{b}/{r}"] for r in (0,1))
            results[f"repeat/{c}/{b}"]={"scope":"repeat_exact","pass":a==bv,
                                      "max_abs":max(abs(x-y) for x,y in zip(a,bv))}
    return results

def audit_run(root,native_root):
    native_audit(native_root)
    log=(root/"external.log").read_text()
    # Only a completed successful run, not mere snapshots.
    if not re.search(r"Summary\s+\[[^\]]+\]\s+34 tests run: 34 passed, 2428 skipped",log):
        raise ValueError("expected completed S6 34/34 nextest run")
    ev=merge(log.splitlines())
    n="candle.native.ptx.v1"
    native=[r for r in ev["rows"] if r["abi"]==n and r["submitted"]]
    observed={r["operation"] for r in native if r["sha256"] and r["target_cc"]==61}
    for r in native:
        if r["operation"] not in FAMILIES:
            raise ValueError("unknown native family")
        m=json.loads((native_root/r["operation"]/"manifest.json").read_text())
        if r["sha256"]!=m["sha256"] or r["target_cc"]!=61:
            raise ValueError("submission provenance mismatch")
    failures=[r for r in ev["rows"] if any(r[k] for k in ("adapter_error","prepare_error","rejected","fallback_selected"))]
    results=oracle(json.loads((root/"native-external.json").read_text()))
    bad=[k for k,v in results.items() if v.get("pass") is False]
    legacy_log=(root/"legacy.log").read_text()
    legacy_abort="Rust cannot catch foreign exceptions" in legacy_log and "SIGABRT" in legacy_log
    report={"status":"EXTERNAL_SMOKE_AND_CPU_SCOPE_PASS_LEGACY_OPEN" if not bad and not failures and observed==FAMILIES and legacy_abort else "REVIEW_REQUIRED",
            "legacy_abort":legacy_abort,"ab_parity_qualified":False,"full_numerical_qualification":False,
            "native_families":sorted(observed),"missing_families":sorted(FAMILIES-observed),
            "native_submissions":sum(r["submitted"] for r in native),
            "typed_adapter_success_rows":ev["typed_adapter_success_rows"],
            "snapshots":ev["processes"],"counter_failures":failures,
            "oracle_scope_counts":dict(collections.Counter(v["scope"] for v in results.values())),
            "oracle_failures":bad,"oracle_results":results}
    (root/"qualification-external-recovery.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="oracle_results"},indent=2))
    print("DKR_S6_EXTERNAL_RECOVERY="+report["status"])
    return 0 if report["status"].startswith("EXTERNAL_SMOKE") else 4

def self_test():
    data={f"{c}/dequant":q8_reference(W) for c in CODECS}
    for k,(v,_,_) in expected_cases(data).items():
        data[k]=v[:]
    r=oracle(data)
    assert not [k for k,v in r.items() if v.get("pass") is False]
    data["F32/affine"][0]+=1
    assert not oracle(data)["F32/affine"]["pass"]
    assert len([v for v in r.values() if v["scope"]=="diagnostic_only"])==9
    print("DKR_S6_EXTERNAL_RECOVERY_SELF_TEST=PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return 0
    if not a.report_dir or not a.native_root:
        p.error("both --report-dir and --native-root required")
    return audit_run(a.report_dir,a.native_root)

if __name__=="__main__":
    raise SystemExit(main())
