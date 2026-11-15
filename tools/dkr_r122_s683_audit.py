#!/usr/bin/env python3
"""Audit S6.8.3 extended result matrix and preserve S6.8 seal."""
import argparse, hashlib, json, math, re, sys
from pathlib import Path
from dkr_r122_candle_integration import FAMILIES, CANDIDATE
from dkr_r122_s6_evidence import native_audit
from dkr_r122_s5_evidence import merge
from dkr_r122_s6_external_recovery import CODECS

PIN="08a36261a6b4dd695c291d37c028b16f019fae46"
QUALIFIED="9ce68d5cc4e13f95baf7fd23d0c62ec41af352a1"
SHAPES=((5,256,(1,8,9)),(64,512,(8,16)),(65,2048,(9,17)),(129,4096,(9,33)),(17,11008,(9,)))
DTYPES=("F32","F16","BF16")
def need(ok,why):
    if not ok: raise ValueError(why)
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def indices(n,limit):
    return list(range(n)) if n<=limit else [i*(n-1)//(limit-1) for i in range(limit)]
def inventory():
    d={f"{c}/{n}x{k}":(c,n,k) for c in CODECS for n,k,_ in SHAPES}
    tests={f"{c}/{n}x{k}/{b}/{t}":(c,n,k,b,t,4 if b<=8 else 1,3 if b%2==0 else 2)
           for c in CODECS for n,k,bs in SHAPES for b in bs for t in DTYPES}
    d["model_width/Q4K/2048x2048"]=("Q4K",2048,2048)
    tests["model_width/Q4K/2048x2048/9/F32"]=("Q4K",2048,2048,9,"F32",1,2)
    for mode in ("shared_device","explicit_stream"):
        for w in range(4):
            for r,b in enumerate((9,33,17,9,33,17)):
                tests[f"concurrent/{mode}/{w}/{r}"]=("Q4K",65,2048,b,DTYPES[(w+r)%3],1,2)
    return d,tests
def verify_seal(seal,prior,native):
    s=read(seal)
    need(s["status"]=="INTEGRATION_GATE_PASS_EXPERIMENTAL_ONLY" and s["source"]["qualified_source"]==QUALIFIED and s["source"]["candle_pin"]==PIN==CANDIDATE,"wrong S68 seal")
    need(s["evidence"]["ab_cases"]==105 and s["evidence"]["static_mmq_calls"]==50,"wrong S68 evidence")
    need(len(s["evidence"]["artifact_sha256"])==12,"incomplete prior artifact seal")
    for name,h in s["evidence"]["artifact_sha256"].items():
        need(Path(name).name==name and sha(prior/name)==h,"prior artifact changed: "+name)
    native_audit(native)
    need(set(s["evidence"]["native_manifest_sha256"])==FAMILIES,"incomplete native seal")
    for name,h in s["evidence"]["native_manifest_sha256"].items():
        need(sha(native/name/"manifest.json")==h,"native manifest changed: "+name)
    print("DKR_S683_PRIOR_SEAL=PASS")
def validate(data,mode):
    d,tests=inventory()
    need(data["schema"]=="DKR-S683-V1" and data["mode"]==mode,"wrong result schema/mode")
    need(set(data["dequant"])==set(d) and set(data["cases"])==set(tests),"wrong result inventory")
    for key,(c,n,k) in d.items():
        v=data["dequant"][key]
        need((v["codec"],v["n"],v["k"],v["elements"])==(c,n,k,n*k),key+" dequant shape")
        need(re.fullmatch("[0-9a-f]{64}",v["weights_sha256"]),key+" missing weight digest")
        need(math.isfinite(v["max_abs"]) and v["max_abs"]>=0 and 0<=v["max_ratio"]<=1,key+" dequant failure")
    values=0;samples=0;worst=0.
    for key,(c,n,k,b,t,o,r) in tests.items():
        v=data["cases"][key]
        need((v["codec"],v["n"],v["k"],v["batch"],v["dtype"],v["offset"],v["rank"])==(c,n,k,b,t,o,r),key+" shape mismatch")
        need(re.fullmatch("[0-9a-f]{64}",v["weights_sha256"]),key+" weight digest")
        if not key.startswith("concurrent/"):
            need(v["weights_sha256"]==data["dequant"][key.rsplit("/",2)[0]]["weights_sha256"],key+" dequant bytes mismatch")
        a=v["values"]
        need(len(a)==n*b and all(math.isfinite(x) for x in a) and v["repeat_exact"] is True,key+" values/repeat")
        expected=[row*n+col for row in indices(b,7) for col in indices(n,17)]
        need([s["index"] for s in v["samples"]]==expected,key+" missing CPU samples")
        for s in v["samples"]:
            ref,lim=s["reference"],s["bound"]
            need(math.isfinite(ref) and math.isfinite(lim) and lim>0,key+" invalid CPU bound")
            err=abs(a[s["index"]]-ref)
            need(err<=lim,key+" CPU oracle failure")
            worst=max(worst,err/lim)
        values+=len(a);samples+=len(expected)
    return {"dequant_cases":len(d),"dequant_elements":sum(n*k for _,n,k in d.values()),"cases":len(tests),"output_elements":values,"cpu_dot_samples":samples,"max_cpu_bound_ratio":worst}
def paired(a,b):
    av,bv=validate(a,"legacy"),validate(b,"external")
    for key,v in a["dequant"].items():
        need(v["weights_sha256"]==b["dequant"][key]["weights_sha256"],key+" A/B weight bytes differ")
    max_abs=0.;exact=0;count=0
    for key,x in a["cases"].items():
        y=b["cases"][key]
        need(x["weights_sha256"]==y["weights_sha256"] and x["samples"]==y["samples"],key+" unequal inputs/references")
        # Exact identity is reported. Tolerance is fixed before observing results:
        # same format and shape, allow small accumulation-order/rounding variation.
        atol,rtol={"F32":(0.002,0.002),"F16":(0.004,0.004),"BF16":(0.015,0.008)}[x["dtype"]]
        for u,v in zip(x["values"],y["values"]):
            e=abs(u-v);max_abs=max(max_abs,e);exact+=int(u==v);count+=1
            need(e<=atol+rtol*max(abs(u),abs(v)),key+" A/B numerical failure")
    return {"legacy":av,"external":bv,"ab_cases":len(a["cases"]),"ab_elements":count,"exact_equal_elements":exact,"max_absolute_ab_difference":max_abs}
def telemetry(run,native):
    result={}
    for mode in ("legacy","external"):
        log=(run/f"{mode}.log").read_text()
        need((run/f"{mode}.exit").read_text().strip()=="0" and re.search(r"Summary\s+\[[^\]]+\]\s+1 test run: 1 passed",log),mode+" test not successful")
        ev=merge(log.splitlines());need(ev["processes"]==1,mode+" snapshot count")
        rows=ev["rows"]
        need(not any(r[k] for r in rows for k in ("adapter_error","prepare_error","rejected")),mode+" runtime counter error")
        if mode=="legacy":
            need(ev["native_external_submission_rows"]==0,"legacy external contamination")
            mmq=[r for r in rows if r["operation"]=="fast_mmq" and r["fallback_selected"]]
            mmv=[r for r in rows if r["operation"]=="fast_mmvq" and r["fallback_selected"]]
            # 210 sequential MMQ (including model-width? computed from inventory),
            # plus concurrent cases, each executed twice; derive exact counts.
            _,inv=inventory()
            expected_mmq=2*sum(v[3]>8 for v in inv.values())
            expected_mmv=2*sum(v[3]<=8 for v in inv.values())
            need(sum(r["fallback_selected"] for r in mmq)==expected_mmq and all(r["entry"]=="static_ffi" for r in mmq),"static MMQ route count")
            need(sum(r["fallback_selected"] for r in mmv)==expected_mmv and all(r["entry"]=="static_ffi" for r in mmv),"static MMVQ route count")
            result[mode]={"static_mmq_calls":expected_mmq,"static_mmvq_calls":expected_mmv}
        else:
            need(not any(r["fallback_selected"] for r in rows),"external fallback")
            native_rows=[r for r in rows if r["abi"]=="candle.native.ptx.v1" and r["submitted"]]
            for r in native_rows:
                need(r["operation"] in FAMILIES,"unknown native family")
                m=read(native/r["operation"]/"manifest.json")
                need(r["sha256"]==m["sha256"] and r["target_cc"]==61,"external artifact provenance")
            observed={r["operation"] for r in native_rows}
            required={"quantized","mmvq_gguf","mmq_quantize"}|{f"mmq_instance_{c.lower().replace('k','_k')}" for c in CODECS}
            need(required<=observed,"missing codec native submission")
            hits=sum(r["plan_hit"] for r in rows if r["operation"]=="fast_mmq")
            need(hits>0,"MMQ plan cache not reused")
            result[mode]={"native_submissions":sum(r["submitted"] for r in native_rows),"families":sorted(observed),"mmq_plan_hits":hits}
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--verify-seal",type=Path)
    p.add_argument("--prior",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--run",type=Path)
    a=p.parse_args()
    if a.self_test:
        d,t=inventory()
        need(len(d)==51 and len(t)==349,"matrix size")
        need(len([k for k in t if k.startswith("concurrent/")])==48,"concurrency size")
        need(indices(5,7)==[0,1,2,3,4] and indices(129,17)[-1]==128,"sample indices")
        need("mmq_instance_q2_k" in FAMILIES,"codec names")
        try: validate({"schema":"DKR-S683-V1","mode":"legacy","dequant":{},"cases":{}},"legacy")
        except ValueError: pass
        else: raise AssertionError("empty report accepted")
        print("DKR_S683_SELF_TEST=PASS")
        return 0
    if a.verify_seal:
        verify_seal(a.verify_seal,a.prior,a.native_root);return 0
    if not a.run or not a.native_root: p.error("--run and --native-root required")
    root=a.run
    native_audit(a.native_root)
    evidence=paired(read(root/"legacy.json"),read(root/"external.json"))
    ev=telemetry(root,a.native_root)
    report={"status":"EXTENDED_GGUF_MATRIX_PASS","numeric":evidence,"telemetry":ev,
            "artifact_sha256":{n:sha(root/n) for n in ("tests.tar.zst","legacy.json","external.json","legacy.log","external.log","closure.json")},
            "distinct_driver_contexts_verified":False,"full_model_inference_verified":False,
            "full_numerical_qualification":False,"production_promotion_approved":False}
    (root/"qualification-s683.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print("DKR_S683=EXTENDED_GGUF_MATRIX_PASS")
    return 0
if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError,KeyError,OSError,TypeError,AssertionError) as e:
        print(f"DKR_S683=REVIEW_REQUIRED {e}",file=sys.stderr);sys.exit(4)
