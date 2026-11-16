#!/usr/bin/env python3
"""Read-only S684B differential diagnosis. Never promotes or launches CUDA."""
import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from dkr_r122_s684b_gate import prior, features, typed_index, trace
from dkr_r122_s684_model_audit import logits, generated
from dkr_r122_s6_evidence import native_audit
from dkr_r122_s683_audit import need, sha
from dkr_r122_s5_evidence import merge

def read(p):
    return json.loads(p.read_text())

def raw_metrics(a,b):
    need(a["prompt_ids"]==b["prompt_ids"],"different prompt IDs")
    need(a["vocab"]==b["vocab"] and a["raw_rows"]==b["raw_rows"] and a["raw_rows"]>0,"different raw layout")
    x,y=a["logits"],b["logits"]
    need(len(x)==len(y)==a["vocab"] and len(x)>1 and all(math.isfinite(v) for v in x+y),"invalid logits")
    d=[u-v for u,v in zip(x,y)]
    rms=math.sqrt(math.fsum(v*v for v in d)/len(d))
    largest=max(range(len(d)),key=lambda i:abs(d[i]))
    max_abs=abs(d[largest])
    denom=math.sqrt(math.fsum(v*v for v in x)*math.fsum(v*v for v in y))
    need(denom>0,"zero logits")
    cos=math.fsum(u*v for u,v in zip(x,y))/denom
    top_a=sorted(range(len(x)),key=x.__getitem__,reverse=True)[:5]
    top_b=sorted(range(len(y)),key=y.__getitem__,reverse=True)[:5]
    try:
        logits(a,b)
        gate=True
        why=None
    except ValueError as e:
        gate=False
        why=str(e)
    return {"pass":gate,"reason":why,"prompt_tokens":len(a["prompt_ids"]),"vocab":len(x),
            "rms":rms,"max_abs":max_abs,"cosine":cos,"identical_elements":sum(u==v for u,v in zip(x,y)),
            "mean_delta":math.fsum(d)/len(d),"max_delta_token":largest,
            "top_a":top_a,"top_b":top_b,
            "top_margin_a":x[top_a[0]]-x[top_a[1]],"top_margin_b":y[top_b[0]]-y[top_b[1]]}

def generation_metrics(a,b):
    try:
        generated(a,b)
        gate=True
        why=None
    except ValueError as e:
        gate=False
        why=str(e)
    x,y=a["tokens"],b["tokens"]
    need(all(math.isfinite(p["logprob"]) for p in x+y),"nonfinite decode probabilities")
    first=next((i for i,(u,v) in enumerate(zip(x,y)) if (u["token"],u["bytes"])!=(v["token"],v["bytes"])),None)
    if first is None and len(x)!=len(y): first=min(len(x),len(y))
    prefix=min(len(x),len(y)) if first is None else first
    return {"pass":gate,"reason":why,"token_counts":[len(x),len(y)],
            "first_token_mismatch_zero_based":first,"common_prefix":prefix,
            "max_logprob_delta_common_prefix":max((abs(x[i]["logprob"]-y[i]["logprob"]) for i in range(prefix)),default=None),
            "same_content":a["content"]==b["content"],"same_reasoning":a["reasoning_content"]==b["reasoning_content"]}

def comparison(a,b,repeat=False):
    need(a["model"]==b["model"] and set(a["cases"])==set(b["cases"])=={"short","medium","long"},"model inventory differs")
    out={}
    for k in ("short","medium","long"):
        x,y=a["cases"][k],b["cases"][k]
        for z in (x,y): need(len(z["raw"])==len(z["generation"])==2,"repeat inventory")
        pairs=[(0,1)] if repeat else [(0,0),(1,1)]
        out[k]={"raw":[raw_metrics(x["raw"][i],y["raw"][j]) for i,j in pairs],
                "generation":[generation_metrics(x["generation"][i],y["generation"][j]) for i,j in pairs]}
    return out

def passed(rows):
    return all(r["pass"] for c in rows.values() for key in ("raw","generation") for r in c[key])

def observed(log):
    ev=merge(log.read_text().splitlines())
    need(ev["processes"]==1,"snapshot count")
    rows=ev["rows"]
    typed=[r for r in rows if r["abi"]=="dkr.typed.v1" and r["adapter_ok"]]
    return {"typed_operations":{op:sum(r["adapter_ok"] for r in typed if r["operation"]==op)
                                  for op in sorted({r["operation"] for r in typed})},
            "typed_success_rows":len(typed),
            "external_native_submissions":sum(r["submitted"] for r in rows if r["abi"]=="candle.native.ptx.v1" and r["sha256"]),
            "runtime_errors":{k:sum(r[k] for r in rows) for k in ("adapter_error","prepare_error","rejected")},
            "paged_allocation_logged":"Validated KV cache allocation" in log.read_text()}

def self_test():
    a={"prompt_ids":list(range(12)),"vocab":3,"raw_rows":12,"logits":[0.,1.,2.]}
    b={**a,"logits":[0.,1.,3.]}
    need(raw_metrics(a,a)["pass"],"positive logits fixture")
    need(not raw_metrics(a,b)["pass"] and raw_metrics(a,b)["max_abs"]==1.,"divergence hidden")
    g={"content":"test","reasoning_content":None,"finish_reason":"length","usage":{"prompt_tokens":12,"completion_tokens":24},
       "tokens":[{"token":"x","bytes":[120],"logprob":-0.3} for _ in range(24)]}
    bad={**g,"tokens":g["tokens"][:3]+[{"token":"y","bytes":[121],"logprob":-0.4}]+g["tokens"][4:]}
    need(generation_metrics(g,g)["pass"] and generation_metrics(g,bad)["first_token_mismatch_zero_based"]==3,"generation fixture")
    print("DKR_S684B_DIAGNOSTIC_SELF_TEST=PASS")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    for x in ("prior","run","native-root","typed-root","output"):
        p.add_argument("--"+x,type=Path)
    a=p.parse_args()
    if a.self_test: self_test();return 0
    if not all((a.prior,a.run,a.native_root,a.typed_root,a.output)):p.error("all paths required")
    native_audit(a.native_root)
    idx=typed_index(a.typed_root)
    prior(a.prior,a.native_root)
    features(a.run/"static-features.txt")
    subprocess.run(["sha256sum","-c",str(a.run/"binaries.sha256")],check=True)
    subprocess.run(["sha256sum","-c",str(a.run/"model.sha256")],check=True)
    external_contiguous = ((a.run, "external-contiguous")
                           if (a.run / "external-contiguous.exit").is_file()
                           else (a.prior, "external"))
    locations={"static-contiguous":(a.run,"static-contiguous"),
               "static-paged":(a.run,"static-paged"),
               "typed-contiguous":(a.prior,"legacy"),
               "external-contiguous":external_contiguous,
               "external-paged":(a.run,"external-paged")}
    ds={};ev={};errors={}
    for name,(root,stem) in locations.items():
        need((root/f"{stem}.exit").read_text().strip()=="0",name+" execution failed")
        data=read(root/f"{stem}.json")
        need(data["schema"]=="DKR-S684-MODEL-V1" and data["paged_requested"]==name.endswith("-paged"),"configuration mismatch "+name)
        ds[name]=data;ev[name]=observed(root/f"{stem}.log")
        if root==a.run:
            try: trace(root,stem,idx,a.native_root,name.startswith("static"))
            except (ValueError,KeyError,OSError) as e: errors[name]=str(e)
    pairs={}
    for left,right in itertools.combinations(ds,2):
        name=left+"__"+right
        try:
            cases=comparison(ds[left],ds[right])
            pairs[name]={"pass":passed(cases),"same_cache_mode":ds[left]["paged_requested"]==ds[right]["paged_requested"],"cases":cases}
        except ValueError as e: pairs[name]={"pass":False,"error":str(e)}
    repeats={k:comparison(v,v,True) for k,v in ds.items()}
    ops=ev["external-paged"]["typed_operations"]
    paged={"allocation":ev["external-paged"]["paged_allocation_logged"],
           "kv_store": "kv_cache.reshape_store.vllm.b256.v1" in ops,
           "attention":any(x.startswith("attention.paged.") for x in ops)}
    report={"status":"DIAGNOSTIC_COMPLETE","production_promotion_approved":False,
            "thresholds_unchanged":True,"comparisons":pairs,"repeats":{k:{"pass":passed(v),"cases":v} for k,v in repeats.items()},
            "telemetry":ev,"provenance_errors":errors,"external_paged_route":paged,
            "input_sha256":{k:{ext:sha(root/f"{stem}.{ext}") for ext in ("json","log","exit")} for k,(root,stem) in locations.items()},
            "new_gpu_execution":False,"new_compilation":False}
    with a.output.open("x") as f: json.dump(report,f,indent=2);f.write("\n")
    for k,v in pairs.items():
        print(k,"PASS" if v["pass"] else "FAIL", "cross-cache" if not v.get("same_cache_mode",False) else "same-cache")
        for name,c in v.get("cases",{}).items():
            raw=c["raw"];gen=c["generation"]
            print(f"  {name:6s} rms={max(z['rms'] for z in raw):.9g} max={max(z['max_abs'] for z in raw):.9g} cosine={min(z['cosine'] for z in raw):.9g} raw={'PASS' if all(z['pass'] for z in raw) else 'FAIL'} decode={'PASS' if all(z['pass'] for z in gen) else 'FAIL'} first_token_mismatch={[z['first_token_mismatch_zero_based'] for z in gen]}")
    print("DKR_S684B_REPEATS",json.dumps({k:passed(v) for k,v in repeats.items()}))
    print("DKR_S684B_TYPED_OPERATIONS",json.dumps({k:v["typed_operations"] for k,v in ev.items()}))
    print("DKR_S684B_PAGED_ROUTE",json.dumps(paged))
    print("DKR_S684B_PROVENANCE_ERRORS",json.dumps(errors))
    print("DKR_S684B_DIAGNOSTIC=COMPLETE promotion_ready=false")
    return 0
if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError,KeyError,OSError,TypeError,AssertionError,subprocess.CalledProcessError) as e:
        print("DKR_S684B_DIAGNOSTIC=REVIEW_REQUIRED",e,file=sys.stderr);sys.exit(4)
