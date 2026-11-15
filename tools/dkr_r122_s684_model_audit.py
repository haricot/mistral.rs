#!/usr/bin/env python3
"""Scoped model-level A/B: fixed typed DKR, Candle native off/on."""
import argparse, json, math, sys
from pathlib import Path
from dkr_r122_s683_audit import sha, need
from dkr_r122_s5_evidence import merge
from dkr_r122_s6_evidence import native_audit
from dkr_r122_candle_integration import FAMILIES

# Exact artifact digests from the successful user-supplied S683 qualification.
PRIOR = {
    "tests.tar.zst":"94cd9025082c82307c3f53021051d3adb3a858d939ca962b46d2777e6de6a8c9",
    "legacy.json":"7917fa751f1914652558f53074832989654f7bfefec95086a7c0d3a3f2cb2375",
    "external.json":"6f09a58310e90e721a2516c48f43775ea6c4e2ad369b89c00430cddfb75f9968",
    "legacy.log":"36fa4cac0617f68d6537c9e9fd7991badaf972d5ca89f47e703a49ebab6cecda",
    "external.log":"650430fd4f62cdfcd9d9636bc9f64cc0699f78bcd87529aac15cc9894f3d4cc3",
    "closure.json":"8c9e4fcccac56992b84f7738dc69dcaf6bc91843629046c06b26e89f9fdd939c",
}
def read(p):
    return json.loads(p.read_text())
def verify_prior(root):
    r=read(root/"qualification-s683.json")
    need(r["status"]=="EXTENDED_GGUF_MATRIX_PASS" and r["artifact_sha256"]==PRIOR,
         "S683 report differs from accepted evidence")
    n=r["numeric"];t=r["telemetry"]
    need(n["ab_cases"]==349 and n["ab_elements"]==346402 and n["exact_equal_elements"]==343903 and n["max_absolute_ab_difference"]==0.015625,"S683 numeric inventory")
    for mode in ("legacy","external"):
        v=n[mode]
        need(v["dequant_cases"]==51 and v["dequant_elements"]==13021184 and v["cpu_dot_samples"]==33071 and 0<=v["max_cpu_bound_ratio"]<=1,"S683 CPU evidence")
    need(t["legacy"]["static_mmq_calls"]==518 and t["legacy"]["static_mmvq_calls"]==180 and t["external"]["native_submissions"]==2836 and t["external"]["mmq_plan_hits"]==423,"S683 telemetry")
    for name,h in PRIOR.items(): need(sha(root/name)==h,"S683 artifact mismatch: "+name)
    print("DKR_S684_S683_SEAL=PASS")
def logits(a,b):
    need(a["prompt_ids"]==b["prompt_ids"] and len(a["prompt_ids"])>8,"prompt token identity")
    need(a["vocab"]==b["vocab"] and a["raw_rows"]==b["raw_rows"] and a["raw_rows"]>0,"logits layout")
    x,y=a["logits"],b["logits"]
    need(len(x)==len(y)==a["vocab"] and len(x)>1 and all(math.isfinite(v) for v in x+y),"invalid logits")
    d=[u-v for u,v in zip(x,y)]
    rms=math.sqrt(math.fsum(v*v for v in d)/len(d));maximum=max(map(abs,d))
    denom=math.sqrt(math.fsum(v*v for v in x)*math.fsum(v*v for v in y))
    need(denom>0,"zero logits")
    cosine=math.fsum(u*v for u,v in zip(x,y))/denom
    ax=max(range(len(x)),key=x.__getitem__);ay=max(range(len(y)),key=y.__getitem__)
    # Fixed prospective thresholds; a near tie is review, never silently accepted.
    need(rms<=0.05 and maximum<=0.25 and cosine>=0.9999 and ax==ay,
         f"raw logits mismatch rms={rms} max={maximum} cosine={cosine} argmax={ax}/{ay}")
    return {"rms":rms,"max_abs":maximum,"cosine":cosine,"argmax":ax,"vocab":len(x)}
def generated(a,b):
    need(a["content"]==b["content"] and a["finish_reason"]==b["finish_reason"],"greedy output differs")
    need(a["usage"]["prompt_tokens"]==b["usage"]["prompt_tokens"],"decode prompt count differs")
    need(a["usage"]["completion_tokens"]==b["usage"]["completion_tokens"]==24,"decode token count")
    need(len(a["tokens"])==len(b["tokens"])==24,"decode logprob inventory")
    worst=0.
    for x,y in zip(a["tokens"],b["tokens"]):
        need(x["token"]==y["token"] and x["bytes"]==y["bytes"],"generated token sequence differs")
        need(math.isfinite(x["logprob"]) and math.isfinite(y["logprob"]),"nonfinite logprob")
        worst=max(worst,abs(x["logprob"]-y["logprob"]))
    need(worst<=0.1,"decode logprob divergence")
    return {"generated_tokens":24,"max_logprob_delta":worst}
def compare_models(a,b):
    need(a["schema"]==b["schema"]=="DKR-S684-MODEL-V1" and a["mode"]=="legacy" and b["mode"]=="external","model schema")
    need(a["model"]==b["model"] and a["paged_requested"]==b["paged_requested"],"model configuration differs")
    need(set(a["cases"])==set(b["cases"])=={"short","medium","long"},"prompt inventory")
    out={}
    for key in ("short","medium","long"):
        x,y=a["cases"][key],b["cases"][key]
        for z in (x,y): need(len(z["raw"])==len(z["generation"])==2,"repeat inventory")
        out[key]={"prompt_tokens":len(x["raw"][0]["prompt_ids"]),
                  "raw_ab":[logits(x["raw"][i],y["raw"][i]) for i in range(2)],
                  "decode_ab":[generated(x["generation"][i],y["generation"][i]) for i in range(2)],
                  "raw_repeat":[logits(z["raw"][0],z["raw"][1]) for z in (x,y)],
                  "decode_repeat":[generated(z["generation"][0],z["generation"][1]) for z in (x,y)]}
    need(out["short"]["prompt_tokens"]<out["medium"]["prompt_tokens"]<out["long"]["prompt_tokens"]<=1000,"prompt length matrix")
    return out
def evidence(root,native):
    out={}
    for mode in ("legacy","external"):
        need((root/f"{mode}.exit").read_text().strip()=="0",mode+" failed")
        ev=merge((root/f"{mode}.log").read_text().splitlines())
        need(ev["processes"]==1,mode+" snapshot inventory")
        rows=ev["rows"]
        need(not any(r[k] for r in rows for k in ("adapter_error","prepare_error","rejected")),mode+" runtime error")
        native_rows=[r for r in rows if r["abi"]=="candle.native.ptx.v1" and r["submitted"]]
        if mode=="legacy": need(not native_rows,"legacy native contamination")
        else:
            need(native_rows,"native provider not exercised")
            for r in native_rows:
                need(r["operation"] in FAMILIES,"unknown family")
                m=read(native/r["operation"]/"manifest.json")
                need(r["sha256"]==m["sha256"] and r["target_cc"]==61,"native provenance differs")
            need(not any(r["fallback_selected"] for r in rows if r["abi"]=="candle.native.ptx.v1"),"external native fallback")
        out[mode]={"native_submissions":sum(r["submitted"] for r in native_rows),
                   "native_families":sorted({r["operation"] for r in native_rows}),
                   "typed_success_rows":ev["typed_adapter_success_rows"]}
    return out
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--verify-prior",type=Path)
    p.add_argument("--run",type=Path)
    p.add_argument("--native-root",type=Path)
    a=p.parse_args()
    if a.self_test:
        x={"prompt_ids":list(range(12)),"vocab":3,"raw_rows":12,"logits":[0.,1.,2.]}
        need(logits(x,x)["max_abs"]==0,"positive fixture")
        try: logits(x,{**x,"logits":[0.,1.,9.]})
        except ValueError: pass
        else: raise AssertionError("divergent fixture accepted")
        print("DKR_S684_SELF_TEST=PASS");return 0
    if a.verify_prior: verify_prior(a.verify_prior);return 0
    if not a.run or not a.native_root: p.error("run and native root required")
    r=a.run;native_audit(a.native_root)
    aa,bb=read(r/"legacy.json"),read(r/"external.json")
    cases=compare_models(aa,bb)
    ev=evidence(r,a.native_root)
    out={"status":"MODEL_NATIVE_AB_PASS_SCOPED","cases":cases,"telemetry":ev,
         "paged_requested":aa["paged_requested"],"model":aa["model"],
         "scope":"same binary, typed DKR fixed, Candle native package disabled/enabled",
         "artifact_sha256":{name:sha(r/name) for name in ("legacy.json","external.json","legacy.log","external.log","model.sha256","binary.sha256","closure.json","revision")},
         "distinct_driver_contexts_verified":False,"all_static_vs_all_external_verified":False,
         "non_dkr_build_verified":False,"full_numerical_qualification":False,"production_promotion_approved":False}
    (r/"qualification-s684-model.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2));print("DKR_S684=MODEL_NATIVE_AB_PASS_SCOPED");return 0
if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError,KeyError,TypeError,OSError,AssertionError) as e:
        print(f"DKR_S684=REVIEW_REQUIRED {e}",file=sys.stderr);sys.exit(4)
