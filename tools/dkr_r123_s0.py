#!/usr/bin/env python3
"""R1.23/S0: controlled archived-binary performance screening, no builds."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import time
import tempfile
from dkr_r122_s683_audit import need, sha
from dkr_r122_s684c_gate import model_gate, cpu_features, TESTS
from dkr_r122_s684_model_audit import compare_models

MODEL_SEAL="e9ea1292e4b15fdf9c7d9bf42446706381cbef00b7cede826460ce6ce6b5bae7"
CPU_LOG="a0e2fdd9f3bd6e4e5865c1fdd90727ba274e40404c97c492dac3d88186130c17"
CPU_FEATURES="ed4eec8a8bcb4f37e645528ff5c725f3e85bbcb25f5b00f09b7872445761a27a"
C_REV="c83f8c981dcd"
KEYS=("short","medium","long")
METRICS=("prefill_ms","completion_ms","wall_ms","prefill_tok_s","completion_tok_s","completion_ms_per_reported_token")
GPU_FIELDS="index,name,compute_cap,driver_version,temperature.gpu,pstate,clocks.sm,clocks.mem,memory.used,memory.total"
def read(p):return json.loads(p.read_text())
def write(p,data):
    with p.open("x") as f:json.dump(data,f,indent=2,allow_nan=False);f.write("\n")
def cmd(*args,**kwargs):return subprocess.check_output(args,text=True,**kwargs).strip()
def gpu():
    rows=list(csv.reader(cmd("nvidia-smi","--query-gpu="+GPU_FIELDS,"--format=csv,noheader,nounits").splitlines(),skipinitialspace=True))
    need(len(rows)==1 and rows[0][2]=="6.1","requires one visible SM61 GPU")
    return dict(zip(GPU_FIELDS.split(","),rows[0]))
def verify_c(orig,seal,native):
    base=orig/"target/dkr-r122-s684c"/C_REV
    d=base/"run-gtIzZC7S"
    dr=read(d/"driver.json")
    need(dr["status"]=="DRIVER_CONTEXT_ISOLATION_PASS" and dr["contexts_simultaneously_live"]==2 and dr["contexts_created"]==3 and dr["total_launches"]==26 and dr["recreated_context_verified"] and dr["surviving_context_after_peer_destruction"],"Driver evidence incomplete")
    need(dr["native_ptx_sha256"]==sha(native/"fill/fill.ptx"),"Driver PTX changed")
    need(read(d/"model-seal.json")==seal,"Driver model seal differs")
    matches=[]
    for p in base.glob("run-*/cpu.json"):
        r=read(p)
        if r.get("log_sha256")==CPU_LOG and r.get("cpu_feature_sha256")==CPU_FEATURES:
            need(r["status"]=="CPU_DEFAULT_SELECTED_REGRESSION_PASS" and r["tests"]==len(TESTS)==19,"CPU report incomplete")
            need(sha(p.parent/"cpu.log")==CPU_LOG and sha(p.parent/"cpu-features.txt")==CPU_FEATURES,"CPU proof changed")
            need(cpu_features(p.parent/"cpu-features.txt")==1683==r["feature_entries"],"CPU feature count")
            need(read(p.parent/"model-seal.json")==seal,"CPU model seal differs")
            matches.append(str(p))
    need(len(matches)>0,"accepted CPU evidence not found")
    return {"driver_report_sha256":sha(d/"driver.json"),"cpu_reports":matches,"cpu_log_sha256":CPU_LOG,"cpu_features_sha256":CPU_FEATURES}
def environment(source,model,prior,native,typed,paged,mode,dest):
    e=dict(source)
    # Avoid inheriting latent experimental switches from previous campaigns.
    for k in list(e):
        if k.startswith("MISTRALRS_DKR_") or k.startswith("DKR_S684_") or k in (
            "DKR_KERNEL_PACKAGES","DKR_TELEMETRY","CANDLE_DKR_S31_DIR","CANDLE_DKR_TRACE",
            "CANDLE_DEQUANTIZE_ALL","CANDLE_DEQUANTIZE_ALL_F16","MISTRALRS_CUDA_PHASE_TIMINGS"):
            e.pop(k,None)
    e.update(HF_HUB_OFFLINE="1",MISTRALRS_CUDA_GRAPHS="0",MISTRALRS_FLASHINFER_DECODE="0",
             DKR_S684_MODE="legacy" if mode=="A" else "external",
             DKR_S684_PAGED=str(int(paged)),DKR_S684_GGUF=model,DKR_S684_OUTPUT=str(dest))
    if (prior/"template.sha256").is_file():
        e["DKR_S684_CHAT_TEMPLATE"]=(prior/"template.sha256").read_text().strip().split("  ",1)[1]
    if mode=="B":
        e.update(CANDLE_DKR_S31_DIR=str(native),DKR_KERNEL_PACKAGES=str(typed),MISTRALRS_DKR_ROUTED_QUANT="1",
                 MISTRALRS_DKR_KV_RESHAPE="1",MISTRALRS_DKR_PAGED_DECODE="1",MISTRALRS_DKR_PAGED_V2="1")
    return e
def extract(data):
    result={}
    for key in KEYS:
        runs=data["cases"][key]["generation"]
        need(len(runs)==2,"repetition inventory")
        result[key]=[]
        for g in runs:
            u=g["usage"]
            n=u["prompt_tokens"];c=u["completion_tokens"]
            p=u["total_prompt_time_sec"];d=u["total_completion_time_sec"];w=g["wall_seconds"]
            need(n=={"short":17,"medium":76,"long":385}[key] and c==24,"token inventory")
            need(all(math.isfinite(x) and x>0 for x in (p,d,w)),"missing or invalid engine timings")
            cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens",0)
            need(cached==0,"unexpected prefix cache hit")
            result[key].append({"prefill_ms":1000*p,"completion_ms":1000*d,"wall_ms":1000*w,
                "prefill_tok_s":n/p,"completion_tok_s":c/d,"completion_ms_per_reported_token":1000*d/c})
    return result
def exact(a,b):
    c=compare_models(a,b)
    return all(z["max_abs"]==0 for v in c.values() for group in ("raw_ab","raw_repeat") for z in v[group]) and all(z["max_logprob_delta"]==0 for v in c.values() for group in ("decode_ab","decode_repeat") for z in v[group])
def evaluate(trials):
    out={}
    for cache in ("contiguous","paged"):
        group=[t for t in trials if t["cache"]==cache]
        need(len(group)%3==0 and len(group)>=3,"incomplete triplets")
        out[cache]={}
        for key in KEYS:
            stats={}
            for m in METRICS:
                ratios=[];drift=[];a_vals=[];b_vals=[]
                for i in range(0,len(group),3):
                    a,b,c=group[i:i+3]
                    need([x["mode"] for x in (a,b,c)]==["A","B","A"] and a["round"]==b["round"]==c["round"],"triplet order")
                    # Primary: second generation only, after raw probes and one generation.
                    x,y,z=[t["metrics"][key][1][m] for t in (a,b,c)]
                    mid=(x+z)/2
                    ratios.append(y/mid);drift.append(abs(z-x)/mid);a_vals.append(mid);b_vals.append(y)
                ratio=st.median(ratios)
                stats[m]={"static_median":st.median(a_vals),"external_median":st.median(b_vals),
                          "external_over_static_median":ratio,"delta_pct":100*(ratio-1),
                          "bracket_drift_max_pct":100*max(drift),"ratios":ratios,
                          "stable_brackets":all(x<=0.10 for x in drift)}
            out[cache][key]=stats
    return out
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--preflight",action="store_true")
    a=p.parse_args();need(1<=a.rounds<=10,"rounds must be 1..10")
    orig=a.original.resolve()
    root=Path(__file__).resolve().parents[1]
    rev=cmd("git","-C",str(root),"rev-parse","HEAD")
    subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--","."],check=True)
    subprocess.run(["git","-C",str(root),"diff","--quiet","c83f8c981dcdd4c744f20755749c1799e7133615","HEAD","--",".",":!tools/**"],check=True)
    modelrun=orig/"target/dkr-r122-s684b/556dca9fa4e3/20261116T162401"
    prior=orig/"target/dkr-r122-s684/d392cb70c155/20261115T224855-paged0"
    native=orig/"target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
    typed=modelrun/"typed/sm61"
    need(sha(modelrun/"qualification-s684b.json")==MODEL_SEAL,"model report differs from accepted seal")
    seal=model_gate(modelrun,prior,native)
    evidence=verify_c(orig,seal,native)
    model=read(prior/"qualification-s684-model.json")["model"]
    need(Path(model).is_file(),"model missing")
    if (prior/"template.sha256").exists():subprocess.run(["sha256sum","-c",str(prior/"template.sha256")],check=True)
    baseline=gpu()
    print("DKR_R123_S0_PREFLIGHT=PASS",json.dumps(baseline),flush=True)
    if a.preflight:return
    need(os.getenv("DKR_APPROVE_R123_S0")=="1","set DKR_APPROVE_R123_S0=1")
    base=orig/"target/dkr-r123-s0"/rev[:12];base.mkdir(parents=True,exist_ok=True)
    run=Path(tempfile.mkdtemp(prefix="run-",dir=base))
    print("DKR_R123_S0_RUN="+str(run),flush=True)
    refs={cache:{mode:read(modelrun/f"{'static' if mode=='A' else 'external'}-{cache}.json")
                 for mode in ("A","B")} for cache in ("contiguous","paged")}
    binaries={mode:modelrun/f"{'static' if mode=='A' else 'external'}-bin" for mode in ("A","B")}
    trials=[]
    for i in range(a.rounds):
        for cache in (("contiguous","paged") if i%2==0 else ("paged","contiguous")):
            for j,mode in enumerate(("A","B","A")):
                stem=f"{i:02d}-{cache}-{j}-{mode}"
                dest=run/f"{stem}.json"
                env=environment(os.environ,model,prior,native,typed,cache=="paged",mode,dest)
                before=gpu()
                start=time.monotonic()
                with (run/f"{stem}.log").open("x") as f:
                    rc=subprocess.run([str(binaries[mode])],env=env,stdout=f,stderr=subprocess.STDOUT).returncode
                elapsed=time.monotonic()-start
                after=gpu()
                need(rc==0,f"{stem} failed; see log")
                data=read(dest)
                need(data["paged_requested"]==(cache=="paged") and data["model"]==model,"model configuration changed")
                # Normalize only the mode label to reuse the established comparator;
                # compare each trial against the *same* provider's sealed output.
                ref=refs[cache][mode]
                need(data["mode"]==ref["mode"],"provider mismatch")
                x,y={**ref,"mode":"legacy"},{**data,"mode":"external"}
                parity=exact(x,y)
                trial={"round":i,"cache":cache,"mode":mode,"stem":stem,"parity_exact":parity,
                       "load_seconds":data["load_seconds"],"process_wall_seconds":elapsed,
                       "gpu_before":before,"gpu_after":after,"metrics":extract(data),
                       "json_sha256":sha(dest),"log_sha256":sha(run/f"{stem}.log")}
                trials.append(trial)
                write(run/f"{stem}.metrics.json",trial)
                print(stem,"exact="+str(parity), "long_decode_tok_s="+str(round(trial["metrics"]["long"][1]["completion_tok_s"],2)),flush=True)
    # Preserve qualified archive and staged package identities.
    need(model_gate(modelrun,prior,native)==seal,"model evidence changed during measurement")
    stats=evaluate(trials)
    all_exact=all(t["parity_exact"] for t in trials)
    stable=all(v["stable_brackets"] for cache in stats.values() for prompt in cache.values() for v in prompt.values())
    report={"status":"SCREENING_COMPLETE" if all_exact else "NUMERICAL_REVIEW_REQUIRED",
            "model_seal":seal,"prerequisites":evidence,"revision":rev,
            "rounds_per_cache":a.rounds,"processes":len(trials),"telemetry_enabled":False,"cuda_graphs_enabled":False,
            "primary_sample":"second nonstreaming generation after two raw probes and first generation",
            "numeric_exact_vs_sealed":all_exact,"all_brackets_stable_at_10pct":stable,
            "results":stats,"trials":trials,
            "ttft_streaming_measured":False,"dispatch_cost_measured":False,"peak_vram_measured":False,
            "performance_promotion_approved":False,"new_compilation":False,
            "scope":"archived optimized dev binaries; engine phase timings with millisecond granularity, not release acceptance"}
    write(run/"performance-screening.json",report)
    for cache,prompts in stats.items():
        for name,metrics in prompts.items():
            print(cache,name,*(f"{k} A={v['static_median']:.3f} B={v['external_median']:.3f} delta={v['delta_pct']:+.2f}% drift={v['bracket_drift_max_pct']:.2f}%" for k,v in metrics.items() if k in ("prefill_tok_s","completion_tok_s","wall_ms")),sep=" | ")
    print("DKR_R123_S0="+report["status"])
    print("DKR_R123_S0_REPORT="+str(run/"performance-screening.json"))
    if not all_exact:sys.exit(4)

if __name__=="__main__":
    try:main()
    except (ValueError,KeyError,OSError,TypeError,subprocess.CalledProcessError) as e:
        print("DKR_R123_S0=REVIEW_REQUIRED",e,file=sys.stderr);sys.exit(4)
