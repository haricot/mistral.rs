#!/usr/bin/env python3
"""DKR R1.23/S1A: archived-binary CUDA attribution (no build).
S1A measures whole-process CUDA activity, observer effect and sampled VRAM.
It deliberately does not report streaming TTFT/TPOT or host dispatch-only
latency: the sealed example has no streaming or per-request NVTX boundaries.
Run --self-test, then --preflight, then DKR_APPROVE_R123_S1A=1.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time

import dkr_r123_s0 as s0
from dkr_r122_s684b_gate import trace, typed_index
from dkr_r122_s683_audit import need, sha
from dkr_r122_s5_evidence import merge

S0_REV = "cc05aab02210fbfdd46660dc50036d30d258f33d"
CACHES = ("contiguous", "paged")
MODES = ("A", "B")
GPU_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
              "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")

def union_ns(rows):
    total = 0
    left = right = None
    for a, b in sorted(rows):
        need(isinstance(a, int) and isinstance(b, int) and 0 <= a <= b,
             "invalid trace interval")
        if left is None:
            left, right = a, b
        elif a > right:
            total += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return total + (right-left if left is not None else 0)

def interval_stats(rows):
    return {"count": len(rows), "sum_ns": sum(b-a for a,b in rows),
            "union_ns": union_ns(rows)}

def analyze_sqlite(path):
    need(path.is_file(), "Nsight sqlite missing")
    db = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        tables = {x[0] for x in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def intervals(table):
            if table not in tables:
                return []
            cols = {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}
            need({"start","end"} <= cols, "unexpected Nsight schema: "+table)
            return list(db.execute(f'SELECT start,end FROM "{table}"'))
        gpu = {t: intervals(t) for t in GPU_TABLES}
        api = {t: intervals(t) for t in API_TABLES}
        kernels = gpu[GPU_TABLES[0]] + gpu[GPU_TABLES[1]]
        need(kernels and any(api.values()), "missing CUDA kernel or API activity")
        need(not (gpu[GPU_TABLES[0]] and gpu[GPU_TABLES[1]]),
             "two kernel tables: possible duplicated activity, review schema")
        all_gpu = [p for v in gpu.values() for p in v]
        all_api = [p for v in api.values() for p in v]
        span = max(b for a,b in all_gpu)-min(a for a,b in all_gpu)
        need(span > 0, "empty GPU span")
        names = {}
        for table in API_TABLES:
            if table not in tables or "StringIds" not in tables:
                continue
            cols = {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}
            if "nameId" not in cols:
                continue
            for name, count, duration in db.execute(
                f'SELECT s.value,COUNT(*),SUM(t.end-t.start) FROM "{table}" t '
                'LEFT JOIN StringIds s ON t.nameId=s.id GROUP BY s.value'):
                names[table+"::"+str(name)] = {"count": count, "sum_ns": duration}
        return {"scope":"whole archived example process; all CUDA activity in trace",
                "gpu_tables":{k:interval_stats(v) for k,v in gpu.items()},
                "api_tables":{k:interval_stats(v) for k,v in api.items()},
                "gpu_kernel":interval_stats(kernels),
                "gpu_activity":interval_stats(all_gpu),
                "api_activity":interval_stats(all_api),
                "api_names":names,"gpu_activity_span_ns":span,
                "gpu_inactive_within_activity_span_ns":span-union_ns(all_gpu),
                "trace_loss_verified":False,
                "per_request_attribution":False}
    finally:
        db.close()

def observer(left, profiled, right):
    mid=(left+right)/2
    need(all(math.isfinite(x) and x>0 for x in (left,profiled,right)), "invalid timing")
    return {"control_midpoint":mid,"profiled":profiled,
            "observer_delta_pct":100*(profiled/mid-1),
            "control_drift_pct":100*abs(right-left)/mid,
            "stable_control":abs(right-left)/mid<=0.10}

def cache_summary(rows):
    events=("plan_hit","plan_miss","plan_evicted","function_hit","function_miss","module_loaded")
    result={}
    for abi in sorted({r["abi"] for r in rows}):
        selected=[r for r in rows if r["abi"]==abi]
        counts={k:sum(r[k] for r in selected) for k in events}
        for prefix in ("plan","function"):
            total=counts[prefix+"_hit"]+counts[prefix+"_miss"]
            counts[prefix+"_hit_rate"]=counts[prefix+"_hit"]/total if total else None
        result[abi]=counts
    return result

def self_test():
    need(union_ns([(0,10),(5,12),(15,20),(20,22)])==19,"union regression")
    need(union_ns([])==0,"empty union")
    need(observer(10,12,10)["stable_control"],"stable control")
    need(not observer(8,10,12)["stable_control"],"unstable control")
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"fixture.sqlite"
        db=sqlite3.connect(path)
        for t in (GPU_TABLES[0],GPU_TABLES[2],API_TABLES[0],API_TABLES[1]):
            db.execute(f'CREATE TABLE "{t}" (start INTEGER,end INTEGER)')
        db.executemany(f'INSERT INTO "{GPU_TABLES[0]}" VALUES (?,?)',[(10,30),(20,40)])
        db.execute(f'INSERT INTO "{GPU_TABLES[2]}" VALUES (50,60)')
        db.execute(f'INSERT INTO "{API_TABLES[0]}" VALUES (1,8)')
        db.execute(f'INSERT INTO "{API_TABLES[1]}" VALUES (3,9)')
        db.commit();db.close()
        r=analyze_sqlite(path)
        need(r["gpu_kernel"]["union_ns"]==30 and r["gpu_activity"]["union_ns"]==40
             and r["gpu_inactive_within_activity_span_ns"]==10
             and r["api_activity"]["union_ns"]==8,"trace aggregation regression")
    c=cache_summary([{"abi":"test","plan_hit":9,"plan_miss":1,"plan_evicted":0,
                      "function_hit":1,"function_miss":1,"module_loaded":1}])["test"]
    need(c["plan_hit_rate"]==0.9 and c["function_hit_rate"]==0.5,"cache counter regression")
    print("DKR_R123_S1A_SELF_TEST=PASS")

def compute_apps():
    return s0.cmd("nvidia-smi","--query-compute-apps=pid,process_name,used_gpu_memory",
                  "--format=csv,noheader,nounits")

def memory_sample(stop, samples, errors):
    fields="memory.used,temperature.gpu,clocks.sm,pstate"
    while not stop.is_set():
        try:
            t=time.monotonic_ns()
            rows=list(csv.reader(s0.cmd("nvidia-smi","--query-gpu="+fields,
                     "--format=csv,noheader,nounits").splitlines(),skipinitialspace=True))
            need(len(rows)==1 and len(rows[0])==4,"memory sampler GPU inventory")
            samples.append({"monotonic_ns":t,"device_used_mib":int(rows[0][0]),
                            "temperature_c":rows[0][1],"sm_clock_mhz":rows[0][2],"pstate":rows[0][3]})
        except Exception as exc:
            errors.append(str(exc))
            return
        stop.wait(0.10)

def prep(orig, s0_report):
    root=Path(__file__).resolve().parents[1]
    rev=s0.cmd("git","-C",str(root),"rev-parse","HEAD")
    subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--","."],check=True)
    subprocess.run(["git","-C",str(root),"diff","--quiet",S0_REV,"HEAD","--",".",":!tools/**"],check=True)
    modelrun=orig/"target/dkr-r122-s684b/556dca9fa4e3/20261116T162401"
    prior=orig/"target/dkr-r122-s684/d392cb70c155/20261115T224855-paged0"
    native=orig/"target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
    typed=modelrun/"typed/sm61"
    need(sha(modelrun/"qualification-s684b.json")==s0.MODEL_SEAL,"wrong model seal")
    seal=s0.model_gate(modelrun,prior,native)
    prereq=s0.verify_c(orig,seal,native)
    report=s0.read(s0_report)
    need(report["status"]=="SCREENING_COMPLETE" and report["numeric_exact_vs_sealed"]
         and report["revision"]==S0_REV and report["model_seal"]==seal
         and report["prerequisites"]==prereq and report["processes"]==18
         and len(report["trials"])==18,"unexpected S0 evidence")
    refs={c:{m:s0.read(modelrun/f"{'static' if m=='A' else 'external'}-{c}.json")
             for m in MODES} for c in CACHES}
    inventory=set()
    for t in report["trials"]:
        stem=t["stem"]
        need(Path(stem).name==stem and stem not in inventory,"invalid S0 stem")
        inventory.add(stem)
        dest=s0_report.parent/f"{stem}.json"
        log=s0_report.parent/f"{stem}.log"
        need(sha(dest)==t["json_sha256"] and sha(log)==t["log_sha256"],"S0 raw evidence changed")
        data=s0.read(dest);ref=refs[t["cache"]][t["mode"]]
        need(data["mode"]==ref["mode"] and data["paged_requested"]==ref["paged_requested"]
             and data["model"]==ref["model"] and t["parity_exact"]
             and s0.exact({**ref,"mode":"legacy"},{**data,"mode":"external"}),
             "S0 parity recomputation failed")
    need(inventory=={f"{i:02d}-{c}-{j}-{m}" for i in range(3) for c in CACHES
                      for j,m in enumerate(("A","B","A"))},"S0 trial inventory mismatch")
    model=s0.read(prior/"qualification-s684-model.json")["model"]
    if (prior/"template.sha256").exists():
        subprocess.run(["sha256sum","-c",str(prior/"template.sha256")],check=True)
    binaries={m:modelrun/f"{'static' if m=='A' else 'external'}-bin" for m in MODES}
    need(all(os.access(p,os.X_OK) for p in binaries.values()),"archive not executable")
    return rev,modelrun,prior,native,typed,seal,prereq,model,refs,binaries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path)
    p.add_argument("--s0-report",type=Path)
    p.add_argument("--rounds",type=int,default=1)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--nsys",default="nsys")
    a=p.parse_args()
    if a.self_test:
        self_test();return
    need(a.original is not None and a.s0_report is not None,"--original and --s0-report required")
    need(1<=a.rounds<=5,"rounds must be 1..5")
    orig=a.original.resolve();s0_report=a.s0_report.resolve()
    nsys=shutil.which(a.nsys)
    need(nsys is not None,"Nsight Systems not found; use --nsys /absolute/path/to/nsys")
    rev,modelrun,prior,native,typed,seal,prereq,model,refs,binaries=prep(orig,s0_report)
    baseline=s0.gpu()
    need(not compute_apps(),"other GPU compute processes active; stop them before qualification")
    version=s0.cmd(nsys,"--version")
    print("DKR_R123_S1A_PREFLIGHT=PASS",json.dumps({"gpu":baseline,"nsys":version}),flush=True)
    if a.preflight:return
    need(os.getenv("DKR_APPROVE_R123_S1A")=="1","set DKR_APPROVE_R123_S1A=1")
    base=orig/"target/dkr-r123-s1a"/rev[:12];base.mkdir(parents=True,exist_ok=True)
    lock=(orig/"target/dkr-r122-s684/.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Path(tempfile.mkdtemp(prefix="run-",dir=base))
    print("DKR_R123_S1A_RUN="+str(run),flush=True)
    trials=[]
    def leg(stem, cache, mode, kind):
        need(not compute_apps(),"unexpected GPU compute process before "+stem)
        dest=run/f"{stem}.json";log=run/f"{stem}.log"
        env=s0.environment(os.environ,model,prior,native,typed,cache=="paged",mode,dest)
        command=[str(binaries[mode])]
        if kind=="profile":
            command=[nsys,"profile","--trace=cuda","--sample=none","--cpuctxsw=none",
                     "--force-overwrite=false","--output="+str(run/stem),*command]
        if kind=="telemetry":
            env["DKR_TELEMETRY"]="1"
            if mode=="B":env["CANDLE_DKR_TRACE"]="1"
        before=s0.gpu()
        samples=[];errors=[];stop=threading.Event();sampler=None
        if kind=="memory":
            sampler=threading.Thread(target=memory_sample,args=(stop,samples,errors),daemon=True)
            sampler.start()
        start=time.monotonic()
        try:
            with log.open("x") as f:
                rc=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        finally:
            elapsed=time.monotonic()-start
            stop.set()
            if sampler is not None:sampler.join()
        after=s0.gpu()
        (run/f"{stem}.exit").write_text(str(rc)+"\n")
        need(rc==0,stem+" failed; inspect "+str(log))
        data=s0.read(dest);ref=refs[cache][mode]
        need(data["mode"]==ref["mode"] and data["model"]==model
             and data["paged_requested"]==(cache=="paged"),"provider/model mismatch")
        need(s0.exact({**ref,"mode":"legacy"},{**data,"mode":"external"}),"numerical mismatch "+stem)
        trial={"stem":stem,"cache":cache,"mode":mode,"kind":kind,"parity_exact":True,
               "gpu_before":before,"gpu_after":after,"process_wall_seconds":elapsed,
               "load_seconds":data["load_seconds"],"metrics":s0.extract(data),
               "json_sha256":sha(dest),"log_sha256":sha(log)}
        if kind=="profile":
            rep=run/f"{stem}.nsys-rep";sql=run/f"{stem}.sqlite"
            need(rep.is_file(),"Nsight report missing")
            subprocess.run([nsys,"export","--type=sqlite","--output="+str(sql),str(rep)],check=True)
            trial["trace"]=analyze_sqlite(sql)
            trial["nsys_report_sha256"]=sha(rep);trial["sqlite_sha256"]=sha(sql)
        if kind=="memory":
            need(not errors and len(samples)>=2,"memory sampling failed: "+str(errors))
            peak=max(x["device_used_mib"] for x in samples)
            trial["memory"]={"scope":"sampled device-wide VRAM; lower bound on true peak, not process allocator peak",
                             "samples":samples,"sampled_peak_device_mib":peak,
                             "device_used_before_mib":int(before["memory.used"]),
                             "sampled_peak_minus_before_mib":peak-int(before["memory.used"])}
        if kind=="telemetry":
            trial["telemetry"]=trace(run,stem,typed_index(typed),native,mode=="A")
            trial["runtime_counters"]=merge(log.read_text().splitlines())
            trial["cache_counters_by_abi"]=cache_summary(trial["runtime_counters"]["rows"])
        s0.write(run/f"{stem}.metrics.json",trial)
        trials.append(trial)
        print(stem,"exact=True",flush=True)
        return trial
    brackets=[]
    for i in range(a.rounds):
        for cache in (CACHES if i%2==0 else CACHES[::-1]):
            for mode in (MODES if i%2==0 else MODES[::-1]):
                prefix=f"{i:02d}-{cache}-{mode}"
                x,y,z=[leg(prefix+"-"+suffix,cache,mode,kind) for suffix,kind in
                       (("control0","control"),("profile","profile"),("control1","control"))]
                measures={}
                for key in s0.KEYS:
                    measures[key]={m:observer(*(t["metrics"][key][1][m] for t in (x,y,z)))
                                   for m in s0.METRICS}
                brackets.append({"round":i,"cache":cache,"mode":mode,"metrics":measures,
                                 "load_seconds":observer(*(t["load_seconds"] for t in (x,y,z))),
                                 "process_wall_seconds":observer(*(t["process_wall_seconds"] for t in (x,y,z)))})
    for cache in CACHES:
        for mode in MODES:
            for kind in ("memory","telemetry"):
                leg(f"{cache}-{mode}-{kind}",cache,mode,kind)
    need(s0.model_gate(modelrun,prior,native)==seal,"model seal changed during S1A")
    report={"status":"ATTRIBUTION_CAPTURE_COMPLETE","revision":rev,"s0_report_sha256":sha(s0_report),
            "model_seal":seal,"prerequisites":prereq,"nsys_version":version,
            "numeric_exact_vs_sealed":True,"new_compilation":False,"cuda_graphs_enabled":False,
            "trials":trials,"observer_brackets":brackets,
            "all_engine_brackets_stable_at_10pct":all(v["stable_control"] for b in brackets
                for d in b["metrics"].values() for v in d.values()),
            "ttft_streaming_measured":False,"tpot_streaming_measured":False,
            "host_dispatch_only_measured":False,"per_request_gpu_attribution":False,
            "true_peak_vram_measured":False,
            "launch_cache_hits_measured":any(c["plan_hit_rate"] is not None for t in trials
                if t["kind"]=="telemetry" for c in t["cache_counters_by_abi"].values()),
            "performance_promotion_approved":False,"production_promotion_approved":False}
    out=run/"attribution-s1a.json";s0.write(out,report)
    for t in trials:
        if t["kind"]=="profile":
            q=t["trace"]
            print(t["stem"],"gpu_union_ms=",round(q["gpu_activity"]["union_ns"]/1e6,3),
                  "api_union_ms=",round(q["api_activity"]["union_ns"]/1e6,3),
                  "gpu_inactive_span_ms=",round(q["gpu_inactive_within_activity_span_ns"]/1e6,3))
        if t["kind"]=="memory":
            print(t["stem"],"sampled_peak_device_mib=",t["memory"]["sampled_peak_device_mib"])
        if t["kind"]=="telemetry":
            print(t["stem"],"cache_counters_by_abi=",json.dumps(t["cache_counters_by_abi"],sort_keys=True))
    print("DKR_R123_S1A="+report["status"])
    print("DKR_R123_S1A_REPORT="+str(out))

if __name__=="__main__":
    try:main()
    except (ValueError,KeyError,OSError,TypeError,sqlite3.Error,subprocess.CalledProcessError) as e:
        print("DKR_R123_S1A=REVIEW_REQUIRED",e,file=sys.stderr);sys.exit(4)
