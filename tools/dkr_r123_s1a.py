#!/usr/bin/env python3
"""DKR R1.23/S1A: archived-binary CUDA attribution (no build).
S1A measures whole-process CUDA activity, observer effect and sampled VRAM.
It deliberately does not report streaming TTFT/TPOT or host dispatch-only
latency: the sealed example has no streaming or per-request NVTX boundaries.
Run --self-test, then --preflight, then DKR_APPROVE_R123_S1A=1.
Use --trace-probe to check one archived CUDA run and --inspect-trace to read an existing export.
Use --resume-run after an interruption during memory sampling or telemetry.
"""
import argparse
import csv
import fcntl
import gzip
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
from urllib.request import urlopen

import dkr_r123_s0 as s0
from dkr_r122_s684b_gate import trace, typed_index
from dkr_r122_s683_audit import need, sha
from dkr_r122_s5_evidence import merge

S0_REV = "cc05aab02210fbfdd46660dc50036d30d258f33d"
LEGACY_CAPTURE_REV = "bfc206e9278dbbc0bd7d55e008df13b6c77776de"
CACHES = ("contiguous", "paged")
MODES = ("A", "B")
GPU_IDENTITY = ("index", "name", "compute_cap", "driver_version", "memory.total")
OBSERVATION_FIELDS = ("gpu_before", "gpu_after", "process_wall_seconds")
GPU_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
              "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
API_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
NSYS_CUDA129_VERSION = "2025.1.3.140"
NSYS_REPO = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/"
NSYS_DEB = f"nsight-systems-2025.1.3_{NSYS_CUDA129_VERSION}-1_amd64.deb"

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

def open_trace(path):
    need(path.is_file(), f"Nsight SQLite missing: {path}")
    return sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)


def trace_tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def trace_inventory(path):
    db = open_trace(path)
    try:
        tables = trace_tables(db)
        counts = {
            table: db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if table in tables else None
            for table in GPU_TABLES + API_TABLES
        }
        diagnostics = []
        if "DIAGNOSTIC_EVENT" in tables:
            columns = {row[1] for row in db.execute("PRAGMA table_info(DIAGNOSTIC_EVENT)")}
            if "text" in columns:
                diagnostics = [str(row[0]) for row in db.execute(
                    "SELECT text FROM DIAGNOSTIC_EVENT WHERE text IS NOT NULL LIMIT 30"
                )]
        return {
            "sqlite": str(path.resolve()),
            "tables": sorted(tables),
            "cuda_rows": counts,
            "nsight_diagnostics": diagnostics,
        }
    finally:
        db.close()


def analyze_sqlite(path):
    db = open_trace(path)
    try:
        tables = trace_tables(db)
        def intervals(table):
            if table not in tables:
                return []
            cols = {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}
            need({"start","end"} <= cols, "unexpected Nsight schema: "+table)
            return list(db.execute(f'SELECT start,end FROM "{table}"'))
        gpu = {t: intervals(t) for t in GPU_TABLES}
        api = {t: intervals(t) for t in API_TABLES}
        kernels = gpu[GPU_TABLES[0]] + gpu[GPU_TABLES[1]]
        missing = []
        if not kernels:
            missing.append("GPU kernels")
        if not any(api.values()):
            missing.append("CUDA API calls")
        need(not missing, "missing CUDA activity: " + ", ".join(missing))
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

        db = sqlite3.connect(path)
        try:
            with db:
                db.execute(f'DROP TABLE "{GPU_TABLES[0]}"')
                db.execute('CREATE TABLE DIAGNOSTIC_EVENT (text TEXT)')
                db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES ('CUPTI test diagnostic')")
            info = trace_inventory(path)
            need(info["cuda_rows"][GPU_TABLES[0]] is None
                 and info["cuda_rows"][API_TABLES[1]] == 1
                 and info["nsight_diagnostics"] == ["CUPTI test diagnostic"], "trace inventory regression")
            try:
                analyze_sqlite(path)
            except ValueError as exc:
                need(str(exc) == "missing CUDA activity: GPU kernels", "wrong API-only trace error")
            else:
                need(False, "API-only trace accepted")

            with db:
                db.execute(f'CREATE TABLE "{GPU_TABLES[1]}" (start INTEGER,end INTEGER)')
                db.execute(f'INSERT INTO "{GPU_TABLES[1]}" VALUES (10,30)')
                db.execute(f'DELETE FROM "{API_TABLES[0]}"')
            need(analyze_sqlite(path)["gpu_kernel"]["count"] == 1, "concurrent kernel/driver trace rejected")
            with db:
                db.execute(f'DELETE FROM "{API_TABLES[1]}"')
            try:
                analyze_sqlite(path)
            except ValueError as exc:
                need(str(exc) == "missing CUDA activity: CUDA API calls", "wrong kernel-only trace error")
            else:
                need(False, "kernel-only trace accepted")
        finally:
            db.close()
        root = Path(d)
        checkpoint = root / "checkpoint.json"
        write_checkpoint(checkpoint, {"completed": True})
        need(s0.read(checkpoint) == {"completed": True}, "checkpoint write regression")
        (root / "failed.exit").write_text("-11\n")
        (root / "failed.log").write_text("example failure\n")
        attempt, previous_exit = archive_failed_attempt(root, "failed")
        need(previous_exit == "-11" and (attempt / "failed.exit").read_text() == "-11\n"
             and (attempt / "failed.log").read_text() == "example failure\n"
             and not (root / "failed.log").exists(), "failed attempt archival regression")
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


def archive_failed_attempt(run, stem):
    files = [run / f"{stem}.{suffix}" for suffix in ("exit", "json", "log")]
    files = [path for path in files if path.exists()]
    if not files:
        return
    archive = run / "previous-attempts"
    archive.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=f"{stem}-", dir=archive))
    exit_file = run / f"{stem}.exit"
    previous_exit = exit_file.read_text().strip() if exit_file.is_file() else "unknown"
    for path in files:
        shutil.copy2(path, attempt / path.name)
    for path in files:
        path.unlink()
    return attempt, previous_exit


def write_checkpoint(path, data):
    need(not path.exists(), f"checkpoint already exists: {path}")
    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as stage:
        temp_path = Path(stage) / path.name
        s0.write(temp_path, data)
        temp_path.rename(path)


def prep(orig, s0_report, *, allowed_source_changes=()):
    root=Path(__file__).resolve().parents[1]
    rev=s0.cmd("git","-C",str(root),"rev-parse","HEAD")
    subprocess.run(["git","-C",str(root),"diff","--quiet","HEAD","--","."],check=True)
    excluded = [":!tools/**", *(f":!{path}" for path in allowed_source_changes)]
    subprocess.run(["git","-C",str(root),"diff","--quiet",S0_REV,"HEAD","--",".",*excluded],check=True)
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

def installed_cuda129_nsys(paths):
    for path in paths:
        path = path.absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            continue
        try:
            version = s0.cmd(str(path), "--version", stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            continue
        if any(word.partition("-")[0] == NSYS_CUDA129_VERSION for word in version.split()):
            return path
    return None


def nsys_package_hash():
    with urlopen(NSYS_REPO + "Packages.gz", timeout=30) as response:
        metadata = gzip.decompress(response.read()).decode("utf-8")
    for entry in metadata.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in entry.splitlines()
                      if line[:1].isalpha() and ": " in line)
        if fields.get("Filename", "").removeprefix("./") == NSYS_DEB:
            digest = fields.get("SHA256", "")
            need(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                 "invalid Nsight package digest in NVIDIA metadata")
            return digest
    raise ValueError("Nsight package missing from NVIDIA metadata")


def unpack_nsys(package, dest):
    if shutil.which("dpkg-deb"):
        subprocess.run(["dpkg-deb", "-x", str(package), str(dest)], check=True)
        return
    members = [name for name in s0.cmd("ar", "t", str(package)).splitlines()
               if name == "data.tar" or name.startswith("data.tar.")]
    need(len(members) == 1, "unexpected Nsight Debian archive")
    payload = package.with_name(members[0])
    with payload.open("xb") as f:
        subprocess.run(["ar", "p", str(package), members[0]], stdout=f, check=True)
    subprocess.run(["tar", "-xf", str(payload), "-C", str(dest)], check=True)


def cuda129_nsys(allow_install):
    cache = Path(os.getenv("XDG_CACHE_HOME") or (Path.home() / ".cache")).resolve() / "dkr-nsys"
    target = cache / NSYS_CUDA129_VERSION
    candidates = [Path(entry) / "nsys" for entry in os.get_exec_path()]
    candidates.extend(Path("/opt").glob("cuda*/bin/nsys"))
    candidates.extend(Path("/opt/nvidia/nsight-systems").glob("*/target-linux-x64/nsys"))
    candidates.extend(Path("/usr/local").glob("cuda*/bin/nsys"))
    candidates.extend(target.rglob("target-linux-x64/nsys"))
    existing = installed_cuda129_nsys(candidates)
    if existing:
        return str(existing)
    need(allow_install, "CUDA 12.9 Nsight missing; run --trace-probe --cuda-129-nsys to cache it")
    need(os.uname().machine == "x86_64", "automatic Nsight extraction requires Linux x86_64; use --nsys")
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / (NSYS_CUDA129_VERSION + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = installed_cuda129_nsys(target.rglob("target-linux-x64/nsys"))
        if existing:
            return str(existing)
        need(not target.exists(), f"Nsight cache is unusable at {target}; select an executable with --nsys")
        with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as temp:
            package = Path(temp) / NSYS_DEB
            digest = nsys_package_hash()
            print(f"DKR_R123_S1A_NSYS_DOWNLOAD={NSYS_CUDA129_VERSION}", flush=True)
            subprocess.run(["curl", "-fL", "--retry", "2", "-o", str(package), NSYS_REPO + NSYS_DEB],
                           check=True)
            need(sha(package) == digest, "NVIDIA Nsight package SHA-256 mismatch")
            extracted = Path(temp) / "extracted"
            extracted.mkdir()
            unpack_nsys(package, extracted)
            paths = [p for p in extracted.rglob("target-linux-x64/nsys")
                     if p.resolve().is_relative_to(extracted)]
            installed = installed_cuda129_nsys(paths)
            need(installed is not None, "Nsight launch/version check failed after extraction")
            relative = installed.relative_to(extracted)
            extracted.rename(target)
            return str(target / relative)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path)
    p.add_argument("--s0-report",type=Path)
    p.add_argument("--rounds",type=int,default=1)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--inspect-trace", type=Path, metavar="SQLITE")
    action.add_argument("--trace-probe", action="store_true", help="profile one archived static run")
    action.add_argument("--resume-run", type=Path, metavar="RUN", help="reuse completed brackets and resume sampling")
    profiler = p.add_mutually_exclusive_group()
    profiler.add_argument("--nsys", metavar="PATH")
    profiler.add_argument("--cuda-129-nsys", action="store_true",
                          help="select Nsight Systems from CUDA 12.9 and cache it if needed for a capture")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.inspect_trace is not None:
        print(json.dumps(trace_inventory(a.inspect_trace), indent=2))
        return
    need(a.original is not None and a.s0_report is not None,"--original and --s0-report required")
    need(1<=a.rounds<=5,"rounds must be 1..5")
    if not (a.preflight or a.trace_probe):
        need(os.getenv("DKR_APPROVE_R123_S1A")=="1","set DKR_APPROVE_R123_S1A=1")
    orig=a.original.resolve();s0_report=a.s0_report.resolve()
    rev,modelrun,prior,native,typed,seal,prereq,model,refs,binaries=prep(orig,s0_report)
    nsys = cuda129_nsys(allow_install=not a.preflight) if a.cuda_129_nsys else shutil.which(a.nsys or "nsys")
    need(nsys is not None,"Nsight Systems not found; use --nsys /absolute/path/to/nsys")
    nsys = str(Path(nsys).absolute())
    baseline=s0.gpu()
    need(not compute_apps(),"other GPU compute processes active; stop them before qualification")
    version=s0.cmd(nsys,"--version")
    print("DKR_R123_S1A_PREFLIGHT=PASS",json.dumps({"gpu":baseline,"nsys":version,"nsys_path":nsys}),flush=True)
    if a.preflight:return
    base=orig/"target/dkr-r123-s1a"/rev[:12]
    lock=(orig/"target/dkr-r122-s684/.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    capture_revision = rev
    if a.resume_run is None:
        base.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    else:
        run = a.resume_run.resolve()
        need(run.is_dir() and run.name.startswith("run-") and run.parent.parent == base.parent.resolve(),
             "expected an S1A run directory under the original repository")
        need(run.parent.name in (rev[:12], LEGACY_CAPTURE_REV[:12]), "unsupported S1A capture revision")
        need(not (run / "attribution-s1a.json").exists(), "S1A report already exists")
        need(not (run / "trace-probe.json").exists(), "a standalone trace probe cannot be resumed as a campaign")
        if run.parent.name == LEGACY_CAPTURE_REV[:12]:
            capture_revision = LEGACY_CAPTURE_REV
    context_path = run / "capture-context.json"
    context = {"capture_revision": capture_revision, "rounds": a.rounds, "nsys_version": version,
               "s0_report_sha256": sha(s0_report), "model_seal": seal}
    if a.resume_run is not None:
        if context_path.is_file():
            need(s0.read(context_path) == context, "S1A capture context changed")
        else:
            need(capture_revision == LEGACY_CAPTURE_REV and a.rounds == 1
                 and any(word.partition("-")[0] == NSYS_CUDA129_VERSION for word in version.split()),
                 "expected a one-round legacy run with the CUDA 12.9 Nsight version")
    print("DKR_R123_S1A_RUN="+str(run),flush=True)
    trials = []
    reused_trials = []

    def checked_trial(leg_spec, observation, samples, *, reused=False):
        stem, cache, mode, kind = leg_spec
        dest, log = run / f"{stem}.json", run / f"{stem}.log"
        rc = (run / f"{stem}.exit").read_text().strip()
        need(rc == "0", f"{stem} failed (exit={rc}); inspect {log}")
        data=s0.read(dest);ref=refs[cache][mode]
        need(data["mode"]==ref["mode"] and data["model"]==model
             and data["paged_requested"]==(cache=="paged"),"provider/model mismatch")
        need(s0.exact({**ref,"mode":"legacy"},{**data,"mode":"external"}),"numerical mismatch "+stem)
        elapsed = observation["process_wall_seconds"]
        need(math.isfinite(elapsed) and elapsed > 0, f"invalid process timing for {stem}")
        trial={"stem":stem,"cache":cache,"mode":mode,"kind":kind,"parity_exact":True,
               **observation,
               "load_seconds":data["load_seconds"],"metrics":s0.extract(data),
               "json_sha256":sha(dest),"log_sha256":sha(log)}
        if kind=="profile":
            rep=run/f"{stem}.nsys-rep";sql=run/f"{stem}.sqlite"
            need(rep.is_file(),"Nsight report missing")
            if not reused:
                subprocess.run([nsys,"export","--type=sqlite","--output="+str(sql),str(rep)],check=True)
            try:
                trial["trace"] = analyze_sqlite(sql)
            except ValueError as exc:
                if reused or not sql.is_file():
                    raise
                diagnostic = run / f"{stem}.trace-diagnostic.json"
                info = trace_inventory(sql)
                info.update(error=str(exc), nsys_version=version, nsys_path=nsys, profile_log=str(log),
                            parity_exact=True)
                s0.write(diagnostic, info)
                raise ValueError(f"{exc}; inspect {diagnostic}") from exc
            trial["nsys_report_sha256"]=sha(rep);trial["sqlite_sha256"]=sha(sql)
        if kind=="memory":
            need(len(samples) >= 2, f"too few memory samples for {stem}")
            peak=max(x["device_used_mib"] for x in samples)
            before = int(observation["gpu_before"]["memory.used"])
            trial["memory"]={"scope":"sampled device-wide VRAM; lower bound on true peak, not process allocator peak",
                             "samples":samples,"sampled_peak_device_mib":peak,
                             "device_used_before_mib":before,
                             "sampled_peak_minus_before_mib":peak-before}
        if kind=="telemetry":
            trial["telemetry"]=trace(run,stem,typed_index(typed),native,mode=="A")
            trial["runtime_counters"]=merge(log.read_text().splitlines())
            trial["cache_counters_by_abi"]=cache_summary(trial["runtime_counters"]["rows"])
        return trial

    def leg(stem, cache, mode, kind):
        leg_spec = (stem, cache, mode, kind)
        checkpoint = run / f"{stem}.metrics.json"
        if a.resume_run is not None:
            if checkpoint.is_file():
                saved = s0.read(checkpoint)
                observation = {key: saved[key] for key in OBSERVATION_FIELDS}
                for key in ("gpu_before", "gpu_after"):
                    need(all(observation[key][field] == baseline[field] for field in GPU_IDENTITY),
                         f"GPU or driver differs from saved trial {stem}")
                samples = saved["memory"]["samples"] if kind == "memory" else []
                trial = checked_trial(leg_spec, observation, samples, reused=True)
                need(trial == saved, f"saved S1A evidence differs for {stem}")
                trials.append(trial)
                reused_trials.append(stem)
                print(stem, "exact=True reused=True", flush=True)
                return trial
            need(kind in ("memory", "telemetry"), f"cannot resume an incomplete profile bracket ({stem})")

        need(not compute_apps(),"unexpected GPU compute process before "+stem)
        if a.resume_run is not None:
            archived = archive_failed_attempt(run, stem)
            if archived is not None:
                attempt, previous_exit = archived
                print(f"{stem} previous_exit={previous_exit} archived={attempt}", flush=True)
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
        (run / f"{stem}.exit").write_text(f"{rc}\n")
        need(rc == 0, f"{stem} failed (exit={rc}); inspect {log}")
        observation = {"gpu_before": before, "gpu_after": s0.gpu(), "process_wall_seconds": elapsed}
        need(not errors, f"{stem} memory sampling failed: {errors}")
        trial = checked_trial(leg_spec, observation, samples)
        write_checkpoint(checkpoint, trial)
        trials.append(trial)
        print(stem, "exact=True", flush=True)
        return trial
    if a.trace_probe:
        trial = leg("00-contiguous-A-profile", "contiguous", "A", "profile")
        report = run / "trace-probe.json"
        s0.write(report, {"status": "TRACE_PROBE_PASS", "revision": rev, "nsys_version": version,
                          "nsys_path": nsys, "trial": trial})
        profile = trial["trace"]
        print("DKR_R123_S1A_TRACE_PROBE=PASS",
              f"kernels={profile['gpu_kernel']['count']}",
              f"api_calls={profile['api_activity']['count']}", flush=True)
        print("DKR_R123_S1A_TRACE_PROBE_REPORT=" + str(report))
        return
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
    if not context_path.is_file():
        write_checkpoint(context_path, context)
    for cache in CACHES:
        for mode in MODES:
            for kind in ("memory","telemetry"):
                leg(f"{cache}-{mode}-{kind}",cache,mode,kind)
    need(s0.model_gate(modelrun,prior,native)==seal,"model seal changed during S1A")
    need(sha(s0_report) == context["s0_report_sha256"], "S0 evidence changed during S1A")
    report={"status":"ATTRIBUTION_CAPTURE_COMPLETE","revision":rev,
            "s0_report_sha256":context["s0_report_sha256"],
            "capture_revision":capture_revision,"reused_trials":reused_trials,
            "model_seal":seal,"prerequisites":prereq,"nsys_version":version,"nsys_path":nsys,
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
    out=run/"attribution-s1a.json"
    write_checkpoint(out, report)
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
