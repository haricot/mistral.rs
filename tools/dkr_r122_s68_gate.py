#!/usr/bin/env python3
"""S6.8 evidence-only integration gate. Never builds, launches or promotes production."""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from dkr_r122_candle_integration import CANDIDATE, FAMILIES, CANDLE_PINS, PROFILE, candidate_gate
from dkr_r122_s6_evidence import native_audit, compare
from dkr_r122_s6_external_recovery import oracle, CODECS
from dkr_r122_s5_evidence import merge
from dkr_r122_s67_promotion import coverage
from dkr_r122_s67_final import NON_GPU

ROOT = Path(__file__).resolve().parents[1]
QUALIFIED = "9ce68d5cc4e13f95baf7fd23d0c62ec41af352a1"
PIN = "08a36261a6b4dd695c291d37c028b16f019fae46"
SYMBOLS = ["launch_mmq_gguf_" + x.lower().replace("k", "_k") if x.endswith("K") else "launch_mmq_gguf_" + x.lower() for x in CODECS]
SYMBOLS += ["launch_mmq_quantize_q8_1_" + x for x in ("D4", "DS4", "D2S6")]
INPUTS = ("closure.json", "archive.key", "archive-build.log", "tests.tar.zst",
          "native-root.txt", "legacy.exit", "external.exit", "legacy.log", "external.log",
          "native-legacy.json", "native-external.json", "qualification-s67.json")
OPEN_GATES = (
    "independent CPU codec oracles for all ten GGUF formats",
    "extended real-model shape and dtype matrix including aligned/tail tiles",
    "multi-thread/context isolation and repeated inference stability",
    "end-to-end model logit and generation parity, memory and performance",
    "non-DKR/default-feature build and regression qualification",
    "explicit production promotion approval",
)

def need(ok, message):
    if not ok:
        raise ValueError(message)

def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def load(p):
    return json.loads(p.read_text())

def host_symbols(candle):
    # Reject the original linker collision between two incompatible C ABIs.
    cdir = candle / "candle-kernels/src/mmq_gguf"
    mdir = ROOT / "mistralrs-quant/kernels/mmq_gguf"
    ffi = (candle / "candle-kernels/src/ffi.rs").read_text()
    ctext = "\n".join(p.read_text() for p in sorted(cdir.glob("*.cu")))
    mtext = "\n".join(p.read_text() for p in sorted(mdir.glob("*.cu")))
    for s in SYMBOLS:
        need(re.search(r'\bfn\s+'+re.escape(s)+r'\s*\(',ffi), "missing Candle Rust declaration: "+s)
        need(f'#[link_name = "candle_{s}"]' in ffi, "missing isolated Candle link name: "+s)
        need(re.search(r'extern\s+"C"\s+int\s+candle_'+re.escape(s)+r'\s*\(',ctext), "missing namespaced C export: "+s)
        need(not re.search(r'extern\s+"C"\s+\w+\s+'+re.escape(s)+r'\s*\(',ctext), "Candle exports old collision: "+s)
        need(re.search(r'extern\s+"C"\s+\w+\s+'+re.escape(s)+r'\s*\(',mtext), "Mistral symbol missing: "+s)
    return SYMBOLS

def source_gate(candle):
    need(CANDIDATE == PIN, "unexpected integration pin constant")
    git("merge-base", "--is-ancestor", QUALIFIED, "HEAD")
    # Only promotion tooling may differ from the exact GPU-qualified revision.
    need(not git("diff", "--name-only", QUALIFIED, "HEAD", "--", ".", ":!tools/**"), "qualified source differs")
    need(not git("status", "--porcelain", "--untracked-files=no"), "tracked worktree changes")
    deps=tomllib.loads((ROOT/"Cargo.toml").read_text())["workspace"]["dependencies"]
    for name in (*CANDLE_PINS, "dkr-kernel-registry"):
        need(deps[name]["rev"]==PIN, "inconsistent Candle pin: "+name)
    core=tomllib.loads((ROOT/"mistralrs-core/Cargo.toml").read_text())["features"]
    need(PROFILE not in core.get("default", []), "experimental profile enabled by default")
    need("dkr-fp4-dequant" not in core[PROFILE], "diagnostic FP4 entered production aggregate")
    changed=candidate_gate(candle)
    names=host_symbols(candle)
    return {"qualified_source":QUALIFIED,"audited_head":git("rev-parse","HEAD"),
            "candle_pin":PIN,"candle_changed_files":changed,
            "isolated_ffi_symbols":names,"experimental_opt_in":True}

def numerical(a,b):
    errs,bad=compare(a,b)
    need(len(errs)==105 and not bad, "A/B numerical failure")
    counts={}
    for name,data in (("legacy",a),("external",b)):
        r=oracle(data)
        need(not [k for k,v in r.items() if v.get("pass") is False],name+" CPU oracle failure")
        counts[name]=dict(collections.Counter(v["scope"] for v in r.values()))
        need(sum(v for k,v in counts[name].items() if k!="diagnostic_only")==126,name+" wrong oracle inventory")
    return {"ab_cases":len(errs),"oracle_scopes":counts,"max_absolute_ab_difference":max(v["max_abs"] for v in errs.values())}

def evidence_gate(run,native):
    for n in INPUTS:
        need((run/n).is_file() and (run/n).stat().st_size>0,"missing artifact: "+n)
    need(run.name==QUALIFIED[:12],"wrong evidence directory")
    closure=load(run/"closure.json")
    need(closure["candidate"]==PIN and closure["cudarc_version"]=="0.19.10","dependency closure mismatch")
    need((run/"legacy.exit").read_text().strip()=="0" and (run/"external.exit").read_text().strip()=="0","test exit failure")
    need(Path((run/"native-root.txt").read_text().strip()).resolve()==native,"native root mismatch")
    native_audit(native)
    a=(run/"legacy.log").read_text()
    b=(run/"external.log").read_text()
    need(re.search(r"Summary\s+\[[^\]]+\]\s+1 test run: 1 passed",a),"legacy nextest missing PASS")
    need(re.search(r"Summary\s+\[[^\]]+\]\s+34 tests run: 34 passed, 2428 skipped",b),"external nextest missing PASS")
    need(not re.search(r"cudaError_t=201|SIGABRT|Rust cannot catch foreign exceptions",a),"legacy error present")
    le,ee=merge(a.splitlines()),merge(b.splitlines())
    cov=coverage(b)
    need(le["processes"]==1 and ee["processes"]==31 and cov["started"]==34 and cov["mapped_snapshots"]==31 and set(cov["without_snapshot"])==NON_GPU,"snapshot coverage mismatch")
    need(le["native_external_submission_rows"]==0,"legacy external contamination")
    mmq=[r for r in le["rows"] if r["operation"]=="fast_mmq" and r["fallback_selected"]]
    need(sum(r["fallback_selected"] for r in mmq)==50 and all(r["entry"]=="static_ffi" for r in mmq),"legacy MMQ route mismatch")
    native_rows=[r for r in ee["rows"] if r["abi"]=="candle.native.ptx.v1" and r["submitted"]]
    need({r["operation"] for r in native_rows}==FAMILIES,"missing native family")
    for r in native_rows:
        m=load(native/r["operation"]/"manifest.json")
        need(r["sha256"]==m["sha256"] and r["target_cc"]==61,"native provenance failure")
    need(not any(r[k] for r in ee["rows"] for k in ("adapter_error","prepare_error","rejected","fallback_selected")),"external counter failure")
    need(ee["typed_adapter_success_rows"]>0,"typed evidence missing")
    num=numerical(load(run/"native-legacy.json"),load(run/"native-external.json"))
    old=load(run/"qualification-s67.json")
    need(old["status"]=="SCOPED_AB_AND_TELEMETRY_PASS" and old["legacy_static_mmq_qualified"] is True and old["legacy_static_mmq_calls"]==50 and not old["failures"],"previous qualification disagreement")
    return {**num,"static_mmq_calls":50,"legacy_external_submissions":0,"external_tests":34,
            "native_families":len(FAMILIES),"native_submissions":sum(r["submitted"] for r in native_rows),
            "typed_adapter_success_rows":ee["typed_adapter_success_rows"],"snapshots":ee["processes"],
            "source_archive_binding":"original campaign path, closure and archive key; no pre-run signed artifact hash",
            "artifact_sha256":{n:digest(run/n) for n in INPUTS},
            "native_manifest_sha256":{f:digest(native/f/"manifest.json") for f in sorted(FAMILIES)}}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        need(len(SYMBOLS)==len(set(SYMBOLS))==13,"wrong ABI inventory")
        need("launch_mmq_gguf_q2_k" in SYMBOLS,"wrong codec spelling")
        from dkr_r122_s6_external_recovery import W,q8_reference,expected_cases
        d={f"{c}/dequant":q8_reference(W) for c in CODECS}
        for k,(v,_,_) in expected_cases(d).items():
            d[k]=v[:]
        need(numerical(d,d)["ab_cases"]==105,"positive fixture failed")
        bad={**d,"F32/affine":[-999.]*8}
        try:
            numerical(d,bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad numerical evidence accepted")
        print("DKR_S68_SELF_TEST=PASS")
        return 0
    if not all((a.report_dir,a.native_root,a.output)):
        p.error("--report-dir, --native-root and --output required")
    run=a.report_dir.resolve()
    out=a.output.resolve()
    need(not out.is_relative_to(run) and not out.is_relative_to(ROOT),"write new report outside source and qualified run")
    need(not out.exists(),"refusing to overwrite promotion report")
    source=source_gate(Path(load(run/"closure.json")["candle_checkout"]))
    evidence=evidence_gate(run,a.native_root.resolve())
    report={"status":"INTEGRATION_GATE_PASS_EXPERIMENTAL_ONLY","source":source,"evidence":evidence,
            "legacy_static_mmq_qualified_for_smoke_scope":True,
            "full_numerical_qualification":False,"production_promotion_approved":False,
            "open_gates":list(OPEN_GATES),"new_compilation":False,"new_gpu_execution":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as f:
        json.dump(report,f,indent=2);f.write("\n")
    print(json.dumps(report,indent=2))
    print("DKR_S68=INTEGRATION_GATE_PASS_EXPERIMENTAL_ONLY")
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,AssertionError,subprocess.CalledProcessError) as e:
        print(f"DKR_S68=REVIEW_REQUIRED {e}",file=sys.stderr)
        raise SystemExit(4)
