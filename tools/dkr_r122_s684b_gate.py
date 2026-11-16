#!/usr/bin/env python3
"""S6.8.4-B: seal previous model evidence and audit independent static build."""
import argparse, json, re, sys
from pathlib import Path
from dkr_r122_s683_audit import sha, need
from dkr_r122_s684b_diagnostic import comparison
from dkr_r122_s684_model_audit import compare_models, evidence, verify_prior
from dkr_r122_s5_evidence import merge
from dkr_r122_s6_evidence import native_audit
from dkr_s9_ptx_manifest_audit import verify as typed_audit

REV="d392cb70c155ffa1a6aa9312cc482f3be5b9f19a"
REFERENCE_LEGS = ("static-contiguous", "external-contiguous", "static-paged")
PAGED_ATTENTION_OPERATIONS = {
    "attention.paged.decode.online.v1",
    "attention.paged.v2.partition.p256.v1",
    "attention.paged.v2.merge.p256.v1",
}
SEAL={
"legacy.json":"74448f48d8aa67a0f0ac5e76b1c599028885b8ac99141177ecf5a0638b5486c1",
"external.json":"6680685411de804cb49b936b69d90e4eb9d156f8f96fccee358e02b23ca68493",
"legacy.log":"81f11df86ec41ad3aa696da4b286e41f417bafbe442510067109d6d128444443",
"external.log":"67300517d0bce4f3e829092d4ea6d88cdd528a64fac420ce3170dac721c14d57",
"model.sha256":"f5a165e89a18a0af6f493590da1c885ce2443b849e46973c8d1259721f75063a",
"binary.sha256":"32fe665c09444ad3aaa4dd713d412292cc302b51325cc3836c7d3631aabbd7da",
"closure.json":"8c9e4fcccac56992b84f7738dc69dcaf6bc91843629046c06b26e89f9fdd939c",
"revision":"d6b32abc7a6723a205d3b9bffe3cf7a00ca095128e3b662f431269757029d2bd",
}
def read(p): return json.loads(p.read_text())
def prior(root,native):
    r=read(root/"qualification-s684-model.json")
    need(r["status"]=="MODEL_NATIVE_AB_PASS_SCOPED" and r["artifact_sha256"]==SEAL,"prior model report mismatch")
    for name,h in SEAL.items(): need(sha(root/name)==h,"sealed artifact differs: "+name)
    need((root/"revision").read_text().strip()==REV,"wrong prior revision")
    a,b=read(root/"legacy.json"),read(root/"external.json")
    need(not a["paged_requested"] and not b["paged_requested"],"prior must be contiguous")
    n=compare_models(a,b);ev=evidence(root,native)
    need(n==r["cases"] and ev==r["telemetry"],"prior raw evidence disagrees")
    need(ev["legacy"]["embedded_submissions"]==34430 and ev["external"]["native_submissions"]==34430 and ev["external"]["embedded_submissions"]==0,"prior submission inventory")
    need(all(z["max_abs"]==0 for c in n.values() for z in c["raw_ab"]),"prior logits not exact")
    return r
def features(path):
    lines=path.read_text().splitlines()
    entries=[(s.split("|",1)[0],s.split("|",1)[1].split(",")) for s in lines if "|" in s]
    need(entries and any("mistralrs v" in p and "cuda-legacy" in f for p,f in entries),"missing legacy SDK feature")
    forbidden=[(p,x) for p,fs in entries for x in fs if x.startswith("dkr-") or x in ("cuda-dkr-s31","shared-registry")]
    need(not forbidden,"DKR feature in static build: "+str(forbidden))
    return len(entries)

def verify_reference(root):
    report = read(root / "qualification-s684b.json")
    need(report["contiguous"] is not None, "archived contiguous comparison is missing")
    files = ["binaries.sha256", "model.sha256", "revision.txt", "static-features.txt"]
    if (root / "static-build-revision.txt").is_file():
        files.append("static-build-revision.txt")
    files.extend(f"{leg}.{suffix}" for leg in REFERENCE_LEGS for suffix in ("exit", "json", "log"))
    for name in files:
        need(sha(root / name) == report["artifact_sha256"][name], f"archived {name} digest mismatch")
    for leg in REFERENCE_LEGS:
        need((root / f"{leg}.exit").read_text().strip() == "0", f"archived {leg} failed")

def typed_index(root):
    typed_audit(root,61)
    return {(m["operation"],m["entry"],m["sha256"],m["target_cc"]) for p in root.rglob("manifest.json") for m in [read(p)]}
def trace(root,mode,index,native,static):
    need((root/f"{mode}.exit").read_text().strip()=="0",mode+" failed")
    log=(root/f"{mode}.log").read_text()
    ev=merge(log.splitlines())
    need(ev["processes"]==1,"snapshot count")
    rows=ev["rows"]
    need(not any(r[k] for r in rows for k in ("adapter_error","prepare_error","rejected")),"runtime counter failure")
    typed=[r for r in rows if r["abi"]=="dkr.typed.v1" and r["adapter_ok"]]
    nat=[r for r in rows if r["abi"]=="candle.native.ptx.v1" and r["submitted"] and r["sha256"]]
    emb=[r for r in rows if r["abi"]=="candle.native.ptx.v1" and r["submitted"] and not r["sha256"]]
    if static:
        need(not typed and not nat and "MISTRAL_DKR_MODULE_LOADED" not in log,"external contamination of static binary")
    else:
        need(typed and nat and not emb,"external route missing or embedded submission")
        for r in typed: need(tuple(r[k] for k in ("operation","entry","sha256","target_cc")) in index,"typed manifest provenance")
        for r in nat:
            m=read(native/r["operation"]/"manifest.json")
            need(r["sha256"]==m["sha256"] and r["target_cc"]==61,"native manifest provenance")
        need(not any(r["fallback_selected"] for r in rows if r["abi"]=="candle.native.ptx.v1"),"native fallback")
    return {"typed_operations":sorted({r["operation"] for r in typed}),
            "typed_adapter_ok":sum(r["adapter_ok"] for r in typed),
            "native_submissions":sum(r["submitted"] for r in nat),"embedded_submissions":sum(r["submitted"] for r in emb),
            "paged_allocation_log":"Validated KV cache allocation" in log}

def audit_run(root, index, native):
    cases = {}
    telemetry = {}
    failures = {}
    diagnostic = {}
    rms_operations = {"tensor.rms_residual.plain.v1", "tensor.rms_residual_then_rms.plain.v1"}
    for cache in ("contiguous", "paged"):
        legs = (f"static-{cache}", f"external-{cache}")
        for leg in legs:
            try:
                entry = trace(root, leg, index, native, leg.startswith("static"))
                telemetry[leg] = entry
                if leg.startswith("external"):
                    missing = rms_operations - set(entry["typed_operations"])
                    need(not missing, "missing RMSNorm execution: " + ", ".join(sorted(missing)))
            except (ValueError, KeyError, OSError) as error:
                failures[leg] = str(error)
        left = right = None
        try:
            left, right = (read(root / f"{leg}.json") for leg in legs)
            need(left["paged_requested"] == (cache == "paged"), "KV configuration mismatch")
            cases[cache] = compare_models(left, right)
        except (ValueError, KeyError, OSError) as error:
            failures[f"{cache}_parity"] = str(error)
            if left is not None and right is not None:
                try:
                    diagnostic[cache] = comparison(left, right)
                except (ValueError, KeyError, TypeError) as diagnostic_error:
                    diagnostic[cache] = {"error": str(diagnostic_error)}

    external_ops = set(telemetry.get("external-paged", {}).get("typed_operations", ()))
    observed_kv = "kv_cache.reshape_store.vllm.b256.v1" in external_ops
    missing_attention = sorted(PAGED_ATTENTION_OPERATIONS - external_ops)
    observed_attention = not missing_attention
    if not observed_kv or not observed_attention:
        failures["external_paged_route"] = (
            f"paged KV store observed={observed_kv}, missing attention operations={missing_attention}"
        )
    observed_static = telemetry.get("static-paged", {}).get("paged_allocation_log", False)
    if not observed_static:
        failures["static_paged_allocation"] = "static paged allocation unobserved"
    return cases, telemetry, failures, diagnostic, observed_static, observed_kv, observed_attention

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--prior",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--typed-root",type=Path)
    p.add_argument("--run",type=Path)
    p.add_argument("--features",type=Path)
    p.add_argument("--verify-reference", type=Path)
    a=p.parse_args()
    if a.self_test:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"features.txt"
            f.write_text("mistralrs v0.10.0|cuda,cuda-legacy\ncandle-core v0.1|cuda\n")
            need(features(f)==2,"feature fixture")
            f.write_text(f.read_text()+"candle-core v0.1|cuda-dkr-s31\n")
            try: features(f)
            except ValueError: pass
            else: raise AssertionError("forbidden feature accepted")
        print("DKR_S684B_SELF_TEST=PASS");return 0
    if a.features:
        print("DKR_S684B_STATIC_FEATURES=PASS", features(a.features))
        return 0
    if a.verify_reference:
        verify_reference(a.verify_reference)
        print("DKR_S684B_REFERENCE_SEAL=PASS")
        return 0
    if not a.prior or not a.native_root or not a.typed_root: p.error("prior and both roots required")
    native_audit(a.native_root);idx=typed_index(a.typed_root)
    prior(a.prior,a.native_root)
    print("DKR_S684B_MODEL_SEAL=PASS")
    if not a.run: return 0
    root=a.run
    count=features(root/"static-features.txt")
    cases, ev, failures, diagnostic, observed_static, observed_kv, observed_attention = audit_run(
        root, idx, a.native_root
    )
    out={"status":"REVIEW_REQUIRED" if failures else "FINAL_MODEL_GATE_PASS_SCOPED",
         "contiguous":cases.get("contiguous"),"paged":cases.get("paged"),"telemetry":ev,
         "failures":failures,
         "numerical_diagnostic":diagnostic,
         "static_cuda_no_dkr_build_verified":True,"static_feature_inventory":count,
         "all_static_vs_all_external_verified_for_model":not failures,
         "static_paged_allocation_observed":observed_static,
         "external_paged_kv_store_observed":observed_kv,
         "external_paged_execution_observed":observed_attention,
         "prior_model_report_sha256":sha(a.prior/"qualification-s684-model.json"),
         "artifact_sha256":{p.name:sha(p) for p in sorted(root.glob("*")) if p.is_file() and p.suffix in (".json",".log",".exit",".txt",".sha256")},
         "distinct_driver_contexts_verified":False,"default_cpu_build_verified":False,
         "full_numerical_qualification":False,"production_promotion_approved":False,
         "open_gates":["distinct driver context isolation","default CPU regression build","performance qualification","explicit production approval"]+([] if observed_static else ["static paged allocation attestation"])}
    (root/"qualification-s684b.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2));print("DKR_S684B="+out["status"])
    return 4 if failures else 0
if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError,KeyError,OSError,AssertionError,TypeError) as e:
        print(f"DKR_S684B=REVIEW_REQUIRED {e}",file=sys.stderr);sys.exit(4)
