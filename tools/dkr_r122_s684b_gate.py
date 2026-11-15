#!/usr/bin/env python3
"""S6.8.4-B: seal previous model evidence and audit independent static build."""
import argparse, json, re, sys
from pathlib import Path
from dkr_r122_s683_audit import sha, need
from dkr_r122_s684_model_audit import compare_models, evidence, verify_prior
from dkr_r122_s5_evidence import merge
from dkr_r122_s6_evidence import native_audit
from dkr_s9_ptx_manifest_audit import verify as typed_audit

REV="d392cb70c155ffa1a6aa9312cc482f3be5b9f19a"
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
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--prior",type=Path)
    p.add_argument("--native-root",type=Path)
    p.add_argument("--typed-root",type=Path)
    p.add_argument("--run",type=Path)
    p.add_argument("--features",type=Path)
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
    if not a.prior or not a.native_root or not a.typed_root: p.error("prior and both roots required")
    native_audit(a.native_root);idx=typed_index(a.typed_root)
    r=prior(a.prior,a.native_root)
    print("DKR_S684B_MODEL_SEAL=PASS")
    if not a.run: return 0
    root=a.run
    count=features(root/"static-features.txt")
    # All comparisons use the exact S684 prompt/threshold contract.
    sa=read(root/"static-contiguous.json");sp=read(root/"static-paged.json");ep=read(root/"external-paged.json")
    old=read(a.prior/"external.json")
    need(not sa["paged_requested"] and sp["paged_requested"] and ep["paged_requested"],"KV configuration mismatch")
    contiguous=compare_models(sa,old)
    paged=compare_models(sp,ep)
    ev={m:trace(root,m,idx,a.native_root,m.startswith("static")) for m in ("static-contiguous","static-paged","external-paged")}
    # Paged routing must be observed, not merely requested.
    ops=set(ev["external-paged"]["typed_operations"])
    need("kv_cache.reshape_store.vllm.b256.v1" in ops,"external paged KV store unobserved")
    need(any(x.startswith("attention.paged.") for x in ops),"external paged attention unobserved")
    # Static allocation log can be absent on certain fixed-budget paths.
    # Do not equate a requested paged configuration with driver-level proof.
    observed_static=ev["static-paged"]["paged_allocation_log"]
    out={"status":"FINAL_MODEL_GATE_PASS_SCOPED" if observed_static else "MODEL_PARITY_PASS_STATIC_PAGED_ATTESTATION_OPEN",
         "contiguous":contiguous,"paged":paged,"telemetry":ev,
         "static_cuda_no_dkr_build_verified":True,"static_feature_inventory":count,
         "all_static_vs_all_external_verified_for_model":True,
         "static_paged_allocation_observed":observed_static,
         "external_paged_execution_observed":True,
         "prior_model_report_sha256":sha(a.prior/"qualification-s684-model.json"),
         "artifact_sha256":{p.name:sha(p) for p in sorted(root.glob("*")) if p.is_file() and p.suffix in (".json",".log",".exit",".txt",".sha256")},
         "distinct_driver_contexts_verified":False,"default_cpu_build_verified":False,
         "full_numerical_qualification":False,"production_promotion_approved":False,
         "open_gates":["distinct driver context isolation","default CPU regression build","performance qualification","explicit production approval"]+([] if observed_static else ["static paged allocation attestation"])}
    (root/"qualification-s684b.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2));print("DKR_S684B="+out["status"])
    return 0 if observed_static else 4
if __name__=="__main__":
    try: sys.exit(main())
    except (ValueError,KeyError,OSError,AssertionError,TypeError) as e:
        print(f"DKR_S684B=REVIEW_REQUIRED {e}",file=sys.stderr);sys.exit(4)
