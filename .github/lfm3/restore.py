#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lfm3_qualification", ROOT / "qualification.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def context_from_report(report):
    return {"policy": q.read(ROOT / "models.json"), "fingerprint": report["fingerprint"],
            "asd_profiles": report["asd_profiles"], "runtimes": {
                r: {"fingerprint": report["runtimes"][r]["fingerprint"],
                    "candidate": report["runtimes"][r].get("candidate")}
                for r in q.RUNTIMES}}


def restore(repo, plan_path, proofs):
    plan = q.read(plan_path)
    expected = q.source_fingerprint(repo, plan["candidate"], plan, q.read(ROOT / "models.json"))
    proofs = Path(proofs)
    page = 1
    while True:
        data = q.api(f"/actions/artifacts?per_page=100&page={page}")
        for artifact in data["artifacts"]:
            if artifact["expired"] or not artifact["name"].startswith("lfm3-model-proof-"):
                continue
            prior = q.api("/actions/runs/" + str(artifact["workflow_run"]["id"]))
            if (prior.get("path") != q.WORKFLOW or prior.get("event") != "workflow_dispatch"
                    or prior.get("head_branch") != "mirror_orch" or prior.get("conclusion") != "success"):
                continue
            with tempfile.TemporaryDirectory(prefix="lfm3-proof-") as tmp:
                folder = Path(tmp)
                q.download_artifact(artifact, folder / "model-proof")
                report = q.read(folder / "model-proof/report.json")
                if report.get("runtimes", {}).get(plan["target"], {}).get("fingerprint") != expected:
                    continue
                if artifact["name"] != "lfm3-model-proof-" + report["fingerprint"]:
                    raise RuntimeError("cached model artifact name and fingerprint differ")
                context = context_from_report(report)
                q.check_report(folder / "model-proof", context)
                q.write(folder / "context.json", context)
                destination = proofs / "model-evidence"
                if destination.exists():
                    raise RuntimeError("refusing to overwrite supplied model evidence")
                shutil.copytree(folder, destination)
                q.attach(repo, plan_path, proofs)
                print("LFM3_MODEL_EVIDENCE_REUSED=PASS original_execution=" + str(report["run_id"])
                      + " candidate=" + plan["candidate"] + " artifact_sha256=" + artifact["digest"])
                return
        if len(data["artifacts"]) < 100:
            break
        page += 1
    raise RuntimeError("no complete real LFM3 model evidence matches this runtime; run the dedicated model qualification")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--proofs", type=Path, required=True)
    args = p.parse_args()
    try:
        restore(args.repo, args.plan, args.proofs)
    except (RuntimeError, KeyError, OSError, ValueError) as error:
        raise SystemExit("LFM3_MODEL_EVIDENCE_REJECT=" + str(error))
