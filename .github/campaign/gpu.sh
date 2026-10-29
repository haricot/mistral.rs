#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plan="$1"
builds="$2"
report="$3"
mkdir -p "$report"
python3 - "$plan" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); plan=json.loads(path.read_text())
actual=hashlib.sha256((path.parent/"candidate.bundle").read_bytes()).hexdigest()
assert actual==plan["bundle_sha256"], "candidate bundle digest mismatch"
PY
test "${EXPECTED_RUN_ID:-}" = "$GITHUB_RUN_ID"
test "${RUNNER_REPO:-}" = "$GITHUB_REPOSITORY"
test "${RUNNER_NO_DEFAULT_LABELS:-}" = true
candidate="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate"])' "$plan")"
test "${EXPECTED_CANDIDATE_SHA:-}" = "$candidate"
asd_required="$(python3 - "$plan" <<'PYCODE'
import json,sys
p=json.load(open(sys.argv[1]))
mandatory=p.get("target")=="extended"
if mandatory:
    components=p["config"]["targets"]["extended"]["components"]
    matrix=[n["component"] for n in p["matrix"]]
    assert "asd" in components and "asd" in matrix and "extended" in matrix
    assert p["config"]["components"]["asd"]["branch"]=="asd_runner_standalone"
    assert p["components"].get("asd"), "ASD SHA missing"
print("true" if mandatory else "false")
PYCODE
)"
if [[ "$asd_required" == true ]]; then
  test "${ASD_REQUIRED:-false}" = true
  test -s /run/haricot/asd/fallbacks/current.json
  test -s /run/haricot/asd/profiles/current.asd
  mkdir -p "$report/asd"
  sha256sum /run/haricot/asd/fallbacks/current.json > "$report/asd/current.json.sha256"
  sha256sum /run/haricot/asd/profiles/current.asd > "$report/asd/current.asd.sha256"
fi
nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv,noheader > "$report/gpu.txt"
python3 - "$report/gpu.txt" <<'PY'
import csv,sys
rows=list(csv.reader(open(sys.argv[1]),skipinitialspace=True))
assert rows and all(row[2].strip()=="6.1" for row in rows), "physical SM61 device required"
PY
while read -r component sha ref; do
  dir="$builds/$component"
  python3 "$script_dir/proof.py" verify "$dir" "$component" --plan "$plan"
  git fetch --no-tags "$(dirname "$plan")/candidate.bundle" "$ref"
  git switch --detach --force "$sha"
  test "$(git rev-parse HEAD)" = "$sha"
  chmod +x "$dir/cargo-nextest"
  python3 "$script_dir/proof.py" config "$report/nextest-$component.toml" gpu --junit "$report/$component-junit.xml"
  "$dir/cargo-nextest" nextest run --archive-file "$dir/tests.tar.zst" \
    --workspace-remap "$PWD" --extract-to "$report/target-$component" \
    --config-file "$report/nextest-$component.toml" --profile gpu --no-tests=fail \
    > "$report/$component.log" 2>&1 || { tail -n 100 "$report/$component.log"; exit 1; }
  python3 "$script_dir/proof.py" gpu-count "$report/$component-junit.xml" > "$report/$component-count.txt"
  if [[ "$(python3 -c 'import json,sys;print(str(json.load(open(sys.argv[1]))["config"].get("lfm3_qualification",{}).get("require_sm61_causal_conv_parity",False)).lower())' "$plan")" == true ]]; then
    python3 "$script_dir/proof.py" gpu-required "$report/$component-junit.xml" cuda_prefill_decode_and_cached_chunks_match_reference > "$report/$component-lfm3-count.txt"
  fi
  if [[ "$asd_required" == true && ( "$component" == asd || "$component" == extended ) ]]; then
    python3 "$script_dir/proof.py" gpu-required "$report/$component-junit.xml" asd_sm61_smoke > "$report/$component-asd-count.txt"
  fi
done < <(python3 -c 'import json,sys;[print(n["component"],n["candidate"],n["ref"]) for n in json.load(open(sys.argv[1]))["matrix"]]' "$plan")
python3 - "$plan" "$builds" "$report" "$GITHUB_RUN_ID" <<'PY'
import json,sys
from pathlib import Path
plan=json.load(open(sys.argv[1])); builds=Path(sys.argv[2]); report=Path(sys.argv[3])
archives={n["component"]:json.load(open(builds/n["component"]/"build.json"))["files"]["tests.tar.zst"] for n in plan["matrix"]}
counts={n["component"]:int((report/(n["component"]+"-count.txt")).read_text()) for n in plan["matrix"]}
asd_required=plan["target"]=="extended"
asd_profiles={}
asd_counts={}
if asd_required:
    import re
    for path,name in (("current.json.sha256","current.json"),("current.asd.sha256","current.asd")):
        fingerprint=(report/"asd"/path).read_text().split()[0]
        assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)
        asd_profiles[name]=fingerprint
    for name in ("asd","extended"):
        asd_counts[name]=int((report/(name+"-asd-count.txt")).read_text())
        assert asd_counts[name]>0
lfm3_conv_gpu_counts={}
if plan["config"].get("lfm3_qualification",{}).get("require_sm61_causal_conv_parity"):
    for name in counts:
        lfm3_conv_gpu_counts[name]=int((report/(name+"-lfm3-count.txt")).read_text())
        assert lfm3_conv_gpu_counts[name]>0
out={"schema_version":2,"candidate":plan["candidate"],"run_id":sys.argv[4],"target":"sm_61","status":"GPU_PASSED","gpu_executed":True,"archives":archives,"test_counts":counts,"asd_required":asd_required,"asd_profiles":asd_profiles,"asd_test_counts":asd_counts,"lfm3_conv_gpu_counts":lfm3_conv_gpu_counts,"full_models_executed":False}
(report/"report.json").write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
PY
