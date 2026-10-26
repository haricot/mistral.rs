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
if [[ "${ASD_REQUIRED:-false}" == true ]]; then
  test -s /run/haricot/asd/fallbacks/current.json
  test -s /run/haricot/asd/profiles/current.asd
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
done < <(python3 -c 'import json,sys;[print(n["component"],n["candidate"],n["ref"]) for n in json.load(open(sys.argv[1]))["matrix"]]' "$plan")
python3 - "$plan" "$builds" "$report" "$GITHUB_RUN_ID" <<'PY'
import json,sys
from pathlib import Path
plan=json.load(open(sys.argv[1])); builds=Path(sys.argv[2]); report=Path(sys.argv[3])
archives={n["component"]:json.load(open(builds/n["component"]/"build.json"))["files"]["tests.tar.zst"] for n in plan["matrix"]}
counts={n["component"]:int((report/(n["component"]+"-count.txt")).read_text()) for n in plan["matrix"]}
out={"schema_version":2,"candidate":plan["candidate"],"run_id":sys.argv[4],"target":"sm_61","status":"GPU_PASSED","gpu_executed":True,"archives":archives,"test_counts":counts,"full_models_executed":False}
(report/"report.json").write_text(json.dumps(out,sort_keys=True,indent=2)+"\n")
PY
