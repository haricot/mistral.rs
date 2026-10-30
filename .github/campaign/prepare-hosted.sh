#!/usr/bin/env bash
set -euo pipefail
: "${CAMPAIGN_TARGET:?}" "${CI_CUDA_IMAGE:?}"
report="$RUNNER_TEMP/campaign"
mkdir -p "$report" "$RUNNER_TEMP/prepare-home" "$RUNNER_TEMP/prepare-cargo"
python3 .github/campaign/campaign.py snapshot --repo "$PWD" --target "$CAMPAIGN_TARGET" \
  --image "$CI_CUDA_IMAGE" --output "$report/snapshot.json"
python3 .github/campaign/verify-candle.py "$report/snapshot.json"
docker run --rm --user "$(id -u):$(id -g)" --security-opt=no-new-privileges \
  --volume "$GITHUB_WORKSPACE:$GITHUB_WORKSPACE:rw" \
  --volume "$RUNNER_TEMP:$RUNNER_TEMP:rw" --workdir "$GITHUB_WORKSPACE" \
  --env "HOME=$RUNNER_TEMP/prepare-home" --env "CARGO_HOME=$RUNNER_TEMP/prepare-cargo" \
  --env RUSTUP_HOME=/opt/rustup --env GITHUB_WORKSPACE --env RUNNER_TEMP \
  --entrypoint /bin/bash "$CI_CUDA_IMAGE" -c '
    set -euo pipefail
    export PATH=/opt/cargo/bin:$PATH
    git config --global --add safe.directory "$GITHUB_WORKSPACE"
    python3 .github/campaign/campaign.py prepare --repo "$PWD" \
      --snapshot "$RUNNER_TEMP/campaign/snapshot.json" --output "$RUNNER_TEMP/campaign"
  '
python3 - "$report/plan.json" <<'PY'
import hashlib,json,os,sys
plan=json.load(open(sys.argv[1]))
def key(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
outputs={"candidate":plan["candidate"],"cache_key":plan["input_key"],"recipe":plan["recipe"],
         "environment_key":key([plan["image"],plan["config"]["toolchain"],plan["config"]["features"],"sm61","bf16,fp8"]),
         "bundle_key":key(sorted(node["build_key"] for node in plan["matrix"]))}
with open(os.environ["GITHUB_OUTPUT"],"a") as stream:
    stream.writelines(f"{name}={value}\n" for name,value in outputs.items())
with open(os.environ["GITHUB_STEP_SUMMARY"],"a") as stream:
    stream.write("Prepared immutable candidate "+plan["candidate"]+" with "+plan["image"]+"\n")
PY
cp -a .github/campaign "$report/orchestrator"
