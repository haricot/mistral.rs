#!/usr/bin/env bash
# R1.26/S1: one isolated NVFP4 GEMV catalogue and bounded Nextest run.
# The only device-code production permitted here is pairing existing PTX
# through the already-built Cudaforge package builder (ptxas, not nvcc).
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

src="${DKR_R125_TYPED_PACKAGE_ROOT:-$PWD/kernel-packages/build/cuda/sm61}"
cache="${MISTRALRS_CUDA_BUILD_ROOT:?set persistent SM61 Cudaforge cache root}"
snapshot="${DKR_R126_NVFP4_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4/snapshots/913d13c49fbbb2b477dce78552a31ab650d5b693}"
pilot="$PWD/target/dkr-r126-nvfp4-pilot"
builder="${DKR_R126_PACKAGE_BUILDER:-$PWD/target/release/dkr_package_builder}"
features="mistralrs-core/dkr-r125-quant-typed-pilot,mistralrs-core/dkr-nvfp4-gemv"

python3 tools/dkr_r126_nvfp4_preflight.py --snapshot "$snapshot" --packages "$src"
python3 tools/dkr_r125_cudaforge_cache_audit.py --cache-root "$cache" \
    --source-root "$PWD" --require-clean

if [[ "${DKR_APPROVE_R126_NVFP4:-0}" != 1 ]]; then
    echo "DKR_R126_NVFP4=DEFERRED preflight=PASS approval_required=true" >&2
    exit 2
fi

# Reconcile the full lock graph without building; Nextest queries all features.
cargo metadata --offline --all-features --format-version 1 >/dev/null
cargo metadata --offline --all-features --locked --format-version 1 >/dev/null
python3 tools/dkr_r125_candle_pin_audit.py
cargo metadata --offline --locked --format-version 1 --features "$features" |
python3 -c '
import json, sys
data=json.load(sys.stdin)
names={p["id"]:p["name"] for p in data["packages"]}
def feats(name):
    rows=[set(n["features"]) for n in data["resolve"]["nodes"] if names[n["id"]]==name]
    if len(rows)!=1:
        raise SystemExit("expected unique package: "+name)
    return rows[0]
candle, quant, paged = map(feats, ("candle-core","mistralrs-quant","mistralrs-paged-attn"))
assert {"cuda-dkr-typed-image","cuda-legacy-bf16","cuda-legacy-fp8"} <= candle
assert not {"cuda-dkr-catalog","cuda-dkr-s31","cuda-legacy-fp4"} & candle
assert "dkr-nvfp4-gemv" in quant
assert not {f for f in quant | paged if f.startswith("cuda-legacy")}
print("DKR_R126_NVFP4_FEATURES=PASS typed=true static_legacy=false")
'

mkdir -p "$pilot/sm61"
if [[ ! -d "$pilot/sm61/nvfp4-gemv" ]]; then
    cp -a "$src/nvfp4-gemv" "$pilot/sm61/"
fi

# Require exact PTX identity across the original and copied catalogue.
pair_status="$(PYTHONPATH=tools python3 - "$src" "$pilot/sm61" <<'PY'
from pathlib import Path
import sys
from dkr_r126_nvfp4_preflight import inspect_packages
source, staged = map(lambda x: inspect_packages(Path(x)), sys.argv[1:])
if source["ptx_sha256"] != staged["ptx_sha256"]:
    raise SystemExit("staged PTX differs from validated source; refusing to overwrite")
print(staged["cubin_status"])
PY
)"
if [[ "$pair_status" == PTX_ONLY ]]; then
    if [[ ! -x "$builder" ]]; then
        echo "DKR_R126_NVFP4=BLOCKED existing package builder missing: $builder" >&2
        exit 3
    fi
    "$builder" --cc 61 --pair-existing --output "$pilot"
fi
python3 tools/dkr_r126_nvfp4_preflight.py --snapshot "$snapshot" --packages "$pilot/sm61"
PYTHONPATH=tools python3 - "$pilot/sm61" <<'PY'
from pathlib import Path
import sys
from dkr_r126_nvfp4_preflight import inspect_packages
p = inspect_packages(Path(sys.argv[1]))
if p["cubin_status"] != "PAIRED" or p["cubin_images"] != 1:
    raise SystemExit("strict NVFP4 CUBIN not ready")
PY

export DKR_KERNEL_PACKAGES="$pilot/sm61"
export MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export MISTRALRS_DKR_NVFP4_GEMV=1
export MISTRALRS_DKR_NVFP4_TYPED_ONLY=1
export DKR_R126_NVFP4_SNAPSHOT="$snapshot"
export RUST_BACKTRACE=1
stamp="$(date -u +%Y%m%dT%H%M%S)-$$"
log="$pilot/nvfp4-gemv-$stamp.log"
echo "DKR_R126_NVFP4_GPU=START test_timeout_s=120 log=$log"
cargo nextest run --profile dkr-r125-quant --release --locked --offline \
    -p mistralrs-core -p mistralrs-quant --lib \
    --features "$features" --run-ignored=all --no-capture --no-fail-fast \
    -E 'test(dkr_nvfp4_gemv_gpu) or test(dkr_nvfp4_checkpoint_layer_gpu)' \
    2>&1 | tee "$log"
python3 tools/dkr_r126_nvfp4_gpu_evidence.py --packages "$pilot/sm61" \
    --log "$log" --json-out "$pilot/nvfp4-gemv-$stamp.json"
