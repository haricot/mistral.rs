#!/usr/bin/env bash
# R1.26/S3: combined catalogue of two already-qualified images.
# No external device compiler, ptxas or package builder is called.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

src="${DKR_R125_TYPED_PACKAGE_ROOT:-$PWD/kernel-packages/build/cuda/sm61}"
cache="${MISTRALRS_CUDA_BUILD_ROOT:?set persistent SM61 Cudaforge cache root}"
snapshot="${DKR_R126_NVFP4_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4/snapshots/913d13c49fbbb2b477dce78552a31ab650d5b693}"
pilot="$PWD/target/dkr-r126-nvfp4-combined-pilot"
s1="$PWD/target/dkr-r126-nvfp4-pilot/sm61"
features="mistralrs-core/dkr-r125-quant-typed-pilot,mistralrs-core/dkr-nvfp4-gemv"

python3 tools/dkr_r126_nvfp4_preflight.py --snapshot "$snapshot" --packages "$src"
python3 tools/dkr_r125_cudaforge_cache_audit.py --cache-root "$cache" \
    --source-root "$PWD" --require-clean

if [[ "${DKR_APPROVE_R126_NVFP4_COMBINED:-0}" != 1 ]]; then
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

python3 tools/dkr_r126_nvfp4_combined.py stage \
    --root "$pilot/sm61" --gemv "$s1" \
    --gemm "$PWD/target/dkr-r126-nvfp4-gemm-pilot/sm61"

export DKR_KERNEL_PACKAGES="$pilot/sm61"
export MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export MISTRALRS_DKR_NVFP4_GEMV=1
export MISTRALRS_DKR_NVFP4_GEMM=1
export MISTRALRS_DKR_NVFP4_TYPED_ONLY=1
export DKR_R126_NVFP4_SNAPSHOT="$snapshot"
export RUST_BACKTRACE=1
stamp="$(date -u +%Y%m%dT%H%M%S)-$$"
log="$pilot/nvfp4-combined-$stamp.log"
echo "DKR_R126_NVFP4_COMBINED_GPU=START test_timeout_s=120 log=$log"
cargo nextest run --profile dkr-r125-quant --release --locked --offline \
    -p mistralrs-core -p mistralrs-quant --lib \
    --features "$features" --run-ignored=all --no-capture \
    -E 'test(dkr_nvfp4_checkpoint_combined_gpu)' \
    2>&1 | tee "$log"
python3 tools/dkr_r126_nvfp4_combined.py verify --root "$pilot/sm61" \
    --log "$log" --json-out "$pilot/nvfp4-combined-$stamp.json"
