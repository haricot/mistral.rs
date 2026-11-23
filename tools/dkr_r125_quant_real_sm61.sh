#!/usr/bin/env bash
# Deferred SM61 quantified-pilot gate: FP8/MXFP4 direct GEMV and GEMM.
# Reuses existing cudaforge PTX, pairs CUBIN through cached ptxas only, and
# executes the TWO previously ignored real GPU numeric tests in one release run.
# Explicit approval required: never accidentally trigger large CUDA rebuilds.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

src="${DKR_R125_TYPED_PACKAGE_ROOT:-$PWD/kernel-packages/build/cuda/sm61}"
root="${MISTRALRS_CUDA_BUILD_ROOT:-}"
if [[ ! -d "$src" ]]; then
    echo "DKR_R125_QUANT=BLOCKED missing existing PTX catalogue $src" >&2
    exit 3
fi
# Fully read-only: no Cargo or nvcc invocation during these checks.
python3 tools/dkr_r125_quant_manifest_preflight.py --cc 61 --root "$src"
if [[ -z "$root" || ! -d "$root" ]]; then
    echo "DKR_R125_QUANT=BLOCKED supply MISTRALRS_CUDA_BUILD_ROOT for existing caches" >&2
    exit 3
fi
python3 tools/dkr_r125_cudaforge_cache_audit.py --cache-root "$root" \
    --source-root "$PWD" --require-clean

if [[ "${DKR_APPROVE_R125_QUANT:-0}" != "1" ]]; then
    echo "DKR_R125_QUANT=DEFERRED manifest_and_cache_preflight_passed=true" >&2
    echo "Explicit approval required: DKR_APPROVE_R125_QUANT=1" >&2
    exit 2
fi
# The checked source/headers/objects are ready. The Cargo feature set can
# nevertheless change native NVCC args; do not claim a guaranteed cache hit.
cargo metadata --format-version 1 \
    --features mistralrs-core/dkr-r125-quant-typed-pilot |
python3 -c '
import json,sys
data=json.load(sys.stdin)
names={p["id"]: p["name"] for p in data["packages"]}
def features(name):
    rows=[set(node["features"]) for node in data["resolve"]["nodes"]
          if names[node["id"]]==name]
    if len(rows)!=1:
        raise SystemExit(f"DKR_R125_QUANT_FEATURES=FAIL expected one {name}")
    return rows[0]
candle=features("candle-core")
quant=features("mistralrs-quant")
paged=features("mistralrs-paged-attn")
assert "cuda-dkr-typed-image" in candle and "cuda-dkr-catalog" not in candle
assert {"cuda-legacy-bf16", "cuda-legacy-fp8"} <= candle
assert {"dkr-quant-gemv", "dkr-quant-gemm"} <= quant
assert not {"cuda-legacy-bf16", "cuda-legacy-fp8"} & quant
assert not {"cuda-legacy-bf16", "cuda-legacy-fp8"} & paged
print("DKR_R125_QUANT_FEATURES=PASS typed=true native_catalog=false candle_fp8_bf16_compat=true")
'
python3 tools/dkr_r125_candle_pin_audit.py

pilot="${DKR_R125_QUANT_PILOT_ROOT:-$PWD/target/dkr-r125-quant-pilot}"
mkdir -p "$pilot/sm61"
for family in quant-gemv quant-gemm; do
    cp -a "$src/$family" "$pilot/sm61/"
done

# No CUDA C++ compilation: --pair-existing uses already compiled PTX.
# The package-builder binary itself is reused from target/release when valid.
cargo run --release --locked -p mistralrs-kernel-runtime \
    --features builder-cudaforge --bin dkr_package_builder -- \
    --cc 61 --pair-existing --output "$pilot"

python3 tools/dkr_r125_quant_manifest_preflight.py --cc 61 \
    --root "$pilot/sm61"

export DKR_KERNEL_PACKAGES="$pilot/sm61"
export MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export MISTRALRS_DKR_QUANT_GEMV=1
export MISTRALRS_DKR_QUANT_GEMM=1
export RUST_BACKTRACE=1

echo "DKR_R125_QUANT=START families=quant-gemv,quant-gemm mode=cubin test_timeout_s=120"
log="$pilot/r125-quant.log"
if ! cargo nextest run --profile dkr-r125-quant --release --locked \
    -p mistralrs-core -p mistralrs-quant --lib \
    --features mistralrs-core/dkr-r125-quant-typed-pilot \
    --run-ignored=all --test-threads=1 --no-fail-fast \
    --no-capture \
    -E 'test(dkr_quantized_gemv_gpu) or test(dkr_quantized_gemm_gpu)' \
    2>&1 | tee "$log"; then
    echo "DKR_R125_QUANT=FAIL stage=GPU_TESTS log=$log" >&2
    exit 1
fi
for name in dkr_quantized_gemv_gpu dkr_quantized_gemm_gpu; do
    if ! grep -Eq "PASS .*${name}\b" "$log"; then
        echo "DKR_R125_QUANT=FAIL missing passed test $name" >&2
        exit 1
    fi
done
loads="$(grep -c 'CANDLE_DKR_TYPED_IMAGE_LOADED.*format=cubin' "$log" || true)"
if (( loads < 4 )); then
    echo "DKR_R125_QUANT=FAIL insufficient CUBIN loads ($loads)" >&2
    exit 1
fi
if grep -q 'CANDLE_DKR_TYPED_IMAGE_FALLBACK' "$log"; then
    echo "DKR_R125_QUANT=FAIL PTX fallback in strict mode" >&2
    exit 1
fi
echo "DKR_R125_QUANT=PASS tests=2 formats=fp8,mxfp4 mode=cubin loads=$loads"
