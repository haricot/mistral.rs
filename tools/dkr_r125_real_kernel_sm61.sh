#!/usr/bin/env bash
# R1.25 real-kernel gate: MoE grouped GEMM + indexed KV row copying +
# short-context paged decode, using Candle's typed CUBIN loader.
# NO native Candle catalogue required. NO NVCC rebuild of DKR source.
# The host can still require a first Cargo release build: guard its cache.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

if [[ "${DKR_APPROVE_R125_REAL:-0}" != "1" ]]; then
    echo "DKR_R125_REAL=DEFERRED set DKR_APPROVE_R125_REAL=1 to run the physical SM61 gate" >&2
    exit 2
fi
if [[ -z "${MISTRALRS_CUDA_BUILD_ROOT:-}" || ! -d "$MISTRALRS_CUDA_BUILD_ROOT" ]]; then
    echo "DKR_R125_REAL=BLOCKED set MISTRALRS_CUDA_BUILD_ROOT to your EXISTING persistent CUDA object cache" >&2
    echo "This script refuses to silently create a new static CUDA compilation tree." >&2
    exit 3
fi

src="${DKR_R125_TYPED_PACKAGE_ROOT:-$PWD/kernel-packages/build/cuda/sm61}"
if [[ ! -d "$src" ]]; then
    echo "DKR_R125_REAL=BLOCKED missing existing PTX catalogue: $src" >&2
    exit 3
fi

for family in moe indexed-copy paged-decode; do
    if [[ ! -d "$src/$family" ]] || ! find "$src/$family" -type f -name manifest.json -print -quit | grep -q .; then
        echo "DKR_R125_REAL=BLOCKED missing verified PTX family: $src/$family" >&2
        exit 3
    fi
done

# Resolve Cargo.lock only if needed. Does not compile Rust or CUDA kernels.
cargo metadata --format-version 1 > /dev/null
python3 tools/dkr_r125_candle_pin_audit.py

tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
mkdir -p "$tmp/sm61"
for family in moe indexed-copy paged-decode; do
    cp -a "$src/$family" "$tmp/sm61/"
done
# Reassemble only selected already-compiled PTX to real CUBIN via ptxas.
# Cudaforge is used here only to compile the small Rust package-builder host.
cargo run --release --locked -p mistralrs-kernel-runtime \
    --features builder-cudaforge --bin dkr_package_builder -- \
    --cc 61 --pair-existing --output "$tmp"

export DKR_KERNEL_PACKAGES="$tmp/sm61"
export MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export RUST_BACKTRACE=1

# One feature-unified host build; three different production CUDA adapter
# tests in one command. Each test performs real CUDA execution and numeric
# checks. Release-only, no new target/debug artifacts.
echo "DKR_R125_REAL=START features=typed-real-pilot modes=cubin families=moe,indexed-copy,paged-decode"
log="$tmp/r125-real.log"
if ! cargo nextest run --release --locked \
      -p mistralrs-core -p mistralrs-paged-attn --lib \
      --features mistralrs-core/dkr-r125-typed-real-pilot \
      --run-ignored=all --test-threads=1 --no-fail-fast \
      --success-output immediate \
      -E 'test(dkr_moe_gpu) or test(dkr_indexed_copy_gpu) or test(dkr_paged_decode_gpu)' \
      > "$log" 2>&1; then
    cat "$log"
    echo "DKR_R125_REAL=FAIL stage=CUDA_tests" >&2
    exit 1
fi
cat "$log"
# Nextest must actually have run all three tests and Candle must have loaded
# at least one physical CUBIN for each; strict mode has no PTX rollback.
for name in dkr_moe_gpu dkr_indexed_copy_gpu dkr_paged_decode_gpu; do
    if ! grep -q "PASS .*\\b${name}\\b" "$log"; then
        echo "DKR_R125_REAL=FAIL missing successful test $name" >&2
        exit 1
    fi
done
loads="$(grep -c 'CANDLE_DKR_TYPED_IMAGE_LOADED.*format=cubin' "$log" || true)"
if (( loads < 3 )); then
    echo "DKR_R125_REAL=FAIL insufficient physical Candle CUBIN admission evidence: $loads" >&2
    exit 1
fi
if grep -q 'CANDLE_DKR_TYPED_IMAGE_FALLBACK' "$log"; then
    echo "DKR_R125_REAL=FAIL unexpected PTX fallback in strict mode" >&2
    exit 1
fi
echo "DKR_R125_REAL=PASS families=3 mode=cubin loader=candle physical_loads=$loads"
