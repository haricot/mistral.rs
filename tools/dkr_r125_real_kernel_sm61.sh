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
for component in quant core paged-attn; do
    if [[ ! -s "$MISTRALRS_CUDA_BUILD_ROOT/$component/kernels/.cudaforge_cache.json" ]]; then
        echo "DKR_R125_REAL=BLOCKED missing persistent cudaforge cache index for $component" >&2
        exit 3
    fi
done
echo "DKR_R125_CUDA_CACHE=REUSED root=$MISTRALRS_CUDA_BUILD_ROOT"

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

# Resolve Cargo.lock and inspect the actual Cargo feature graph before
# any CUDA build script. This catches the previous candle-nn missing ffi issue:
# typed-only must NOT activate native cuda-dkr-catalog.
cargo metadata --format-version 1 \
    --features mistralrs-core/dkr-r125-typed-real-pilot |
python3 -c '
import json, sys
data = json.load(sys.stdin)
ids = {p["id"]: p["name"] for p in data["packages"]}
nodes = data["resolve"]["nodes"]
def features_for(name):
    matches = [set(n["features"]) for n in nodes if ids[n["id"]] == name]
    if len(matches) != 1:
        raise SystemExit(f"DKR_R125_CARGO_FEATURES=FAIL expected one {name}, got {len(matches)}")
    return matches[0]
core = features_for("candle-core")
nn = features_for("candle-nn")
kernels = features_for("candle-kernels")
quant = features_for("mistralrs-quant")
paged = features_for("mistralrs-paged-attn")
if "cuda-dkr-typed-image" not in core or "cuda-dkr-catalog" in core or "cuda-dkr-catalog" in nn:
    raise SystemExit("DKR_R125_CARGO_FEATURES=FAIL typed loader is still coupled to native catalog")
if "cuda-legacy-bf16" not in core or "cuda-legacy-bf16" not in kernels:
    raise SystemExit("DKR_R125_CARGO_FEATURES=FAIL Candle BF16 fill/cast are absent on SM61")
if "cuda-legacy-bf16" in quant or "cuda-legacy-bf16" in paged:
    raise SystemExit("DKR_R125_CARGO_FEATURES=FAIL generic Candle BF16 fallback must not enable Mistral legacy producer flags")
print("DKR_R125_CARGO_FEATURES=PASS typed_image=true native_catalog=false candle_bf16_compat=true mistral_legacy_bf16=false")
'
python3 tools/dkr_r125_candle_pin_audit.py

# Keep the three selected CUBIN packages and ptxas image cache persistent.
# Never erase this directory automatically. Its contents are external artifacts,
# not host target/release objects or native CUDA static fallback files.
pilot="${DKR_R125_PILOT_ROOT:-$PWD/target/dkr-r125-real-pilot}"
mkdir -p "$pilot/sm61"
for family in moe indexed-copy paged-decode; do
    cp -a "$src/$family" "$pilot/sm61/"
done
# Reassemble only selected already-compiled PTX to real CUBIN via ptxas.
# Cudaforge is used here only to compile the small Rust package-builder host.
cargo run --release --locked -p mistralrs-kernel-runtime \
    --features builder-cudaforge --bin dkr_package_builder -- \
    --cc 61 --pair-existing --output "$pilot"

export DKR_KERNEL_PACKAGES="$pilot/sm61"
export MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export RUST_BACKTRACE=1

# One feature-unified host build; three different production CUDA adapter
# tests in one command. Each test performs real CUDA execution and numeric
# checks. Release-only, no new target/debug artifacts.
echo "DKR_R125_REAL=START features=typed-real-pilot modes=cubin families=moe,indexed-copy,paged-decode"
log="$pilot/r125-real.log"
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
    if ! grep -Eq "PASS .*\\b${name}\\b" "$log"; then
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
