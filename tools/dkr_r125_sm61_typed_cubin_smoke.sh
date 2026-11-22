#!/usr/bin/env bash
# One CUDA host build; three separate process launches over the same
# exact PTX/CUBIN pair. No Python CUDA compilation and no static rebuild
# between image modes. This is a narrow transport smoke, NOT model parity.
set -euo pipefail

# Never produce target/debug artifacts for this qualification.
# Keep CARGO_TARGET_DIR unchanged so the persistent release cache is reused.
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

cd "$(dirname "$0")/.."
python3 tools/dkr_r125_candle_pin_audit.py

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/sm61"
cp -a kernel-packages/cuda/sm61/elementwise-scale "$tmp/sm61/"

cargo nextest run --release --locked -p mistralrs-kernel-runtime --lib

# Assemble exactly the existing PTX without invoking the CUDA C++ compiler.
cargo run --release --locked -p mistralrs-kernel-runtime \
  --features builder-cudaforge --bin dkr_package_builder -- \
  --cc 61 --pair-existing --output "$tmp"

# Build the executable once, with the new Candle typed-image transport.
cargo build --release --locked -p mistralrs-kernel-runtime \
  --features cuda,shared-registry --example dkr_scale_cuda

for mode in ptx cubin auto; do
    logfile="$tmp/${mode}.log"
    echo "DKR_R125_SM61_TRANSPORT_RUN mode=$mode"
    DKR_KERNEL_PACKAGES="$tmp/sm61" \
    CANDLE_DKR_IMAGE_MODE="$mode" \
    CANDLE_DKR_TRACE=1 \
      cargo run --release --locked -p mistralrs-kernel-runtime \
        --features cuda,shared-registry --example dkr_scale_cuda \
        >"$logfile" 2>&1 || {
            cat "$logfile"
            echo "DKR_R125_SM61_TRANSPORT=FAIL mode=$mode"
            exit 1
        }
    cat "$logfile"
    grep -q 'DKR_AUTO_SCALE_GPU=PASS' "$logfile"
    case "$mode" in
        ptx)
            grep -Eq 'CANDLE_DKR_TYPED_IMAGE_LOADED.*format=ptx' "$logfile"
            ;;
        cubin|auto)
            grep -Eq 'CANDLE_DKR_TYPED_IMAGE_LOADED.*format=cubin' "$logfile"
            if grep -q 'CANDLE_DKR_TYPED_IMAGE_FALLBACK' "$logfile"; then
                echo "DKR_R125_SM61_TRANSPORT=FAIL unexpected_ptx_fallback mode=$mode"
                exit 1
            fi
            ;;
    esac
done
echo "DKR_R125_SM61_TRANSPORT=PASS modes=ptx,cubin,auto image=candle-typed"
