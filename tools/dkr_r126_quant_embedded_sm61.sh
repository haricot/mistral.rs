#!/usr/bin/env bash
# R1.26/S5: build once with the four qualified images, then run Nextest
# against prebuilt binaries with the exact embedded build input absent.
set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

src="$PWD/target/dkr-r126-quant-catalog/sm61"
evidence="$PWD/target/dkr-r126-quant-catalog/evidence.json"
cache="${MISTRALRS_CUDA_BUILD_ROOT:?set persistent SM61 Cudaforge cache root}"
snapshot="${DKR_R126_NVFP4_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4/snapshots/913d13c49fbbb2b477dce78552a31ab650d5b693}"
pilot="$PWD/target/dkr-r126-quant-embedded"
input="$pilot/build-input/sm61"
hidden="$pilot/build-input/sm61.hidden"
features="mistralrs-core/dkr-r125-quant-typed-pilot,mistralrs-core/dkr-nvfp4-gemv"

python3 tools/dkr_r126_nvfp4_preflight.py --snapshot "$snapshot" --packages "$src"
python3 tools/dkr_r125_cudaforge_cache_audit.py --cache-root "$cache" \
    --source-root "$PWD" --require-clean

if [[ "${DKR_APPROVE_R126_QUANT_EMBEDDED:-0}" != 1 ]]; then
    echo "DKR_R126_QUANT_EMBEDDED=DEFERRED preflight=PASS approval_required=true" >&2
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

# Never rename the S4 root or any other qualified original.
# A previous interrupted run may have left this dedicated copy hidden.
if [[ -e "$hidden" || -L "$hidden" ]]; then
    echo "DKR_R126_QUANT_EMBEDDED=BLOCKED previous hidden copy exists: $hidden" >&2
    exit 3
fi
python3 tools/dkr_r126_quant_embedded.py stage --root "$src" \
    --evidence "$evidence" --output "$input"

export DKR_EMBED_TYPED_DIR="$input"
export MISTRALRS_DKR_ARTIFACT_SOURCE=embedded
unset DKR_KERNEL_PACKAGES
export CANDLE_DKR_IMAGE_MODE=cubin
export CANDLE_DKR_TRACE=1
export MISTRALRS_DKR_QUANT_GEMV=1
export MISTRALRS_DKR_QUANT_GEMM=1
export MISTRALRS_DKR_NVFP4_GEMV=1
export MISTRALRS_DKR_NVFP4_GEMM=1
export MISTRALRS_DKR_NVFP4_TYPED_ONLY=1
export DKR_R126_NVFP4_SNAPSHOT="$snapshot"
export RUST_BACKTRACE=1

stamp="$(date -u +%Y%m%dT%H%M%S)-$$"
run="$pilot/$stamp"
mkdir -p "$run"
echo "DKR_R126_EMBED_BUILD=START"
# Save both metadata files before hiding the build input. The nextest
# reuse-build mode cannot run a Cargo rebuild during the isolation window.
cargo metadata --offline --locked --all-features --format-version 1 \
    --filter-platform "$(rustc -vV | sed -n 's/^host: //p')" > "$run/cargo-metadata.json"
cargo nextest list --release --locked --offline --color never \
    -p mistralrs-core -p mistralrs-quant --lib --features "$features" \
    --list-type binaries-only --message-format json \
    > "$run/binaries-metadata.json" 2> >(tee "$run/build.log" >&2)
python3 -m json.tool "$run/binaries-metadata.json" >/dev/null
echo "DKR_R126_EMBED_BUILD=PASS"

restore_input() {
    if [[ -d "$hidden" && ! -e "$input" && ! -L "$input" ]]; then
        mv -T -- "$hidden" "$input"
    elif [[ -e "$hidden" ]]; then
        echo "DKR_R126_EMBED_RESTORE=BLOCKED inspect $hidden and $input" >&2
        return 1
    fi
}
trap restore_input EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mv -T -- "$input" "$hidden"
unset DKR_EMBED_TYPED_DIR DKR_KERNEL_PACKAGES
log="$run/embedded.log"
{
    test ! -e "$input" && test ! -L "$input" || exit 3
    echo "DKR_R126_EMBED_INPUT_ABSENT=PASS"
    cargo nextest run --profile dkr-r125-quant --color never \
        --cargo-metadata "$run/cargo-metadata.json" \
        --binaries-metadata "$run/binaries-metadata.json" \
        --run-ignored=all --no-capture --no-fail-fast \
        -E 'test(dkr_quantized_gemv_gpu) or test(dkr_quantized_gemm_gpu) or test(dkr_blockwise_fp8_layer_forward_gpu) or test(dkr_nvfp4_gemv_gpu) or test(dkr_nvfp4_checkpoint_gemm_gpu) or test(dkr_nvfp4_checkpoint_combined_gpu)' || exit $?
    test ! -e "$input" && test ! -L "$input" || exit 3
    echo "DKR_R126_EMBED_INPUT_STILL_ABSENT=PASS"
} 2>&1 | tee "$log"
restore_input
trap - EXIT INT TERM
python3 tools/dkr_r126_quant_embedded.py verify --root "$src" \
    --evidence "$evidence" --log "$log" --json-out "$run/evidence.json"
