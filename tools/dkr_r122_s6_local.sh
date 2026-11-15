#!/usr/bin/env bash
# Local-only S6. Resume by rerunning the same command; no cargo clean, no CI.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set to original qualified checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
[[ "$ORIG" != "$ROOT" ]]
CACHE="$ORIG/target/dkr-cuda-cache/sm61"
export MISTRALRS_CUDA_BUILD_ROOT="$CACHE"
for component in quant core paged-attn; do
    [[ -s "$CACHE/$component/kernels/.cudaforge_cache.json" ]] || { echo "missing cache index $component"; exit 3; }
done
objects="$(find "$CACHE" -name '*.o' -type f | wc -l)"
[[ "$objects" -ge 82 ]] || { echo "insufficient persistent objects: $objects"; exit 3; }
echo "DKR_S6_CACHE_GUARD=PASS objects=$objects root=$CACHE"
echo "CUDA_COMPUTE_CAP unchanged: ${CUDA_COMPUTE_CAP-<unset>}"
if [[ -n "${CUDA_COMPUTE_CAP:-}" && "$CUDA_COMPUTE_CAP" != 61 ]]; then
    echo "Refusing non-SM61 CUDA_COMPUTE_CAP"; exit 3
fi
command -v nvcc; command -v jq; command -v nvidia-smi
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')" == 6.1 ]] || { echo "Expected one physical SM61 GPU"; exit 3; }
NEXT_VERSION="$(cargo nextest --version)"
printf '%s\n' "$NEXT_VERSION"
# S6 policy floor: a recent release with modern target metadata support.
# The old 0.9.47 graph parser fails on arm64ec-pc-windows-msvc even on Linux.
python3 - "$NEXT_VERSION" <<'PYVERSION'
import re, sys
m = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", sys.argv[1])
if not m or tuple(map(int, m.groups())) < (0, 9, 145):
    sys.exit("DKR_S6_NEXTEST=FAIL use recent nextest (S6 policy >=0.9.145); see https://nexte.st/docs/installation/pre-built-binaries/")
print("DKR_S6_NEXTEST=PASS")
PYVERSION
RUN="$ORIG/target/dkr-r122-s6/$(git rev-parse --short=12 HEAD)"
mkdir -p "$RUN"
exec 9>"$RUN/.lock"
flock -n 9 || { echo "S6 already running"; exit 3; }
python3 tools/dkr_r122_s4_dependency_closure.py --locked --output "$RUN/closure.json"
CANDLE="$(jq -er .candle_checkout "$RUN/closure.json")"
python3 tools/dkr_r122_s5_evidence.py --self-test
python3 tools/dkr_r122_s6_evidence.py --self-test
python3 tools/dkr_s9_ptx_manifest_audit.py --self-test
python3 tools/dkr_r121_manifest_preflight.py
python3 tools/dkr_r121_v2_numeric_preflight.py
python3 tools/dkr_r121_s7_cpu_preflight.py
python3 tools/dkr_r121_s8_cpu_preflight.py
python3 tools/dkr_nvcc_flag_audit.py
# Original typed package sources must be unchanged.
BASE=e74f6c0583b9c1c55ed5d9aa057f56cf5fb2f02c
TYPED="$ORIG/kernel-packages/build/cuda/sm61"
if ! git diff --quiet "$BASE" HEAD -- kernel-packages/cuda mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs; then
    echo "Typed package source provenance differs; review/rebuild separately"; exit 3
fi
python3 tools/dkr_s9_ptx_manifest_audit.py --root "$TYPED" --cc 61
df -h "$ORIG/target" /tmp
echo "DKR_S6_PREFLIGHT=PASS run=$RUN"
if [[ "${1:-}" == "--preflight" ]]; then exit 0; fi
[[ "${DKR_APPROVE_S6:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S6=1 for native PTX and grouped build"; exit 2; }
export DKR_KERNEL_PACKAGES="$TYPED"
export DKR_TELEMETRY=1 CANDLE_DKR_TRACE=1 MISTRALRS_DKR_ROUTED_QUANT=1
# Test the actual GGUF paths rather than an inherited dequantize-all override.
unset CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
NATIVE="$RUN/native-sm61"
if [[ -n "${DKR_S6_REUSE_NATIVE_RUN:-}" ]]; then
    PRIOR="$(realpath "$DKR_S6_REUSE_NATIVE_RUN")"
    [[ "$PRIOR" == "$ORIG/target/dkr-r122-s6/"* ]] || { echo "Invalid prior run path"; exit 3; }
    PRIOR_PIN="$(jq -er .candidate "$PRIOR/closure.json")"
    CURRENT_PIN="$(jq -er .candidate "$RUN/closure.json")"
    if [[ "$PRIOR_PIN" != "$CURRENT_PIN" ]]; then
        # Exact reviewed transition: candidate_gate already validates the
        # three-file delta and unchanged CUDA device code prefix.
        [[ "$PRIOR_PIN" == b231d3c4e07b04c0df788ee252979b1890430dd6 &&
           "$CURRENT_PIN" == 63217e3a0ea8a91b0630acbcbc047c471f43761a ]] ||
           { echo "Prior Candle candidate differs"; exit 3; }
        echo "DKR_S67_NATIVE_REUSE=HOST_ONLY_DELTA device source unchanged"
    fi
    grep -Fxq 'CANDLE_DKR_S31_PACKAGES=STAGED target=sm61 families=25' "$PRIOR/native-build.log"
    NATIVE="$PRIOR/native-sm61"
    python3 tools/dkr_r122_s6_evidence.py --native-root "$NATIVE"
    echo "DKR_S6_NATIVE_PTX=REUSED root=$NATIVE"
else
    mkdir -p "$NATIVE"
    # Cudaforge owns incremental native PTX build and its ABI checks.
    cargo run --manifest-path "$CANDLE/tools/dkr-s31-builder/Cargo.toml" \
      --target-dir "$ORIG/target/dkr-s31-builder" --release -- --output "$NATIVE" 2>&1 | tee "$RUN/native-build.log"
    python3 tools/dkr_r122_s6_evidence.py --native-root "$NATIVE"
fi
printf '%s\n' "$NATIVE" > "$RUN/native-root.txt"
# Fingerprint archive inputs; failed archive is never used.
export CARGO_TARGET_DIR="$ORIG/target/dkr-r122-s6-cargo"
KEY="$( { git rev-parse HEAD; printf '%s\n' "$ROOT" "$CACHE" "$CARGO_TARGET_DIR"; sha256sum Cargo.lock; rustc -Vv; nvcc --version; cargo nextest --version; printf '%s\n' "${CUDA_COMPUTE_CAP-}" "${RUSTFLAGS-}" "${CUDA_HOME-}" "${CUDA_PATH-}" "${CUDA_VISIBLE_DEVICES-}" "${CARGO_ENCODED_RUSTFLAGS-}" "${CC-}" "${CXX-}" "${NVCC-}" "${CARGO_BUILD_TARGET-}"; } | sha256sum | cut -d' ' -f1 )"
ARCHIVE="$RUN/tests.tar.zst"
FILTER='test(dkr_) or test(prevalidated_) or test(pads_decode_input_by_aliasing_row_zero_values)'
if [[ ! -s "$ARCHIVE" || ! -f "$RUN/archive.key" || "$(cat "$RUN/archive.key")" != "$KEY" ]]; then
    cargo nextest archive --locked -p mistralrs-kernel-runtime -p mistralrs-quant -p mistralrs-paged-attn -p mistralrs-core \
      --lib --features 'mistralrs-core/dkr-sm61-experimental,mistralrs-core/dkr-fp4-dequant' \
      --archive-file "$RUN/tests-building.tar.zst" 2>&1 | tee "$RUN/archive-build.log"
    mv "$RUN/tests-building.tar.zst" "$ARCHIVE"
    printf '%s\n' "$KEY" > "$RUN/archive.key"
else
    echo "DKR_S6_ARCHIVE=REUSED"
fi
# Same exact binaries, legacy Candle vs external Candle in different processes.
rm -f "$RUN/native-legacy.json" "$RUN/native-external.json" "$RUN/qualification.json"
env -u CANDLE_DKR_S31_DIR DKR_S6_NATIVE_OUTPUT="$RUN/native-legacy.json" \
    cargo nextest run --archive-file "$ARCHIVE" --run-ignored=all --no-capture --test-threads=1 \
    -E 'test(dkr_s6_native_ab_gpu)' 2>&1 | tee "$RUN/legacy.log"
CANDLE_DKR_S31_DIR="$NATIVE" DKR_S6_NATIVE_OUTPUT="$RUN/native-external.json" \
    cargo nextest run --archive-file "$ARCHIVE" --run-ignored=all --no-fail-fast --no-capture --test-threads=1 \
    -E "$FILTER" 2>&1 | tee "$RUN/external.log"
python3 tools/dkr_r122_s6_evidence.py --report-dir "$RUN"
python3 tools/dkr_r122_s67_final.py --report-dir "$RUN" --native-root "$NATIVE"
echo "DKR_S67_FINAL_SMOKE=PASS report=$RUN/qualification-s67.json"
