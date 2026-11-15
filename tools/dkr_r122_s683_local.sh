#!/usr/bin/env bash
# S6.8.3: extended numerical matrix; same archive for A and B.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set to original qualified checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
[[ "$ORIG" != "$ROOT" ]]
git diff --quiet HEAD -- . || { echo "Tracked source differs from HEAD"; exit 3; }
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
RUN="$ORIG/target/dkr-r122-s683/$(git rev-parse --short=12 HEAD)"
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
# Reuse only the exact S6.8 sealed native and prior qualification artifacts.
SEALED="$ORIG/target/dkr-s68/promotion-gate.json"
QUALIFIED="$ORIG/target/dkr-r122-s6/9ce68d5cc4e1"
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
python3 tools/dkr_r122_s683_audit.py --self-test
python3 tools/dkr_r122_s683_audit.py --verify-seal "$SEALED" --prior "$QUALIFIED" --native-root "$NATIVE"
echo "DKR_S683_NATIVE_PTX=REUSED root=$NATIVE"
echo "DKR_S683_PREFLIGHT=PASS run=$RUN"
if [[ "${1:-}" == "--preflight" ]]; then exit 0; fi
[[ "${DKR_APPROVE_S683:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S683=1 for grouped build"; exit 2; }
export DKR_KERNEL_PACKAGES="$TYPED"
export DKR_TELEMETRY=1 CANDLE_DKR_TRACE=1 MISTRALRS_DKR_ROUTED_QUANT=1
# Test the actual GGUF paths rather than an inherited dequantize-all override.
unset CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
printf '%s\n' "$NATIVE" > "$RUN/native-root.txt"
# Fingerprint archive inputs; failed archive is never used.
export CARGO_TARGET_DIR="$ORIG/target/dkr-r122-s6-cargo"
KEY="$( { git rev-parse HEAD; printf '%s\n' "$ROOT" "$CACHE" "$CARGO_TARGET_DIR"; sha256sum Cargo.lock; rustc -Vv; nvcc --version; cargo nextest --version; printf '%s\n' "${CUDA_COMPUTE_CAP-}" "${RUSTFLAGS-}" "${CUDA_HOME-}" "${CUDA_PATH-}" "${CUDA_VISIBLE_DEVICES-}" "${CARGO_ENCODED_RUSTFLAGS-}" "${CC-}" "${CXX-}" "${NVCC-}" "${CARGO_BUILD_TARGET-}"; } | sha256sum | cut -d' ' -f1 )"
ARCHIVE="$RUN/tests.tar.zst"
FILTER='test(=dkr_s683::dkr_s683_extended_gpu)'
if [[ ! -s "$ARCHIVE" || ! -f "$RUN/archive.key" || "$(cat "$RUN/archive.key")" != "$KEY" ]]; then
    cargo nextest archive --locked -p mistralrs-core \
      --lib --features 'mistralrs-core/dkr-sm61-experimental,mistralrs-core/dkr-fp4-dequant' \
      --archive-file "$RUN/tests-building.tar.zst" 2>&1 | tee "$RUN/archive-build.log"
    mv "$RUN/tests-building.tar.zst" "$ARCHIVE"
    printf '%s\n' "$KEY" > "$RUN/archive.key"
else
    echo "DKR_S6_ARCHIVE=REUSED"
fi
# Archive is immutable across both legs. Fresh output per execution.
sha256sum "$ARCHIVE" > "$RUN/archive.sha256"
# Shared target and cache are not cleaned. No change to CUDA_COMPUTE_CAP.
set +e
for mode in legacy external; do
    rm -f "$RUN/$mode.json"
    export DKR_S683_MODE="$mode" DKR_S683_OUTPUT="$RUN/$mode.json"
    if [[ "$mode" == external ]]; then
        export CANDLE_DKR_S31_DIR="$NATIVE"
    else
        unset CANDLE_DKR_S31_DIR
    fi
    cargo nextest run --archive-file "$ARCHIVE" --workspace-remap "$ROOT" \
        --run-ignored=all --no-capture -E "$FILTER" 2>&1 | tee "$RUN/$mode.log"
    codes=("${PIPESTATUS[@]}")
    rc="${codes[0]}"
    if [[ "${codes[1]}" != 0 ]]; then rc=3; fi
    printf '%s\n' "$rc" > "$RUN/$mode.exit"
    echo "DKR_S683_${mode}_EXIT=$rc"
done
set -e
sha256sum -c "$RUN/archive.sha256"
python3 tools/dkr_r122_s683_audit.py --run "$RUN" --native-root "$NATIVE"
echo "DKR_S683_REPORT=$RUN/qualification-s683.json"
