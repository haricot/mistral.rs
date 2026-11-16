#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "DKR_S684B_ERROR line=$LINENO exit=$rc command=$BASH_COMMAND" >&2; exit "$rc"' ERR
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set original checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
[[ "$ROOT" != "$ORIG" ]]
git diff --quiet HEAD -- .
RMS_SOURCE="kernel-packages/cuda/rms-variants/rms_variants.cu"
REPLAY_EXCLUDES=(
    ':!tools/**'
    ":!$RMS_SOURCE"
    ':!kernel-packages/cuda/paged-decode/**'
    ':!kernel-packages/cuda/paged-v2/**'
    ':!kernel-packages/cuda/paged_vllm_compat.cuh'
    ':!mistralrs-paged-attn/src/cuda/pagedattention.cuh'
    ':!mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs'
)
git diff --quiet d392cb70c155ffa1a6aa9312cc482f3be5b9f19a HEAD -- . \
    ':!mistralrs/Cargo.toml' "${REPLAY_EXCLUDES[@]}"
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
TYPED="$ORIG/kernel-packages/build/cuda/sm61"
BASE="$ORIG/target/dkr-r122-s684/d392cb70c155"
if [[ -n "${DKR_S684B_PRIOR:-}" ]]; then
    PRIOR="$(realpath "$DKR_S684B_PRIOR")"
else
    mapfile -t hits < <(find "$BASE" -name qualification-s684-model.json -type f)
    [[ "${#hits[@]}" == 1 ]] || { echo "Expected one prior report under $BASE; set DKR_S684B_PRIOR" >&2; exit 3; }
    PRIOR="$(dirname "${hits[0]}")"
fi
python3 tools/dkr_r122_s684b_gate.py --self-test
python3 tools/dkr_r122_s684_model_audit.py --verify-prior "$ORIG/target/dkr-r122-s683/d3dff22d83ab"
python3 tools/dkr_r122_s683_audit.py --verify-seal "$ORIG/target/dkr-s68/promotion-gate.json" --prior "$ORIG/target/dkr-r122-s6/9ce68d5cc4e1" --native-root "$NATIVE"
python3 tools/dkr_r122_s684b_gate.py --prior "$PRIOR" --native-root "$NATIVE" --typed-root "$TYPED"
sha256sum -c "$PRIOR/model.sha256"
export DKR_S684_GGUF="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$PRIOR/qualification-s684-model.json")"
[[ -f "$DKR_S684_GGUF" && "$DKR_S684_GGUF" == *.gguf ]]
if [[ -s "$PRIOR/template.sha256" ]]; then
    sha256sum -c "$PRIOR/template.sha256"
    export DKR_S684_CHAT_TEMPLATE="$(cut -c67- "$PRIOR/template.sha256")"
else unset DKR_S684_CHAT_TEMPLATE; fi
export CARGO_TARGET_DIR="$ORIG/target/dkr-r122-s6-cargo"
[[ -z "${CUDA_COMPUTE_CAP:-}" || "$CUDA_COMPUTE_CAP" == 61 ]]
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')" == 6.1 ]]
command -v nvcc >/dev/null

file_sha() {
    sha256sum -- "$1" | cut -d' ' -f1
}

ARCHIVE=""
REUSE_REFERENCE=0
if [[ -n "${DKR_S684B_REPLAY_FROM:-}" ]]; then
    ARCHIVE="$(realpath "$DKR_S684B_REPLAY_FROM")"
    archived_revision="$(cat "$ARCHIVE/revision.txt")"
    git diff --quiet "$archived_revision" HEAD -- . "${REPLAY_EXCLUDES[@]}"
    static_build_revision="$archived_revision"
    if [[ -s "$ARCHIVE/static-build-revision.txt" ]]; then
        static_build_revision="$(cat "$ARCHIVE/static-build-revision.txt")"
    fi
    archived_static_sha="$(awk '$2 ~ /\/static-bin$/ { print $1 }' "$ARCHIVE/binaries.sha256")"
    [[ -x "$ARCHIVE/static-bin" && "$(file_sha "$ARCHIVE/static-bin")" == "$archived_static_sha" ]]
    python3 tools/dkr_r122_s684b_gate.py --features "$ARCHIVE/static-features.txt"
    EXT="$ARCHIVE/external-bin"
    if [[ -f "$ARCHIVE/qualification-s684b.json" && -d "$ARCHIVE/typed/sm61/rms-variants" ]] &&
       git diff --quiet "$archived_revision" HEAD -- "$RMS_SOURCE"; then
        python3 tools/dkr_r122_s684b_gate.py --verify-reference "$ARCHIVE"
        cmp --silent "$ARCHIVE/model.sha256" "$PRIOR/model.sha256"
        TYPED="$ARCHIVE/typed/sm61"
        REUSE_REFERENCE=1
    fi
else
    EXT="${DKR_S684B_EXTERNAL_BIN:-$CARGO_TARGET_DIR/debug/examples/dkr_s684_model}"
fi
expected="$(cut -d' ' -f1 "$PRIOR/binary.sha256")"
[[ -x "$EXT" && "$(file_sha "$EXT")" == "$expected" ]] || {
    echo "DKR_S684B_ERROR prior external binary unavailable or hash mismatch; do not substitute an unqualified binary" >&2; exit 3; }
echo "DKR_S684B_PREFLIGHT=PASS prior=$PRIOR"
if [[ "${1:-}" == --preflight ]]; then exit 0; fi
[[ "${DKR_APPROVE_S684B:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S684B=1"; exit 2; }
exec 9>"$ORIG/target/dkr-r122-s684/.lock"
flock -n 9
RUN="$ORIG/target/dkr-r122-s684b/$(git rev-parse --short=12 HEAD)/$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
echo "DKR_S684B_RUN=$RUN"
cp --reflink=auto "$EXT" "$RUN/external-bin"
[[ "$(file_sha "$RUN/external-bin")" == "$expected" ]]
cp "$PRIOR/model.sha256" "$RUN/model.sha256"
printf '%s\n' "$(git rev-parse HEAD)" > "$RUN/revision.txt"
export HF_HUB_OFFLINE=1 DKR_TELEMETRY=1
unset DKR_KERNEL_PACKAGES CANDLE_DKR_S31_DIR CANDLE_DKR_TRACE MISTRALRS_DKR_ROUTED_QUANT CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
if [[ -n "$ARCHIVE" ]]; then
    cp --reflink=auto "$ARCHIVE/static-bin" "$RUN/static-bin"
    [[ "$(file_sha "$RUN/static-bin")" == "$archived_static_sha" ]]
    cp "$ARCHIVE/static-features.txt" "$RUN/static-features.txt"
    printf '%s\n' "$static_build_revision" > "$RUN/static-build-revision.txt"
    echo "DKR_S684B_BINARIES=REUSED"
else
    shared_cache="$ORIG/target/dkr-cuda-cache/sm61"
    export MISTRALRS_CUDA_BUILD_ROOT="$ORIG/target/dkr-cuda-cache/sm61-legacy"
    if [[ ! -d "$MISTRALRS_CUDA_BUILD_ROOT" && -d "$shared_cache" ]]; then
        cp -a --reflink=auto "$shared_cache" "$MISTRALRS_CUDA_BUILD_ROOT"
    fi
    cargo tree --locked -p mistralrs --no-default-features --features cuda-legacy \
        -e normal,build,dev --prefix none --format '{p}|{f}' > "$RUN/static-features.txt"
    python3 tools/dkr_r122_s684b_gate.py --features "$RUN/static-features.txt"
    cargo build --locked -p mistralrs --example dkr_s684_model --no-default-features --features cuda-legacy 2>&1 | tee "$RUN/static-build.log"
    cp --reflink=auto "$CARGO_TARGET_DIR/debug/examples/dkr_s684_model" "$RUN/static-bin"
    cp "$RUN/revision.txt" "$RUN/static-build-revision.txt"
fi
sha256sum "$RUN/static-bin" "$RUN/external-bin" > "$RUN/binaries.sha256"
STAGED_TYPED="$RUN/typed/sm61"
mkdir -p "$STAGED_TYPED"
for family in "$TYPED"/*; do
    case "$(basename "$family")" in
        paged-decode|paged-v2|_cudaforge_cache) continue ;;
        rms-variants) [[ "$REUSE_REFERENCE" == 1 ]] || continue ;;
    esac
    cp -a --reflink=auto "$family" "$STAGED_TYPED/"
done
if [[ "$REUSE_REFERENCE" == 0 ]]; then
    python3 kernel-packages/cuda/rms-variants/build.py --cc 61 --output "$RUN/typed"
else
    echo "DKR_RMS_VARIANTS=REUSED"
fi
for family in paged-decode paged-v2; do
    python3 "kernel-packages/cuda/$family/build.py" --cc 61 --output "$RUN/typed"
done
python3 tools/dkr_s9_ptx_manifest_audit.py --root "$STAGED_TYPED" --cc 61

# The DKR paged kernels use VLLM pages. Both paged legs must use that layout.
export MISTRALRS_FLASHINFER_DECODE=0
for leg in static-contiguous static-paged external-contiguous external-paged; do
    if [[ "$REUSE_REFERENCE" == 1 && "$leg" != external-paged ]]; then
        cp --reflink=auto "$ARCHIVE/$leg."{exit,json,log} "$RUN/"
        echo "DKR_S684B_$leg=REUSED"
        continue
    fi
    export DKR_S684_OUTPUT="$RUN/$leg.json"
    export DKR_S684_PAGED=0
    [[ "$leg" != *-paged ]] || export DKR_S684_PAGED=1
    if [[ "$leg" == external-* ]]; then
        export DKR_S684_MODE=external CANDLE_DKR_S31_DIR="$NATIVE" DKR_KERNEL_PACKAGES="$STAGED_TYPED" CANDLE_DKR_TRACE=1 MISTRALRS_DKR_ROUTED_QUANT=1
        export MISTRALRS_DKR_KV_RESHAPE=1 MISTRALRS_DKR_PAGED_DECODE=1 MISTRALRS_DKR_PAGED_V2=1
        BIN="$RUN/external-bin"
    else
        unset CANDLE_DKR_S31_DIR DKR_KERNEL_PACKAGES CANDLE_DKR_TRACE MISTRALRS_DKR_ROUTED_QUANT
        export DKR_S684_MODE=legacy
        BIN="$RUN/static-bin"
    fi
    if "$BIN" 2>&1 | tee "$RUN/$leg.log" | awk '!/DKR_RUNTIME_EVIDENCE / && !/Discovered and using GGUF chat template/ {print; fflush()}'; then
        codes=("${PIPESTATUS[@]}")
    else
        codes=("${PIPESTATUS[@]}")
    fi
    rc="${codes[0]}"; [[ "${codes[1]}" == 0 && "${codes[2]}" == 0 ]] || rc=3
    printf '%s\n' "$rc" > "$RUN/$leg.exit"
    echo "DKR_S684B_$leg=$rc"
done
sha256sum -c "$RUN/binaries.sha256"
sha256sum -c "$RUN/model.sha256"
if [[ -s "$PRIOR/template.sha256" ]]; then sha256sum -c "$PRIOR/template.sha256"; fi
python3 tools/dkr_r122_s684b_gate.py --prior "$PRIOR" --native-root "$NATIVE" --typed-root "$STAGED_TYPED" --run "$RUN"
