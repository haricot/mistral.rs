#!/usr/bin/env bash
# S6.8.4-A: model parity on the same executable; no production promotion.
set -Eeuo pipefail
trap 'rc=$?; printf "DKR_S684_ERROR line=%s exit=%s command=%s\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2; exit "$rc"' ERR
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set original qualified checkout}"
: "${DKR_S684_GGUF:?set absolute local single-file GGUF path}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
# Keep the snapshot filename and directory; HF cache files are symlinks
# to extensionless blobs. Only use the resolved path for diagnostics.
export DKR_S684_GGUF="$(realpath -s -- "$DKR_S684_GGUF")"
[[ -f "$DKR_S684_GGUF" ]] || { echo "DKR_S684_ERROR missing model: $DKR_S684_GGUF" >&2; exit 3; }
[[ "$ORIG" != "$ROOT" ]] || { echo "DKR_S684_ERROR use separate worktree" >&2; exit 3; }
[[ "$DKR_S684_GGUF" == *.gguf ]] || { echo "DKR_S684_ERROR choose the named GGUF snapshot, not its blob" >&2; exit 3; }
printf 'DKR_S684_MODEL_PATH=%s\nDKR_S684_MODEL_TARGET=%s\n' "$DKR_S684_GGUF" "$(realpath -- "$DKR_S684_GGUF")"
git diff --quiet HEAD -- . || { echo "dirty tracked tree"; exit 3; }
git merge-base --is-ancestor d3dff22d83ab48c046539818c0a1f1fd597240f2 HEAD
git diff --quiet d3dff22d83ab48c046539818c0a1f1fd597240f2 HEAD -- . ':!tools/**' ':!mistralrs/examples/advanced/dkr_s684_model/**' ':!mistralrs/Cargo.toml'
CACHE="$ORIG/target/dkr-cuda-cache/sm61"
export MISTRALRS_CUDA_BUILD_ROOT="$CACHE"
for component in quant core paged-attn; do
    [[ -s "$CACHE/$component/kernels/.cudaforge_cache.json" ]]
done
[[ "$(find "$CACHE" -name '*.o' -type f | wc -l)" -ge 82 ]]
[[ -z "${CUDA_COMPUTE_CAP:-}" || "$CUDA_COMPUTE_CAP" == 61 ]]
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')" == 6.1 ]]
export DKR_S684_PAGED="${DKR_S684_PAGED:-0}"
[[ "$DKR_S684_PAGED" == 0 || "$DKR_S684_PAGED" == 1 ]]
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
TYPED="$ORIG/kernel-packages/build/cuda/sm61"
PRIOR="$ORIG/target/dkr-r122-s683/d3dff22d83ab"
python3 tools/dkr_r122_s684_model_audit.py --self-test
python3 tools/dkr_r122_s684_model_audit.py --verify-prior "$PRIOR"
python3 tools/dkr_r122_s683_audit.py --verify-seal "$ORIG/target/dkr-s68/promotion-gate.json" \
    --prior "$ORIG/target/dkr-r122-s6/9ce68d5cc4e1" --native-root "$NATIVE"
python3 tools/dkr_s9_ptx_manifest_audit.py --root "$TYPED" --cc 61
HEAD="$(git rev-parse HEAD)"
RUN="$ORIG/target/dkr-r122-s684/${HEAD:0:12}/$(date -u +%Y%m%dT%H%M%S)-paged$DKR_S684_PAGED"
mkdir -p "$RUN"
exec 9>"$ORIG/target/dkr-r122-s684/.lock"
flock -n 9
python3 tools/dkr_r122_s4_dependency_closure.py --locked --output "$RUN/closure.json"
sha256sum "$DKR_S684_GGUF" > "$RUN/model.sha256"
if [[ -n "${DKR_S684_CHAT_TEMPLATE:-}" ]]; then
    export DKR_S684_CHAT_TEMPLATE="$(realpath -s -- "$DKR_S684_CHAT_TEMPLATE")"
    sha256sum "$DKR_S684_CHAT_TEMPLATE" > "$RUN/template.sha256"
fi
echo "DKR_S684_PREFLIGHT=PASS model=$DKR_S684_GGUF"
if [[ "${1:-}" == --preflight ]]; then exit 0; fi
[[ "${DKR_APPROVE_S684:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S684=1"; exit 2; }
export HF_HUB_OFFLINE=1
export DKR_KERNEL_PACKAGES="$TYPED"
export DKR_TELEMETRY=1 CANDLE_DKR_TRACE=1 MISTRALRS_DKR_ROUTED_QUANT=1
unset CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
export CARGO_TARGET_DIR="$ORIG/target/dkr-r122-s6-cargo"
# Same prior test-profile dependency features; only this executable is new.
cargo build --locked -p mistralrs --example dkr_s684_model \
    --features 'dkr-sm61-experimental,mistralrs-core/dkr-fp4-dequant' 2>&1 | tee "$RUN/build.log"
BIN="$CARGO_TARGET_DIR/debug/examples/dkr_s684_model"
[[ -x "$BIN" ]]
sha256sum "$BIN" > "$RUN/binary.sha256"
printf '%s\n' "$HEAD" > "$RUN/revision"
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,memory.used --format=csv > "$RUN/gpu-before.csv"
set +e
for mode in legacy external; do
    export DKR_S684_MODE="$mode" DKR_S684_OUTPUT="$RUN/$mode.json"
    if [[ "$mode" == external ]]; then export CANDLE_DKR_S31_DIR="$NATIVE"; else unset CANDLE_DKR_S31_DIR; fi
    "$BIN" 2>&1 | tee "$RUN/$mode.log"
    codes=("${PIPESTATUS[@]}")
    rc="${codes[0]}"
    [[ "${codes[1]}" == 0 ]] || rc=3
    printf '%s\n' "$rc" > "$RUN/$mode.exit"
    echo "DKR_S684_${mode}_EXIT=$rc"
done
set -e
sha256sum -c "$RUN/binary.sha256"
sha256sum -c "$RUN/model.sha256"
if [[ -s "$RUN/template.sha256" ]]; then sha256sum -c "$RUN/template.sha256"; fi
python3 tools/dkr_r122_s684_model_audit.py --run "$RUN" --native-root "$NATIVE"
echo "DKR_S684_REPORT=$RUN/qualification-s684-model.json"
