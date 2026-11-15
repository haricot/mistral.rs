#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "DKR_S684B_ERROR line=$LINENO exit=$rc command=$BASH_COMMAND" >&2; exit "$rc"' ERR
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set original checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
[[ "$ROOT" != "$ORIG" ]]
git diff --quiet HEAD -- .
git diff --quiet d392cb70c155ffa1a6aa9312cc482f3be5b9f19a HEAD -- . ':!tools/**' ':!mistralrs/Cargo.toml'
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
TYPED="$ORIG/kernel-packages/build/cuda/sm61"
BASE="$ORIG/target/dkr-r122-s684/d392cb70c155"
mapfile -t hits < <(find "$BASE" -name qualification-s684-model.json -type f)
[[ "${#hits[@]}" == 1 ]] || { echo "Expected one prior report under $BASE; use DKR_S684B_PRIOR to select it" >&2; }
if [[ -n "${DKR_S684B_PRIOR:-}" ]]; then PRIOR="$(realpath "$DKR_S684B_PRIOR")"; else [[ "${#hits[@]}" == 1 ]]; PRIOR="$(dirname "${hits[0]}")"; fi
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
export MISTRALRS_CUDA_BUILD_ROOT="$ORIG/target/dkr-cuda-cache/sm61"
[[ -z "${CUDA_COMPUTE_CAP:-}" || "$CUDA_COMPUTE_CAP" == 61 ]]
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')" == 6.1 ]]
# Copy exact prior external binary before static link can overwrite it.
EXT="${DKR_S684B_EXTERNAL_BIN:-$CARGO_TARGET_DIR/debug/examples/dkr_s684_model}"
expected="$(cut -d' ' -f1 "$PRIOR/binary.sha256")"
[[ -x "$EXT" && "$(sha256sum "$EXT" | cut -d' ' -f1)" == "$expected" ]] || {
    echo "DKR_S684B_ERROR prior external binary unavailable or hash mismatch; do not substitute an unqualified binary" >&2; exit 3; }
echo "DKR_S684B_PREFLIGHT=PASS prior=$PRIOR"
if [[ "${1:-}" == --preflight ]]; then exit 0; fi
[[ "${DKR_APPROVE_S684B:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S684B=1"; exit 2; }
RUN="$ORIG/target/dkr-r122-s684b/$(git rev-parse --short=12 HEAD)/$(date -u +%Y%m%dT%H%M%S)"
mkdir -p "$RUN"
exec 9>"$ORIG/target/dkr-r122-s684/.lock"
flock -n 9
cp --reflink=auto "$EXT" "$RUN/external-bin"
[[ "$(sha256sum "$RUN/external-bin" | cut -d' ' -f1)" == "$expected" ]]
cp "$PRIOR/model.sha256" "$RUN/model.sha256"
printf '%s\n' "$(git rev-parse HEAD)" > "$RUN/revision.txt"
# A separate cargo invocation is essential: unsetting an environment
# variable cannot prove compile-time DKR features are inactive.
cargo tree --locked -p mistralrs --no-default-features --features cuda-legacy \
    -e normal,build,dev --prefix none --format '{p}|{f}' > "$RUN/static-features.txt"
python3 -c 'import sys;sys.path.insert(0,"tools");from dkr_r122_s684b_gate import features;from pathlib import Path; print("DKR_S684B_STATIC_FEATURES=PASS",features(Path(sys.argv[1])))' "$RUN/static-features.txt"
export HF_HUB_OFFLINE=1 DKR_TELEMETRY=1
unset DKR_KERNEL_PACKAGES CANDLE_DKR_S31_DIR CANDLE_DKR_TRACE MISTRALRS_DKR_ROUTED_QUANT CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
cargo build --locked -p mistralrs --example dkr_s684_model --no-default-features --features cuda-legacy 2>&1 | tee "$RUN/static-build.log"
cp --reflink=auto "$CARGO_TARGET_DIR/debug/examples/dkr_s684_model" "$RUN/static-bin"
sha256sum "$RUN/static-bin" "$RUN/external-bin" > "$RUN/binaries.sha256"
for leg in static-contiguous static-paged external-paged; do
    export DKR_S684_OUTPUT="$RUN/$leg.json"
    if [[ "$leg" == external-paged ]]; then
        export DKR_S684_MODE=external DKR_S684_PAGED=1 CANDLE_DKR_S31_DIR="$NATIVE" DKR_KERNEL_PACKAGES="$TYPED" CANDLE_DKR_TRACE=1 MISTRALRS_DKR_ROUTED_QUANT=1
        BIN="$RUN/external-bin"
    else
        unset CANDLE_DKR_S31_DIR DKR_KERNEL_PACKAGES CANDLE_DKR_TRACE MISTRALRS_DKR_ROUTED_QUANT
        export DKR_S684_MODE=legacy DKR_S684_PAGED=0
        [[ "$leg" != static-paged ]] || export DKR_S684_PAGED=1
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
echo "DKR_S684B_RUN=$RUN"
sha256sum -c "$RUN/binaries.sha256"
sha256sum -c "$RUN/model.sha256"
for leg in static-contiguous static-paged external-paged; do [[ "$(cat "$RUN/$leg.exit")" == 0 ]] || exit 4; done
python3 tools/dkr_r122_s684b_gate.py --prior "$PRIOR" --native-root "$NATIVE" --typed-root "$TYPED" --run "$RUN"
