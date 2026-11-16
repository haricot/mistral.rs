#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "DKR_S684C_ERROR line=$LINENO command=$BASH_COMMAND" >&2' ERR
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set original checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
[[ "$ORIG" != "$ROOT" ]]
git diff --quiet HEAD -- .
git diff --quiet 556dca9fa4e31ec80b77c642ded8c03a76d690e1 HEAD -- . ':!tools/**'
MODEL="$ORIG/target/dkr-r122-s684b/556dca9fa4e3/20261116T162401"
PRIOR="$ORIG/target/dkr-r122-s684/d392cb70c155/20261115T224855-paged0"
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
BASE="$ORIG/target/dkr-r122-s684c/$(git rev-parse --short=12 HEAD)"
mkdir -p "$BASE"
RUN="$(mktemp -d "$BASE/run-XXXXXXXX")"
python3 tools/dkr_r122_s684c_gate.py --model-run "$MODEL" --prior "$PRIOR" --native-root "$NATIVE" --output "$RUN/model-seal.json"
echo "DKR_S684C_RUN=$RUN"
case "${1:---preflight}" in
    --preflight) exit 0;;
    --driver)
        python3 tools/dkr_r122_s684c_driver.py --native-root "$NATIVE" --output "$RUN/driver.json"
        ;;
    --cpu)
        [[ "${DKR_APPROVE_S684C_CPU:-0}" == 1 ]] || { echo "Set DKR_APPROVE_S684C_CPU=1"; exit 2; }
        # Separate target: never overwrite either archived GPU executable.
        export CARGO_TARGET_DIR="$ORIG/target/dkr-r122-cpu-cargo"
        export HF_HUB_OFFLINE=1
        unset CANDLE_DKR_S31_DIR DKR_KERNEL_PACKAGES DKR_TELEMETRY CANDLE_DKR_TRACE
        pkgs=(-p mistralrs -p mistralrs-core -p mistralrs-quant -p mistralrs-kernel-runtime)
        cargo tree --locked "${pkgs[@]}" -e normal,build,dev --prefix none --format '{p}|{f}' > "$RUN/cpu-features.txt"
        python3 tools/dkr_r122_s684c_gate.py --cpu-features "$RUN/cpu-features.txt"
        FILTER="$(python3 tools/dkr_r122_s684c_gate.py --filter)"
        cargo nextest run --locked "${pkgs[@]}" --lib -E "$FILTER" --status-level pass --final-status-level none --color never 2>&1 | tee "$RUN/cpu.log"
        python3 tools/dkr_r122_s684c_gate.py --cpu-features "$RUN/cpu-features.txt" --cpu-log "$RUN/cpu.log" --output "$RUN/cpu.json"
        ;;
    *) echo "usage: $0 [--preflight|--driver|--cpu]" >&2; exit 2;;
esac
