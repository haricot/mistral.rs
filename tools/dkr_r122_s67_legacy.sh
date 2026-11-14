#!/usr/bin/env bash
# Diagnose the original S6 legacy crash using the archived executable.
# No build, no changes to CUDA flags, no mutation of the original evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?set original qualified checkout}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
RUN="$ORIG/target/dkr-r122-s6/8dba9606087b"
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
ARCHIVE="$RUN/tests.tar.zst"
OUT="$RUN/s67"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9
[[ -s "$ARCHIVE" && -s "$RUN/archive.key" && -s "$RUN/qualification-external-recovery.json" ]]
[[ "$(jq -er .candidate "$RUN/closure.json")" == b231d3c4e07b04c0df788ee252979b1890430dd6 ]]
git diff --quiet 8dba9606087b HEAD -- . ':!tools/**' || {
    echo "Source revision differs from S6 archive; stop"; exit 3; }
python3 tools/dkr_r122_s6_external_recovery.py --report-dir "$RUN" --native-root "$NATIVE" > "$OUT/recovery-recheck.log"
command -v gdb >/dev/null || { echo "Install gdb before diagnosis (Arch: sudo pacman -S gdb)"; exit 3; }
command -v cargo >/dev/null
df -h /tmp "$OUT"
# nextest handles archive remapping and extraction. List only, no compilation.
EXTRACT="$(mktemp -d /tmp/dkr-s67-archive.XXXXXX)"
trap 'rm -rf -- "$EXTRACT"' EXIT
cargo nextest list --archive-file "$ARCHIVE" --extract-to "$EXTRACT" \
    --list-type binaries-only --message-format json > "$OUT/binaries.json"
mapfile -d '' BINS < <(find "$EXTRACT/target" -type f -name 'mistralrs_core-*' -executable -print0)
[[ "${#BINS[@]}" == 1 ]] || { echo "Expected exactly one archived core test binary, got ${#BINS[@]}"; exit 3; }
BIN="${BINS[0]}"
sha256sum "$BIN" > "$OUT/core-binary.sha256"
# Include archived shared libraries; preserve the inherited system CUDA path.
LIBDIRS="$(find "$EXTRACT/target" -type f \( -name '*.so' -o -name '*.so.*' \) -printf '%h\n' | sort -u | paste -sd: -)"
export LD_LIBRARY_PATH="${LIBDIRS:+$LIBDIRS:}${LD_LIBRARY_PATH:-}"
export DKR_KERNEL_PACKAGES="$ORIG/kernel-packages/build/cuda/sm61"
export DKR_S6_NATIVE_OUTPUT="$OUT/native-legacy-debug.json"
export DKR_TELEMETRY=1 MISTRALRS_DKR_ROUTED_QUANT=1
unset CANDLE_DKR_S31_DIR CANDLE_DKR_TRACE CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
# Preserve original asynchronous behavior on first diagnostic.
if [[ "${DKR_S67_SYNC_DIAGNOSTIC:-0}" == 1 ]]; then
    export CUDA_LAUNCH_BLOCKING=1
    LOG="$OUT/legacy-gdb-sync.log"
else
    LOG="$OUT/legacy-gdb.log"
fi
rm -f "$DKR_S6_NATIVE_OUTPUT"
cat > "$OUT/legacy.gdb" <<'GDB'
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set breakpoint pending on
catch throw
commands
silent
printf "\nDKR_S67_CPP_THROW\n"
bt 40
continue
end
handle SIGABRT stop print pass
run
bt 40
GDB
cd "$ROOT/mistralrs-core"
set +e
gdb -nx -batch -x "$OUT/legacy.gdb" --args "$BIN" \
    --exact dkr_s6_native::dkr_s6_native_ab_gpu --ignored --nocapture \
    2>&1 | tee "$LOG"
GDB_RC="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$GDB_RC" > "$LOG.exit"
cd "$ROOT"
python3 tools/dkr_r122_s67_promotion.py --report-dir "$RUN" --gdb-log "$LOG"
