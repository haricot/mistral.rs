#!/usr/bin/env bash
# S6.7.5: replay the already built candidate; keep original failed A evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd -P)"
: "${DKR_S6_ORIGINAL_REPO:?original qualified checkout required}"
ORIG="$(realpath "$DKR_S6_ORIGINAL_REPO")"
RUN="$ORIG/target/dkr-r122-s6/2a1b85394468"
NATIVE="$ORIG/target/dkr-r122-s6/7b3b19cf8b62/native-sm61"
OUT="$RUN/s675"
mkdir -p "$OUT"
exec 9>"$OUT/.lock"
flock -n 9
[[ -s "$RUN/tests.tar.zst" && -s "$RUN/archive.key" ]]
[[ "$(jq -er .candidate "$RUN/closure.json")" == 63217e3a0ea8a91b0630acbcbc047c471f43761a ]]
git diff --quiet 2a1b85394468 HEAD -- . ':!tools/**'
python3 tools/dkr_r122_s6_evidence.py --native-root "$NATIVE"
export DKR_KERNEL_PACKAGES="$ORIG/kernel-packages/build/cuda/sm61"
export DKR_TELEMETRY=1 MISTRALRS_DKR_ROUTED_QUANT=1
unset CANDLE_DEQUANTIZE_ALL CANDLE_DEQUANTIZE_ALL_F16
# Independent B even when A fails; no Cargo or NVCC build.
if [[ "${1:-}" == "--trace-only" ]]; then
    [[ "$(cat "$OUT/external.exit")" == 0 && -s "$OUT/native-external.json" && -s "$OUT/external.log" ]]
    echo "DKR_S675_EXTERNAL_EXIT=0 (preserved)"
else
set +e
CANDLE_DKR_S31_DIR="$NATIVE" CANDLE_DKR_TRACE=1 DKR_S6_NATIVE_OUTPUT="$OUT/native-external.json" \
    cargo nextest run --archive-file "$RUN/tests.tar.zst" --workspace-remap "$ROOT" \
    --run-ignored=all --no-fail-fast --no-capture \
    -E 'test(dkr_) or test(prevalidated_) or test(pads_decode_input_by_aliasing_row_zero_values)' \
    2>&1 | tee "$OUT/external.log"
B_RC="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$B_RC" > "$OUT/external.exit"
echo "DKR_S675_EXTERNAL_EXIT=$B_RC"
fi
# GDB trace: stop on the first legacy SIMT invocation then instrument only
# Runtime API calls in that wrapper; don't call CUDA from the debugger.
command -v gdb
EXTRACT="$(mktemp -d /tmp/dkr-s675-archive.XXXXXX)"
trap 'rm -rf -- "$EXTRACT"' EXIT
cargo nextest list --archive-file "$RUN/tests.tar.zst" --workspace-remap "$ROOT" \
    --extract-to "$EXTRACT" --list-type binaries-only --message-format json > "$OUT/binaries.json"
mapfile -d '' BINS < <(find "$EXTRACT/target" -type f -name 'mistralrs_core-*' -executable -print0)
[[ "${#BINS[@]}" == 1 ]]
LIBDIRS="$(find "$EXTRACT/target" -type f \( -name '*.so' -o -name '*.so.*' \) -printf '%h\n' | sort -u | paste -sd: -)"
export LD_LIBRARY_PATH="${LIBDIRS:+$LIBDIRS:}${LD_LIBRARY_PATH:-}"
unset CANDLE_DKR_S31_DIR CANDLE_DKR_TRACE
export DKR_S6_NATIVE_OUTPUT="$OUT/native-legacy-debug.json"
cat > "$OUT/trace.gdb" <<'GDB'
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set breakpoint pending on
break moe_gemm_simt_f16
run
python
class ReturnTrace(gdb.FinishBreakpoint):
    def __init__(self, name):
        super().__init__(gdb.newest_frame(), internal=True)
        self.name = name
    def stop(self):
        # x86-64 SysV int return register, also works without cudart DWARF.
        print("DKR_S675_API_RETURN %s status=%d" % (self.name, int(gdb.newest_frame().read_register("eax")) & 0xffffffff))
        return False
class EntryTrace(gdb.Breakpoint):
    def __init__(self, name):
        super().__init__(name, internal=True)
        self.name = name
    def stop(self):
        print("DKR_S675_API_ENTER "+self.name)
        ReturnTrace(self.name)
        return False
print("DKR_S675_SIMT_ENTRY pc="+str(gdb.newest_frame().read_register("pc")))
ReturnTrace("moe_gemm_simt_f16")
for name in ("cudaMemsetAsync", "cudaLaunchKernel", "cudaGetLastError", "cudaPeekAtLastError", "cudaLaunchKernel_ptsz", "cudaMemsetAsync_ptsz"):
    EntryTrace(name)
end
disable 1
continue
GDB
cd "$ROOT/mistralrs-core"
set +e
gdb -nx -batch -x "$OUT/trace.gdb" --args "${BINS[0]}" \
    --exact dkr_s6_native::dkr_s6_native_ab_gpu --ignored --nocapture 2>&1 | tee "$OUT/legacy-trace.log"
GDB_RC="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$GDB_RC" > "$OUT/gdb.exit"
echo "DKR_S675_TRACE_EXIT=$GDB_RC"
echo "=== CUDA API RETURN TRACE ==="
grep -E '^DKR_S675_(API_RETURN|SIMT_ENTRY)' "$OUT/legacy-trace.log" || true
if grep -q "Python Exception" "$OUT/legacy-trace.log" || ! grep -q "^DKR_S675_API_RETURN moe_gemm_simt_f16 status=" "$OUT/legacy-trace.log"; then
    echo "DKR_S675_TRACE=INCOMPLETE"
    exit 3
fi
echo "DKR_S675_TRACE=COMPLETE"
echo "DKR_S675_REPORT=$OUT"
# Do not claim promotion even if B succeeds.
exit 4
