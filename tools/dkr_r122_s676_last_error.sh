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
# Preserve B and original A traces; new diagnostics use a separate directory.
[[ "$(cat "$OUT/external.exit")" == 0 && -s "$OUT/native-external.json" && -s "$OUT/external.log" ]]
B="$OUT"
OUT="$RUN/s676"
mkdir -p "$OUT"
python3 tools/dkr_r122_s676_audit.py --external "$B" --native-root "$NATIVE" --output "$OUT/external-audit.json"
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
# First trace all relevant Runtime calls before SIMT; no injected calls.
cat > "$OUT/history.gdb" <<'GDB'
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set breakpoint pending on
start
python
class ReturnTrace(gdb.FinishBreakpoint):
    def __init__(self, name):
        super().__init__(gdb.newest_frame(), internal=True)
        self.name = name
        self.caller = gdb.newest_frame().older()
        self.site = self.caller.name() if self.caller else "unknown"
    def stop(self):
        status = int(gdb.newest_frame().read_register("eax")) & 0xffffffff
        if status or self.name == "moe_gemm_simt_f16":
            print("DKR_S676_HISTORY name=%s status=%d caller=%s" % (self.name,status,self.site))
            if status:
                gdb.execute("bt 5")
        return False
class EntryTrace(gdb.Breakpoint):
    def __init__(self, name):
        super().__init__(name, internal=True)
        self.name = name
    def stop(self):
        if self.name == "moe_gemm_simt_f16":
            print("DKR_S676_SIMT_ENTRY")
        ReturnTrace(self.name)
        return False
for name in ("cudaMemsetAsync", "cudaMemsetAsync_ptsz", "cudaLaunchKernel", "cudaLaunchKernel_ptsz",
             "cudaGetLastError", "cudaPeekAtLastError", "cudaMallocAsync", "cudaMallocAsync_ptsz",
             "cudaFreeAsync", "cudaFreeAsync_ptsz", "cudaMalloc", "cudaFree",
             "cudaGetDevice", "cudaSetDevice", "cudaDeviceGetAttribute", "cudaGetDeviceProperties_v2",
             "cudaStreamSynchronize", "cudaStreamSynchronize_ptsz", "cudaDeviceSynchronize",
             "moe_gemm_simt_f16"):
    EntryTrace(name)
end
continue
GDB
# Second run: inspect the slot at the SIMT boundary. Only if it is already
# 201, explicitly consume it in this disposable process, record the result,
# then allow the complete Rust test (including final synchronize) to run.
# This is a controlled experiment, NEVER a production error suppression.
cat > "$OUT/probe.gdb" <<'GDB'
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set breakpoint pending on
break moe_gemm_simt_f16
run
set language c
set $before = ((int (*)(void)) cudaPeekAtLastError)()
printf "DKR_S676_PREEXISTING=%d\n", $before
if $before == 201
    set $consumed = ((int (*)(void)) cudaGetLastError)()
    printf "DKR_S676_CONSUMED=%d\n", $consumed
    set $after = ((int (*)(void)) cudaPeekAtLastError)()
    printf "DKR_S676_AFTER_CLEAR=%d\n", $after
end
disable 1
set language auto
continue
GDB
cd "$ROOT/mistralrs-core"
for mode in history probe; do
    export DKR_S6_NATIVE_OUTPUT="$OUT/native-legacy-$mode.json"
    rm -f "$DKR_S6_NATIVE_OUTPUT"
    set +e
    gdb -nx -batch -x "$OUT/$mode.gdb" --args "${BINS[0]}" \
       --exact dkr_s6_native::dkr_s6_native_ab_gpu --ignored --nocapture \
       > "$OUT/$mode.log" 2>&1
    rc=$?
    set -e
    echo "DKR_S676_${mode}_GDB_EXIT=$rc"
    grep -E '^DKR_S676_|^Error:|^test result:|^Python Exception|^.*Error in sourced command file' "$OUT/$mode.log" || true
done
cd "$ROOT"
python3 tools/dkr_r122_s676_audit.py --external "$B" --native-root "$NATIVE" --output "$OUT/external-audit.json" --probe "$OUT"
echo "DKR_S676_REPORT=$OUT"
exit 4
