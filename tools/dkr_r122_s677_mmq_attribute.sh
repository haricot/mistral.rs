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
# Preserve original evidence, including the completed B qualification.
B="$OUT"
[[ "$(cat "$B/external.exit")" == 0 && -s "$B/native-external.json" ]]
[[ "$(jq -er .external_scope_ok "$RUN/s676/external-audit.json")" == true ]]
[[ "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '[:space:]')" == 6.1 ]]
OUT="$RUN/s677"
mkdir -p "$OUT"
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
# Trace the attribute setters and MMQ launches in the original A binary.
# Then do one counterfactual replay without the redundant ordinary-limit
# attribute setter. No last-error clearance and no forced launch success.
cat > "$OUT/trace.gdb" <<'GDB'
set pagination off
set confirm off
set print thread-events off
set debuginfod enabled off
set breakpoint pending on
start
set language c
python
import os
def mmq_stack():
    f = gdb.newest_frame()
    names = []
    while f is not None and len(names) < 16:
        names.append(f.name() or "")
        f = f.older()
    return any("instantiate_mmq_" in n or "launch_mmq_case_" in n or "launch_mmq_gguf_" in n for n in names)
class MmqCaller(gdb.Function):
    def __init__(self):
        super().__init__("dkr_mmq_caller")
    def invoke(self):
        return int(mmq_stack())
MmqCaller()
class ReturnTrace(gdb.FinishBreakpoint):
    def __init__(self, name, detail):
        super().__init__(gdb.newest_frame(), internal=True)
        self.name, self.detail = name, detail
    def stop(self):
        status=int(gdb.newest_frame().read_register("eax")) & 0xffffffff
        print("DKR_S677_RETURN name=%s status=%d %s" % (self.name,status,self.detail))
        return False
class EntryTrace(gdb.Breakpoint):
    def __init__(self, name, attribute=False):
        super().__init__(name, internal=True)
        self.name,self.attribute=name,attribute
    def stop(self):
        if not mmq_stack():
            return False
        f=gdb.newest_frame()
        if self.attribute:
            detail="attr=%d bytes=%d" % (int(f.read_register("esi")),int(f.read_register("edx")))
        else:
            caller=f.older()
            detail="caller="+(caller.name() or "unknown" if caller else "unknown")
        ReturnTrace(self.name,detail)
        return False
if os.environ.get("DKR_S677_MODE")=="observe":
    EntryTrace("cudaFuncSetAttribute",True)
for name in ("cudaLaunchKernel_ptsz","cudaLaunchKernel"):
    EntryTrace(name)
end
GDB
cp "$OUT/trace.gdb" "$OUT/observe.gdb"
printf 'continue\n' >> "$OUT/observe.gdb"
cp "$OUT/trace.gdb" "$OUT/omit.gdb"
cat >> "$OUT/omit.gdb" <<'GDB'
# On Linux x86-64, Runtime signature is (function, enum attr, int value).
# MaxDynamicSharedMemorySize == 8. Pascal ordinary per-block limit is 48 KiB.
# Refuse to intercept any other call, or any larger/invalid request.
break cudaFuncSetAttribute if $esi == 8 && $edx > 0 && $edx <= 49152 && $dkr_mmq_caller()
commands
silent
printf "DKR_S677_OMIT_ATTRIBUTE attr=%d bytes=%d\n", $esi, $edx
return (int)0
continue
end
continue
GDB
cd "$ROOT/mistralrs-core"
for mode in observe omit; do
    export DKR_S677_MODE="$mode"
    export DKR_S6_NATIVE_OUTPUT="$OUT/native-legacy-$mode.json"
    rm -f "$DKR_S6_NATIVE_OUTPUT"
    set +e
    gdb -nx -batch -x "$OUT/$mode.gdb" --args "${BINS[0]}" \
        --exact dkr_s6_native::dkr_s6_native_ab_gpu --ignored --nocapture \
        > "$OUT/$mode.log" 2>&1
    rc=$?
    set -e
    echo "DKR_S677_${mode}_GDB_EXIT=$rc"
    grep -E '^DKR_S677_|^Error:|^test result:|^Python Exception|Error in sourced command file|^Cannot ' "$OUT/$mode.log" || true
done
cd "$ROOT"
python3 tools/dkr_r122_s677_audit.py --report-dir "$OUT" --external "$B/native-external.json"
echo "DKR_S677_REPORT=$OUT"
# Even successful counterfactual is diagnostic only.
exit 4
