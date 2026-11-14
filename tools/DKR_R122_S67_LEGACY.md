# S6.7 — Legacy Isolation & Promotion Readiness

Use the already compiled S6 archive at 8dba9606087b, native packages at
7b3b19cf8b62, and unchanged Candle pin b231d3c. No CUDA/Rust compilation,
no CI and no source modification. The script uses nextest list to extract
the archive, GDB catch throw and SIGABRT backtraces, with separate diagnostic
outputs. It does not overwrite the original A/B logs or JSON. The original
external evidence is re-audited before debugging. On Arch, GDB is available
as the gdb package. The default run does NOT enable CUDA_LAUNCH_BLOCKING;
if the stack is ambiguous, repeat with DKR_S67_SYNC_DIAGNOSTIC=1.

    python3 tools/dkr_r122_s67_promotion.py --self-test
    DKR_S6_ORIGINAL_REPO=/path/to/original bash tools/dkr_r122_s67_legacy.sh

Expected exit 4: promotion blocked, regardless of whether the throw is
localized. The GDB stack identifies the actual C++ frame if symbols survive;
a scan stack does NOT identify the underlying CUDA error code. If the
exception is not reproducible or symbols are missing, keep it inconclusive.
After reviewing the stack, design a minimal legacy-only correction and
separate numeric qualification. Do not repin Candle or rebuild on speculation.
The report lists tests without telemetry, not necessarily GPU tests.
