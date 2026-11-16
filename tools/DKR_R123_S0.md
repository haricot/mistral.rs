# DKR R1.23/S0 — Archived-binary performance screening

S684C: scoped model seal, raw Driver context isolation and 19 selected
default CPU tests passed. Verify the precise accepted model report hash,
CPU log and feature graph, Driver report and final model raw evidence.
No new build, no CUDA source change, no download, no graph capture.

    export DKR_S6_ORIGINAL_REPO="$HOME/tmp/mistralrs_provider/mistral.rs"
    python3 tools/dkr_r123_s0.py --original "$DKR_S6_ORIGINAL_REPO" --preflight
    DKR_APPROVE_R123_S0=1 python3 tools/dkr_r123_s0.py --original "$DKR_S6_ORIGINAL_REPO"

Three ABA brackets per cache (18 fresh processes in all). Both providers run
on the same model, token counts, prefix cache disabled, graphs disabled and
VLLM layout. Use only the staged corrected RMSNorm and paged PTX. Each
process has two raw probes and two 24-token nonstreaming generations per
prompt. Report primary second generation separately from first; compare all
numeric values to the corresponding accepted provider. Timing uses engine
prefill/completion times (millisecond granularity) and whole request wall
clock. Completion ms per reported token is an engine proxy, not a streaming
inter-token latency or TTFT. GPU occupancy, clocks and temperature before
and after are snapshots, not a peak VRAM measurement. Keep the GPU otherwise
idle. No telemetry during timing. Each B is normalized by the mean of its
two adjacent A measurements; report median ratios and max A/A drift, flag
>10% drift. Screening does not establish a release benchmark or automatic
performance acceptance. A numerical change requires review. Real TTFT,
TPOT, peak VRAM, dispatch profiling, longer decode and release-grade
measurement remain for R1.23/S1+. Nonprimary Candle registry integration
remains open; production promotion not authorized.
