# DKR R2 CUDA Graphs on SM61

This campaign uses the sealed R1.23/S0 LFM3 GGUF and embeds the same native
and typed PTX. Build the dedicated candidate once from the graph worktree:

```bash
python3 tools/dkr_r2_sm61_graph.py --build \
  --original "$DKR_S6_ORIGINAL_REPO" --s0-report "$DKR_S0_REPORT"
```

Run the same command without `--build`, headless on the SM61 GPU. The
existing S6 Cargo target and SM61 CUDA object cache are reused. The runner
checks the seal again after the experiment.

The candidate initially captures an actual DKR KV write and paged attention
at batch 1. It changes Q, K, V, the page table, the slot and context length
over four replays. It compares the output and final KV contents bitwise
against the same DKR calls on a separate eager cache.

The model legs use the R1.23/S1B SDK streaming benchmark. Every 24-token and
128-token stream checks the graph event inventory and preserves exact output
and logprobs. The runner also checks raw logits and short generations against
the seal. Separate VRAM and route audits run before the timed ABBA brackets.
The route audit compares operation, symbol and PTX hash; a graph launch
replays nodes without calling the Rust adapters again.

The report records host capture time including preparation and warmup, SDK
TTFT and inter-token time, host forward time, and sampled device peak VRAM.
Host forward time includes metadata preparation and submission. The first
and repeat streams are reported separately, along with control drift and
raw per-leg JSON and logs. `dkr-sm61-graph-comparison.json` is written after
the replay, parity and route checks pass.

LFM3 graph admission uses the experimental SM61 feature and
`MISTRALRS_DKR_LFM3_CUDA_GRAPHS=1` in the candidate environment. Graph
contexts keep DKR's 256-token boundaries so the short route and partition
count agree with eager dispatch.
