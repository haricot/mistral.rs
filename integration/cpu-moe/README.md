# CPU-MoE-S0: Standalone Recovery and Exact Baseline

This standalone feature starts from `haricot/mistral.rs` `master` at
`095991a7d680a6994e6f1554156ebef5d09db049`. The historical source is
[haricot/mistral.rs PR #4](https://github.com/haricot/mistral.rs/pull/4),
head `55d14313be867bc6f88605df21f467f344403ecf`.

S0 recovers the CPU expert execution contract against current upstream MoE
interfaces. It does not transplant the old PR, rewrite the router, or change
the default CUDA dispatch. The other features in PR #4 (KVarN, TurboQuant,
FlashInfer, MTP, modality selection) are out of scope.

## Runtime contract

Set `MISTRALRS_CPU_MOE_BASELINE=1` before starting mistral.rs to opt in.
The flag is read once at process startup. Unset or `0` keeps upstream behavior.

In baseline mode:

- Routed expert weights are loaded on CPU; the router, attention and shared
  expert stay on their existing model device.
- The selected activations, top-k expert indices and routing weights are
  transferred to CPU for expert execution. The reduced expert output is moved
  back to the original device.
- Transfers are synchronous by design. S0 is a correctness and cost baseline,
  not an asynchronous execution or GPU resident expert cache.
- Multi-rank execution and active expert LoRA fail closed in baseline mode.
- Each model format and precision still requires its own real-model gate.

This baseline relies on the existing Candle and mistralrs-quant MoE backends.
There is no imported RustingStrata implementation or license-specific code.

## Checks

The branch-scoped `cpu-moe-s0.yml` workflow checks formatting, CPU compilation
and synthetic numerical routes. For a real-model qualification, compare the
same checkpoint and SHA with the flag off and on, using the same prompts,
sampling, quantization, context, and paged-attention configuration.

Record at least:

| Evidence | Baseline control |
| --- | --- |
| Model load | Resolved CPU expert storage and no unexpected VRAM OOM |
| Prefill | Prompt throughput and logits finite |
| Decode | TPOT, CPU time, GPU time, and synchronization cost |
| Numeric | Logits and greedy output vs the non-offloaded reference |
| Resources | Peak host RAM, peak VRAM, transfers and available VRAM |

No runtime performance claim or SM61 qualification is made by the hosted CPU
tests alone. The physical GTX 1080 Max-Q gate stays authoritative for SM61.

## Following stages

1. S1: Per-expert VRAM residency with slots keyed by stable weight identity,
   device, layer and expert id. Bound residency by measured available memory.
2. S2: Asynchronous host/GPU expert split with bounded waits and a safe fallback.
   Preserve the existing SM61 SIMT CUDA kernels for resident GPU experts.
3. S3: Frequency-based adaptive placement with hysteresis and measurable
   swap overhead.
4. S4: Compose the standalone feature with `extended`, verify the exact patch
   and SHA, and run CPU, CUDA and physical SM61 qualification before promotion.

ASD is an independent Candle feature. CPU-MoE has no build-time or runtime ASD
dependency: integration exercises both only after their standalone gates pass.
