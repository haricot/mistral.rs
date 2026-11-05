# CPU-MoE-S0: Standalone Recovery and Exact Baseline

This standalone feature starts from `haricot/mistral.rs` `master` at
`095991a7d680a6994e6f1554156ebef5d09db049`. The historical source is
[haricot/mistral.rs PR #4](https://github.com/haricot/mistral.rs/pull/4),
head `55d14313be867bc6f88605df21f467f344403ecf`.

S0 recovers the CPU expert execution contract against current upstream MoE
interfaces. It does not transplant the old PR, rewrite the router, or change
the default CUDA dispatch. The other features in PR #4 (KVarN, TurboQuant,
FlashInfer, MTP, modality selection) are out of scope.

S1/S2 add an opt-in, persistent per-layer Q4K expert GPU cache and separate
CPU compute/GPU prefetch worker threads. New code is specific to current Candle
and mistralrs-quant interfaces, not copied from RustingStrata.

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

## S1/S2 resident expert mode

For an explicitly experimental GGUF Q4K decode path, enable both flags before
model loading:

```sh
MISTRALRS_CPU_MOE_BASELINE=1 MISTRALRS_CPU_MOE_RESIDENT_SLOTS=1 mistralrs run -m MODEL
```

`MISTRALRS_CPU_MOE_RESIDENT_SLOTS` defaults to `0`, so S0 is unchanged. Its
nonzero value is the maximum cached experts **per MoE layer**, not a global
VRAM quota. The limit is 32. On an 8 GiB Pascal card, start with 1 slot and
measure total peak VRAM across all MoE layers before increasing it.

- The canonical packed Q4K stacks stay on CPU. Each successful prefetch creates
  three packed per-expert projections on the original GPU; they survive token
  boundaries and use existing GGUF CUDA kernels, including SM61 paths.
- Cache hits compute on GPU while the missing routes are sent to a persistent
  CPU worker. CPU and GPU partial outputs are combined with routing weights
  using F32 accumulation before casting back to the caller's dtype.
- Misses execute correctly on CPU immediately; a separate worker stages the
  highest-weight missing experts for subsequent decode tokens. Residency is
  bounded by per-layer slots, with LRU eviction and no duplicate prefetches.
- Cache hits, misses, promotions, evictions and transfer failures are logged
  every 128 decode forwards per layer. Prefill and unsupported formats use S0.
- The worker threads are concurrent. Whether CUDA copies overlap kernels on
  the actual SM61 stream is a separate, instrumented device-level measurement.
- LoRA, multi-rank and unsupported weight layouts retain the S0 restrictions.

This mode is not yet promoted. Asynchronous staging, CUDA memory lifetime,
numeric parity and decode TPOT must still pass on the physical GTX 1080 Max-Q.

## Checks

The branch-scoped `cpu-moe-qualification.yml` workflow runs one combined hosted
campaign: format/compile, S0 functional parity, cache/worker contract, and
lossless per-expert Q4K extraction. For a real-model SM61 qualification, compare the
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
| Cache | Per-layer hits, misses, promotions, evictions, failed prefetches |
| Concurrency | CPU worker intervals, GPU compute, H2D transfers and sync points |

No runtime performance claim or SM61 qualification is made by the hosted CPU
tests alone. The physical GTX 1080 Max-Q gate stays authoritative for SM61.

## Following stages

1. S1/S2: Qualify per-layer Q4K residence, asynchronous routing and CUDA
   stream lifetime on the actual SM61 GPU. Current implementation is opt-in.
2. S3: Add total-device memory budgeting, frequency-based admission and
   hysteresis only if the measurements justify the extra complexity.
3. S4: Compose with `extended`, verify the exact patch and SHA, then run the
   physical SM61 compatibility gate before promotion.

ASD is an independent Candle feature. CPU-MoE has no build-time or runtime ASD
dependency: integration exercises both only after their standalone gates pass.
