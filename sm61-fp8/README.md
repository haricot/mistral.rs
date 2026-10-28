# MistralRS FP8 standalone — one packed CUDA copy, BF16/F16 and MoE

Base: `master@28ed6daca455b4ae063896014d1ea698b912c539` plus the exact
Legacy source delta `allow_old_card@31556971b6a655883f452d76c78ebef772f53c38`.
Candle dependency: `haricot/candle:fp8_sm61_bf16_standalone@9cba2e898af3d25a89257f1f2ec5acece0404dcb`.
Activation: explicit Cargo feature `cuda-legacy-fp8` and Pascal SM61 only.

## Implemented in this candidate
- Blockwise E4M3FN weight 2D, F16 or BF16 activations, F32 accumulation;
  BF16 is widened directly from its bit pattern and never passed through F16.
- All packed CUDA weights and F32 dequant scales have **one** resident
  instance per projection. Original checkpoint weight and scale tensors move
  to **CPU before** the Candle packed device allocation, preventing steady-state
  double residency in VRAM. This still keeps a host-side copy, uses CPU upload
  during load and requires explicit high-water VRAM profiling to confirm.
- Rank-3 grouped MoE weights [experts,N,K], scales [experts,ceil(N/bn),ceil(K/bk)].
  Tokens are grouped by expert, one packed GEMM per active expert; no
  `dequantize_w()` in the forward/gather path.
- ModelOpt-style `weight_scale_inv` semantics are maintained; this is
  **not** a generic loader for all FP8 scale conventions.
- Dynamic activation checkpoint formats are admitted to F16/BF16
  W8A16 serving, not a dynamic FP8 W8A8 CUDA path. Numerical parity must be
  tested against the baseline model because activation quantization differs.

## Limits and qualification blockers
- Model-wide performance and logits parity are not yet qualified.
- Current MoE routing copies expert indices GPU→CPU and assembles output
  views; measure synchronization and allocation overhead before enabling
  it as a default. A device-only grouped routing kernel is a next optimization.
- The *weights* remain packed FP8, while scale factors remain F32;
  activations/outputs still occupy F16/BF16.
- Explicit ISQ/dequantization operations are allowed to materialize a
  dense tensor; ordinary SM61 forward/gather must not do so.
- Cargo.lock still reflects the previous Candle revision until Cargo
  re-resolves the pinned git revision on a developer machine.
- Changes are **not compiled or executed** in this session.

## Local qualification on GTX 1080 Max-Q
```bash
git fetch origin
git switch fp8_mistralrs_standalone
export CUDA_COMPUTE_CAP=61
export NVCC_CCBIN=gcc-14

# First regenerate the lock file for the exact Candle source revision.
cargo +1.99.0 update
cargo +1.99.0 fmt --all -- --check
cargo +1.99.0 check -p mistralrs-quant --features 'cuda cuda-legacy-fp8'
cargo +1.99.0 test --release -p mistralrs-quant \
  --features 'cuda cuda-legacy-fp8' \
  --test fp8_sm61_standalone -- --nocapture
```

Next: full-model reference logits, prefill/decode throughput, CUDA memory
high-water mark and MoE routing CPU/GPU correlation. Promotion only after
independent legacy/extended proofs from a **shared** SM61 campaign.
