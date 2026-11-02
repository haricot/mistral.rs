# FP4 MistralRS standalone (experimental)

## Provenance
Created from `master@28ed6daca455b4ae063896014d1ea698b912c539`.
Layers the prior `nemotron_bolt_dflash_standalone@1319e272149f41e7ce1648b2b52514877403fb44` source delta on top, to retain Bolt's DFlash2 config, own LM head, and full source compatibility. Depends on `haricot/candle@40b5a8acc43b5f8ef925370ae12daddd929f39f2`.

## Implemented (source only)
- Explicit `cuda-legacy-fp4` route in `Nvfp4Layer` using Candle E7 W4A16.
- Transfer E2M1 weights and E4M3 scales as packed bytes. Per-output-row F32 scale multipliers remain distinct.
- DFlash Q/K/V, gate/up, and context K/V avoid dense weight concatenation when legacy FP4 is enabled and ISQ is disabled.
- Retain old dense fallback when the feature is disabled.
- Reserve VRAM for two packed copies in the current bridge (one held by MistralRS, one by Candle E7) rather than assuming one copy or counting a permanently decoded F16 matrix.

## Limitations
**Not yet compiled or validated on physical SM61.** The Candle E7 W4A16 execution currently uses a scalar reference kernel, not a performance-optimized tiled GEMM. BF16 inputs are converted to F16 for this backend; compare numerical results to CPU/F16 reference. Only rank-2 W4A16 is supported on legacy CUDA; W4A4 and packed MoE expert tensors remain outside this initial route. Do not describe this as native Pascal FP4 or as final peak-memory optimization.

## Local qualification (operator-run)
```bash
export CUDA_COMPUTE_CAP=61
export NVCC_CCBIN=gcc-14
cargo fmt --all -- --check
cargo test -p mistralrs-core separated_projection_matches_dense_concatenation
cargo test -p mistralrs-quant --features 'cuda cuda-legacy-fp4'
cargo check -p mistralrs-cli --features 'cuda cuda-legacy-fp4 flash-attn'
```

Run a small real Bolt/DFlash2 smoke only after these pass, using `MISTRALRS_DFLASH_ISQ=none` to preserve original NVFP4 packed tensors. Record CUDA version, actual VRAM, draft acceptance, F16 parity and the absence of dense GEMM weight materialization. Compilation and GPU qualification are deliberately left to the operator.

## Integration
Review and qualify first. Integration staging branch for the legacy dispatch: `fix/allow_old_card-nvfp4-legacy`. Integration staging branch for DFlash: `fix/extended-nvfp4-dflash`. Neither `allow_old_card` nor `extended` is modified by these stages. The inherited `mistral-integration/sources.json` identifies the Bolt prerequisite and must be regenerated before orchestration/promotion.
