# FP8 MistralRS standalone — Pascal SM61 (candidate, NOT qualified)

## Scope and provenance
Created from `haricot/mistral.rs:master@28ed6daca455b4ae063896014d1ea698b912c539`.
The shared CUDA legacy source delta is imported from
`allow_old_card@31556971b6a655883f452d76c78ebef772f53c38`.
Candle is pinned to `haricot/candle:cuda_legacy@bfce966ad4d61a48801eb6af4db8b8eabebc01a8`.

Only `cuda-legacy-fp8` should be enabled for this standalone's qualification.
The shared source overlay contains other legacy backends but they are opt-in.

## First bridge
`mistralrs-quant::BlockwiseFP8Linear` can select Candle's
`Fp8Sm61Linear` **SIMT W8A16** in the opt-in `cuda-legacy-fp8` build
when weight is 2D E4M3, activation/output dtype is F16, no dynamic
activation scheme is active, scales have two block dimensions, and K
is divisible by 128. The checkpoint's inverse scales are handed to
Candle without reinterpreting unqualified scale conventions.

The existing high-end CUDA providers remain intact.
The F16 activation contract is explicit; BF16 activations, dynamic W8A8,
rank-3 MoE and advanced residency selection are NOT yet routed to Candle
SM61. Unsupported shapes use the previous path. Do not claim universal
FP8 model support from this bridge.

## Known memory limitation
This first bridge still holds the original MistralRS FP8 weight and
an additional packed SM61 device allocation; it avoids persistent
F16 expansion in the new forward provider, but it does not yet achieve
single-copy resident FP8. Next gate: eliminate the duplicate packed
representation and integrate global KV/workspace/weight VRAM admission.

## Local qualification — operator only
```bash
git fetch origin
git switch fp8_mistralrs_standalone
export CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14
cargo +1.99.0 fmt --all -- --check
cargo +1.99.0 check -p mistralrs-quant --features 'cuda cuda-legacy-fp8'
cargo +1.99.0 test --release -p mistralrs-quant --features 'cuda cuda-legacy-fp8' --test fp8_sm61_standalone -- --nocapture
```

Then qualify model-weight inventory, numerical logits/argmax and independent
prefill/decode timing. Record packed device allocation, peak VRAM, backend
selected, model config and exact GPU/driver/Candle SHA. Tests have not been
executed simply by committing this branch.

## Promotion
Standalone first; candidate integration into `extended` should be
part of one Extended SM61 campaign with independently executed component
archives. Do not promote on the strength of the Extended binary alone.
