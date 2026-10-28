# BF16 MistralRS standalone — Pascal SM61 (candidate, NOT qualified)

## Scope and provenance
Created from `haricot/mistral.rs:master@28ed6daca455b4ae063896014d1ea698b912c539`.
The shared legacy source is imported from
`allow_old_card@31556971b6a655883f452d76c78ebef772f53c38`.
Candle pinned at `haricot/candle:cuda_legacy@bfce966ad4d61a48801eb6af4db8b8eabebc01a8`.
Only `cuda-legacy-bf16` needs to be enabled for the standalone test.

## Model and numerical contract
Pascal has no native BF16 GEMM. The Candle legacy path widens BF16
operands to F32, uses SGEMM/FP32 accumulation and rounds the result
once to BF16. This preserves numbers outside FP16's dynamic range.
This is not a native BF16 speedup and it consumes temporary FP32
buffers; the original BF16 weights remain 16-bit.

A MistralRS-level regression test verifies an actual BF16 matrix product
on the Pascal card and checks a result larger than the maximum finite
FP16 value against a CPU F32 accumulation of BF16-rounded operands.
Candle already contains wider independent tests (strides, offsets,
broadcast, reductions and transposes).

## Local qualification — operator only
```bash
git fetch origin
git switch bf16_mistralrs_standalone
export CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14
cargo +1.99.0 fmt --all -- --check
cargo +1.99.0 check -p mistralrs-core --features 'cuda cuda-legacy-bf16'
cargo +1.99.0 test --release -p mistralrs-core --features 'cuda cuda-legacy-bf16' --test bf16_sm61_standalone -- --nocapture
```

Later qualify actual model forward, attention and KV-cache dtype boundaries,
CPU/GPU numerical parity, peak temporary workspace and latency.

Source-only candidate. Compilation and physical GPU execution are NOT claimed.
Promotion should reuse the Extended campaign's independent Legacy archive proof,
not the Extended binary's smoke result by inference.
