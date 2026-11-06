# FP4 standalone v2

Base: master@095991a7d680a6994e6f1554156ebef5d09db049.

The current NVFP4 implementation is retained, with an opt-in Pascal E7 adapter.
No Mistral Legacy branch is a prerequisite or ancestor.
The FP4 opt-in also enables Candle software BF16 conversion primitives for
BF16 outputs; it does not import the Mistral BF16 standalone algorithms. Candle is pinned to
9cba2e898af3d25a89257f1f2ec5acece0404dcb, shared with BF16/FP8 v2.

Enable `cuda-legacy-fp4` in the CLI, API, core or Python crate. SM61 uses E7 for
dense rank-2 W4A16 only. Quantized activations and stacked NVFP4 experts fail
explicitly. Other GPUs retain their existing routing. Raw E4M3 scale bytes are
preserved; per-row global multipliers precede bias addition. E7 arithmetic uses
F16 activations, including explicit conversion for BF16 output: the full BF16
activation range is not preserved. At most 65535 flattened input rows are
accepted per E7 call. Offset/strided inputs receive an owned contiguous copy.

Compatibility changes cover Candle GGUF/MXFP4 types, CPU/CUDA/Metal embeddings,
UQFF mappings and minimal SM61 MoE/FlashInfer compilation support. BF16 and FP8
algorithms, ASD, DSpark and Bolt/DFlash are not imported.

CPU validation covers the CLI, NVFP4 unit/checkpoint contracts and synthetic
NVFP4/MXFP4 loading/embedding fixtures. One external-checkpoint test is ignored
without its fixtures. CUDA compilation, GPU execution, models and performance
remain unqualified. Feature-gated hardware tests are included for the later
integrated campaign, including offset input views.

The future assembly combines the three v2 standalones and residual support.
Old Legacy supplies selected changes, not a wholesale merge input.
