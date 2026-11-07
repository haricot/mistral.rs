# FP8 standalone v2

Base: master `095991a7d680a6994e6f1554156ebef5d09db049`. Source provenance is recorded in `sources.json`.
This branch has no dependency on the Mistral Legacy branch and imports no
Legacy merge commit, ASD, DSpark, Bolt/DFlash, NVFP4 adapter, campaign producer,
promotion job or runner registration.

## Component

Packed SM61 FP8 W8A16/W8ABF16 dense and routed-expert weights, single CUDA packed-weight residency, and the real FP8 KV-cache admission and Standard-layout backend selected before allocation. F16 KV caches keep their normal backend. MLA remains unsupported by the Standard backend.

## Declared backend dependency

Candle is pinned to `9cba2e898af3d25a89257f1f2ec5acece0404dcb` for its opt-in software CUDA APIs.
The Candle core, NN, FlashAttention and Metal packages use the same revision.
The checked-in lockfile is generated for this exact source, without updating
unrelated registry dependencies. The root ZIP dependency uses the official
zip-rs 9.1.3 source commit `af2ed2a3e12af1fdd733b1886b3e9067bf510be5`:
Candle requires the incompatible registry ZIP 9.0.3, so this separate source
avoids a same-source resolver conflict. This backend currently identifies as 0.11.0;
the Mistral source remains the current 0.10.0 master.

The minimal shared compatibility includes the Candle GGUF/MXFP4 enum and row
codec extensions, SM61 MoE SIMT, and pre-native FlashInfer math. These are
required to compile/use the selected Candle and SM61 architecture; they are
identified separately from the FP8 algorithms. There is no new FP4
serving implementation in this branch. Shared compatibility can later be
factored into the integration support without merging the old Legacy tree.

The software path is enabled explicitly with `cuda-legacy-fp8`. Native
CUDA builds without that opt-in retain their modern architecture routing.
The pinned CUDA SDK can be reused unchanged; sources/features belong to the
binary and kernel cache identities, not a forced SDK rebuild.

## Verification

The branch workflow runs only hosted CPU checks and tests. It performs no
GPU initialization, model download, publication or promotion. Physical tests
are retained in the source for the later integrated operator-run campaign;
their presence is not an execution claim or a separate GPU prerequisite.

```bash
cargo check --locked -p mistralrs-cli --no-default-features --features code-execution
cargo test --locked -p mistralrs-quant --test fp8_standalone_cpu
```

Full-model validation, memory high-water marks and performance remain to be
measured on the final integrated candidate. Packed storage or synthetic tests
do not establish model-wide speed or numerical parity. Existing numerical
observations remain available; this branch introduces no VL/GGUF threshold.

The software FP8 weights use ModelOpt block-scale semantics and rank-2/rank-3
packed tensors. Expert routing still copies indices to CPU. Dynamic activation
checkpoints are served as W8A16/W8ABF16, not as a native W8A8 Tensor Core path.
Unsupported SM61 geometries fail instead of silently expanding dense weights.
BF16 Candle conversions are enabled as a technical dependency of W8ABF16;
general Mistral BF16 linear dispatch remains owned by the BF16 standalone.
