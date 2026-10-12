# MEXT-G3N — Native Mistral.rs EmbeddingGemma 2 integration contract

Status: **DESIGN / NOT IMPLEMENTED**. This document records the exact work
needed to extend the native Mistral.rs embedding pipeline; it is not an
integration success claim.

## Source and model identities

- Code baseline: `haricot/mistral.rs:master`,
  `20f607a77abd4738aa5594ee93a393a9fd284605`.
- Runtime model implementation: `haricot/candle:extended`,
  `66c4651cc83f9a43e79a4aa09302aac9bc763d2c`.
- `EmbeddingGemma2Model` is **not** the older `Gemma3TextModel` /
  `EmbeddingGemma`. The existing
  `EmbeddingLoaderType::EmbeddingGemma` and automatic matching cannot
  serve EmbeddingGemma 2 without a separate implementation.
- `embeddinggemma2_standalone` already contains the separate F32-only
  `mistralrs-embeddinggemma2` crate and CLI. Do not copy the model
  internals into Mistral.rs; reuse pinned Candle.

## Reviewed native call chain

1. `mistralrs/src/embedding_model.rs`:
   `EmbeddingModelBuilder::with_loader_type` sets the model family.
2. `mistralrs-core/src/pipeline/loaders/embedding_loaders.rs`:
   `AutoEmbeddingLoader` selects by `architectures` and expects an
   `EmbeddingModelLoader` returning a model compatible with
   `ShardedVarBuilder`, `IsqModel` and model device mapping.
3. `mistralrs-core/src/pipeline/embedding.rs`:
   `EmbeddingLoaderBuilder::build` maps loader enums. The existing
   loader handles `modules.json` and subsequently applies optional
   sentence-transformers `Pooling`, `Dense`, `Normalize` modules.
4. `EmbeddingPipeline::forward_inputs` calls
   `self.model.forward(&input_ids, &flash_meta)` and applies the
   modules before returning `ForwardInputsResult::Embeddings`.

## Required design choices before native API wiring

- **Explicit family:** introduce `EmbeddingGemma2` with distinct
  `architectures` matching and explicit errors for unsupported source
  configurations. Never reuse the legacy EmbeddingGemma model.
- **Weight adapter:** Candle's `embedding_gemma2::Model::load` currently
  takes `candle_nn::VarBuilder`; Mistral's general loader supplies
  `mistralrs_quant::ShardedVarBuilder`. Implement a tested adapter or
  a dedicated native F32 path before enabling `EmbeddingModelBuilder`.
- **Embedding semantics:** Candle `Model::embed(input, dims)` already
  performs the architecture's pooling/Matryoshka and normalization. The
  embedding pipeline must not repeat pooling or normalization blindly.
  Inspect `modules.json` carefully; don't insist upon sentence-transformers
  metadata that may not exist in a native Candle checkpoint.
- **Requested dimension:** the native `EmbeddingModel::forward` currently
  does not accept a dimension. Thread 128/256/512/768 from request to
  `Model::embed` or explicitly use a fixed pipeline dimension. Do not
  silently return a different width.
- **Tokenizer and prompt policy:** preserve exact tokenizer IDs and a
  versioned `Query:` versus `Passage:` policy for persisted indexes;
  no silent prompt insertion.
- **Initial runtime scope:** single text sequence, F32 CPU; CUDA SM61
  only after CUDA compile + physical runtime proof; no ISQ, quantization,
  multi-GPU, image/audio or batching in the first native milestone.
- **API:** ensure the Mistral `/v1/embeddings` endpoint reports the
  model identity, dimensions and total token usage coherently; reject
  unsupported encoding formats/dimensions.

## Qualification and promotion gates

1. Cargo.lock exact dependency graph (already resolved in candidate).
2. Local adapter unit tests and both CLI builds (CPU-only).
3. Pinned real checkpoint inference with fixed input/tokenizer IDs.
4. Separately sourced Google/Transformers embeddings in the same
   prompt policy and Matryoshka dimensions; compare via
   `embeddinggemma2-qualify`. Synthetic unit test data is not parity.
5. Native endpoint OpenAI API contract tests.
6. Hosted CUDA 12.9.2 sm_61 compilation and physical Pascal GPU run.
7. Only then consider `mistral.rs:extended` composition/promotion,
   with every proof bound to the same full SHA and Cargo.lock digest.

No operations on `master`, `mirror_orch`, Candle `extended`, validated
archives, or physical Pascal runner are requested by this contract.
