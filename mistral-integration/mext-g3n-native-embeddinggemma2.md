# MEXT-G3N — Native Mistral.rs EmbeddingGemma 2 (candidate)

**Status: UNVALIDATED_NATIVE_CANDIDATE.**
This integration branch starts from the separately CPU-validated
`embeddinggemma2_standalone` (source commit
`641dd2541406476203c286c8d4cd5e431951c103`).
It is **not** a merge to `master`, `mirror_orch` or `extended`.

## Intended behavior

- Register distinct Mistral.rs `EmbeddingLoaderType::EmbeddingGemma2`,
  including automatic detection of HF `architectures=["EmbeddingGemma2Model"]`.
- Reuse the native Candle implementation in
  `candle-transformers::models::embedding_gemma2`; a
  `candle_nn::SimpleBackend` bridges the existing Mistral
  `ShardedVarBuilder` to Candle's native weight names.
- Treat the checkpoint as `language_model.*`, not as
  `Gemma3TextModel` or the old `EmbeddingGemma`.
- Allow native HF checkpoints without `modules.json`, while preserving
  the legacy sentence-transformers loader for other models.
- Skip every external pooling/dense/normalization module for the native
  Gemma 2 path. Candle performs mean pooling, dimension truncation and L2
  normalization in this order.
- Recognize the native architecture in `EmbeddingModelBuilder`.
  Automatic single-device assignment is converted to an explicit dummy
  mapper; actual cross-device layer assignments are rejected.
- Force **FP32**, one text sequence and fixed **768-dimensional** output.
  Quantization/ISQ, UQFF, batch size >1, distributed layers and other
  modalities are not yet qualified.

## Explicit SDK construction

Select the loader with
`EmbeddingModelBuilder::with_loader_type(EmbeddingLoaderType::EmbeddingGemma2)`.
Choose F32 with `EmbeddingModelBuilder::with_dtype(...)` and optionally
`with_force_cpu()`. Automatic architecture discovery is also registered,
but explicit loading is recommended during qualification.

The native pipeline supplies `ForwardInputsResult::Embeddings` and can use
Mistral.rs' existing embedding serving path. This is a **source-level
integration claim only** until checkpoint and API response tests are run.

### Important limitations

- The initial native `EmbeddingModel::forward` interface does not carry
  the per-request Matryoshka output dimension. It therefore returns
  **768 dimensions** unconditionally. It must not advertise arbitrary
  128/256/512-dimensional native API output yet.
- The native Mistral engine's sequence scheduler should be limited to one
  request at a time. The explicit builder route caps its scheduler at one;
  automatic paths must be verified before concurrent deployment.
- The absence of duplicate normalization is required for correctness;
  adding arbitrary sentence-transformers module stacks is rejected.
- GPU/F16/BF16, multi-GPU, tensor parallel, ISQ, performance and actual
  Google checkpoint numerical parity are not validated by a CPU build.
- Do not use this unqualified native endpoint to populate Haricot Memory
  Dynamics persistent indexes.

## Qualification gates

1. Cargo.lock exact dependency graph + source formatting.
2. `cargo check --locked --no-default-features -p mistralrs-core`.
3. `embedding_gemma2_routes_to_a_distinct_native_family` unit test.
4. HF real checkpoint tokenizer IDs and embeddings against the independent
   pinned reference via `embeddinggemma2-qualify`.
5. HTTP `/v1/embeddings` response and one-request scheduler test.
6. CUDA 12.9.2 `sm_61` compile and physical Pascal execution.
7. Only then review candidate composition into `mistral.rs:extended`.

The related `MEXT-G3N — Native EmbeddingGemma 2 CPU` workflow does **not**
perform gates 4–7. No promotional or GPU runner workflow is activated here.
