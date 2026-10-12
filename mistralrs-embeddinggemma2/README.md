# EmbeddingGemma 2 — Haricot Mistral.rs standalone (MEXT-G3, initial milestone)

Source branch: `haricot/mistral.rs:master`.
Model implementation: `haricot/candle:extended`, pinned SHA
`66c4651cc83f9a43e79a4aa09302aac9bc763d2c`.

This is an independently testable, local-only Rust adapter around
`candle_transformers::models::embedding_gemma2` and does not duplicate its
attention, per-layer-embedding (PLE), pooling or Matryoshka code.

## Run with a local checkpoint

```bash
cargo run --release --locked -p mistralrs-embeddinggemma2 --bin embeddinggemma2-text -- \
  --local-dir /path/to/embeddinggemma-2 \
  --text 'Passage: Exemple de mémoire persistante Haricot.' --dims 256 --json
```

Requires `config.json`, `tokenizer.json` and safetensors shards. The wrapper
uses **F32** and the native Candle model; the caller supplies the query/document
prompt policy explicitly. Only text input is supported; one sequence per call.


## G3-P — numerical/reference qualification (Rust-only)

The local embedding JSON now includes `token_ids` (exact tokenization) and
the normalized `vector`. Use an **independently generated** reference JSON:

```json
{
  "schema_version": 1,
  "model_revision": "<explicit pinned Google model revision>",
  "tokenizer_revision": "<explicit pinned tokenizer revision>",
  "prompt_policy": "document: Passage: prefix",
  "dimensions": 256,
  "token_ids": [/* exact reference token IDs */],
  "vector": [/* 256 reference floats */]
}
```

This is a **schema example**, not a measured or validated reference. Create
the full reference with a separately pinned implementation and exact matching
input; never use the observed vector as a reference.

```bash
cargo run --release --locked -p mistralrs-embeddinggemma2 \
  --bin embeddinggemma2-text -- \
  --local-dir /path/to/embeddinggemma-2 \
  --text 'Passage: Mémoire persistante Haricot.' --dims 256 --json \
  > observed.json

cargo run --release --locked -p mistralrs-embeddinggemma2 \
  --bin embeddinggemma2-qualify -- \
  --observed observed.json --reference reference.json \
  --min-cosine 0.999 --max-abs 0.01
```

The comparison requires exact token IDs and token counts, matches the
embedding dimension, checks finite values, unit L2 norm and the supplied
cosine/max-absolute-error thresholds. The report **always** records
`reference_origin_independently_verified: false`; the tool cannot authenticate
where the provided reference came from. Real Google parity has not yet been
performed. A CPU test with synthetic fixtures is not a real checkpoint proof.

## Not yet complete

- This is **not yet wired into** Mistral.rs's `EmbeddingLoaderType`,
  `EmbeddingModelBuilder` or OpenAI-compatible `/v1/embeddings` endpoint.
- Checkpoint inference and numerical parity against a fixed Google/Transformers
  model revision have not been demonstrated.
- No image/audio towers, quantization or GPU SM61 qualification.
- Never mix these 128/256/512/768-dimensional vectors with older embedding
  indexes without model revision and prompt-policy metadata.
- Cargo.lock has been regenerated against the pinned Candle graph by
  candidate CI run `38237724116` (`cargo metadata --locked` passed). It still
  requires CPU test/typecheck qualification before this branch is build-ready.

The base `master`, `mirror_orch`, `asd_runner` and Candle branches are
unchanged. Promotion requires separate CPU, parity, CUDA and SM61 evidence.
