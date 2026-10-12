# MEXT-G3V — Official Google EmbeddingGemma 2 Runtime Qualification

Status: **CPU COMPILE / TEST CANDIDATE, NOT QUALIFIED CHECKPOINT**.

The G3N branch has passed Rust SDK and core CPU compilation. G3V
focuses on **real-model loading** while keeping G3N's qualified commit
untouched. G3V's target model is
[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2);
the model's published `config.json` declares `EmbeddingGemma2Model`.
The repository is approximately 1.5 GB and contains multimodal
components; this candidate only uses the text encoder.

## Why G3V is necessary

Google's published `modules.json` explicitly contains:
`Transformer -> Pooling -> Normalize`.
Its `1_Pooling/config.json` contains
`{"embedding_dimension":768,"pooling_mode":"mean","include_prompt":true}`.

Our native Candle text model already does mean pooling and L2
normalization. Earlier G3N would **reject** the official `modules.json`,
despite passing compilation. G3V checks the official metadata
semantics, allows them if equivalent, then omits their execution to
prevent double pooling/normalization. It refuses Dense modules,
different order, non-mean pooling, other dimensions, unexpected
configuration keys, or `include_prompt=false`. A checkpoint without
`modules.json` is also supported.

## First real-checkpoint local smoke (after CPU CI is green)

Obtain the official checkpoint using the HF CLI. Preserve an immutable
40-character model commit revision in `MODEL_REV`; do not treat
`main` as a reproducible model identity. Confirm any model access
terms in your own HF account.

```bash
export MODEL_REV='<exact-full-40-character-HF-model-commit>'
hf download google/embeddinggemma-2 \
  --revision "$MODEL_REV" \
  --local-dir ./assets/embeddinggemma-2 \
  --include 'config.json' 'tokenizer.json' \
            'model.safetensors' 'modules.json' '1_Pooling/config.json'

git fetch origin integration/mext-g3v-official-model-runtime
git switch integration/mext-g3v-official-model-runtime

MEXT_G3V_MODEL="$PWD/assets/embeddinggemma-2" \
  cargo run --release --locked --no-default-features \
  -p mistralrs --example embedding_gemma2_native_smoke
```

Or set `MEXT_G3V_MODEL=google/embeddinggemma-2` and
`MEXT_G3V_REV=$MODEL_REV` to allow the native HF loader to fetch
files. The smoke uses explicit `EmbeddingLoaderType::EmbeddingGemma2`,
forces F32 CPU, processes **one text request**, checks an exactly
768-wide finite result and L2 norm near 1. It never records a
Google-reference numeric parity claim or SM61 success.

### Follow-on numerical qualification

1. Preserve **exact tokenizer IDs**, prefix policy, model revision,
   architecture and 768-dim output under a single evidence manifest.
2. Run the standalone `embeddinggemma2-text` on the same local
   checkpoint and prompt, preserving `token_ids`.
3. Generate an **independent** Google/Transformers reference at the
   same pinned revision, with the same prompt and token IDs. Compare
   with `embeddinggemma2-qualify`, never with a reference copied from
   the observed vector.
4. Perform a real `/v1/embeddings` HTTP request against the running
   Mistral server and verify schema, width and usage.
5. Validate CUDA 12.9.2 sm_61 compile and physical Pascal runtime on
   exactly the same candidate commit and Cargo.lock checksum.

The G3V GitHub workflow **does not** download the 1.5-GB model.
Its tests only assert that published module strings parse, matching
pooling metadata is accepted and unsupported metadata is rejected,
and that the real-model smoke example compiles.

This is an isolated, draft-only feature branch. No merge into master,
mirror_orch, or extended, and no physical GPU dispatch.
