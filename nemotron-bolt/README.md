# Nemotron Bolt E4B and native DFlash

This standalone branch starts at mistral.rs master `28ed6daca455` and pins every
Candle workspace package to haricot/candle Extended `40b5a8acc43b`. The same
Candle revision is used by the native Myna ASR provider. The final integration
target is mistral.rs `extended`; runtime qualification must precede promotion.

Bolt uses its published Q4_K_M GGUF. DFlash initially uses the official native
safetensors checkpoint, not a GGUF drafter. The native NVIDIA checkpoint is
W4A16 NVFP4 and has its own quantized output head. The portable loader expands
NVFP4 weights on CPU once and transfers floating point weights to the execution
device. On Pascal, select F16. This is software dequantization, not a native
Blackwell FP4 kernel and not recovery of the original BF16 weights. W4A4 is
rejected because its activation quantization is not implemented by this path.

The target GGUF stays quantized. `MISTRALRS_GEMMA4_PLE_CPU=1` optionally keeps
its roughly 3 GB per-layer token embedding table in host RAM. Only selected
rows are transferred to the GPU; the projection and decoder remain on their
mapped devices. Device mapping excludes that table from the GPU reservation.
The transfer latency and combined ASR memory use still need physical measurement.
The existing Gemma 4 decoder handles PLE,
proportional RoPE, shared KV, final logit softcapping and streaming. Without a
projector the loader explicitly loads only the text model. DFlash captures
the configured post-layer states (Bolt: 5, 11, 17, 23, 35), keeps prompt and
verification captures separate, and appends only the anchor and accepted
verification rows to its context. Each sequence has an independent drafter
context. Bypassed sequences become dormant; finished sequences are released.

The first Gemma 4 DFlash implementation uses eager execution. Target decode
graphs and the shared-KV prefill tail optimization are disabled while its
layer taps are active. The existing Google Gemma 4 MTP path is preserved.
All production inference remains Rust/Candle.

## Build and first qualification

On a Pascal machine with CUDA 12.9 and GCC 14:

```bash
CUDA_COMPUTE_CAP=61 NVCC=/opt/cuda/bin/nvcc NVCC_CCBIN=gcc-14 \
ALLOW_LEGACY=bf16,fp8 \
cargo build --release --locked -p mistralrs-cli --no-default-features \
  --features cuda-legacy-bf16,cuda-legacy-fp8,cuda-legacy-fp4,cudnn
```

Start with the target alone, one sequence, and a bounded KV cache:

```bash
MISTRALRS_GEMMA4_PLE_CPU=1 ./target/release/mistralrs serve \
  -m nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB \
  -f Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf \
  --dtype f16 --max-seqs 1 --max-seq-len 4096 --pa-context-len 4096
```

Then restart with the same arguments plus:

```bash
--mtp-model nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4 \
--mtp-n-predict 7 --mtp-draft-sampling greedy
```

Use `chat_template_kwargs` with `enable_thinking: false` and
`conversational_mode: true` for initial voice-like requests. The normal server
tool-call parser and streaming endpoint remain available. For reproducible
measurements, download the revisions recorded in `sources.json` and supply
local target/drafter paths. The convenience repository commands above follow
the remote default revisions.

Keep the working ASR provider's `RAYON_NUM_THREADS=1 CANDLE_NUM_THREADS=1`
settings. Full-model target/drafter inference and combined ASR/LLM memory and
latency need measurement on the physical GPU; unit tests and a CUDA compile
do not establish these results. Native W4A16 expansion reserves the expanded
floating point weight footprint before device mapping, including the draft
output head. That reservation is conservative if draft ISQ is requested.

## Qualification and integration

`nemotron-bolt-dflash.yml` checks the CLI on CPU, the real NVFP4 loader using
small native checkpoints, Gemma 4 capture/decoding contracts and the existing
speculative tests. A separate hosted CUDA 12.9 job compiles the complete
legacy CLI and tests for sm_61; it executes no GPU instructions.

Promotion to Extended requires the physical sm_61 build and a real-checkpoint
comparison with/without the drafter: greedy outputs, sampling behavior,
cache reuse and cancellation, thinking modes, tool calls, TTFT, accepted
drafts per verification, decode throughput and peak memory. The combined
half-duplex qualification adds the ASR and TTS residents and measures the
delay from confirmed end of speech to the first audible answer. Source and
Candle revisions must match across the candidate and its proofs.

The model metadata in the Rust fixtures comes from the pinned NVIDIA files;
it contains no weights. CPU fixture parity covers native NVFP4 decoding,
not an end-to-end qualification of the published full model. See
`sources.json` for pins and qualification status.
