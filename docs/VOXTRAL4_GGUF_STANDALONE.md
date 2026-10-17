# Voxtral4 GGUF — standalone implementation and evidence

Branch: voxtral4_gguf
Base: haricot/mistral.rs master at 28ed6daca455b4ae063896014d1ea698b912c539
Model: cstr/voxtral-mini-4b-realtime-GGUF
File: voxtral-mini-4b-realtime-q4_k.gguf

## B0 — structural binding contract

- Recognize GGUF general.architecture voxtral4 and voxtral4b without
  advertising either as a normal text-only architecture in --arch.
- Validate bounded, explicit Voxtral audio and LLM layer counts.
- Map CrispASR names to the native Voxtral encoder, projector, embedding,
  decoder, and adaptive-RMSNorm weight names.
- Fail on missing required tensors instead of silently accepting them.
- Isolate changes on a standalone master-derived branch.
- Run model-free CPU tests before advancing.

## What B0 does not do

This branch DOES NOT YET transcribe from GGUF. The original native Voxtral
loader expects params.json and native audio/streaming processing. CrispASR
stores its Tekken vocabulary in tokenizer.tekken.vocab_tensor as floats,
and its Slaney mel filters and Hann window as tensors. The GGUF Q4_K data
must be connected to the native pipeline without losing the audio/token
alignment, RoPE conventions, or adaptive norm behavior.

## Required gated milestones

B1: Test real GGUF tensors and metadata for both architecture aliases.
Check shapes and converter-specific Q/K conventions.

B2: Decode Tekken vocabulary, restore GGUF mel filters/window and
synthesize Voxtral native configuration from GGUF metadata.

B3: Wire GgufWeightSource quantized Q4_K weights, model loader and
streaming audio processing through the native Voxtral path.

B4: Quantified parity on a common French WAV with CrispASR, then hosted
CUDA compilation and physical Pascal SM61 run (accuracy/latency evidence).

B1 (2026-10-17): real Q4_K header PASSED all 714 names, shapes checked,
and 303 F32 + 2 F16 + 409 Q4_K. Inventory fingerprint
775c2b3010a52d3a11d61f89b62ab2124c3fd4f29366a1e648736552f7a2dc9e.
This fingerprint is not a whole-file content digest.

B2-A: tools/voxtral4_gguf_assets.py decodes the converter's packed
150000-entry Tekken BPE byte stream, all 1000 special-token strings, and the
original F32 Slaney filterbank/Hann window from their GGUF tensor offsets,
without fetching or decoding the 2.52 GB of Q4_K model weights. The tool
records SHA256 digests and validates the 400-sample periodic Hann window.
It does NOT prove tokenizer ID mapping: 150000 BPE entries vs 131072 model
embedding rows remains an explicit unresolved blocker.

    python3 -B tools/voxtral4_gguf_assets.py "$MODEL" --report voxtral4-b2-assets.json

Only add to Mistral Extended after completing B2/B3/B4 with real forward
and physical SM61 parity. The B1 header pass alone is not an inference pass.

## Model-free test

    cargo test --locked -p mistralrs-core --lib gguf::voxtral4

## Final target command (NOT YET SUPPORTED)

    ./target/release/mistralrs serve \
      -m cstr/voxtral-mini-4b-realtime-GGUF \
      -f voxtral-mini-4b-realtime-q4_k.gguf
