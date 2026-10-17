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

Only add to Mistral Extended after passing B1 through B4. Never promote
or advertise GGUF inference from the B0 inventory-only milestone.

## Model-free test

    cargo test --locked -p mistralrs-core --lib gguf::voxtral4

## Final target command (NOT YET SUPPORTED)

    ./target/release/mistralrs serve \
      -m cstr/voxtral-mini-4b-realtime-GGUF \
      -f voxtral-mini-4b-realtime-q4_k.gguf
