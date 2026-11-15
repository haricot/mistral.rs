# S6.8.4-A — Full model native-provider parity
S683 passed: 349 cases, 346402 compared values, 343903 exactly equal,
51 CPU/GPU dequant checks (13021184 elements), 33071 CPU dot samples per
leg, 518 static MMQ and 180 static MMVQ calls, 2836 native submissions,
423 MMQ plan hits. Seal against the exact artifact digests.

Use a local single-file text GGUF with embedded tokenizer/chat template,
small enough to stay on one SM61 GPU. Prefer a dense instruct Q4_K_M
1–4B model already cached (no automatic download). Optional local
chat template override. Build a single Rust SDK example in the existing
debug Cargo target. This is correctness, not a performance benchmark.
The typed Mistral DKR configuration is fixed in both legs; only Candle
native root changes. No claim of purely static Mistral A vs fully
external B. Raw last-vocabulary-row logits, prompt IDs and deterministic
24-token generation with logprobs, repeat twice for three prompt lengths.
Fixed prospective logit thresholds: RMS <=0.05, max <=0.25, cosine >=0.9999,
same argmax; generation token sequence identical and chosen-token logprob
delta <=0.1. Near ties or other divergence require review.
Default contiguous KV cache; optionally DKR_S684_PAGED=1 for an independent
paged run if supported. Both use the same binary in separate processes.
Manual device placement avoids CPU layer offload; no model config changes.
No production promotion. Default/no-DKR build, distinct Driver contexts,
broader model matrix, performance remain open.

    export DKR_S6_ORIGINAL_REPO="$HOME/tmp/mistralrs_provider/mistral.rs"
    export DKR_S684_GGUF=/absolute/local/model.gguf
    bash tools/dkr_r122_s684_model.sh --preflight
    DKR_APPROVE_S684=1 bash tools/dkr_r122_s684_model.sh

The example uses existing APIs; it still needs compilation and local GPU
execution. Model raw output may contain model-specific unsupported shapes
or logprob semantics; fail rather than relax the qualification silently.
