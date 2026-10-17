# LFM2.5 DSpark standalone

This branch starts directly from the fork's pinned `master` commit
`20f607a77abd4738aa5594ee93a393a9fd284605`, the same Mistral.rs 0.9.2 baseline
as the other standalone feature branches. Current `master` has advanced to
Mistral.rs 0.9.4; porting to that newer baseline requires separate work.

The functional payload contains the six native DSpark commits through
`7501527eb3633e9ea020a9b6559d11c6634287d0` and the CUDA SM61 prerequisites.
These include pre-load reservation of nine recurrent checkpoint lanes for
eight drafts, gather metadata, multi-token query geometry, causal masks,
native Markov token conditioning, and ShortConv state rollback.

The standalone commit has only the pinned master commit as its parent.
Its source and compatibility pins are recorded in `sources.json`.

Candle is pinned to `haricot/candle:extended` at
`f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3`. CUDA compatibility includes the
legacy BF16/FP8/FP4 feature forwarding, Pascal fallbacks, and the corresponding
checkpoint-loading support. A model run does not require an ASD store.

Build on a CUDA 12.9 SM61 host with GCC 14:

```bash
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=/usr/bin/g++-14 \
cargo build --release --locked -p mistralrs-cli --no-default-features \
  --features cuda-legacy-bf16,cuda-legacy-fp8,cuda-legacy-fp4
```

Run a single prompt:

```bash
./target/release/mistralrs run -m LiquidAI/LFM2.5-2.6B \
  --mtp-model LiquidAI/LFM2.5-2.6B-DSpark \
  --mtp-n-predict 8 --mtp-draft-sampling greedy --dtype f16 \
  --input "Describe the model's agent capabilities."
```

For an HTTP server, use `serve` with the same model options and send prompts
to its API. `--input` is a `run` option. Keep PagedAttention enabled and use
the default static speculation schedule; native DSpark supports at most
eight draft tokens for this checkpoint's block size of nine.

The offline unit tests cover native DSpark configuration, Markov conditioning,
recurrent state reservation and indexing, speculative metadata, and CLI
parsing. CUDA attention tests, the standalone SM61 device smoke test, and
actual LFM2.5 + DSpark generation require local CUDA validation. Earlier
Extended qualifications do not qualify this standalone branch.

The branch is independently testable. Future composition into Extended is a
separate integration step.
