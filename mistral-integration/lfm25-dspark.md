# LFM2.5 DSpark in the Extended seed

The Extended campaign checks out `asd_runner`, composes a candidate with the
validated Candle Extended pin, and promotes `extended` only after its CPU,
hosted CUDA compilation, and physical SM61 gates pass. Native LFM2.5 DSpark
must be present in `asd_runner` to survive this composition.

The complete standalone payload is `d3813494903e5cfca01e12df596f3fe14a5d22ba..7501527eb3633e9ea020a9b6559d11c6634287d0`
from `lfm25_dspark`. Its six commits include the native proposer, documentation,
nine recurrent checkpoint lanes for eight draft tokens, materialized gather
metadata, multi-token query geometry, and causal mask normalization. The source
pin is recorded in `sources.json`.

The integration starts from Extended `71c0e13021a4544c1b475eff136f807e42bfe91c`,
with Candle `f5ce7f6f8b7a68d14a55419f9f221e1ae3537fd3`. That earlier Extended
qualification did not contain DSpark and does not qualify DSpark inference.
The next campaign supplies its own candidate SHA and campaign identifier.

Build for a CUDA 12.9 SM61 host with GCC 14:

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

For an HTTP server, use `serve` with the same model and MTP options, omit
`--input`, and send requests to its API. Both modes require a rebuilt binary
that contains the native LFM2 speculative implementation.

Keep PagedAttention enabled and use the default static speculation schedule.
The native checkpoint has a block size of nine, including the anchor token;
its maximum draft depth is eight. LFM2 DSpark does not support ragged
speculation scheduling.

Offline regression coverage includes the native checkpoint configuration,
Markov token conditioning, recurrent checkpoint reservation and indexing,
gather metadata, multi-token decode geometry and masks, and CLI parsing.
The physical SM61 smoke tests cover device operations; successful loading of
both models and generated output with DSpark require a separate model test.
