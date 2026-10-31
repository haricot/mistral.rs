# Reusable real-model LFM3 qualification

This supplementary workflow qualifies the original immutable candidates from
Legacy 41080696084 and Extended 41080753574. It never reruns their successful
physical kernel jobs, rewrites their reports, or rebuilds the CUDA environment.
The compiler image remains pinned to its original registry manifest digest.

`lfm3-model-sm61-qualification.yml` restores and hashes the original bundles,
checks hosted and physical job identities, rechecks Candle physical proofs, and
compiles two additional CLI executables using the prior hosted compilation and
CUDA caches. Cargo check is mandatory for both executables. No local compiler
is needed and no model weights are downloaded on GitHub.

One dedicated `mistral/extended` runner executes all tests sequentially. Models
and per-file SHA256 hashes are pinned in `models.json`: official LFM3-1.7B,
LFM3-VL-2B, and LFM3-1.7B-Q4_K_M.gguf. Model directories are mounted read-only
using the runner's existing opt-in three-repository HF policy. Existing labels,
attempt checks, ASD isolation, and token transport remain unchanged.

For each final executable, Text, image-based VL and GGUF load real weights and
execute CPU and CUDA inference on the same fixed input. This short compatibility
qualification uses F16, a 2048-token context cap, four greedy output tokens, five
top token logprobs, and absolute logprob error at most 0.05. Generated token divergence fails the test. At least four of five top tokens
must agree. A single boundary replacement is allowed only when each unmatched
token has probability at most 0.001 and is within 0.05 logprob of its list's
cutoff; all common-token logprob errors must still be at most 0.05. CPU references and CUDA results are preserved as raw
JSON, rehashed, and rechecked before promotion. This is an initial acceptance
policy, with physical measurements still pending; no model quality benchmark,
full-context coverage, VL video coverage or performance claim is made.

Full model evidence is cached by executable source inputs, Cargo dependencies,
features, Rust toolchain, the CUDA image, harness, fixed model and numeric policy,
and ASD profile hashes. Documentation and orchestration metadata do not alone
invalidate model execution. Reused reports retain their actual execution run,
candidate, GPU and driver; reuse does not fabricate a new execution. Original
per-candidate CPU and physical kernel proofs remain mandatory.

With the explicit `collection_only=true` workflow input, all six model pairs
are attempted even after a numerical mismatch or a model server error. Each
device's raw JSON and log is retained, along with an incremental `report.json`
and a final `summary.md` also visible in the GitHub job summary. Metrics separate
greedy token identity, generated token logprob error, common top-token logprob
error and absolute probability error. Positions after a greedy divergence are
marked as having different prefixes and cannot calibrate parity.

A complete collection can finish successfully while its numeric verdict is
FAIL under the unchanged 0.05 threshold. Its report status is COLLECTED and its
artifact prefix is `lfm3-model-measurements-`, never a promotable model proof.
Execution errors or missing numeric references still fail the workflow after
the remaining cases have been attempted. Both the local launcher and promotion
verifier refuse collection-only evidence, including a collection whose six
numeric comparisons happen to pass. Acceptance is evaluated from the retained
measurements separately; this mode does not adjust the threshold automatically.

The normal Legacy and Extended promotion jobs also restore complete model
evidence automatically when their final runtime fingerprint matches. Original
candidate-specific CPU and CUDA proofs still have to pass. A missing or stale
model proof refuses publication; it never becomes a kernel-only validation.
Legacy retains its existing automatic promotion policy after all gates pass;
Extended retains its explicit dispatch policy. Model artifacts are retained
for 90 days, and unavailable evidence cannot satisfy the gate.

The updated runner supplies these commands, from its extracted directory:

```bash
./lfm3.sh check
./lfm3.sh start
./lfm3.sh promote
```

Missing pinned snapshots require the separate explicit `./lfm3.sh fetch-models`
command, using an installed Hugging Face `hf` CLI. Tests remain offline. The
normal four profiles and `morning.sh` retain their original behavior; do not
restart completed Mistral campaigns to obtain these supplementary proofs.

`lfm3-model-sm61-promote.yml` is an explicit subsequent dispatch. It requires a
successful dedicated qualification, verifies the model artifact's GitHub digest
and raw numerical references, checks source equivalence for reuse, and invokes
the existing promotion code with the original CPU/GPU archives and run IDs.
All source and Candle leases, SHA256 checks, archival tags, ASD gates and atomic
pushes remain in effect. Missing initial model evidence still refuses promotion.

Local validation:

GGUF uses its embedded tokenizer fallback instead of forcing the Text
`tokenizer.json`. The local Text snapshot remains the explicit base-model
asset directory; upstream vocabulary checks remain enabled.

The collection-only `resume_measurements_run` input can reuse the established
Text/VL observations from the pinned measurement harness revision. Preparation
verifies the original workflow, hosted jobs, artifact digests, model policy,
candidate identities, ASD hashes and raw references. Hosted compilation must
produce byte-identical executables before a GPU job becomes eligible. Reused
cases retain their original run, GPU, driver and archive digest in the report;
only GGUF runs again. This does not authorize promotion or change tolerance.

```bash
python3 -B -m unittest discover -v -s .github/lfm3/tests
python3 -B -m unittest discover -v -s .github/campaign/tests
```


