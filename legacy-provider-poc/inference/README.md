# F32 provider inside Mistral inference

This lot connects the external F32 ABI to `UnquantLinear`, then runs the normal
Mistral SDK loader, pipeline, scheduler and engine on a generated local Llama
checkpoint. It performs raw-logit inference and deterministic four-token chat
completion. The one-layer checkpoint has 32 hidden units, 64 intermediate units
and 32 vocabulary entries. It is a functional fixture, not a trained language
model or a performance benchmark. No model download is needed.

Candle remains pinned to `9cba2e898af3d25a89257f1f2ec5acece0404dcb`.
Only `cuda` is required. No BF16, FP8, FP4 or tuner feature is needed.

## Runtime and Mistral connection

`mistralrs-kernel-runtime` owns the shared F32 contract, PTX verification/loading,
minimal profile adapter and immutable per-layer plans. The earlier standalone
executable reuses this library rather than maintaining another CUDA executor.

A CUDA F32 `UnquantLinear` snapshots the explicit profile during construction.
It resolves the required modules on its Candle device and keeps exact row-count
plans. Forward flattens batch dimensions, normalizes layout/offset, executes the
selected module, adds bias after the F32 matrix product and restores the output
shape. Selected layers do not expose raw weights through optimization hooks
which would bypass their executor. Delta-weight reconstruction prepares a new
plan through the same constructor.

Without the environment variable, existing selection remains active. CPU,
Metal, non-F32 and stacked expert weights are outside this contract.

## Minimal manual ASD profile

Set `MISTRALRS_ASD_F32_PROFILE=/absolute/path/A.profile.json` before loading the
model. Configuration is snapshotted once per process. Switching profiles uses
a new process with the same executable. There is no tuner or automatic search.

The JSON schema is `ASD-LINEAR-PROFILE-V1`, with:

- `profile_id`, `abi=mistral.linear.f32.v1`, `purpose=validation_only`, `target_cc`;
- exact decisions binding `[M,N,K]`, decision id and opaque implementation id;
- a relative module-manifest path, its SHA256 and the PTX SHA256.

Each decision receives a canonical identity hash. Duplicate exact keys/ids,
ABI, target, manifest/artifact identity and resource incompatibilities fail
closed. A configured dense F32 layer or row count without a decision is an
error; it cannot silently use a built-in GEMV/cuBLAS path.

This is an explicit linear adapter following ASD's exact-decision and identity
model. It is not a parser for Candle's convolution profile wire format, and does
not modify or promote the production ASD profile/store. A new operation ABI
would need its own runtime support.

Profile, manifest and PTX reads/hashes happen during preparation. Forward uses
in-memory exact lookup, a cached CUDA function and atomic submission counters.
It does not consult files or environment variables. Module launch contracts are
cached by Candle device identity and manifest hash; Candle modules themselves
remain content addressed by PTX hash.

## Validation

From the repository root, Rust 1.99 and Python 3.11+:

```bash
python3 legacy-provider-poc/inference/run.py cpu
```

This runs runtime/parser tests, existing UnquantLinear CPU regressions, the actual
QuantMethod operator contract and a full local Mistral CPU inference. CUDA is not
initialized. `cargo check` and formatting checks are included.

On the physical SM61 machine, CUDA 12.9 in PATH:

```bash
set -o pipefail
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 \
  python3 legacy-provider-poc/inference/run.py gpu 2>&1 | tee /tmp/provider-f32-inference.log
```

The script builds the inference executable once, then separately builds the two
PTX implementations and their manual exact profiles. The same executable runs:

1. CPU reference and normal CUDA reference, without an external profile;
2. profile A, profile B, then profile A2 replay, each in a fresh process;
3. operator tests through QuantMethod, with/without bias, batch, stride and offset;
4. logits and four generated tokens through the normal Mistral engine;
5. a missing exact decision which must fail instead of falling back;
6. A2 after deleting its isolated profile, manifest and PTX following model load.

The reviewer checks profile/decision/artifact/entry identities, actual device
UUID and compute capability, submission counters for both prefill and decode,
CPU/CUDA baseline parity, generated-token/logprob parity and unchanged executable
SHA256. A2 logits must equal A logits exactly. Model comparisons use absolute
and relative tolerance 1e-4. This does not qualify model quality or performance.

Success ends with `PROVIDER_F32_INFERENCE_GPU=PASS`. `OUTPUT=...` is printed before
building. That temporary directory retains individual operator/model reports,
the generated checkpoint, profiles, artifacts, negative-case log and
`evidence.json`. A2 deletes only its dedicated replay files. No runner is
registered and the user's ASD store is not accessed.

`CARGO_TARGET_DIR` can select a persistent build cache. `cuda-compile` builds the
actual inference executable and external modules without executing on a GPU.
The hosted workflow runs CPU inference and CUDA compilation for SM61/75/80;
physical CUDA inference remains a separate operator-run qualification.
