# DKR R1.25 — Cudaforge cache misses and quantified SM61 preparation

**Status:** source audit and staged/preflight tools only. Do not launch an
unapproved NVCC build or claim quantized GPU parity.

## 1. What the previous 49/49 and 15/15 warnings actually mean

In cudaforge 0.1.6, \`KernelBuilder::build_lib\` loads
\`.cudaforge_cache.json\` from the **builder's out_dir**, then creates
\`compile_jobs\` only for individual kernel sources whose
\`BuildCache::needs_rebuild\` returns true. The warning
\`Compiling N of M kernels\` prints \`compile_jobs.len()\`, not cache size.
Therefore:

- \`mistralrs-quant: Compiling 49 of 49 kernels\` means **49 CUDA
  translation units were selected to run nvcc**.
- \`mistralrs-paged-attn: Compiling 15 of 15 kernels\` means 15 selected.
- \`mistralrs-core: Compiling 0 of 15 kernels\` indicates 0 source units
  selected; relinking an archive is possible even with zero compiles.

The cache key is a composite \`source_path:object_path\`. The predicate checks
whether the \`.o\` exists and whether cached GPU architecture, args hash,
watched-header hash, source SHA256 and object path match. A mere cache-index
file is **not** a cache-hit proof.

### Likely invalidation mechanisms requiring local evidence

Both CUDA build scripts explicitly supply a **whole-directory**
\`.watch(["kernels"])\` in quant or \`.watch(["src/cuda"])\` in paged attention.
Cudaforge hashes all \`.h\`, \`.cuh\` and \`.hpp\` files in those directories
into a single watch hash, used for every compiled \`.cu\` within the crate.
A single watched-header edit can invalidate every object, even if many CUDA
units never include it.

Both crates also inject a *global* header-hash
\`-DMISTRALRS_*_CUDA_HEADER_HASH=...\` into every NVCC invocation. This
causes the **args hash to change for all units** when the header tree hash
changes. Any CUDA architecture, feature-derived macro or external
dependency include-path change can similarly invalidate every object.

A persistent cache at
\`/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61\`
is also shared across worktrees unless given separate roots. Switching
between \`extended\`, \`legacy\` and \`r125\` source trees can overwrite the
*same* cached object identities with different content/flags. This is a
**hypothesis**, not a proof of what happened during the previous run.
We need cache snapshots before and after a branch switch to identify the
specific cause. No automatic cleanup is appropriate.

## 2. No-build diagnostics implemented

Read-only script:
\`tools/dkr_r125_cudaforge_cache_audit.py\`.

\`\`\`bash
cd /home/np/dkr-worktrees/r125-candle-integration
export MISTRALRS_CUDA_BUILD_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61

python3 tools/dkr_r125_cudaforge_cache_audit.py \
    --cache-root "$MISTRALRS_CUDA_BUILD_ROOT" --source-root "$PWD" \
    --snapshot "$PWD/target/dkr-r125-cache-before-next-worktree-switch.json"
\`\`\`

The JSON snapshot includes per-entry cached and current source hashes,
watched-header hashes, argument-hash fingerprints, GPU architecture and
object presence. \`--compare SNAPSHOT\` compares those records with a later
read-only measurement. The script never calls Cargo, nvcc or ptxas.
\`--require-clean\` conservatively blocks if the cached state is already
stale. **The audit cannot reconstruct next-build NVCC flags; a clean report
does not guarantee a Cargo cache hit.**

Do not erase or move the existing \`target/dkr-*\` or Cudaforge caches.
Avoid changing \`CUDA_COMPUTE_CAP\` to compensate for a cache mismatch.
Prefer physically separated persistent object caches for repeated
Legacy / Extended / DKR cross-branch A/B runs **after** establishing a
baseline and calculating the additional disk usage.

## 3. Quantized GEMV/GEMM existing operation coverage

The R1.25 code already contains actual production adapter implementations,
with real CUDA numerical tests in \`mistralrs-quant\`:

| Consumer | Operation | Existing test | Supported fixtures |
| --- | --- | --- | --- |
| \`quantized_gemv_dynamic.rs\` | \`quant.fp8.blockwise.gemv.b128.v1\`, \`quant.mxfp4.gemv.{plain,bias}.b128.v1\` | \`dkr_quantized_gemv_gpu\` | F16/BF16, decode M=1,3, FP8 direct weights and MXFP4 packed nibbles |
| \`quantized_gemm_dynamic.rs\` | \`quant.fp8.blockwise.gemm.t16.v1\`, \`quant.mxfp4.gemm.{plain,bias}.t16.v1\` | \`dkr_quantized_gemm_gpu\` | F16/BF16, prefill M=17, FP8 direct weights and MXFP4 packed nibbles |

Manifest families: \`quant-gemv\` and \`quant-gemm\`, **6 typed profiles
each** = 12 manifests. The files may already exist in the persistent SM61
PTX package directory. **Do not rebuild them blindly.**

Read-only source and artifact gate:

\`\`\`bash
python3 tools/dkr_r125_quant_manifest_preflight.py \
    --cc 61 --root /home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61
\`\`\`

The script checks the ABI, operation, PTX symbol/SM target, SHA256, and
optional CUBIN pair's SHA256 and ELF SM flag. Missing manifests report
\`DKR_R125_QUANT_PREFLIGHT=BLOCKED\` rather than compiling. They must be
generated explicitly in a controlled, single-target, cache-aware build.

## 4. New but unqualified feature + deferred test gate

\`mistralrs-core/dkr-r125-quant-typed-pilot\` enables only these two
quantized adapters plus Candle typed transport. It enables generic
\`candle-core/cuda-legacy-bf16\` and \`candle-core/cuda-legacy-fp8\`
because test fixture conversions to BF16/FP8 need Candle generic
tensor kernels on SM61, which are otherwise absent.
**These are not feature flags on \`mistralrs-quant\` or \`mistralrs-paged-attn\`**,
and they do not enable native S31 or its sealed static fallback.

The new \`tools/dkr_r125_quant_real_sm61.sh\`:

1. Runs both read-only preflights; refuses a stale cache, missing manifests
   or missing cached objects.
2. Does **not run any Cargo or GPU tests by default**.
3. Only when \`DKR_APPROVE_R125_QUANT=1\` is provided: verifies Cargo feature
   closure, pairs the existing two families with CUBIN via \`ptxas\`,
   persists them under \`target/dkr-r125-quant-pilot\`, then performs the two
   existing ignored numeric tests using a **single**
   \`cargo nextest --release --locked\` run.
4. Requires real Candle CUBIN loads and rejects image-mode fallback.

An approval may still cause Candle's **generic FP8 compatibility kernels**
to compile once, and may relink the Mistral host. This has **not** yet
been executed or numerically qualified. A future quantitative baseline
must retain model logits, quantization format, clocks, VRAM, GPU timing,
and headless-relogin A1/B/A2 for the comparison to Extended.

## 5. Acceptance sequence

- First obtain the read-only cache audit and 12-manifest preflight result.
- Inspect any source/header/object mismatch **before** changing native
  build scripts, then snapshot exact cache source + args fingerprints.
- Only after preflight passes, explicitly approve the 2-test quant pilot;
  require 2/2 PASS and verified real CUBIN loads.
- Then integrate other S1–S8 production families under separate gates,
  without conflating selected numerical fixture proof with model inference
  or end-to-end performance certification.

## 6. Incident 2026-11-23: quantized GEMM test did not terminate

On user's SM61 GTX 1080 Max-Q, the 12-manifest read-only preflight and
Cudaforge cache preflight passed. The subsequent Cargo release build completed
in 7m25s. **Actual CUDA recompilation counts** were:

- `mistralrs-quant`: **0 / 49**;
- `mistralrs-paged-attn`: **0 / 15**;
- `mistralrs-core`: **3 / 18**.

The Nextest worker `quantized_gemm_dynamic::tests::dkr_quantized_gemm_gpu`
reported repeated `SLOW` at 11,520 seconds (over three hours). No numerical
success/failure proof exists for the quantized pilot. This is an
**execution-time stall**, not a slow NVCC compile, and a prolonged test must
be stopped rather than awaited. Pending a process-level diagnostic, the stage
is unknown (input conversion, CUBIN loader/dispatch, GPU readback, etc.).
No claim is made that GEMV passed: a serial Nextest run may not have reached
that worker before GEMM stalled.

**Additional audit pitfall:** a cache comparison later reported 0/0 entries
across all crates because `MISTRALRS_CUDA_BUILD_ROOT` had not been re-exported
in that terminal. This was a false comparison against the wrong root, not
evidence of deleted cached objects. The audit tool now rejects empty and
relative paths.

**Recovery / safeguards committed:** `.config/nextest.toml` now defines
`dkr-r125-quant` as an isolated Nextest profile with
`slow-timeout = { period = "20s", terminate-after = 6, grace-period = "10s" }`.
The quant pilot uses `--profile dkr-r125-quant` and `tee` to stream build and
test output live. GEMM test diagnostics mark the precise stage (dtype, input
cast, FP8 cast, dispatch return, GPU readback, MXFP4 dispatch and readback).
These mitigations are **source-only**, not yet GPU-executed. They do not change
kernel algorithms, PTX artifacts, `CUDA_COMPUTE_CAP`, or Cudaforge caches.

**Next action:** safely terminate the outstanding Nextest/test process, verify
no lingering test worker, inspect GPU state, and review the existing log before
any bounded reproduction. Avoid running the previous unbounded Nextest binary
again. A failed/timeout quantized test must not be promoted to a DKR PASS.

## 7. Follow-up (2026-11-23): GEMM numeric PASS; count-based evidence gate false negative

The bounded GEMM-only Nextest run
`b4bab107-3049-477b-84d4-40202077db45` (branch `214c0247`,
Candle `d40ac1b4`) finished normally:

- `dkr_quantized_gemm_gpu` **1/1 PASS in 0.230s**;
- FP8 blockwise and MXFP4, F16 and BF16, plain/bias where applicable;
- stage markers cover conversion, typed launch return and successful GPU
  readback for all variants;
- the device loaded one verified SM61 CUBIN image, sha256
  `3017abe46a75675321e5ef5beb8092de3c582999cbc26ebb491025fdfee3affb`,
  no PTX fallback;
- the original wrapper subsequently printed
  `DKR_R125_QUANT=FAIL insufficient CUBIN loads (1)`. **This is a
  wrapper/evidence false negative, not a numerical CUDA failure.**
- Native NVCC build objects: quant **0/49**, paged-attn **0/15**, core
  **0/18**, despite host release build taking 4m11s.

### Root cause verified in the code

`kernel-packages/cuda/quant-gemm/build.py` emits six different typed
manifests with different entry names **from the same quant_gemm.ptx**.
All six variants share one CUBIN image, and Candle's
`candle-core/src/cuda_backend/dkr_typed_image.rs` caches modules by
verified PTX artifact key and optional CUBIN SHA. It prints
`CANDLE_DKR_TYPED_IMAGE_LOADED` **only on module admission**, not whenever
a different entry/function of that same module is used.

The old pilot `min_loads=2` for GEMM-only (or GEMV-only) and `min_loads=4`
for both was incorrect. It conflated the number of CUDA entry functions
with the number of unique loaded image modules.

### Permanent fix

`tools/dkr_r125_quant_evidence.py` now independently checks existing
Nextest logs against the six per-family manifest CUBIN SHA256 identities
and hashes of the on-disk image bytes. It requires:

- Nextest numerical PASS for each selected family;
- an actual `format=cubin target=sm61` load with the **expected SHA**;
- no PTX load or fallback;
- the exact strict-CUBIN policy banner.

Because GEMM-alone has already passed, requalifying it is unnecessary to
correct the wrapper record. Evidence can be rehabilitated **offline**:

```bash
python3 tools/dkr_r125_quant_evidence.py \
  --root target/dkr-r125-quant-pilot/sm61 \
  --log target/dkr-r125-quant-pilot/r125-quant-gemm.log \
  --scope gemm
```

`tools/dkr_r125_quant_real_sm61.sh` uses this validator after each run,
rather than a hard-coded module-load count. Offline regression tests are
staged in `tools/test_dkr_r125_quant_evidence.py`. GEMV remains **unqualified**;
do not label the whole quantized pilot PASS until its numerical Nextest test
also succeeds. The earlier multi-hour hang is **not reproduced** by the
bounded, live-traced GEMM-only run; its original cause remains unknown.
