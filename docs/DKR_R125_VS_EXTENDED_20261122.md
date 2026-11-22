# DKR R1.25 vs Extended: Candle first, then Mistral.rs

**Code audit:** 2026-11-22. This is a source/branch comparison, **not** a speed or GPU-model qualification.

## Immutable comparison anchors

| Project | Extended baseline | DKR working branch | Git divergence (Extended...DKR) |
| --- | --- | --- | --- |
| Candle | `haricot/candle:extended` at `86b3ef0a17bd` | `feature/dkr-r125-typed-image-transport` descended from `feature/dkr-r124-neutral-catalog` | 132 DKR-only / 25 Extended-only commits |
| Mistral.rs | `haricot/mistral.rs:extended` at `3966d0ad0ba5` | `feature/dkr-r125-sm61-cubin` | 181 DKR-only / 63 Extended-only commits |

Divergence counts describe Git commit history only. Both branches have code absent
from the other. The diff touches many files; committing an unreviewed whole-tree
merge is **not** a valid preservation strategy.

## 1. Candle DKR vs Candle Extended

| Domain | DKR R1.25 branch | Candle Extended |
| --- | --- | --- |
| GPU image producer | Neutral Cudaforge catalogue with architecture-owned recipes and an attested native catalogue; qualified embedded PTX/CUBIN | ASD external modules including PTX and CUBIN, managed under ASD-specific artifact and profile contracts |
| CUDA module runtime | `CudaDevice::get_or_load_dkr_func` (native) and `get_or_load_dkr_typed_image_func` (typed); exact-SM CUBIN and single `CANDLE_DKR_IMAGE_MODE` | `device/asd_modules.rs` with external module lookup, candidate/decision binding and invalidatable ASD runtime cache |
| Decision / tuning | Validated image admission and fixed mode, **not** a complete performance tuner | `ASD-EXACT-PROFILE-V3` and runtime extensions, exact-key decision cache, profile refresh and multiple CUDA/conv experiments |
| Stability evidence | Native SM61 R1.24 and R1.25 P1/P2 reports; typed image **scalar scale** GPU proof completed in Mistral | ASD V3 validation examples and tuning history; no proof from this comparison that DKR performance is superior |
| Native artifacts | May demand sealed native catalogue under `cuda-dkr-qualified-default`; typed-only transport now explicitly allows *no native S31 catalogue* | ASD owns separate directories, manifest identities and promotion gates |

**Conclusion:** Candle DKR's reusable *transport, provenance and cataloguing* are
the appropriate base for Mistral, but Extended's ASD *decision, evidence and
profile-refresh* algorithms are still required for a later ASD integration.
Do not transplant the entire Extended CUDA device/runtime implementation;
reconcile the interfaces and add an ASD-to-DKR decision boundary only after
physical numeric equivalence.

## 2. Mistral DKR vs Mistral Extended

| Domain | DKR R1.25 branch | Mistral Extended |
| --- | --- | --- |
| Candle dependency | `0.11.1`, pinned to Candle DKR typed transport | `0.11.0`, pinned to **`e0dbd926ae211e`** (not Candle's current `extended` head) |
| DKR ABI / packages | `mistralrs-kernel-runtime`, versioned `dkr.typed.v1`, Cudaforge PTX and optional hashed CUBIN | No equivalent generic typed runtime crate on Extended |
| Production operation adapters | Existing DKR adapters for MoE, indexed copy, gated-add, norms, dense/quantized GEMM/GEMV, FP8/FP4, paged attention and KV operations **behind explicit features**; not all end-to-end qualified | Existing inference and CUDA / GGUF / paged attention paths, plus standalone optimizations; these are valid behavioral baselines |
| Native/typed deployment | `dkr-sm61-cubin-qualified` is native + typed opt-in and may require a sealed native catalogue. New `dkr-r125-typed-real-pilot` uses typed-only Candle transport without S31/static-native coupling | Legacy/static and dedicated paths with separate performance/compatibility history |
| ASD | Historic independent F32 exact-profile proof and deferred ASD selection for typed catalogues | Integrated application history built around older Candle/ASD assumptions; not compatible with DKR ABI by simple feature re-enable |

**Important:** Mistral Extended pins Candle `0.11.0@e0dbd926`, whereas
Mistral DKR pins `0.11.1`. Compare ABI and lock resolution before porting
inference routes; never simply copy `Cargo.lock` between branches.

## 3. Real-kernel pilot now staged

`mistralrs-core/dkr-r125-typed-real-pilot` ties **three pre-existing production
operation adapters** to Candle's typed-only PTX/CUBIN loader:

1. Grouped MoE GEMM (F16 and BF16, gated and ungated).
2. Indexed KV-cache row copy (F16, BF16, F32; repeated launch-cache check).
3. Short-context paged-attention decode (F16, BF16, F32; repeated plan check).

The source tests are already marked `#[ignore]` because they launch a physical
GPU. New `tools/dkr_r125_real_kernel_sm61.sh` assembles **only these existing
PTX families** through `ptxas`, persists the selective CUBIN cache, uses
`CANDLE_DKR_IMAGE_MODE=cubin` strictly and executes the three CUDA tests with
one release-only `cargo nextest` invocation.

To avoid accidental large static builds, it requires an existing
`MISTRALRS_CUDA_BUILD_ROOT` with three cached Cudaforge indexes and an
explicit `DKR_APPROVE_R125_REAL=1`. It does not alter `CUDA_COMPUTE_CAP`.

**Evidence boundary:** The earlier scale-image physical SM61 test passed in PTX,
CUBIN, AUTO, `max_error=0`. It proves the shared transport. The real-kernel
pilot in this section has been **committed but not executed or numerically
qualified** in the GitHub editing environment.

## 4. Integration order and acceptance gates

1. **Typed real pilot:** run MoE/indexed-copy/paged-decode; require 3/3 GPU
   tests PASS, confirmed `CANDLE_DKR_TYPED_IMAGE_LOADED ... format=cubin`,
   no driver fallback, stable prevalidated plans.
2. **Remaining Mistral operation families:** gradually enable the already
   written adapters; pair existing compiled PTX to CUBIN without rebuilding
   all C++ kernels. Prioritize quantized GEMV/GEMM, FP8/FP4, routed MoE,
   paged v2/FP8 and logits. Keep explicit rollback for unsupported layouts.
3. **Extended A/B:** use identical model SHA, quantization, batch, context,
   prompt, output tokens, GPU clocks and CUDA runtime. Compare logits and
   numeric error **before** throughput, TPOT/TTFB/VRAM and cache hits. For
   each tested case record whether Extended used static/ASD or DKR external
   CUBIN.
4. **ASD adaptation:** port the exact-key profile/evidence principles from
   Extended, but have ASD select a *verified DKR implementation identity*,
   not load or compile CUDA artifacts independently. Preserve immutable
   snapshots, invalidation semantics and producer attestation.
5. **Merge criteria:** do not replace either `extended` branch until model
   parity, decoder/quantized-path coverage, fallback policy, memory pressure
   and performance regression thresholds are independently demonstrated.

## Quick branch comparisons (read-only)

```bash
git -C /path/to/candle fetch origin extended feature/dkr-r125-typed-image-transport
git -C /path/to/candle log --oneline --left-right --cherry-pick \
  origin/extended...origin/feature/dkr-r125-typed-image-transport

cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin extended feature/dkr-r125-sm61-cubin
git log --oneline --left-right --cherry-pick \
  origin/extended...origin/feature/dkr-r125-sm61-cubin
```

This report is based on current source inspection and Git compare metadata,
not on executing a branch merge or benchmark.

## 5. Deferred headless-relogin performance comparison

**Do not run this benchmark until real typed operations and model logits are
qualified.** Use the established `headless-relogin` v0.1.4 workflow with
`--with-after-run-openlog` after closing the desktop session, and remove
`DISPLAY` / `WAYLAND_DISPLAY` from the benchmark command.

Candidate common workload: the exact same
`LiquidAI/LFM3-1.7B-Q4_K_M.gguf` file (record its SHA256), same tokenizer,
GGUF route, prompt tokens, prefill and decode lengths and seeds on GTX 1080
Max-Q (SM61). Before performance analysis, compare final logits/numeric
parity and report *actual* provider decisions, not merely enabled features.

Benchmark **two different questions separately**:

- Same-host **DKR CUBIN vs existing static/legacy path** under the same pinned
  Candle revision (best isolation of loader/dispatch overhead).
- Cross-branch **DKR vs Mistral Extended**, noting Extended's older Candle
  version and feature history (system-level comparison, **not** a controlled
  causal estimate of DKR overhead). Optionally include Candle Extended on
  matching synthetic operations in an independent producer comparison.

Use alternating A1/B/A2 run order and at least three paired rounds for
stabilized results; preserve separate cold module-load and steady-state
measurements, with warmups and synchronizations. Collect TTFT/TTFB, TPOT,
throughput, p50/p95 latency, peak/resident VRAM, cache/plan hit and fallback
telemetry, driver/CUDA versions, GPU P-state, clock and temperature. Exclude
runs with thermal throttling or mismatched provider/quantization routes.

Store immutable input manifests, executable/source/artefact hashes, JSON
results, and complete logs. Keep all builds strictly `--release`, use the
existing persistent Cudaforge cache, and do not change global
`CUDA_COMPUTE_CAP` just to compare architectures. This is a future campaign,
not permission to run current tests in the background.
