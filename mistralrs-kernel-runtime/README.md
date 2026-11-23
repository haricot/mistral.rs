# mistralrs-kernel-runtime

Reusable runtime substrate for external, versioned CUDA kernel providers.

## Implemented: legacy F32 reference plus independent DKR-R1 loader

The original F32 execution path remains intentionally narrow:

- `mistral.linear.f32.v1` ABI and schema-1 PTX module manifests (`src/lib.rs`).
- Strict manifest, PTX-target, hash, entry-symbol, path and resource verification.
- `ASD-LINEAR-PROFILE-V1` exact F32 decisions with immutable preparation (`src/profile.rs`).
- Device/context-bound CUDA execution and content-addressed module reuse (`src/cuda.rs`, `src/session.rs`).
- No silent fallback for a selected F32 shape; no hot-path filesystem reads after preparation.
- Physical SM61 A/B/A2 proof through a local deterministic Mistral Llama fixture, archived in [PR #34](https://github.com/haricot/mistral.rs/pull/34).

`legacy-provider-poc/inference` is a **historical conformance fixture**, not the permanent runtime API. Keep it until equivalent conformance tests are established outside the demonstration directory.

## DKR-R1 independent typed module ABI

The new `src/universal.rs` provides a versioned, content-addressed PTX package
catalogue and an ordered typed tensor/scalar ABI, without importing ASD.
Under the CUDA feature, `src/universal_cuda.rs` loads an explicitly selected
verified function into Candle's context-bound CUDA module cache and validates
the launch configuration. Executable operation-specific launch adapters are
still required; their unsafe argument lifetime/order contract is not hidden
behind the manifest verifier. The original F32 path is unchanged.

See [DKR-R1 typed ABI and complete coverage inventory](../docs/DKR_R1_TYPED_ABI_AND_COVERAGE.md)
for the 196 classified CUDA source units and 44 Candle backend methods, operation families, open adapter work,
SM61 restrictions and qualification gates.


## R1.25: paired CUBIN/PTX images (SM61 qualification candidate)

The *typed* DKR catalogue now supports an optional, verified CUBIN image paired
with each PTX module. The PTX remains the source of the typed entry signature,
and the binary must have an independent SHA-256, a valid CUDA ELF64 header and
the exact target architecture. Existing PTX-only manifests remain readable.
Neither Candle nor the operation adapters contain a hardcoded Pascal dispatch.

The binary builder still generates PTX via the pinned `cudaforge` 0.1.6.
For CUBIN, `ptxas` assembles those exact PTX bytes rather than rebuilding the
CUDA C++ kernels. It then records `cubin_artifact` and `cubin_sha256` in each
typed manifest. Compilation is deduplicated by PTX, toolchain and target;
reusing existing PTX packages avoids the Cudaforge NVCC pass entirely.

```bash
# All R1.25 checks are release-only: do not create target/debug.
# Reconcile the Candle git revision in Cargo.lock with Cargo itself.
# This is dependency resolution only, with no Rust/CUDA compilation.
cargo metadata --format-version 1 > /dev/null
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_DEBUG=0
export CARGO_PROFILE_RELEASE_STRIP=debuginfo

# Run the CPU-only typed-manifest / policy gates first.
cargo nextest run --release --locked -p mistralrs-kernel-runtime --lib

# In an SM61 worktree with *already qualified* PTX packages:
cargo run --release --locked -p mistralrs-kernel-runtime --features builder-cudaforge \
  --bin dkr_package_builder -- \
  --cc 61 --pair-existing --output kernel-packages/build/cuda

# Or request a complete Cudaforge PTX build + paired CUBIN generation:
# cargo run --release --locked -p mistralrs-kernel-runtime --features builder-cudaforge \
#   --bin dkr_package_builder -- --cc 61 --cubin
```

The focused SM61 transport check is also release-only and performs this
one-time Cargo.lock metadata reconciliation automatically:
`bash tools/dkr_r125_sm61_typed_cubin_smoke.sh`. Release builds
reuse the existing Cargo target directory and do **not** recreate
`target/debug`. Before reclaiming old debug artifacts, inspect the
worktree's `target/debug` and never erase persistent
`target/dkr-*`, `_cudaforge_cache`, model files, or qualified PTX/CUBIN
packages. The `release-with-debug` profile is intentionally **not**
used for this qualification.

Use `DKR_KERNEL_PACKAGES=kernel-packages/build/cuda/sm61` for the filesystem
catalogue. At build time, `DKR_EMBED_TYPED_DIR` can seal the verified CUBIN
and PTX bytes together in the executable. Set
`MISTRALRS_DKR_ARTIFACT_SOURCE=filesystem` or `embedded` as appropriate.

`CANDLE_DKR_IMAGE_MODE` is shared by Candle and Mistral DKR and is fixed
on first use, so **restart the process** to change it:

- `auto` (default): prefer an exact-target paired CUBIN. Only a *cold CUDA
  module/entry admission failure* may revert to the already verified PTX.
  Missing CUBIN on a PTX-only package uses PTX.
- `cubin`: require a paired CUBIN for the exact GPU compute capability.
  Missing or rejected images cause errors, without PTX fallback.
- `ptx`: explicit rollback, using the verified PTX.

With `mistralrs-kernel-runtime/shared-registry`, the typed runtime delegates
CUBIN/PTX image loading, image policy and physical CUDA-module caches to Candle's
`get_or_load_dkr_typed_image_func`. It has **no private CUDA CUBIN cache or
independent image-mode parser**. It retains Mistral's typed manifest/ABI
validation, operation selection, and prevalidated launch-plan cache. Native
Candle and external Mistral operations have separate cache namespaces in the
same Candle CUDA device, so they cannot collide on identical PTX content.

This branch is pinned to Candle commit
`d40ac1b4d69a209d08f76c5b5bd4e267ec63b4f0` (branch
`feature/dkr-r125-typed-image-transport`), which is a strict descendant of
`feature/dkr-r124-neutral-catalog` and includes the full R1.24/R1.25
native catalogue qualification history.

`mistralrs-core/dkr-sm61-experimental` enables typed images with PTX
fallback in auto mode. The additional **opt-in**
`mistralrs-core/dkr-sm61-cubin-qualified` activates the Candle native
`cuda-dkr-qualified-default` deployment as well. This requires verified,
attested native **and** typed catalogues at build time, and is not a claim of
end-to-end Mistral model performance qualification.

### R1.25 Cudaforge cache-miss audit and deferred quantized pilot

The existing Cudaforge cache indices do **not** themselves prove reuse.
`tools/dkr_r125_cudaforge_cache_audit.py` is a read-only check of cached
object presence, per-source SHA256 and directory-wide watched-header hash,
with optional immutable snapshots for cross-worktree comparisons.

`tools/dkr_r125_quant_manifest_preflight.py` checks 12 existing SM61 typed
FP8/MXFP4 GEMV/GEMM package manifests without compilation.
The `mistralrs-core/dkr-r125-quant-typed-pilot` feature now has **2/2
independent physical SM61 numerical GPU tests PASS**: one GEMM and one GEMV,
collectively attesting twelve manifest variants to two distinct CUBIN images
via Candle's typed loader. This is a *family-scoped functional* qualification,
not a successful combined Nextest run, a model-level certification or a
performance A/B result. The broader R1.25 audit remains
`gpu_qualified=false`.

The gated `tools/dkr_r125_quant_real_sm61.sh` still refuses missing
prebuilt PTX packages and stale cache state, and requires explicit
`DKR_APPROVE_R125_QUANT=1` to launch GPU tests. Existing logs may instead
be re-attested without another GPU run using
`tools/dkr_r125_quant_pair_evidence.py`, which checks the two independent
Nextest PASS records and their two on-disk SHA256-attested CUBIN images.

See [cache failure diagnosis and quantified preparation](../docs/DKR_R125_CACHE_AND_QUANT_PREPARATION_20261123.md).
Do not modify `CUDA_COMPUTE_CAP`, clear `target/dkr-*` or start another
49/49+15/15 compilation before reviewing the read-only audit.

### R1.25 real production-adapter pilot (SM61 functional PASS)

The new `mistralrs-core/dkr-r125-typed-real-pilot` feature enables real grouped
MoE GEMM, indexed KV row-copy, and short-context paged decode through the
shared Candle typed-image transport. It does **not** automatically enable
`cuda-dkr-s31`, the full `cuda-legacy` feature or a sealed native catalogue.
On SM61 it **does** enable `candle-core/cuda-legacy-bf16`, narrowly for generic
BF16 tensor fill/cast operations: `const_set_bf16` and `cast_bf16_f32` are
not compiled for pre-Ampere devices without that compatibility definition.
The three migrated operation kernels still execute as external typed CUBIN;
this temporary generic BF16 compatibility is not a completed native DKR
replacement for every standard Candle tensor utility.
Existing typed PTX families must already have been generated and verified.

Run one focused release-only GPU gate, rather than repeating the full
25-family static build:

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
# Point these at your EXISTING cached artifacts (never a fresh directory).
export MISTRALRS_CUDA_BUILD_ROOT=/path/to/existing/target/dkr-cuda-cache/sm61
export DKR_R125_TYPED_PACKAGE_ROOT=/path/to/existing/kernel-packages/build/cuda/sm61
DKR_APPROVE_R125_REAL=1 bash tools/dkr_r125_real_kernel_sm61.sh
```

It refuses to launch if the persistent Cudaforge indexes or three PTX families
are missing. `ptxas` produces selected CUBINs under persistent
`target/dkr-r125-real-pilot`. Tests are GPU-real, ignored by default,
run through `cargo nextest --release --locked` once, and require strict
`CANDLE_DKR_IMAGE_MODE=cubin` with actual Candle load evidence.
**Physical SM61 pilot passed on 2026-11-23:** Nextest run `43d7d6aa-1e98-488c-8948-2b673ce2ebbe` reports three real GPU tests PASS, four Candle CUBIN image loads, and `DKR_R125_REAL=PASS families=3 mode=cubin loader=candle physical_loads=4`. This qualifies only these test shapes and numerical assertions, not full-model inference or performance. [Qualification report](../docs/DKR_R125_SM61_REAL_GPU_20261123.md).
Detailed branch-by-branch comparison:
[DKR R1.25 vs Candle/Mistral Extended](../docs/DKR_R125_VS_EXTENDED_20261122.md).

**Release gate:** after local compilation, run a single physical SM61
qualification comparing `auto`, `cubin` and `ptx`, checking that the
actual module load is `format=cubin` for the selected operations, full
logits/FP8/FP4/paged parity, launch-plan cache reuse, and no static CUDA
fallback for the qualified routes. Until that gate passes, keep the
experimental feature flag and do not retarget the production Candle pin.

## Historical RFC and ASD integration

See **[DKR-R0 historical ASD-first RFC](../docs/DYNAMIC_KERNEL_RUNTIME_R0.md)** for:

1. Typed per-operation ABIs and manifest versions.
2. SM61-first, per-architecture content-addressed kernel packages.
3. A generic ASD selection boundary and explicit fallback rules.
4. Separate host build, kernel build and GPU qualification cache keys.
5. Sequenced migration of F32, quantized GEMV/GEMM, convolution, MoE and other validated families.
6. Physical-gate requirements before any Legacy/Extended integration.

**Current status:** the generic verified catalog exists; executable multi-operation adapters and GPU qualification remain pending. Do not interpret the inventory as proof that FP8/FP4/MoE kernels run dynamically today, and do not merge to `extended` until their targeted gates pass.

The permanent objective is to compile the Rust host environment only when its own sources/ABI change, then select and qualify architecture-specific kernels via external artifacts and ASD profiles without rebuilding that host for an ordinary kernel/profile change.
