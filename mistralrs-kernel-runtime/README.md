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
# Run the CPU-only typed-manifest / policy gates first.
cargo nextest run -p mistralrs-kernel-runtime --lib

# In an SM61 worktree with *already qualified* PTX packages:
cargo run -p mistralrs-kernel-runtime --features builder-cudaforge \
  --bin dkr_package_builder -- \
  --cc 61 --pair-existing --output kernel-packages/build/cuda

# Or request a complete Cudaforge PTX build + paired CUBIN generation:
# cargo run -p mistralrs-kernel-runtime --features builder-cudaforge \
#   --bin dkr_package_builder -- --cc 61 --cubin
```

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

Mistral keeps a bounded, per-CUDA-device CUBIN module cache (512 physical
modules) and reuses the existing typed prevalidated launch plans. The PTX
path continues through Candle's shared DKR module cache. Both image formats
must pass the same numerical and route-selection qualification; successful
loading does **not** establish numeric parity or performance promotion.

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
