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

## Historical RFC and ASD integration

See **[DKR-R0 historical ASD-first RFC](../docs/DYNAMIC_KERNEL_RUNTIME_R0.md)** for:

1. Typed per-operation ABIs and manifest versions.
2. SM61-first, per-architecture content-addressed kernel packages.
3. A generic ASD selection boundary and explicit fallback rules.
4. Separate host build, kernel build and GPU qualification cache keys.
5. Sequenced migration of F32, quantized GEMV/GEMM, convolution, MoE and other validated families.
6. Physical-gate requirements before any Legacy/Extended integration.

**Current status:** the generic verified catalog exists; executable multi-operation adapters and GPU qualification remain pending. Do not interpret the inventory as proof that FP8/FP4/MoE kernels run dynamically today, and do not merge to `extended` until their targeted gates pass.

## DKR-R2 batched source preparation

See [DKR-R2 single-batch preparation](../docs/DKR_R2_SINGLE_BATCH_PREPARATION.md)
for the exhaustive operation-accounting matrix, the first 22 external SM61 CUDA
entry candidates, fixed optional RMSNorm signatures, sparse logits contracts,
and the prevalidated launch-plan API. This is **source-only** progress:
no Rust/NVCC build, GPU qualification or runtime activation has occurred.

The permanent objective is to compile the Rust host environment only when its own sources/ABI change, then select and qualify architecture-specific kernels via external artifacts and ASD profiles without rebuilding that host for an ordinary kernel/profile change.
