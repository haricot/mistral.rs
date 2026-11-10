# mistralrs-kernel-runtime

Reusable runtime substrate for external, versioned CUDA kernel providers.

## Implemented now: F32 reference

The current implementation is intentionally narrow:

- `mistral.linear.f32.v1` ABI and schema-1 PTX module manifests (`src/lib.rs`).
- Strict manifest, PTX-target, hash, entry-symbol, path and resource verification.
- `ASD-LINEAR-PROFILE-V1` exact F32 decisions with immutable preparation (`src/profile.rs`).
- Device/context-bound CUDA execution and content-addressed module reuse (`src/cuda.rs`, `src/session.rs`).
- No silent fallback for a selected F32 shape; no hot-path filesystem reads after preparation.
- Physical SM61 A/B/A2 proof through a local deterministic Mistral Llama fixture, archived in [PR #34](https://github.com/haricot/mistral.rs/pull/34).

`legacy-provider-poc/inference` is a **historical conformance fixture**, not the permanent runtime API. Keep it until equivalent conformance tests are established outside the demonstration directory.

## Intended generalization

See **[DKR-R0 — ASD-first Dynamic Kernel Runtime](../docs/DYNAMIC_KERNEL_RUNTIME_R0.md)** for:

1. Typed per-operation ABIs and manifest versions.
2. SM61-first, per-architecture content-addressed kernel packages.
3. A generic ASD selection boundary and explicit fallback rules.
4. Separate host build, kernel build and GPU qualification cache keys.
5. Sequenced migration of F32, quantized GEMV/GEMM, convolution, MoE and other validated families.
6. Physical-gate requirements before any Legacy/Extended integration.

**RFC status:** the generic catalog and multi-operation dispatcher are not implemented. Do not interpret the design document as proof that FP8/FP4/MoE kernels are dynamically loadable today, and do not merge to `extended` until the targeted gates pass.

The permanent objective is to compile the Rust host environment only when its own sources/ABI change, then select and qualify architecture-specific kernels via external artifacts and ASD profiles without rebuilding that host for an ordinary kernel/profile change.
