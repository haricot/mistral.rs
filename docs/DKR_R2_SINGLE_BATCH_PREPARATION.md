# DKR-R2: exhaustive accounting, SM61 source batch and deferred final gate

Status: **source preparation only; not a GPU qualification**. Branch: `dynamic-kernel-runtime`.

## Ground truth before this batch

The branch head `2e19b827ba77605a3d0f5798b95a07a7b2973efc` contained the independent typed R1
catalogue, the historical F32 provider and a source/backend classification audit. It did **not**
contain executable R1.6/R1.7/R1.8 operation adapters. Statements from an earlier conversation
about those stages being published must not be treated as repository evidence.

The pinned Candle source is `haricot/candle:cuda_legacy` at
`d99f6d77e507688bd9e4b654fbe20ac9e18238fc`. On that revision the established inventory is
196 checked-in CUDA compilation units (including 53 SM80-only units) and 44 backend methods.
The counts are audit scope, **not** 196 executable dynamic kernel implementations.

## What is added without a Rust or CUDA compilation

1. `tools/dkr_operation_matrix.py` cross-references every classified CUDA source family and
   backend method. Every family has one explicit route: candidate device kernel,
   host-and-device operation, runtime-only, vendor provider, incompatible SM80, or historical.
   It fails if any family or inspected Candle method is unclassified. An absent pinned Candle
   checkout produces a partial result, never a complete PASS.
2. `kernel-packages/sm61/dkr_ops.cu` holds 22 CUDA entry-point candidates in **one** SM61
   compilation unit: F32 copy/affine, four binary ops, four unary ops, sum/max/softmax,
   four RMSNorm signatures, causal mask, sparse logits penalty and bias, gather rows and
   pair-interleaved RoPE. The operation-specific normalizations and logits kernels remain
   versioned rather than silently substituting for different model semantics.
3. `tools/dkr_sm61_package.py` consumes **already compiled** PTX; it never calls NVCC or Cargo.
   It emits one immutable PTX artifact and 22 separate `DKR-KERNEL-MODULE-V1` manifests whose
   tensor/scalar argument order is specific to each entry. A missing entry refuses packaging.
   Each entry is marked `UNQUALIFIED` until numerical and physical SM61 testing.
4. `universal_cuda::PrevalidatedLaunchPlan` freezes an exact, already-checked launch and
   refuses a different module or CUDA device on reuse. No runtime filesystem lookup or repeated
   manifest admission is needed on that path. The launch remains `unsafe`: the caller must
   supply correct CUDA arguments and guarantee buffer lifetime, access, and stream ordering.

## Boundaries and unresolved execution work

This batch does not wire the 22 entries into Candle's backend or make them the active runtime
path. The original F32 execution proof remains unchanged. There is no default fallback from a
missing DKR provider, and no experimental source package may claim GPU PASS merely because its
manifest can be loaded.

The full 196-source inventory includes substantially more than these entry points:
quantized formats, full dense GEMM/GEMV, MoE routing and cache, attention and paged KV,
convolutions, state-space, arbitrary views/strides, random streams, pooling/upsampling, sorting,
multi-GPU operations, allocator and CPU/GPU transfers. These require separate typed adapters,
provider bundles or runtime capabilities. SM80/SM90 kernels must not be mislabeled as Pascal
implementations. Host-only operations are **accounted for**, not scheduled for PTX generation.

## Semantic requirements before activation

- Manifest pointers and scalar order must match the actual compiled PTX, and the host adapter
  must validate exact ranks, dimensions, dtype, device/context and aliasing.
- For the F32 row reductions, softmax and RMSNorm, launch precisely one 256-thread CTA per row.
  Width and row products must fit 32-bit indexing; epsilon must be positive and finite.
- Sparse logits ID arrays must be unique, coalesced, in-bounds and on the same GPU. Apply
  repetition penalties, then frequency/presence, and apply sparse bias in a separate ordered
  launch. Repetition must be finite and positive; the caller chooses any additional processors'
  order explicitly. These kernels do not implement all sampler semantics.
- For normalization, choose one of four **fixed** signatures: ordinary/residual and
  unscaled/scaled. No nullable scale pointer or branch on a per-call option is introduced.
  Further two-stage residual-then-normalization variants must have their own explicit ABI.
- The RoPE candidate is **interleaved pairs only**. Its output tail is not written when
  `rotary_width < width`; the adapter must pre-copy it or restrict to full-width output. It is
  not interchangeable with split-half or specialized positional conventions.
- Gather indices must be validated before launch. Reject empty/zero dimensions. Causal-mask
  dimensions and position offset must be checked for overflow.
- Numerical references must cover F32 rounding, all-masked softmax rows, negative logits,
  token repetition, non-finite values, arbitrary long rows and adversarial indexing.

## One final local qualification, only when all adapters are ready

No Rust toolchain installation, Cargo check, NVCC build, test, or CI campaign is required to
prepare source changes. The **future** local campaign should perform exactly one SM61 external
kernel compilation, one cached Cargo/nextest build of the selected host target, then a single
grouped correctness/negative/parity/ASD/graph-capability GPU qualification on the physical
GTX 1080 Max-Q. Its gate must refuse to report PASS for every unimplemented adapter; all
host/runtime-only methods require a separate explicit coverage statement. This campaign is
**not yet ready to run** and no executable GPU PASS is claimed by this document.

The manifest preparation command, after the one final NVCC compilation, will be:

```bash
python3 tools/dkr_sm61_package.py \
  --ptx /path/to/final/dkr_ops.ptx \
  --out-dir /path/to/final/dkr-modules
```

The classification command (no build) will be:

```bash
python3 tools/dkr_operation_matrix.py \
  --candle-dir /path/to/candle-at-d99f6d7 \
  --output /path/to/final/dkr-matrix.json
```

Neither command substitutes for the eventual grouped GPU test.
