# External kernel packages (DKR generic loading proof)

The Rust runtime does not compile or embed the kernels in this directory. Each
architecture package contains a PTX artifact and a `manifest.json` with the same
versioned operation ABI and named implementation family.

```text
kernel-packages/
  cuda/
    sm61/elementwise-scale/{manifest.json,scale.ptx}
    sm80/elementwise-scale/{manifest.json,scale.ptx}
```

The architecture folders are package organisation, **not hard-coded runtime
branches**. `KernelCatalog::discover` walks the directory tree and checks hashes
and typed signatures. `KernelCatalog::resolve` selects the compatible package
with the highest PTX target capability; equal-priority compatible providers are
rejected as ambiguous. The CUDA executor checks the actual device capability.

The PTX scale examples intentionally implement exactly the same algorithm. Only
the `.target` directive differs. These are demonstration artifacts, not tuned or
numerically qualified production implementations.

## Local proof without GPU

From the repository root:

```sh
cargo test -p mistralrs-kernel-runtime --no-default-features --locked
```

Tests discover both checked-in packages, confirm automatic selection for
capabilities 61, 80 and 90, verify semantic PTX identity aside from target,
and refuse unsupported capability 60. They cannot prove device execution.

## Physical GPU proof (one host build)

From the repository root, with an already configured CUDA environment:

```sh
cargo build -p mistralrs-kernel-runtime --example dkr_scale_cuda --features cuda --locked
./target/debug/examples/dkr_scale_cuda
```

Expected result: `DKR_AUTO_SCALE_GPU=PASS` with `device_cc`, `selected_target_cc`
and the numerical error against a scalar oracle. No architecture name appears
in the executable adapter. Set `DKR_KERNEL_PACKAGES` to an alternative external
directory to test a new package without rebuilding the host executable.

Replace a package by creating a new trusted manifest with the SHA256 of its PTX;
never hand-edit a hash to bypass an integrity failure. Other GPU architecture
families can be supplied through new folders without changing Rust source,
provided they respect the same ABI. Native CUBIN and code signatures are beyond
this PTX-only R1 proof.

## DKR-R1.1: external MoE reference package

The MoE source lives in `kernel-packages/cuda/moe/moe.cu` and is not part of
`mistralrs-core/src/cuda/*.cu`. Its external package builder targets only the
requested architecture, and reuses previously compiled artifacts when the
source, NVCC version and compilation arguments are unchanged:

```sh
python3 kernel-packages/cuda/moe/build.py --cc 61
export DKR_KERNEL_PACKAGES="$PWD/kernel-packages/build/cuda"
cargo test -p mistralrs-core --features 'cuda-legacy,dkr-moe' \
  --lib dkr_moe_gpu -- --ignored --nocapture
```

`dkr-moe` opts in to the dynamically loaded provider for `moe_gemm`. The old
FFI-based MoE is retained in `moe_legacy.rs` for comparison without this
feature. The generic Rust dispatch knows only the logical operation
`moe.grouped_gemm.v1`, dtype, shapes and validated launch arguments. It has no
GPU-model-specific name or architecture switch. The caller must supply the
external catalog path via `DKR_KERNEL_PACKAGES`.

The generic `KernelExecutionCache` discovers and verifies packages once, then
keeps up to 128 prepared CUDA functions per exact launch contract and device.
On a warm hit, no filesystem access, PTX hashing, catalogue resolution, CUDA
capability query or module loading is repeated. The launch itself still
validates its typed arguments and CUDA context. A new request shape or device
must undergo cold resolution before it can enter the cache.

An identical-output repeated launch in the ignored CUDA test verifies cache
reuse. This establishes the cold/hot separation, **not** zero measured dispatch
overhead. Before claiming performance parity with static dispatch, benchmark
the *same kernel algorithm* with warmed modules, equal shapes, identical CUDA
streams and an A1/B/A2 order. Compare host launch latency and GPU kernel time
separately, including p50, p95 and per-token decode cost; report the measured
delta and its variance. Scalar reference PTX and optimized legacy HFMA2 are not
equivalent workloads for such a comparison.

The external implementation is a scalar, numerically checkable reference for
F16 and BF16, not a promoted HFMA2/WMMA optimization. It does not use ASD and
is not numerically qualified on real hardware until the ignored GPU test passes.
Routed token indices must satisfy the same upstream invariants as the legacy
MoE: valid expert IDs and a permutation of output token indices. The current
typed ABI accepts contiguous zero-offset tensors; other layouts are rejected.

## DKR-R1.3: external indexed row copy

The optional `dkr-indexed-copy` feature routes cache-row copying through the
same generic `KernelExecutionCache` as MoE. It excludes the static
`indexed_copy.cu` kernel from the Mistralrs CUDA build, while the legacy Rust
adapter remains available when the feature is disabled.

```sh
python3 kernel-packages/cuda/indexed-copy/build.py --cc 61
export DKR_KERNEL_PACKAGES="$PWD/kernel-packages/build/cuda"
cargo test -p mistralrs-core --locked \
  --features 'cuda-legacy,dkr-moe,dkr-indexed-copy' \
  --lib dkr_ -- --ignored --nocapture
```

The indexed-copy source compiles exactly once per requested compute capability.
Three typed manifests (F16, BF16, F32) reference the same PTX bytes, with a
16-bit or 32-bit entry point according to element size. The kernel copies bits
without floating-point arithmetic, rejects destination indices outside the
output row count, and preserves nonzero storage offsets through checked device
pointers. The test validates in-place results and repeated-launch cache reuse.
It still requires real-GPU qualification before use in model inference.

## DKR-R1.4: prevalidated launch plans

MoE and indexed-copy now keep a bounded, thread-local plan cache keyed by the
validated operation signature and CUDA device. A cold signature resolves its
external module once, validates the launch geometry and typed ABI, and retains
the CUDA function and launch configuration in a `PrevalidatedLaunchPlan`.

On a warm signature, neither provider reconstructs `LaunchRequest`, allocates
argument descriptors, locks the shared module cache, checks compute capability,
nor repeats manifest admission. Host-side tensor invariants and CUDA buffer
lifetimes remain checked by the adapters before dispatch. The hot cache lookup
uses no operating-system mutex and does not allocate after warming.

The generic plan does not require ASD. CUDA Graphs are reserved for later
sequence-level qualification, after the remaining operation families have been
migrated. The first and repeated launches are not assumed latency-equivalent
to the legacy path until the final paired CPU/GPU benchmark is run. One combined
SM61 CUDA qualification build is planned after all provider migrations; smaller
source/ABI checks can run between changes without GPU CI.

## DKR-R1.5: external fused sigmoid-gated addition

The optional `dkr-gated-add` feature replaces the statically linked
`gated_add.cu` launch with an external CUDA package, while preserving the
legacy adapter when disabled. This follows the same bounded, thread-local
prevalidated plan path as MoE and indexed-copy.

The package covers F16, BF16 and F32 with the existing dtype-specific sigmoid
rounding sequence. F32 retains both flat and row-tiled kernels, selected by
the generic column count rather than a GPU architecture name. The ABI flattens
the contiguous leading dimensions into rows while preserving storage offsets.
Nonmatching/broadcast or strided layouts continue through Candle's existing
fallback without discovering a DKR package.

The eventual single batch command for all external packages is:

```sh
python3 kernel-packages/cuda/build_all.py --cc 61
export DKR_KERNEL_PACKAGES="$PWD/kernel-packages/build/cuda"
cargo test -p mistralrs-core --locked \
  --features 'cuda-legacy,dkr-moe,dkr-indexed-copy,dkr-gated-add' \
  --lib dkr_ -- --ignored --nocapture
```

The batch builder reuses packages whose sources, compilation options and NVCC
version have not changed. These commands are reserved for final qualification,
not for each migration increment. The new ignored GPU test and the existing
exact-rounding and graph-capture tests still require SM61 verification.

## DKR-R1.6: MoE weighted reduction and sink-aware softmax

Enable `mistralrs-core/dkr-reduce-norm` to route weighted reduction and
GPT-OSS sink-aware softmax CUDA entry points through external PTX packages.
The quant crate's own `dkr-reduce-norm` feature also works independently.
Default builds retain the original CUDA kernels. General RMSNorm/LayerNorm,
FP8 fused normalization and generic reductions are not migrated yet.

One NVCC invocation compiles `kernel-packages/cuda/reduce-norm/reduce_norm.cu`
for the requested architecture. Its 24 typed manifests represent nine MoE
weighted input/output precision pairs plus three softmax precisions at five
block widths. R1.4 plans are cached per thread, device and exact signature;
the CUDA source and generic Rust path have no GPU-model branches.

At the **final** qualification point, not after every source increment, run
the single batch package build followed by one test compilation session:

```sh
python3 kernel-packages/cuda/build_all.py --cc 61
export DKR_KERNEL_PACKAGES="$PWD/kernel-packages/build/cuda"
cargo nextest run -p mistralrs-core -p mistralrs-quant --locked \
  --features 'mistralrs-core/cuda-legacy,mistralrs-core/dkr-moe,mistralrs-core/dkr-indexed-copy,mistralrs-core/dkr-gated-add,mistralrs-core/dkr-reduce-norm' \
  --run-ignored ignored-only -E 'test(/dkr_/)' --no-fail-fast
```

The ignored GPU tests cover MoE, indexed-copy, gated-add, weighted reduction
and sink-aware softmax. End-to-end model validation, exact-rounding/graph
capture checks and legacy/DKR host/GPU A1/B/A2 measurements are additional
final promotion gates, not performance claims for this increment.
