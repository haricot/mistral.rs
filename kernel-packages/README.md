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
  --features 'mistralrs-core/cuda-legacy,mistralrs-core/dkr-moe,mistralrs-core/dkr-indexed-copy,mistralrs-core/dkr-gated-add,mistralrs-core/dkr-reduce-norm,mistralrs-core/dkr-add-rms-norm,mistralrs-core/dkr-causal-mask' \
  --run-ignored ignored-only -E 'test(/dkr_/)' --no-fail-fast
```

The ignored GPU tests cover MoE, indexed-copy, gated-add, weighted reduction
and sink-aware softmax. End-to-end model validation, exact-rounding/graph
capture checks and legacy/DKR host/GPU A1/B/A2 measurements are additional
final promotion gates, not performance claims for this increment.

## DKR-R1.7: fused add + RMSNorm

`mistralrs-core/dkr-add-rms-norm` replaces the CUDA `cuda_add_rms_norm`
adapter without changing its call signature. A single
`kernel-packages/cuda/add-rms-norm/build.py --cc 61` invocation produces
one PTX artifact and 70 typed manifests: scalar blocks 32/1024 for all three
float types and all 32 warp-aligned vector8 block widths for F16/BF16.
The vector path is selected only for 16-byte-aligned device pointers and a
column count divisible by eight; no architecture-specific dispatch is needed.

The legacy fused kernel and C FFI wrappers stay available when the feature
is off; `sort.cu` still contains other fused normalization and sorting
kernels. When enabled, a scoped preprocessor guard avoids linking duplicate
static add+RMSNorm definitions. Host-only contract review precedes the single
deferred CUDA qualification.

## DKR-R1.8: external in-place F32 causal masking

`mistralrs-core/dkr-causal-mask` replaces the 256-thread in-place F32 causal
mask kernel from `sort.cu`. Its signed 32-bit query/key geometry and offsets
are ABI-typed scalars; its scores tensor is declared `read_write`. The
prevalidated R1.4 plan is reused for each shape/device, while `q_offset` and
`prefix_len` remain dynamic on the warm path. A contiguous nonzero view
offset is applied to the device pointer before dispatch.

`kernel-packages/cuda/causal-mask/build.py` emits one PTX and one manifest
per requested architecture and is included automatically by `build_all.py`.
The original FFI remains available by disabling the feature. Correctness and
launch-overhead parity will be checked with the other operations only in the
final GPU qualification session.


## DKR-R1.9: sparse logits penalties and bias

The optional `mistralrs-core/dkr-logits` feature retains the existing validated
tensor allocation, offsets and sampler entry points, but routes F32 sparse
penalties and sparse bias through the R1.4 prevalidated launch-plan cache.
The external `kernel-packages/cuda/logits` source emits three entry points:
a contiguous F32 copy, a sparse penalty update and a sparse bias update.
Both sampling paths launch copy before update on the same CUDA stream.

The logits contract requires unique token IDs for race-free sparse updates,
valid in-vocabulary IDs or explicit out-of-range ignore behavior, and a
contiguous F32 logits tensor. The sampler's sparse count/bias construction
provides unique IDs; direct callers must preserve that invariant.
Repetition penalties are applied after frequency/presence penalties,
matching the previous kernel. The penalty factors themselves remain
dynamic scalar arguments, not hot-cache keys. The legacy FFI path remains
selected when the feature is off.

The ignored `dkr_logits_penalties_and_bias_gpu` test has been staged, not
executed. The feature is not yet numerically or performance qualified.

## DKR-R1.10 source staging: residual RMSNorm variants

The `kernel-packages/cuda/rms-variants` builder emits a single CUDA PTX
compilation unit with twelve candidate entry points: three dtypes
(F16, BF16, F32), two residual-normalization forms and two *fixed* ABI
variants with/without a scale tensor. This avoids nullable CUDA pointers,
unused scale-buffer allocations and per-element optional checks.

The forms are:
- residual + RMSNorm(x) * weight, optionally scaled;
- the preceding residual output followed by a second RMSNorm, with
  separate residual/normalization epsilon and weight tensors.

These are source-only candidates. The Candle/Mistral host adapters and their
feature gates have **not** been migrated yet. The existing
`dkr-add-rms-norm` integration is unchanged. Strided 4D normalization
requires a new layout-aware typed ABI and is deliberately not advertised as
supported by the current contiguous-only contract.

Both new builders are discovered by the existing
`kernel-packages/cuda/build_all.py` batch, so **do not** invoke them
individually for this migration. The final SM61 compilation and one grouped
qualification remain deferred until the remaining operation adapters are
ready. Python remains a small packaging/compilation orchestrator for now;
no new Rust toolchain, cudaforge dependency, CI trigger or NVCC invocation is
required for these source edits.

### DKR-R1.10 host integration

The optional `mistralrs-core/dkr-rms-variants` feature now connects both
existing Mistral.rs residual RMSNorm entry points to the external package.
The original host validation, contiguous storage normalization, storage-offset
handling and CUDA stream remain in place. Four fixed launch signatures
(residual/residual-then, with/without the scale tensor) avoid a nullable
device pointer or a runtime scale branch inside the CUDA loop.
F16, BF16 and F32 are selected by the generic typed manifest, not by an
SM-specific host branch. Each (dtype, shape, form, scale-presence, device)
combination has a prevalidated plan; epsilon values remain dynamic scalars.

`MISTRALRS_DKR_RMS_VARIANTS` excludes both legacy residual RMSNorm
function families from the statically compiled `sort.cu` unit when the
feature is enabled. Strided 4D RMSNorm stays on its existing ABI pending
an explicit noncontiguous-tensor extension; it has not silently changed
to a packed copy. The replacement is a scalar CUDA reference, while the
legacy SIMD vec8 fast path is kept when the feature is disabled.
Throughput parity is **not** assumed.

The optional `dkr-logits` and `dkr-rms-variants` features should be
added to the grouped SM61 `cargo nextest` invocation at final
qualification. No individual CUDA compilation or CI is authorized
during staging. The two new ignored GPU tests are
`dkr_logits_penalties_and_bias_gpu` and `dkr_rms_variants_gpu`.


## DKR-R1.11: external U32 decode input padding

The optional `mistralrs-core/dkr-input-pad` feature routes the
`pad_decode_input_u32` path through a dedicated external PTX entry.
It reuses the original zero-offset-adjusted input storage pointer and
output allocation; the CUDA operation duplicates row zero for additional
decode rows. The launch contract includes two U32 tensors of rank two
and three signed I32 dimensions. A per-thread prevalidated plan is cached
by exact input/output shape and CUDA device.

Only `pad_decode_input_u32` is removed from the static
`input_packing.cu` compilation under this feature. The completion
input packing kernel stays static because it receives up to 64
different device pointers by value, which cannot safely be represented
as an ordinary contiguous tensor in `dkr.typed.v1`.
No extra dynamic pointer-table allocation is introduced to obscure
that ABI limitation.

The existing ignored GPU test
`pads_decode_input_by_aliasing_row_zero_values` covers the operation
under the feature. It has not run as part of the deferred qualification.


## DKR-R1.12: external F32 selective SSM scan

The opt-in `mistralrs-core/dkr-ssm` feature routes the Mamba-style
F32 selective scan through the typed DKR execution cache. The existing
public `selective_scan_cuda` signature is preserved. The selected
package is keyed by the state-lane factor (1, 2, 4 or 8), with exact
batch, sequence, head, head-width and state shape in the hot plan.
Clamp bounds remain dynamic arguments. One CUDA warp owns each
(head, head-channel, batch) state slice for the entire sequence.

Unlike the previous reference scan, the new source explicitly masks
state entries beyond `d_state` in the final warp, so sizes such as 16
do not read or write beyond the state buffer. This candidate is limited
to F32, contiguous state storage and `1 <= d_state <= 256`.
The adapter checks tensor shapes, equal device and 32-bit index budget
before selecting the external package. The source-only package builder
compiles all four entry points in **one** PTX invocation for SM61.

When the feature is enabled, `ssm.cu` is excluded from the static Mistral
CUDA build; the CUDA legacy implementation remains selected when the
feature is off. The ignored `dkr_ssm_scan_gpu` oracle tests all four
lane factors and an odd, sub-warp state width. No GPU run has occurred.


## DKR-R1.13: shared launch-plan cache and cudaforge package builder

The per-thread launch-plan cache implementation is now shared by
`mistralrs-core` and `mistralrs-quant`, via the public
`mistralrs-kernel-runtime::launch_plan_cache::LaunchPlanCache`.
The local cache instances are called `THREAD_LAUNCH_PLANS`.
Preparation is performed only on a new shape/variant/device key, with
a per-cache capacity limit of 64. The generic `PrevalidatedLaunchPlan`
still enforces module/entry/configuration validation before the hot path.

**Build backend**: `cudaforge = 0.1.6` is already pinned in the workspace.
The optional Cargo feature `mistralrs-kernel-runtime/builder-cudaforge`
makes `dkr_package_builder` available. The builder calls
`KernelBuilder::build_ptx()` for each of the ten external CUDA source
families, always for an explicitly specified architecture. The
`cudaforge` per-source hash cache handles incremental rebuilds.

Each Python `kernel-packages/cuda/*/build.py` emitter now accepts
`--ptx-file` to **package a PTX produced by cudaforge** and skip NVCC.
MoE uses `--ptx-dir` to preserve its two compile-time F16 and BF16
specializations. Python still generates the typed manifest JSON, SHA256,
variant paths and content cache; it does not compile kernels in this mode.
The previous Python+NVCC path remains available for rollback.

The new Rust builder is intentionally independent of the
`mistralrs-core` application build: no JIT compilation takes place in the
token decoding hot path, and **no GPU target other than the explicit
`--cc` is generated**.

To invoke it in the eventual SM61 qualification campaign (not yet run):

```sh
CUDA_COMPUTE_CAP=61 cargo run -p mistralrs-kernel-runtime \
  --features builder-cudaforge --bin dkr_package_builder -- \
  --cc 61 --output kernel-packages/build/cuda
```

The source-coverage audit is `tools/dkr_operation_matrix.py`, which
separates *source classified*, *external adapter staged*, and
*SM61 qualified*. Its verdict deliberately remains
`MIGRATION_INCOMPLETE` while any source families remain static or
their GPU parity gates are untested.

A combined, opt-in final-test entry point is
`tools/dkr_sm61_final_qualification.sh`. It refuses to launch unless
`DKR_APPROVE_FINAL=1`; the test campaign must only be approved once all
targeted migrations are complete. It uses one SM61 package build and
one feature-unified `cargo nextest` campaign, not per-operation CI.


## DKR-R1.14: contiguous fused RMSNorm + partial NeoX rotary

The opt-in `mistralrs-core/dkr-rounded-rope` routes **contiguous**
inputs for the existing `try_rms_norm_partial_rotary_neox` operation
to an external typed PTX package, with F16/BF16/F32 entries and
prevalidated 32-thread and 1024-thread block profiles. The adapter
constructs *views* of the caller's contiguous buffers, not copies;
their normalized 4D/2D ranks match the typed manifest and keep the
original storage offset.

The source retains explicit F32 rounding instructions and the
round-to-input-dtype step between RMSNorm and rotary multiplication.
It also preserves the original warp reduction strategy. The output
is restored to the caller's original rank/shape.

**Noncontiguous input stays on the original strided CUDA path**.
The typed DKR ABI currently only admits contiguous tensors; claiming
a noncontiguous view as contiguous would be invalid, and forcing a
packed copy would add an unacceptable hidden GPU allocation. A
versioned strided-tensor ABI is therefore a separate migration.

The ignored `dkr_rounded_rms_rope_gpu` candidate test covers three
dtypes and both block sizes. Its numerical agreement and performance
have **not** been qualified on SM61. The cudaforge external package
builder now handles 11 CUDA source families.
