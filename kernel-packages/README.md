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


## DKR-R1.15 — Dense decode GEMV (source-staged, unqualified)

A new opt-in feature, `mistralrs-quant/dkr-gemv` (also forwarded by
`mistralrs-core/dkr-gemv`), provides an **external, Pascal-compatible
SIMT GEMV** for contiguous F16/BF16/F32 tensors. The shapes are weight
[M, K], input [B, K] (B <= 8), optional bias [M], output [B, M].
Contiguous view offsets are supported without copying data.

`linear.gemv.plain.b128.v1` and
`linear.gemv.bias.b128.v1` are **different typed ABI signatures**.
No nullable bias tensor is used; the separate kernels and manifests
avoid optional runtime argument overhead. All kernels use [128, 1, 1]
threads and a [M, B, 1] grid, performing warp SIMT reductions and F32
accumulation. No SM70+ instructions or tensor cores are required.
The thread-local `LaunchPlanCache` prevalidates the resolved variant.

**Scope:** This is the narrow decode-phase GEMV route, not dense
prefill GEMM or the legacy static GEMV tuning matrix. The original
static dispatcher remains the fallback for noncontiguous and other
unsupported cases. A deliberate reference SIMT implementation gives
a parity baseline; the new route is **not yet performance-qualified**
against the previous optimized GEMV, cuBLAS(Lt), or Pascal DP4A.
Do not enable the opt-in route by default before the grouped GPU
parity/performance tests pass.

Test candidate: `dkr_dense_gemv_gpu` (F16/BF16/F32, batches 1/3/8,
with/without bias, odd K). It is ignored until the next grouped GPU
qualification. The previous local **15/15 DKR-R1.14-SM61** result
remains the reference baseline.

**Build-cache invariant:** do not set, unset, or export
`CUDA_COMPUTE_CAP` from these build scripts. Pass only
`--cc 61` to the stand-alone PTX builder, preserving the user's
existing Cargo/NVCC environment and static-kernel build cache.


## DKR-R1.16 — FP8 KV-cache scale update (source-staged, unqualified)

The opt-in feature `mistralrs-paged-attn/dkr-kv-scales`
(forwarded through `mistralrs-core/dkr-kv-scales`) routes the
**contiguous, zero-offset F16/BF16/F32 input** scale-update kernel to
a typed external PTX module. This is the FP8 KV scaling prerequisite,
**not** an FP8 page quantizer, cache gather, or paged-attention decode
kernel.

The external ABI is `kv_cache.scale_update.atomic_max.v1`:
two same-dtype flat K/V input buffers, two F32 scalar read-write scale
buffers, and a U64 element count. Layout normalization is a zero-copy
logical view, admitted only after the adapter checks the actual Candle
layout. The original atomic-max semantics are preserved:
`k_scale = max(k_scale, max(abs(K)) / 240.0)`, likewise for V.
Each block uses 512 threads and 4096 bytes of dynamic shared memory.
The grid is clamped to 65,535 blocks as in the static kernel.
Prevalidated launch plans are keyed by dtype/element count/device.

The old FFI entry remains available on non-eligible layouts, without
introducing a new implicit packing step. The test
`dkr_kv_scales_gpu` is an ignored GPU candidate; it covers
F16/BF16/F32, nonzero existing scales and repeat-launch plan reuse.
It has **not been compiled or qualified** on SM61 yet.

### Remaining FP8/FP4 and attention architecture work

The DKR-v1 `ElementType` supports `fp8_e4m3` and `u8` as typed
tensor storage, but weight formats such as NVFP4/MXFP4 also need
**explicit storage representations**, group-scale tensor layouts,
block/group sizes and packed-byte strides. Do not treat packed E2M1
bytes as a plain `u8` tensor with no representation contract.
The next step is a versioned `representation` descriptor
(`nvfp4_e2m1:group16`, `mxfp4:group32`, etc.), with a
separate conversion/scale-parity fixture before GEMM/GEMV dispatch.
SM61 uses software/SIMT decoding; SM80+ tensor-core entries cannot
be advertised as SM61-compatible kernels.

For paged attention, `reshape_and_cache`, `gather_kv_cache`,
`paged_attention_v1/v2`, MLA and FlashInfer layouts are separate
operations with different strides and page tables. The fixed
`dkr.typed.v1` contiguous argument ABI cannot honestly represent
all their noncontiguous/indirect accesses. Move them only after an
explicit indirect-layout/strided ABI and individual kernel parity
tests. CUDA Graph capture remains deferred.

**Important:** The existing R1.14 15/15 SM61 proof is unchanged. New
R1.15 and R1.16 optional routes remain staged until the *single*
grouped GPU qualification. No change to `CUDA_COMPUTE_CAP` is
performed by the DKR final qualification script.


## DKR-R1.17–R1.20 — New source-staged families (SM61 unqualified)

The R1.16 baseline is **17/17 tests PASS** locally on an NVIDIA SM61
GTX 1080 Max-Q; that result does **not** certify the new R1.17–R1.20
features. No new NVCC build, GPU test, or GitHub CI was run while
implementing these additional source/ABI migrations.

| Milestone | Feature in mistralrs-core | Package | Coverage | Test |
|---|---|---|---|---|
| R1.17 | `dkr-fp8-cast` | `fp8-cast` | six E4M3 F16/BF16/F32 encode/decode signatures | `dkr_fp8_cast_gpu` |
| R1.18 | `dkr-kv-gather` | `kv-gather` | VLLM layout, contiguous F16/BF16/F32, i32 device page tables | `dkr_kv_gather_gpu` |
| R1.19 | `dkr-gemm` | `dense-gemm` | opt-in dense prefill GEMM, 16x16 SM61 SIMT, optional bias | `dkr_dense_gemm_gpu` |
| R1.20 | `dkr-fp4-dequant` | `fp4-dequant` | packed reference NVFP4 (E4M3 group16/global F32) and MXFP4 (E8M0 group32) | `dkr_fp4_dequant_gpu` |

All four are compiled through the existing `dkr_package_builder --cc 61`
when that deferred step is authorized. Total external source families:
**17**. PTX packaging and test selection remain separate from
automatic runtime activation.

### Source/ABI contracts

- **FP8 E4M3 casts:** 256-thread grid, two input/output pointer
  arguments plus U64 count; exact separate operation identities for
  encode and decode. The existing static conversion is preserved for
  noncontiguous and zero-element layouts.
- **Paged K/V gather:** `kv_cache.gather.vllm.contiguous.v1`;
  manifest representations `vllm_k_blocked_x.v1` and
  `vllm_v_token_strided.v1`. Device page tables and prefix offsets
  remain on GPU. **FP8 scaled gathers, MLA and FlashInfer are not
  migrated**; these paths retain their existing backend.
- **GEMM:** `linear.gemm.plain.t16.v1` and
  `linear.gemm.bias.t16.v1` have distinct signatures, as the GEMV
  variant does. This new shared-memory SIMT baseline must be compared
  against cuBLAS(Lt) before production. Its existence is not a claim of
  faster prefill throughput.
- **Packed FP4:** `quant.fp4.dequant.nvfp4.v1` and
  `quant.fp4.dequant.mxfp4.v1` are **reference/parity operators**, not
  an inference hot-path conversion. The former consumes U8 packed
  E2M1, E4M3 group16 and per-row F32 global multipliers. The latter
  consumes U8 packed E2M1 and U8 E8M0 group32. Separate
  `representation` names and typed manifests prevent silent format
  interchangeability. No implicit unpacking is introduced into the
  optimized SM61 GEMM/GEMV paths.

### Still outside these migrations

1. **Quantized linear:** fused FP8 SIMT/DP4A GEMM/GEMV, fused NVFP4 and
   MXFP4 GEMM/GEMV, routed experts, GGUF MMQ/MMVQ, GPTQ, HQQ and
   Marlin. Packed FP4 parity operators above are prerequisites only.
2. **Attention:** SDPA, paged-attention v1/v2, FlashInfer alternatives,
   MLA decode, heterogeneous masks/strides and SM80+ specialized paths.
3. **KV cache:** in-place reshape/cache writes, block copy/swap,
   quantized FP8 gather, page compaction and scale encoding. Only
   scale-update and non-quantized VLLM gather are staged as DKR.
4. **Candle backend:** general elementwise/cast/reduction, indexing,
   layout-specialized kernels; the DKR catalog audit classifies them
   but does not imply migration.
5. **Performance qualification:** exact/parity edge cases,
   token-throughput/prefill comparisons, and CUDA Graph capture once
   all launch plans are prevalidated. No benchmark regression was
   assumed away.

**Cache policy:** never set, unset or export `CUDA_COMPUTE_CAP` in
DKR scripts or shell instructions. Only the independent PTX builder's
`--cc 61` targets Pascal. Do not run `cargo clean` or repeatedly
compile/test between source-staging milestones.


## DKR-R1.20.1 — Preserve static CUDA objects between Cargo feature builds

**Problem verified:** The R1.20 SM61 21/21 campaign built 53 quant, 13 core
and 15 paged-attention CUDA objects (81 in total), despite all 13 older
external PTX families being reused. The local environment had no
`MISTRALRS_CUDA_BUILD_ROOT`. By default these three `build.rs` use
Cargo's feature-specific `OUT_DIR` for cudaforge's object cache; a
different feature unification gets a different cache directory.

The pinned cudaforge 0.1.6 implementation stores
`.cudaforge_cache.json` in `KernelBuilder::out_dir` and keys entries
by the source/object path, SHA-256 of the source, NVCC argument hash,
target architecture, and hash of watched headers. It cannot reuse a
previous Cargo `OUT_DIR` solely because the old objects still exist.
The existing `MISTRALRS_CUDA_BUILD_ROOT` build-script support gives
all Cargo builds a stable out_dir for **objects only**, while the actual
archive `lib*.a` is still linked inside the current Cargo `OUT_DIR`.

**First-time bootstrap (do not rebuild anything yet):**

```bash
cd /home/np/tmp/mistralrs_provider/mistral.rs
python3 tools/dkr_cuda_cache_bootstrap.py
```

This is a dry-run: look for the cached sources from
`mistralrs-quant` (~53), `mistralrs-core` (~13) and
`mistralrs-paged-attn` (~15). The number may differ if the latest
cache was built with other features. For a successful current cache,
adopt the valid objects, **without nvcc or Cargo**:

```bash
python3 tools/dkr_cuda_cache_bootstrap.py --apply
```

This creates `target/dkr-cuda-cache/sm61/{quant,core,paged-attn}/kernels`
and rewrites only cudaforge cache metadata pointing at the copies.
It does not touch existing Cargo build artifacts or CUDA environment
variables, and refuses to overwrite an initialized cache.

**Next combined DKR campaign (not on every source change):**

```bash
MISTRALRS_CUDA_BUILD_ROOT="$PWD/target/dkr-cuda-cache/sm61" \
  DKR_APPROVE_FINAL=1 tools/dkr_sm61_final_qualification.sh
```

This scopes the opt-in environment setting to that command. The
script must not set/unset `CUDA_COMPUTE_CAP`. On the first invocation,
Cargo can rebuild its Rust crates once because the environment changed,
but cudaforge should reuse valid *static CUDA objects* rather than
recompile them. `Compiling 0 of N` or `All library kernels up-to-date`
is the expected NVCC result **if the flags, architecture, sources,
headers and object metadata still match**. There is no guarantee of a
cache hit if the compiler configuration changes. Do not use
`cargo clean` or move/delete the shared root.

**Do not disable all statics on SM61:** DKR still covers only a subset
of input formats/layouts. Dense fallback, indirect paged-attention
variants, quantized linear operators, copy/swap, scale/FP8 paths, and
many Candle backend routines require the old library symbols. The
existing feature-controlled `exclude` filters in the quant/core
builders already omit several proven-replaced source files. More
pruning needs a symbol/dispatch audit and a grouped full-inference
test; it must not break the fallback on legacy GPUs.

**Note on concurrency:** cudaforge 0.1.6 stores a single cache manifest
per build directory. Use one Cargo build at a time when sharing this
cache root, especially with differing feature configurations. The
bootstrap utility should only run when no build is active.


### DKR-R1.20.2 — Guard against global NVCC flag cache invalidation

Before any R1.21 quantized-linear/paged-attention source patch, run the
read-only flag audit on the current checkout:

    python3 tools/dkr_nvcc_flag_audit.py

The audit lists every existing MISTRALRS_DKR_* argument in the three
static CUDA builders, the source files that reference it, and its header
references. DKR_NVCC_FLAG_AUDIT=PASS means **no new global DKR macro
was introduced compared with the frozen R1.20 baseline**. It does not
certify runtime parity or imply that global flags no longer exist.

The existing four core DKR guards are confined to definitions in
src/cuda/sort.cu and src/cuda/input_packing.cu. Their macros are
currently passed as global KernelBuilder::arg() values, so they
invalidate the hash of all core object files on a feature change.
The existing MISTRALRS_DKR_REDUCE_NORM argument in quant has the
same builder-wide cache impact. These are **pre-existing** features:
do not remove or rewrite those compiler options during cache
bootstrap or R1.21 source staging, or the 82 preserved object-cache
entries could be invalidated before benefit is measured.

For a future isolated compile domain, keep unchanged files with the
same baseline NVCC arguments in the stable cache. Only build
sort.cu and input_packing.cu (or extracted smaller translation
units) in their own cudaforge builder and static archive with their
per-family defines, and link both required libraries. Preserve their
FFI symbols and static fallback layout when an external DKR operation
is not selected. Any actual build-script split requires a **separate**
ABI/link parity qualification: it is not introduced by this audit.
For now, new DKR-R1.21 families must use external PTX packages and
optional Rust dispatch **without adding any new global -D DKR flags**.

The safe order remains: bootstrap once with
tools/dkr_cuda_cache_bootstrap.py --apply; inspect the feature-flag
audit; then run the next grouped campaign only after R1.21 source
staging. Do not change CUDA_COMPUTE_CAP, the user's feature set, or
the cached CUDA build root during routine patches.


## DKR-R1.21/S1-S3 — Quantized GEMV and short paged decode (staged)

The SM61 R1.20 baseline **21/21** remains authoritative. R1.21 adds
three **opt-in, not yet GPU-qualified** external PTX source families:
`quant-gemv`, `paged-decode` and `nvfp4-gemv`. All three use cudaforge PTX build,
typed manifests, thread-local prevalidated launch plans, and existing
CUDA static FFI fallback. No DKR-specific NVCC flag is added to
the static builders. Total staged external source families: **20**.

- **DKR-R1.21/S1** (`mistralrs-core/dkr-quant-gemv`, forwarding to
  `mistralrs-quant`): FP8 E4M3 blockwise dense decode GEMV with
  F32 block scales `[ceil(N/block_y), ceil(K/block_x)]`, and
  packed E2M1 MXFP4 decode GEMV with E8M0 group32 scales.
  Separate MXFP4 plain/bias CUDA symbols avoid optional-argument ABI
  ambiguity. F16 and BF16 inputs, contiguous tensors, batch 1..8,
  and dimension bounds are checked before selecting the DKR route.
  Values are decoded and multiplied **within the dot-product kernel**;
  there is no intermediate dense dequantization allocation.
  The original CUDA matmul is retained for M>8, unsupported dtypes,
  layouts and feature-disabled builds.
  GPU parity test: `dkr_quantized_gemv_gpu`.

- **DKR-R1.21/S2** (`mistralrs-core/dkr-paged-decode`, forwarding to
  `mistralrs-paged-attn`): VLLM K/V cache short-context decode
  (max context 256), online softmax, 128 CUDA threads per
  (sequence, query head), with K blocked-x and V token-strided
  representations and U32 device page table/context lengths.
  F16/BF16/F32 query, key, value must share the same dtype.
  **ALiBi, sinks, softcapping different from 1.0, sliding windows,
  FP8 cached K/V and long partitioned attention** remain on the
  existing static paged attention v1/v2 path.
  GPU parity test: `dkr_paged_decode_gpu`.

  This first short-context kernel favors ABI/functional coverage,
  **not** superior performance over the heavily optimized
  paged-attention v1/v2 kernels. Its per-token block reductions can
  be expensive. Benchmark the static route before considering
  production promotion.

- **DKR-R1.21/S3** (`mistralrs-core/dkr-nvfp4-gemv`): fused SM61
  NVFP4 W4A16 decode GEMV with packed E2M1 weights, E4M3 scales in
  groups of 16 and F32 per-row global multipliers, with separate
  bias/plain typed ABIs. The new opt-in route is selected only for
  CUDA F16/BF16 activations that match the layer output dtype, rank-2
  weights, unquantized activations, and up to eight rows. Other
  shapes, input types, and modes retain the resident Candle E7
  or existing portable implementation.
  GPU parity test: `dkr_nvfp4_gemv_gpu`. This is an experimental
  reference kernel; it has not been performance-qualified against E7.

- **Not yet ported:** NVFP4 E7-scale-rounding parity and its per-row
  global scaling/bias semantics, MXFP4 large-M GEMM, FP8 indexed
  expert GEMM, routed FP8/MXFP4/NVFP4 GEMV with arbitrary expert
  indirection, scaled FP8 KV attention, MLA, FlashInfer and
  speculative/partitioned attention. These are follow-on R1.21
  stages and **cannot** be classified as migrated by the first
  three source families.

For the *next* grouped qualification, use the stable cache root
`MISTRALRS_CUDA_BUILD_ROOT=.../target/dkr-cuda-cache/sm61`
without changing `CUDA_COMPUTE_CAP` or resetting `target/`.
When testing new features, retain the same feature list and use the
existing DKR single-campaign test script; inspect the static cudaforge
log for "All library kernels up-to-date" before inferring cache hits.


### R1.21 CPU-only preflight and same-binary A/B selection

The new families are not performance-qualified yet. Run the
CPU-only manifest preflight **before** the next grouped CUDA build:

    python3 tools/dkr_r121_manifest_preflight.py
    python3 tools/dkr_nvcc_flag_audit.py

These commands use synthetic PTX bytes only to exercise Python
manifest packaging and the source contract. They do not verify real
NVCC/PTX signatures, Rust linking, GPU parity or performance.

Each new R1.21 feature also has a once-per-process runtime escape
hatch, so no second Cargo feature compilation is required for
static-vs-DKR performance comparisons:

| Environment for a separate process | With the R1.21 feature compiled |
|---|---|
| MISTRALRS_DKR_QUANT_GEMV=0 | Use existing static FP8/MXFP4 path |
| MISTRALRS_DKR_PAGED_DECODE=0 | Use existing static paged-attention v1/v2 |
| MISTRALRS_DKR_NVFP4_GEMV=0 | Use existing resident NVFP4 E7 path |

Without those environment settings, an explicitly compiled DKR
feature enables the new route on eligible inputs. Settings are
sampled once per process and must be specified **before launching**
each benchmark. Change them between process invocations (not during
a running process). Their effect is only on the DKR routes whose
features were compiled. The static implementations remain linked
and available as fallback; no global NVCC flag is added.

Preserve the same binary, same device, model, input shapes, warmup
and memory configuration across both runs. Compare decode TPOT,
prefill TTFB, kernel time, VRAM, and semantic/numeric parity. Only
promote a DKR route if both accuracy and performance acceptance
criteria are met. The R1.20 21/21 test baseline is unchanged until
the next grouped campaign qualifies these R1.21 kernels.


### DKR-R1.21/S4 — Fused quantized prefill GEMM, experimental

- Feature: `mistralrs-core/dkr-quant-gemm`, forwarding to
  `mistralrs-quant/dkr-quant-gemm`.
- External cudaforge PTX: `kernel-packages/cuda/quant-gemm`.
  F16/BF16 activations, FP8 E4M3 blockwise weight + F32 scale
  matrix (ceil(N/by), ceil(K/bx)), or packed MXFP4 E2M1 weights
  with E8M0 scales (N,K/32). MXFP4 has two *different* plain/bias
  signatures. No intermediate float weight tensor is allocated.
- Strict shape gate: 2D contiguous tensors, valid dtype/layout,
  K aligned to 32 for MXFP4, `M > 8` (the earlier GEMV path handles
  M=1..8), and checked u32 launch dimensions. Fallback: existing
  specialized static GEMM or vendor implementation.
- Kernel geometry: 16x16 thread blocks, one output element/thread,
  F32 accumulation. This intentionally prioritizes a transparent
  parity oracle. It is **not** a claim to outperform tiled/specialized
  static SM61 GEMM, cuBLAS(Lt), or existing DP4A variants.
- Test: `dkr_quantized_gemm_gpu`, with odd M/N tiles, FP8 row
  blocking and both MXFP4 bias signatures. Remains ignored until
  the next grouped SM61 qualification.
- Once-per-process runtime comparison:
  `MISTRALRS_DKR_QUANT_GEMM=0` disables this route even with the
  feature compiled, for a same-binary DKR-versus-static performance
  A/B. By default the route is enabled only if its feature is built.

S4 adds exactly one external source family (six PTX manifests); it does
**not** change any static CUDA build.rs or add a global NVCC macro.
The first grouped CUDA qualification should confirm real PTX compilation,
Rust type safety, numerical parity, and whether this generic GEMM
should remain a testing reference rather than a production dispatch.


### DKR-R1.21/S5 — Explicitly scaled FP8 paged KV decode

This second R1.21 follow-up adds the optional
`mistralrs-core/dkr-paged-fp8` feature and external
`kernel-packages/cuda/paged-fp8` PTX package. It extends the
short-context online attention reference to **FP8 E4M3 cached K/V**
without allocating F16/BF16 copies of KV pages:

- F16/BF16/F32 query and output, typed E4M3 K/V pages using the
  VLLM K-blocked-x and V-token-strided representations.
- Mandatory **separate F32 K/V scale pointers**, each with one
  element (scalar or one-element contiguous view, normalized to [1] in the plan); the CUDA kernel applies the K scale in each dot product
  and the V scale in the online weighted sum. No optional/null scale
  argument is admitted in the new ABI.
- U32 page tables and context lengths, contiguous tensors, head
  size 64..512, context up to 256 tokens, stable online softmax.
- ALiBi, sinks, softcapping != 1.0, sliding windows, absent scales,
  non-E4M3 cache types and longer/partitioned contexts retain
  the previously qualified static paged-attention v1/v2 path.
- One test `dkr_paged_fp8_decode_gpu` with both nontrivial scale
  values. Separate runtime A/B using
  `MISTRALRS_DKR_PAGED_FP8=0` in a new process to force static
  CUDA dispatch without rebuilding the Rust binary.

This is a parity/ABI source candidate, **not GPU-qualified yet**.
The R1.20 GPU result remains 21/21. With S4 and S5, the branch now
stages **22 external source families** and two additional ignored GPU
tests beyond the three already introduced in R1.21/S1-S3.

**Still outstanding in R1.21:** long/partitioned attention v2,
MLA, per-token dynamic FP8 scales, copy/swap/reshape KV kernels,
indirect expert grouped GEMM, and comprehensive performance
qualification on the Pascal device. Do not recompile the static
CUDA fallback during source-only staging; keep all 82 persisted
objects and all existing static NVCC flags unchanged.


### DKR-R1.21/S6 — Partitioned long-context paged attention v2

**Staged, not GPU-qualified.** This source migration adds
`mistralrs-core/dkr-paged-v2` (forwarding to
`mistralrs-paged-attn/dkr-paged-v2`) with a new `paged-v2`
cudaforge external source family. The **R1.20 21/21** SM61 baseline
is still the only authoritative GPU test result.

Two individually typed and prevalidated CUDA kernel ABIs are used:

| Operation | Arguments | CUDA grid | Purpose |
|---|---|---|---|
| `attention.paged.v2.partition.p256.v1` | Q, K, V, U32 page table and context lens, F32 partial output, F32 logsumexp, eight U32 scalars, one F32 scale | [query_heads, sequences, ceil(max_context/256)] | Online softmax over a 256-token partition, normalized partial vector and stable LSE |
| `attention.paged.v2.merge.p256.v1` | F32 partial output, F32 logsumexp, output, four U32 scalars | [query_heads, sequences, 1] | Merge normalized partial vectors weighted by exp(LSE-part - LSE-max) |

Each stage has **three typed profiles**: F16/BF16/F32 query and K/V.
CUDA block size is 128 threads. The partition size is compiled into
the entry identity and kept constant at 256, not supplied as a
mutable runtime kernel parameter.

The scratch layout is F32
`[sequence, q_head, partition, head_dim]` for partial outputs
and F32 `[sequence, q_head, partition]` for log-normalizers.
The implementation reuses `paged_attention.rs::workspace_ensure`:
the **same long-lived GPU scratch pool as legacy v2**, with 16-byte
separation and capacity growth that retains previous buffers. Two
CUDA launches are enqueued in the same current stream and the pool
guard is held across both launch submissions. CPU-side plan selection
and PTX module resolution happen only on cache misses via the
thread-local `LaunchPlanCache`. Output allocation remains the
usual one output tensor per invocation.

**Admitted runtime scope:** dense VLLM K-blocked-x and V-token-strided
layouts, contiguous F16/BF16/F32 tensors with matching Q/K/V types,
U32 page tables and lengths, grouped-query attention with whole
heads-per-KV-head ratio, supported head widths 64..512, and requested
maximum context in **257..32768** tokens within the page table
capacity. A zero context length for a particular sequence writes
zero output; empty trailing partitions produce LSE=-infinity and
zero partial outputs, rather than NaNs.

**Explicit fallback:** FP8 K/V, windowing, ALiBi, attention sinks,
non-default softcapping, unsupported strides/shapes, and out-of-range
contexts are dispatched through the original static v1/v2 path.
No legacy NVCC build flag or `CUDA_COMPUTE_CAP` change is introduced.
The new feature is opt-in at Cargo compile time and
`MISTRALRS_DKR_PAGED_V2=0` disables it **at process startup**
for a same-binary static-vs-DKR A/B experiment. The existing static
v2 code and its workspace remain linked and available.

**Source-only preflight** (never compiles CUDA):
```bash
python3 tools/dkr_r121_manifest_preflight.py
python3 tools/dkr_r121_v2_numeric_preflight.py
python3 tools/dkr_nvcc_flag_audit.py
```
The numeric oracle checks exact log-sum-exp partition/merge identities
for empty/ragged contexts, 256-token boundaries, up to 8193 tokens,
and logits with high dynamic range. It does **not** verify NVCC,
the Rust adapters or the actual device output. The ignored
`dkr_paged_v2_gpu` test covers F16/BF16/F32, grouped-query
attention, 777/513-token contexts, partial tails, and launch-plan
reuse. It is selected by the existing single grouped GPU campaign.

With S6, **23 external source families** and **28 new R1.21 manifest
profiles** (22 previous + 6) are staged. The next DKR GPU suite
should select 27 targeted tests (21 previously qualified + six new).
Do not claim runtime qualification or throughput improvement until
that grouped campaign runs. Performance of the reference 128-thread
per-partition reduction may still be inferior to the specialized
static v2 implementation; compare steady-state TPOT, first-token
latency, VRAM and long-context scaling before promotion. CUDA Graph
capture and cross-stream reuse of the shared pool require independent
qualification before production promotion.


### DKR-R1.21/S7 — KV page copy + VLLM slot write (staged)

**Source-stage only**; neither new symbol has been compiled or qualified
on SM61. The authoritative GPU baseline remains R1.20 21/21.

One external cudaforge PTX source family
`kernel-packages/cuda/kv-cache-ops` provides **four fixed ABI
manifests** (two operations; three typed variants for writing):

| Operation | ABI | Runtime scope |
|---|---|---|
| `kv_cache.copy_blocks.bytes.b256.v1` | i64 per-layer K/V pointer tables, i64 [pair,2] page mappings, 2 u32 sizes, 2 u64 per-page byte counts | Device-resident metadata already built by `copy_blocks`; F16/BF16/F32/E4M3 pages are copied *bit-exactly*. |
| `kv_cache.reshape_store.vllm.b256.v1` | key/value inputs, K/V caches with VLLM layout representations, i64 slots, 6 u32 sizes | Contiguous F16/BF16/F32 token-to-page writes (including -1 padding). F16/BF16/F32 signatures are distinct; no runtime optional scale pointer. |

- Feature: `mistralrs-core/dkr-kv-cache-ops` forwarding to
  `mistralrs-paged-attn/dkr-kv-cache-ops`.
- Cached typed plans with fixed 256-thread launch blocks; source CUDA
  units compiled with the same `cudaforge` PTX builder as existing DKR.
  No new `build.rs` NVCC option and no change to
  `CUDA_COMPUTE_CAP` or the 82-object static CUDA build cache.
- `copy_blocks` validates dtype/device/stream/capability, mapping
  bounds and cache layout exactly as before. DKR reuses its single
  existing device metadata allocation and keeps the static FFI for
  repeated destinations and cross-copy dependencies. Graph capture
  remains explicitly rejected by copy_blocks, as in the original path.
  The DKR copy operates at the byte level, preserving FP8 payloads,
  including NaNs/signs, without decoding or quantizing.
  Because the ABI passes **indirect raw pointers**, the manifest's
  pointer-table tensors are read-only but the referenced cache pages
  are modified. The Rust adapter enforces their lifecycle and
  provenance; this operation must not be exposed to unchecked
  arbitrary pointers.
- `reshape_and_cache` uses the signed i64 slot mapping directly,
  skips negative/padding entries and checks out-of-range slots before
  writing. For contiguous, same-dtype, supported VLLM layouts it
  writes K/V without intermediate buffers or host readbacks.
  FP8 E4M3 *scaled writes* still require the existing
  `scaled_convert` static backend; similarly rank/stride variants
  excluded by the ABI retain that backend. Slot mappings are assumed
  to have unique non-padding destinations, as in the original kernel;
  host-side DKR dispatch does not read back device metadata.
- **Still not migrated**: arbitrary `swap_blocks` between CPU and GPU,
  cross-device transfers, dependent/cyclic block
  permutations, and scoped FP8 quantized cache writing. Those are
  transport/synchronization contracts, not safe byte-copy
  substitutions; retain the original path until separately proven.
- Runtime same-binary opt-out: set `MISTRALRS_DKR_KV_COPY=0` or
  `MISTRALRS_DKR_KV_RESHAPE=0` before launching the process.
  Production should keep DKR features off until GPU parity and A/B
  profiling pass.
- Two ignored GPU tests: `dkr_kv_cache_copy_gpu` exercises the
  view-offset/FP8 exact-copy path, and `dkr_kv_cache_reshape_gpu`
  exercises page layout, padding and plan residency.
- Source-only tests (no Cargo/NVCC): `python3
  tools/dkr_r121_manifest_preflight.py`, `python3
  tools/dkr_r121_s7_cpu_preflight.py`, and `python3
  tools/dkr_nvcc_flag_audit.py`. They validate packaging, a
  standalone CPU byte/layout oracle, and static NVCC flags, but
  cannot prove GPU execution or precision. When all staged S1–S7
  packages are included, expect **24 PTX source families and 32 new
  R1.21 manifest profiles**. The deferred grouped GPU campaign
  would select **29 GPU-designated DKR tests**, plus one CPU-only
  mapping-independence test covered by the same test-name filter,
  provided no further tests are added in S8.
