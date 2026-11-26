# S9C.2E.2 — MoE Router & Small Top-K

Source delivery. No native CUDA compilation, device execution, link or numerical
qualification is claimed. Preserve the grouped S10 campaign.

## Shared device code and scope

The historical router and small Top-K device bodies have been extracted from
sort.cu into router_topk_device.cuh. The legacy wrappers still use the same
templates; the typed wrapper translation unit uses inlined device bodies and
C-linkage entries. The router launch bounds remain 128 threads, minimum one
block per SM; launch block is (32,4,1), four rows per block. The number of
values per thread remains compile-time specialized for each supported expert
count, without runtime switching among expert counts inside the kernel.

There is one deliberate numerical/safety correction shared by both legacy
and typed paths: padded lanes of sub-warp routers remain masked after score
transformation, including when a negative selection bias is present. Previously
a padded lane could obtain a zero sigmoid/softmax score and beat a valid
negative biased score, potentially indexing the expert scale out of range.
This does not redefine legacy degenerate-row behavior (all -infinity/NaNs).

The master has 132 router profiles (11 expert counts × 3 dtypes × 4
bias/scale cases). Each expert count has one physical image with 12 entries.
Small Top-K has 12 logical profiles because the historical host selects four
different block sizes (64/128/256/512) according to input width. Preserve those
launch geometries rather than forcing one block size. All small Top-K entries
are in one physical image. Both producers retain the legacy fast-math flag.

Total: 144 new logical profiles, 12 physical images per target CC. Maximum
ABI arguments: 15. No dummy device pointers for absent bias/scale; wrappers
supply nullptr internally only to the compile-time-disabled template branches.
Flags are transported as U32 and converted to bool inside the wrapper.

Small Top-K dynamic shared bytes = columns * (sizeof(dtype) + sizeof(bool)).
Manifest ceilings encode the selected block tier, capped at 48 KiB for the
largest tier. The typed plan checks actual dynamic + static shared bytes
against the ordinary device limit at preparation, not at every launch.
The typed adapter checks positive dimensions, signed indexing overflow,
contiguity, actual storage range and optional tensor device/dtype/length.
It keeps the historical final_logits_row policy of the public CUDA Top-K
wrapper; it does not change it into batched Top-K.

## Model capability producer

The following exports are available with dkr-static-closure-probe:

```rust
dkr_moe_router_requirement(experts, dtype, bias_present, scale_present)
dkr_small_topk_requirement(columns, dtype)
```

These return exact OperationRequirement identities accepted by the native
ModelCapabilityDescriptor resolver. Use actual route activation dtype, actual
expert count and optional tensor presence for every layer. Union requirements
over target/draft and all allowed modes. For the Small Top-K width policy,
select all reachable tiers conservatively if actual widths are not known.
Unsupported router counts follow the pre-existing generic algorithm rather
than an invented specialized profile, so that path's dependencies must also
be declared. These helpers are producer APIs, not yet automatically called
by every model loader.

Example: expert count 256, BF16, no bias/scale, resolves exactly
cuda.dkr_moe_router_e256_bf16_b0_s0.v1. The source audit constructs a
synthetic 144-profile master, materializes that single identity with an
active scope receipt, and verifies that the view has one logical profile
and one physical manifest. The underlying expert-256 image still has all
12 entries. This is a family-level synthetic test, not actual model coverage.

## Closure and budget

Six historical FFI declarations and their calls are guarded with
dkr-static-closure-probe; legacy public wrappers remain unchanged except the
padded-lane safety correction in shared code. Earlier cumulative source audits
recognize the new guards without changing their historical budget baseline.

Remaining declarations: 37 = 14 argsort + 21 sampling (including the host-only
workspace query) + 2 graph copies. No full link closure yet.

Master projection: 4042+144 = 4186 logical profiles; 534+12 = 546 physical
manifest upper bound. These are projections, not measured final package
inventory. The master must not be used as the runtime discovery or embedded
root. Active limits remain 4096 profiles / 512 manifests, with the other
per-profile/shard/image limits unchanged. Actual active model fit remains
unproven; materialization before runtime/embedded admission is mandatory
when the master exceeds those bounds.

## Qualification

The producer self-tests use synthetic PTX for both SM61 and SM80; the source
audit checks all explicit entry signatures, template parameters, image
assignment, optional ABI, shared-memory contract, guards and scoped fixture.
The CUDA-gated Rust tests cover capability identity enumeration and an
ignored GPU oracle for all expert sizes and optional combinations, several
score modes, offsets, ties, four block tiers and negative-bias padding.
S10 should supply the union of the profiles required by the qualification
tests rather than a single-model subset. These tests are not run during
source qualification. Actual Cudaforge compilation, CUBIN pairing, link
qualification and numerical/throughput measurements remain required.
