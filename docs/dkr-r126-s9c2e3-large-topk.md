# S9C.2E.3 — Large Top-K & Cooperative Radix

Source delivery. No CUDA compilation, device execution, link or numerical
qualification is claimed. Preserve the grouped S10 campaign.

## Shared kernels, typed profiles

The historical iterative and CUB ranked Top-K device bodies have been
extracted into large_topk_device.cuh. The legacy wrappers and the external
typed translation unit share the bodies. radix_topk.cuh exposes the same
select and final pack-sort bodies in its external mode, omitting CUDA
Runtime host launch wrappers from that translation unit. The historical
numerical algorithms have not been replaced.

| Family | Logical profiles | Block |
| --- | ---: | --- |
| Iterative first stage | 7 | 256 |
| Iterative second stage | 4 | 256 |
| CUB ranked two-stage radix | 4 | 256 |
| Cooperative ranked selector | 11 | 1024 |
| Ordinary final pack sort | 1 | 32 |

The seven first-stage profiles keep the single F32 softmax mode, the
three batched softmax dtypes, and the three batched ranked dtypes.
The four second stages keep single split, single packed, batched packed,
and ranked packed output. Missing softmax bindings are absent from the
typed ABI; the wrapper supplies nullptr only to template-disabled branches.
The CUB radix path retains 256 threads and eight items per thread.

27 new logical profiles, two physical images per target CC, maximum nine
ABI arguments. One ordinary image and one cooperative/select-sort image.
Cudaforge owns compilation and paired CUBIN staging. Both preserve the
historical fast-math flag and watch the shared device headers.

## Cooperative launch admission

The generic runtime reserves the manifest operation namespace
cuda.cooperative.*. This is included in the existing manifest and scope
identity, without a manifest schema change or per-GPU code. The ordinary
launch_config and ordinary PrevalidatedLaunchPlan APIs reject that namespace.

PrevalidatedCooperativeLaunchPlan is a distinct type; it never exposes
the CudaFunction to a launch callback. Its submit method binds the admitted
arguments using cudarc's launch_cooperative. On plan miss:

1. Query cooperative-launch support, SM count, and opt-in shared limit.
2. Bound actual shared memory by the existing manifest/device ceiling.
3. Set a stable maximum dynamic shared attribute for the function, rather
   than lowering the attribute for another cached shape.
4. Calculate actual active blocks per SM with the function's occupancy API.
5. Admit only complete groups:
   resident = active_blocks_per_sm * sms;
   groups = min(rows, floor(resident / ctas_per_group));
   grid = groups * ctas_per_group.
6. Verify the adjusted launch request and keep the exact grid in the plan.
   The launch method checks the stream/context and submits cooperatively.

No residence, unavailable cooperation, or out-of-range shape can reject
the cooperative algorithm before launch. A driver or profile error is
not an eligibility signal; it propagates. No fallback is attempted after
a cooperative launch has been submitted. Prepare the final sort before
any launch as well. Graph capture behavior and residency/performance
qualification remain pending. The 96 KiB typed manifest ceiling is not
increased to accommodate devices with larger shared-memory limits; their
largest legacy chunk sizes can therefore differ.

## Workspace and vector alignment

C++ RowState has 3*256 histogram words, a 32-bit arrival counter, and
two arrays of 256 words. Raw size is 5124 bytes, padded to 5136 by
16-byte alignment, i.e. 1284 U32 words. Exact size, alignment and member
offsets are static_asserted in the shared CUDA header. Strict Rust
uses RADIX_STATE_WORDS_PER_ROW = 1284; the legacy host query remains only
outside strict closure. These static assertions still need actual NVCC
compilation in S10.

The cooperative path retains its 1184-byte fixed shared scratch and
2048-byte headroom. It calculates vector width from the gcd of the
historical maximum vector width, row width, and actual device-address
alignment, so an offset contiguous view cannot request a misaligned
vector load. Vector variants: F32 1/2/4, F16 and BF16 1/2/4/8.

The public four large Top-K routes preserve their existing output formats,
allocation and workspace-reuse scheme, retaining CUDA storage/stream
guards across both launches. Checked element spans and device-address
arithmetic replace the previous unbounded raw pointer offsets. Single
packed output now rejects vocabulary indices outside the exact F32
integer bound, as the batched paths already did. Both iterative and
CUB stages have static plus actual dynamic shared memory admission.

## Algorithm selection

For ranked requests: cooperative if eligible and fully resident, otherwise
historical CUB two-stage radix when chunk=2048, 8<=k<=16 and candidate count
<=2048, otherwise the historical iterative two-stage algorithm. Softmax
requests use the original iterative two stages. The entire pair of launch
plans is prepared before the first launch, then cached by geometry, dtype,
mode, device, and vector width. No legacy CUDA FFI fallback is used by
strict mode. The shared workspace is initialized by the existing allocator
and reused on the same stream, relying on the historical successful radix
state cleanup. CUDA errors must not be retried on a potentially dirty state.

## Model-scoped requirements and budget

The producer dkr_large_topk_requirements(dtype, DkrLargeTopKMode) returns
exact typed profile requirements. For ranked mode, it conservatively includes
all compatible vector widths, final sort, CUB and iterative fallback; request
shape, alignment and actual occupancy can differ at inference time.
F32 ranked uses eight profiles; BF16/F16 ranked use nine each. F32+BF16
ranked union = 14 profiles, two physical manifests, in the synthetic source
fixture. The images still contain all compiled entries. This does not prove
automatic model-loader coverage or actual full model fit.

9 additional guarded FFI declarations: eight launch wrappers and one host
workspace query. Residual 28 = 14 Argsort + 12 sampling/DFlash + 2 graph copies.
Master projection is 4186+27 = 4213 logical profiles, 546+2 = 548 physical
manifests (upper bound). The active 4096/512 limits are unchanged.

## Qualification

Python recipe self-tests and audit use synthetic PTX for SM61/SM80.
The source audit checks 27 wrapper signatures, modes, optional ABI,
vector widths, guards, runtime admission markers and a scoped fixture.
The CUDA-free runtime tests dkr_cooperative_grid_residency and
dkr_cooperative_namespace_is_not_ordinary check arithmetic and identity.

The CUDA-gated core tests dkr_large_topk_geometry_and_requirements and
dkr_large_topk_cooperative_contract_gpu remain for S10. The latter is
ignored by default and requires actual cooperative selection, rejects
ordinary API admission, checks the CUB and iterative routes, and replays
the same cooperative workspace twice. Existing tests
cuda_ranked_topk_matches_cpu_across_dtypes_ties_and_offsets,
cuda_ranked_topk_radix_matches_realistic_vocab_and_cross_chunk_ties,
cuda_ranked_topk_radix_preserves_special_value_contract and
cuda_ranked_topk_cooperative_boundaries_and_fallback_match_cpu also apply.
Use a qualification scope containing every required profile. Actual GPU
capability, ABI, compilation, link, numerical parity, graphs and throughput
remain to be established.
