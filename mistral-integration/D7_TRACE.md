# D7 generic diagnostic consumer

Candidate based on `asd_runner_standalone`, with a composed Candle Extended pin.
The new revision requires fresh CUDA/sm_61 tests. Historical validation is not
transferred to it. Production Extended is not updated by this branch.

Enable `d7-trace` on `mistralrs-cli` to compile the producer hooks. Set
`D7_TRACE_DIR`, `D7_TRACE_RUN_ID` and `D7_TRACE_NONCE` to capture a one-shot CLI
run. `D7_TRACE_CAPACITY` limits retained records; `D7_NVTX=1` tries optional NVTX.
The CLI owns capture and writes events/manifest at shutdown, including errors
that unwind through main. A killed process may leave no complete manifest.

Hooks cover normal text forward steps, LFM layer and ShortConv signatures,
existing CUDA graph lifecycle/capture/replay and existing allocator queries.
Request identity across workers, speculative checkpoint commit/rollback and
CUDA graph node attribution are unavailable. No extra allocator query, tensor
value read or per-operation device synchronization is inserted.

LFM decode keeps the existing arithmetic and state update. Its observed
projected input, prior state and weights describe a future stateful candidate.
The generic Candle stateful reference is tested independently; no fused kernel
is selected by these hooks.

The coordinated launcher/reader and hardware commands are in
[Flow-Adaptive-Tuner D7_TRACE.md](https://github.com/haricot/flow-adaptive-tuner/blob/d7-engine-objective-abcd/docs/D7_TRACE.md).
Detailed capture is diagnostic only. D7 timing rejects its variables/injected
collectors. A trace does not qualify a gain or establish GPU optimality.


## Native CUPTI collector recovery

`flow-tuner-cupti` reuses the in-process Rust collector recovered from
`flow-adaptive-tuner_v0.3.19-r2d_fresh-post-c3-residual.zip`
(SHA-256 `26a7127789bca33866d0cc030c4e465d599545555223b52ebc0a339cbfa1c8a1`).
The historical source used cudarc 0.19.8. This branch uses the existing Candle
lockfile's cudarc 0.19.10 with its `cupti` feature and dynamic linking; Candle
and its CUDA kernels are not downgraded. The recovered module typechecks with
that version. Historical hardware success does not qualify this new integration.

Build the CLI with its existing CUDA features plus `d7-trace flow-tuner-cupti`.
Set `CUDA_PATH`, `CUDA_HOME` and `CUDA_ROOT` to the CUDA 12.9 installation and
put its CUPTI 12 library directory first in `LD_LIBRARY_PATH`.
`FLOW_TUNER_CUPTI_OUT` enables collection; when absent, the module returns
without initializing CUPTI. `FLOW_TUNER_CUPTI_MODE=runtime-driver-diagnostic`
collects kernel/copy/memset and runtime/driver activities.

The benchmark starts collection after model loading and benchmark warmups,
and calls `finish()` before returning its results. JSON keeps the existing
`mistralrs.flow-tuner-cupti.v2` schema and adds the D7 nonce/run ID plus a flush
completion flag. A support query rejects an incompatible sm_61 collector.

The coordinated tuner command `tools/d7-t123-local.py --resume <report-dir>
--native-cupti` prepares this feature, deriving the toolkit root from the active
`nvcc`. An explicit `--cuda-root` remains available. Its first native
run builds the engine integration, reuses the verified Candle reference cases
and existing Cargo target, and emits the new executable at a separate path.
Nsight and NVTX are not required. The GPU capture covers the bench region;
T1 remains a separate semantic capture. No exact semantic-to-kernel attribution,
engine speedup or optimized ShortConv kernel is claimed.

`mistral-integration/cupti-check` typechecks the actual recovered module and runs
its disabled path on CPU in CI. Its dynamic-loading check is independent of
the production CLI's existing dynamic-linking configuration. Actual collection
still requires the CUDA 12.9/sm_61 machine.

The 2026-10-18 native hardware run produced 73,751 activity records, including
9,903 kernels and zero dropped records, then exited with SIGSEGV after printing
benchmark results. Of these records, 39 kernels and 58 copies have zero
timestamps and remain incomplete. That run remains `COLLECTOR_FAILED`; JSON output
does not override a failing process receipt.

The collector now disables activity production, synchronizes all visible CUDA
primary contexts used by the Candle workload, restores the caller's original
context and flushes the buffers. This is one diagnostic boundary, not a
per-operation synchronization. It then calls `cuptiFinalize` from a CUDA driver
API exit callback before writing `finalize_complete: true` and
`cuda_synchronized: true`.
This follows NVIDIA's [CUDA 12.9 detach guidance](https://docs.nvidia.com/cupti/12.9.1/main/main.html#dynamic-attach-and-detach)
so teardown is serialized with other CUDA threads. Successful finalization
invalidates the subscriber; Drop never unsubscribes or reinitializes CUPTI
after it. Cleanup is attempted once even when flushing or JSON writing fails.
The missing explicit detach is a code defect. The successful follow-up below
confirms normal termination in the tested workload; it does not establish the
original crash's stack trace.

`python3 mistral-integration/cupti-check/test_lifecycle.py` runs the actual Rust
module with synthetic CUDA/CUPTI shared libraries. It checks finish, Drop,
sync/flush/write/finalize errors and a missing exit callback, including ordering and
absence of duplicate cleanup. This is CPU FFI validation, not a GPU test.

## Successful CUDA 12.9/sm_61 capture

The follow-up `d7-T123-20261018T193823.655236Z.zip` confirms that tested code
pin `642ff4198383c6a9647770f359f7fb07e0360275` completes native collection
and exits normally for the bounded LFM2.5-2.6B F16 bench workload. Archive SHA-256:
`ac1e66053dcb1a29bfcd50c1b302492997992579da0960fadbd6016b71bab9d5`.
Both inventories verify, and replaying raw T1/CUPTI data reproduces the stored
reports exactly.

T1 records 17,680 events on its first attempt with zero drops/open scopes.
Native T2 records 73,158 activities, including 9,877 kernels, with zero drops,
zero incomplete/invalid timestamps, successful CUDA synchronization, flush and
finalization, and process exit 0. The preceding SIGSEGV and zero timestamps are
absent in this corrected run. These results validate this tested diagnostic
scope; they do not establish universal stability or a performance gain.

T3 observes 330 F16 stateful ShortConv decode calls with contiguous x
`[1,2048,1]`, state `[1,2048,3]` and weights `[2048,1,3]`, K=3 and zero
offsets. The separate 48-case reference pass is hash-verified reused evidence.
ASD candidate submissions remain zero because the stateful F16 ABI is not yet
implemented. Exact semantic-to-kernel attribution and untraced engine A/B
qualification remain separate work. The native collector is now hardware
validated for this scope; Nsight is optional.

## Stateful F16 ShortConv ABI candidate

The new Candle candidate pin is
`84a0b24c030d3af912dcab9a3939f8b0de1dbd32` on `d7-stateful-f16`
([Candle #28](https://github.com/haricot/candle/pull/28)). It adds the external
`asd.xswyn.f16.k3.v1` ABI: read-only x/state/weight, fresh output and next-state
buffers, and batch/channel scalars. F16 products are rounded before the same
F32 reduction used by the tensor reference. Bias and the other LFM operations
retain their existing arithmetic.

`decode_conv` now delegates to Candle's stateful dispatcher. Its LFM scope
describes model context; the nested stateful scope owns the generic operation
signature and actual selection/submission events, avoiding duplicate counting
of a single decode operation. Without an eligible explicit profile it keeps
the reference path, including autograd and unsupported layouts/kernels/dtypes.

A `candidate` profile requires `CANDLE_ASD_ALLOW_STATEFUL_CANDIDATE=1`, scoped
to this stateful F16 raw-CUDA operation. The tuner stages it under the new
report, preserves the production ASD store, and verifies the external artifact
hash and decision identity. It never labels the candidate as promoted.

The CUDA + `d7-trace` CLI command
`mistralrs d7-shortconv-f16-validate --out DIRECTORY` calls Candle's shared
validator using this same compiled engine. It checks bit-exact same-device
output/state parity over 768 steps and 48 forks, immutable retained states,
10 layout/autograd fallbacks, and 408 actual traced candidate submissions.
The report binds the fresh validation nonce and verified artifact SHA-256.
The coordinated `--stateful-f16` launcher runs it before real-model capture,
then requires both semantic submissions and the candidate kernel in native
CUPTI. It reuses the Cargo target and emits the new engine at a separate path.

The hardware-validated collector code pin remains
`642ff4198383c6a9647770f359f7fb07e0360275`. Its successful diagnostic evidence
does not validate this new Candle/kernel/dispatch candidate. Fresh physical
parity and real-engine evidence are required, followed by separate untraced
A/B gain qualification. No optimized candidate performance is claimed yet.

## Physical candidate parity and T1 drain correction (2026-10-20)

Archive `d7-T123-20261020T152235.349360Z.zip` (SHA-256
`252a751ea8bf7f81142a6eb1e78392b2c25bca50a191829b324d3175cc7b30f0`)
verifies all 66 outer and 31 engine inventory entries. Replaying each T1 and
native T2 report reproduces the stored evidence exactly.

The same-device candidate validator passes: 48 cases, 768 steps, 48 forks,
10 fallbacks, immutable inputs/retained states, bit-exact outputs and states,
408 traced candidate submissions and a complete 3,802-event validation trace.
This physical result applies to Mistral `7d2eff3156c6fb085fb17c7b3fb9eb98e13d20b2`
and Candle `84a0b24c030d3af912dcab9a3939f8b0de1dbd32`.
All three real-model T1 attempts observe 330 active submissions of the staged
artifact. The separate native T2 contains 330 `d7_shortconv_f16_k3_v1` kernels
among 65,887 activities, with no drops/invalid timestamps and successful
synchronization, flush, finalization and process exit.

T1 still remains incomplete: each attempt loses one final `op.end` from an
unclassified worker `index_select`. `Capture::finish` previously closed the
collector while admitted work was still alive. The coordinated Candle pin
`8a0e6d099e680496c623c116e0ebc1fd516ec3ce` stops admission and drains existing
scopes/payload builders for up to one second before freezing records. A timeout
stays explicitly incomplete. The regression fails on the old collector and
passes on the corrected code. Fresh hardware capture must confirm this repair;
the uploaded archive is not relabeled as complete. Native CUPTI is unchanged.
No exact per-operation GPU attribution, full-engine output parity or untraced
engine gain is established by these separate diagnostic captures.

## Generic CPU clocks and same-process CPU/GPU correlation

Candle pin `5d166459144ed2b830f1d237286e851e654efb7b` adds system thread IDs,
Linux monotonic timestamps and per-thread CPU-time counters to semantic events.
A generic application-owned `ScopeObserver` brackets actual submitting threads;
Candle does not depend on CUPTI or select a model-specific correlation policy.

With CLI features `d7-trace flow-tuner-cupti`, an active semantic capture and
`FLOW_TUNER_CUPTI_MODE=cpu-gpu-correlated`, the existing native collector enables
external-correlation activity. The bridge pushes/pops Candle scope IDs with
CUPTI CUSTOM0, exports external-ID/API correlation records and system API thread
IDs, and retains two host/CUPTI clock calibration brackets. CPU and GPU evidence
share one run, nonce and process. Clock interpolation is only for visualization;
attribution uses observed IDs. Kernel names or interval containment are not
substitutes for correlation. CUDA graph node attribution and implicit propagation
across asynchronous workers remain unavailable.

The bridge stops admission, drains admitted scopes for up to one second and
closes its CUPTI gate before finalization. A late scope destructor cannot invoke
CUPTI after detach. Push/pop errors, drain failure, unknown coverage or discarded
records cannot become a complete correlated capture. Native storage is bounded
by `D7_TRACE_CAPACITY` (default one million records per record family); losses
are counted. Existing noncorrelated collector modes keep their teardown protocol.

The generic tuner CPU adapter adds procfs thread counters/states, scheduler and
context-switch counters where available, plus optional perf call stacks. These
are diagnostic captures. Scope CPU time covers its own submitting thread, not
unobserved worker CPU time; sampling gaps and missing capabilities are explicit.
The local launcher `--cpu-gpu` rebuilds the revised pinned engine using the
verified Cargo cache and captures one diagnostic pass, without an A/B campaign.
Previous native GPU hardware evidence stays attached to its old code pin.
Fresh physical CPU/GPU correlation on CUDA 12.9/sm_61 is still required.
