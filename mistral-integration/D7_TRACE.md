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
