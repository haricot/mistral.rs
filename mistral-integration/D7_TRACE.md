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
--native-cupti --cuda-root /opt/cuda129` prepares this feature. Its first native
run builds the engine integration, reuses the verified Candle reference cases
and existing Cargo target, and emits the new executable at a separate path.
Nsight and NVTX are not required. The GPU capture covers the bench region;
T1 remains a separate semantic capture. No exact semantic-to-kernel attribution,
engine speedup or optimized ShortConv kernel is claimed.

`mistral-integration/cupti-check` typechecks the actual recovered module and runs
its disabled path on CPU in CI. Its dynamic-loading check is independent of
the production CLI's existing dynamic-linking configuration. Actual collection
still requires the CUDA 12.9/sm_61 machine.
