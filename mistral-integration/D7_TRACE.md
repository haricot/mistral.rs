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
