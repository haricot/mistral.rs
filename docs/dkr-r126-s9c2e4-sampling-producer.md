# S9C.2E.4 — Sampling & DFlash: producer source delivery

This is an intermediate producer commit, not FFI closure. The existing Rust dispatch remains unchanged. No CUDA compilation, device execution, link, numerical qualification or model fit is claimed.

| Image | Entries | Profiles |
| --- | --- | ---: |
| 1 | Top-1 stage 1 (six dtype/mode variants), stage 2 (five optional-output variants), categorical stages (two), ranked sample (one) | 14 |
| 2 | DFlash greedy and probabilistic, independent F32/BF16 hidden, predecessor and successor types | 16 |
| 3 | Categorical and bounded top-k sparse rejection | 2 |

The three existing translation units and the typed image wrappers share device bodies; historical algorithms and supplied uniform inputs are unchanged. Stage 2 Top-1 excludes absent output pointers from its ABI. Single-row packed-only is not an existing route; the batched packed-only path covers that request. Categorical stage 2 retains 16 KiB dynamic scratch for the existing 2048-element chunk. The rejection top-k stage retains its one-thread block. All images preserve historical fast math.

The new producer contains 32 profiles, three physical manifests per target and at most 16 ABI arguments. Projected master inventory is 4245 logical / 551 physical, an upper bound; active limits remain 4096 / 512. The 28 residual FFI declarations remain unchanged, including the 12 belonging to this phase.

Cudaforge is the compiler owner. The Python producer stages verified PTX and synthetic self-tests; no Python compiler invocation. The source audit verifies the wrapper inventory and typed argument names/types/order and tests a negative ABI fixture.

Next: checked spans and stream guards; prevalidated cached Rust plans (all stages admitted before submission); exact capability producers; guarded legacy FFI and cumulative audit updates. Preserve ring reservation/abort/event behavior and rejection status semantics. DFlash predecessor dependency must remain within each sequence. Group actual compilation and numerical tests with S10.
