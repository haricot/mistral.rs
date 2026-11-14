# DKR-R1.22/S5 — Unified runtime evidence

Opt in with DKR_TELEMETRY=1 before first use. It is read once per process.
No extra CUDA events, GPU synchronization or per-launch IO. Disabled handles
are empty. Enabled counters use atomic increments; handle registration uses
a bounded map (4096 rows). Not intended for performance benchmarks.

Native source sites record submitted only after successful driver launch
submission, not GPU completion. Embedded PTX and static FFI selections
are counted separately; a void FFI call is not proof of submission.
Typed adapters record adapter_ok/adapter_error after callback completion,
not a GPU completion or independently verified launch count.
Physical cache events use SHA-256, target and symbol; do not sum these as
logical dispatch counts. Plan cache hit/miss/eviction rows are independent.

The runtime exposes runtime_evidence() and attaches a snapshot to
execution_evidence(). 3 selected test functions emit snapshots via a scope
guard. Extend scope guards to remaining tests in S6. For eventual nextest
qualification use DKR_TELEMETRY=1 and --no-capture; each test process must emit
one cumulative snapshot. Parse with tools/dkr_r122_s5_evidence.py --log FILE.
No automatic graph replay accounting.

The previous S4 candidate passed dependency closure and eight CPU tests.
S5 advances the Candle pin; re-resolve Cargo.lock with the S4 tool and repeat
the CPU tests. No GPU build yet. Do not run cargo clean or set CUDA_COMPUTE_CAP.
