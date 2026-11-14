# DKR-R1.22/S4 — Local-first dependency closure

Run in an isolated worktree of feature/dkr-r122-s35-shared-registry.
The qualified branch and persistent target/dkr-cuda-cache/sm61 are untouched.

    python3 tools/dkr_r122_s4_dependency_closure.py --self-test
    python3 tools/dkr_r122_s4_dependency_closure.py

The second command runs Cargo metadata (no compilation / no build.rs / no CI),
resolves the existing lockfile locally and checks the Linux SM61 feature graph.
It checks exact git sources, single reachable core/NN/kernels/cudarc/registry,
shared registry identity and features, and the exact Candle source candidate.
A second run with --locked can confirm lockfile stability. --offline is optional
after the git/index crates have been fetched. Review git diff --stat and the
lockfile changes before promotion. A passing report is dependency closure only.

Do not use cargo update without a specific reason; it updates unrelated packages.
Do not set CUDA_COMPUTE_CAP; do not run cargo clean. No new static CUDA build until
the grouped qualification is explicitly approved. The existing S9 script is
for the previous Mistral-only configuration, not the joint campaign.
On the eventual joint run, retain the original persistent Mistral cache root
and a stable target dir, then build the Candle external PTX packages and run
the integrated tests. Candle's new revision may still trigger static compilation:
a cache hit is not guaranteed by presence of the prior objects.
