# Mistral.rs mirror orchestration

The default branch is `mirror_orch`. `master` remains the Mistral.rs code baseline.
`.github/workflows/mistral-master-sync.yml` fetches the current `master` SHA (or a supplied exact SHA) and merges it into `mirror_orch` without changing `master`; scheduling is weekly, with manual dispatch supported. Conflicts stop synchronization and are saved as an artifact. No force rewrite of source history is allowed.

The permanent feature integration branch is `asd_runner`. Its initial source is the existing Pascal adaptation `cuda_sm61_runner` (`d3813494903e5cfca01e12df596f3fe14a5d22ba`), plus pinned Candle `cuda_asd_runner_v2` (`ac983d16750a136a8563def01926ee83a7831d22`). The original PR #5 remains intact.

From Actions on the default `mirror_orch` branch, run **Mistral ASD Integration** via `workflow_dispatch` and supply the current full `asd_runner` HEAD SHA as `expected_seed_sha`. That pin prevents ambiguous campaigns and stale branch checkouts. The workflow resolves `Cargo.lock` and publishes a temporary `integration/asd_runner-<run_id>` ref. It then runs CPU checks, hosted CUDA 12.9.2 `sm_61` compilation, and the physical Pascal GPU smoke test. All three reports must match the same candidate and lock hash before it fast-forwards `asd_runner`, publishes a proof tag and removes the temporary ref. The CUDA compilation gate is not a GPU execution gate.

Do not enable implicit GPU dispatch on ordinary `master` syncs. Do not silently advance the pinned Candle SHA when `master` changes. Future `cuda_asd_runner_extended` and `cpu_moe_expert` integration require their own explicit source manifests and coverage. The smoke test proves basic device operation, not ASD policy or latency; expand the runtime proof before claiming performance improvements.
