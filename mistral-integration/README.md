# Mistral.rs ASD integration

- Orchestration branch: mirror_orch (source of the workflow).
- Permanent integration branch: asd_runner. Do not call this branch cuda_asd_runner_v2.
- Mistral.rs baseline: master at 20f607a77abd4738aa5594ee93a393a9fd284605.
- Reused Pascal adaptation: cuda_sm61_runner at d3813494903e5cfca01e12df596f3fe14a5d22ba (existing PR #5 remains unchanged).
- Candle source: cuda_asd_runner_v2 at ac983d16750a136a8563def01926ee83a7831d22, pinned as a commit, never followed as a floating Cargo dependency.
- Hosted SM61 compiler recipe: Candle mirror_orch at c565a96bd05374706bd13951f6d126f331f749ae.

The workflow is bootstrapped in asd_runner because GitHub workflow_dispatch only registers workflows present on the default branch. A push to asd_runner starts one campaign. The prepare job resolves Cargo.lock from the pinned Candle commit and pushes a unique temporary integration ref. CPU format/check/tests, hosted CUDA 12.9.2 sm_61 compilation and physical Pascal smoke test must all pass for the same candidate SHA and lock hash. Only the last job fast-forwards asd_runner; no force-push or source branch rewrite. The temporary ref is deleted only after promotion, while the annotated proof tag remains.

This initial gate proves packaging and basic CUDA tensor execution; it does not establish end-to-end ASD dispatch latency or model accuracy. Those require separate benchmarks after the bootstrap campaign. The future cuda_asd_runner_extended or cpu_moe_expert integration should add separately pinned source manifests and tests rather than silently tracking moving branches.

Workflow and bootstrap policy live on mirror_orch. When the branch is reviewed, merge its workflow into master to enable explicit manual dispatch and expand orchestration. Until then the asd_runner copy is executable via its push trigger.
