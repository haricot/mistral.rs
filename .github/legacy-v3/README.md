# Legacy V3 pilot campaign

This is the validation campaign for the future common orchestrator. Its only
publication target is legacy_v3. Legacy, Extended and the standalone maintenance
branches are not promotion targets of this pilot.

The four source inputs are the reviewed BF16/FP8/FP4 v2 branches and residual
support, pinned in config.json and the support source contract. Candidate merge
timestamps are fixed from the inputs: the same inputs produce the same commit,
including the revision embedded by the application build. No campaign run ID or
attempt number is injected into application sources.

Stages:

1. Check protocol contracts and independently resolve the existing compiler SDK.
2. Verify original source CPU qualifications, lock inputs and compose the candidate.
3. Verify an original matching CPU-only baseline, or run cargo check on that exact
   configuration and retain its separate proof.
4. Resolve a matching original build, or compile one release CLI and a nextest
   archive with the same features/profile. Execute CPU contracts with CUDA linked
   through the hosted link stub; this is not a physical CUDA test.
5. Resolve complete previous physical evidence for the exact binary/archive/model
   inputs. A verified reuse skips GPU registration; missing or invalid evidence
   cannot authorize publication.
6. Otherwise execute the archived BF16, compact FP8/expert, NVFP4/MXFP4, SIMT MoE
   and actual FP8 KV tests on SM61. Run Text, VL and GGUF on CPU and CUDA sequentially
   with the same release CLI and pinned read-only offline model snapshots.
7. Reverify raw evidence and publish only legacy_v3, with a lease, atomic backup
   and receipt retaining the original producer run/attempt for reused evidence.

Text numerical checks remain blocking. VL/GGUF numerical differences, including
invalid numeric reference values, remain recorded as FAIL and are informative.
Runtime errors, absent responses, weight/payload integrity errors, missing tests
and incomplete executions remain blocking for every mode.

The NVFP4 Pascal adapter supports dense rank-2 W4A16; no general NVFP4 MoE support
is claimed. Integrated tests qualify the consumed Candle paths at its exact
revision, not every unrelated Candle API or standalone configuration.

Use workflow_dispatch on mirror_orch, with portable or avx2. A source publication
whose commit message includes [legacy-v3-full] also starts a complete campaign.
Pushes to ci/legacy-v3-campaign run protocol, preparation and CPU baseline checks.
All producers queue without cancelling consumers. A failed or interrupted full
campaign can be dispatched again with unchanged inputs: complete build/physical
artifacts are verified against the original producer and reused. Cargo caches
alone never count as evidence. Changed inputs require a complete fresh identity.
Do not start another campaign merely because an existing GPU job awaits Nicolas.

The current compatibility runner is updated in the existing V6 archive. The
dedicated legacy-v3.sh check/start commands require an explicit run ID, reuse the
existing local image and shared GPU lock, and expose only the pinned model files
read-only. The assistant never registers or starts it.

Runner V6.4.7 is required for this pilot: it verifies each pinned model file
before a read-only file mount, and supplies official GitHub CLI 2.103.3 through
a checksum-verified read-only tool mount. An offline CLI execution probe runs
before registration. This preserves the existing image and the source/build
identity. Campaign 42285229267 failed in artifact restoration because its image
lacked gh, before physical tests; its successful hosted CPU/build artifacts
remain eligible for original-producer reuse in a fresh complete campaign.

Artifacts live for 90 days. Missing or expired artifacts cause requalification;
durable artifact publication and the full Candle/Extended orchestration remain
separate acceptance criteria for the definitive workflow. A green pilot receipt
does not declare that definitive cross-project workflow implemented.
