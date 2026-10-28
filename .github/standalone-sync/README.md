# Tracking upstream master changes for SM61 MistralRS standalones

This is a **read-only** preflight, not an automatic rebase or promotion.

- The repo `master` branch is the upstream fork code baseline.
- `mirror_orch` is the orchestration branch; the independent
  `Mistral Master Sync` workflow merges `master` into it.
- After a successful sync, `Mistral standalone master compatibility`
  compares the exact remote `master` SHA to every tracked standalone SHA.
- It uses `git merge-tree --write-tree` to detect **textual** conflicts
  without changing the index, a worktree, or any branch.
- For each branch it reports changed upstream paths, overlap with local
  standalone changes, and the recorded source `base_master_sha`.
- `MERGE_CLEAN_UNTESTED` does **not** mean compatible at the API, numerical
  or GPU levels. Upstream LFM3 introduces GDN, memory planning, attention,
  KV cache and GGUF changes that require real compilation and SM61
  qualification after composition.

The existing `Mistral Extended SM61` campaign currently composes four
historical components (legacy, ASD, LFM2.5 DSpark, Nemotron Bolt DFlash).
It does **not** yet automatically compose/promote the newly introduced
`fp8_mistralrs_standalone`, `bf16_mistralrs_standalone`,
`fp4_mistralrs_standalone` and `UQFF_draft_standalone`.
They are monitored rather than added prematurely to the shared GPU gate.
Each standalone has a different Candle pin/runtime contract.

## Manual actions (no GPU)
```bash
gh workflow run mistral-master-sync.yml --repo haricot/mistral.rs \
  --ref mirror_orch -f master_sha=0494ee4a2e3ffbb0817142f903a535b47495c06a
```

The read-only compatibility workflow is triggered by a completed successful
Master Sync workflow and can also be dispatched on its own.

## Suggested promotion policy
1. Merge `master` into `mirror_orch` (fail closed on conflicts).
2. Inspect the branch-by-branch impact report.
3. For each requested standalone, prepare a new immutable candidate from
   its previous SHA and the pinned new master SHA; do not overwrite the
   original branch before qualification.
4. Run CPU checks → CUDA SM61 compile → physical GPU test of the exact
   candidate archive → optional LFM3 text/VL/GGUF model-level tests.
5. Publish only with exact branch leases and validated per-component proof;
   one shared GPU session may cover multiple independent binaries.

The script never downloads model weights and never runs GPU tests.

## Immutable candidate staging (after preflight)

After a successful `Mistral Master Sync`, the follow-up workflow
fetches the exact SHA report and creates immutable refs **only** when
`git merge-tree --write-tree` is clean. It publishes:

`integration/staged-master/<master12>/<source12>/<standalone-branch>`

Each staged candidate is an actual merge commit with the standalone
source SHA and the exact master SHA as its two parents; its tree is the
git-computed merge result. A source lease change or ref collision rejects
staging. No source branch advances, and no compilation or GPU qualification
is inferred from staging.

Only a later isolated CI qualification should advance the real branches,
using build archive hashes and full SM61 proof, with explicit scope.
A failed merge remains in the conflict report, not in a staged ref.

The PR-level `contracts` job uses a synthetic Git repository and runs
CPU-only tests of merge conflict classification, source lease changes,
staging ref uniqueness and non-mutation of tracked source refs.

## GreenIT
A single master snapshot feeds all eight read-only checks. Repeated
synchronization with the same SHA is a no-op. Staged refs are immutable
and reused only when their exact candidate commit SHA matches.
This does not reuse an unrelated executable or skip required physical GPU
tests.
