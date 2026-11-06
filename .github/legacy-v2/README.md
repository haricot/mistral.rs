# Legacy v2: separate source preparation

This workflow prepares three new source branches. It does not replace `legacy`,
`extended`, any existing standalone, a release, or an OCI image.

| Branch | Composition | Explicit prerequisite |
| --- | --- | --- |
| `legacy_v2` | pinned master + reviewed Legacy compatibility layer + migrated packed FP8 provider + BF16/FP8 standalone tests | none |
| `contrib/lfm25_dspark` | only the pinned DSpark functional delta applied to the new base | `legacy_v2` |
| `contrib/nemotron_bolt_dflash` | only the pinned Bolt/DFlash functional delta applied to the new base | `legacy_v2` |

The contrib branches are source inputs for a future integration campaign. They are
not copies of Extended and do not contain each other's functional delta or ASD.
This first version deliberately pins the existing source SHAs. If an input moves,
the workflow stops until its manifest and migration have been reviewed.

## Formats and retained Legacy changes

BF16 and generic NVFP4 were already migrated to master 0.10.0 in the qualified
Legacy source. `sources.json` lists all 49 retained files and the exact source
patch checksum. The layer contains feature propagation, Pascal BF16 MoE and
paged-attention support, scalar/blockwise FP8, SM80-only grouped FP8 guards,
packed NVFP4 dispatch, lossless E4M3 scale handling, CPU/GPU-separated tests,
the real FP8 KV Standard-backend layout fix, the pinned official zip dependency,
and the pinned Candle dependency/lockfile. It excludes old campaign metadata.

The older FP8 standalone adds a separate packed W8A16 linear/grouped-expert
provider. Its migration retains the newer upstream dynamic provider selection,
Metal path, 3-D dequantization and generic routed kernels. The SM61 packed path
runs before generic routing. It retains the one-copy residency policy and fails
closed on unsupported packed geometry. Original BF16/FP8 standalone GPU tests
are included in the generated source but are not run by this CPU workflow.

The FP4 standalone also contains Bolt/DFlash work. Its historical module blob is
recorded as source provenance, while the reviewed, newer Legacy NVFP4 module is
used in `legacy_v2`. The entire historical FP4 branch is not merged into Legacy.
Bolt/DFlash-specific code remains in its own contrib branch.

## Execution and publication

The workflow is `.github/workflows/mistral-legacy-v2-prepare.yml`, on the isolated
branch `ci/legacy-v2-preparation`. A push changing its recipe starts a complete
preparation; it can also be dispatched manually:

```bash
gh workflow run mistral-legacy-v2-prepare.yml \
  --repo haricot/mistral.rs --ref ci/legacy-v2-preparation
```

The preparation is deterministic for identical source/recipe/toolchain inputs.
Run IDs and attempts are attached to evidence, not the source identity. Candidate
commits, trees, dependency locks and the full Git bundle are hashed. The CPU jobs
run the actual locked CLI check, with portable x86-64 flags and no GPU/model load.
Cargo caches only accelerate the check; they never replace its successful result.
Each job emits evidence for the exact candidate/run/attempt. A final atomic push
creates or updates only the three new branches and their immutable preparation
refs. Replacements require destination leases and retain immutable backup refs.
Source movement, missing CPU evidence or altered bundles block the whole update.
All three CPU checks must succeed before any destination branch is changed.

This is **CPU source preparation**, not CUDA/physical/model qualification. It does
not register a runner, load weights, start containers, create a GPU job, promote
a production target, publish an image, or rebuild the CUDA SDK. The existing SDK
digest is recorded for the subsequent integrated CUDA campaign. New physical
format tests belong to that integrated campaign, not a separate FP4 prerequisite.
Actual model-execution errors remain distinct from informational numerical
metrics. Existing accepted VL/GGUF reports are not changed by this workflow.
