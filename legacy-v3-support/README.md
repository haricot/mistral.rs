# Legacy V3 residual support

This branch starts on the same pinned master as the three v2 standalones.
No production format algorithm or wholesale old-Legacy import is added.

The standalones already contain shared SIMT, architecture and Candle
GGUF/embedding compatibility. The remaining integration support covers:

- feature unions across seven entry-point/dependency crates;
- three build-script resolutions combining BF16 and FP8 enablement on SM61;
- a shared MoE hardware-test gate, including BF16 only when enabled;
- CPU override and mixed-format device tests from old Legacy;
- locked sources, payload checks and all 50 old-Legacy path dispositions.

assemble.py is a local composition check, with no remote writes or promotion.
It verifies the locked standalone scopes and payload hashes before merging,
rejects unknown conflicts and preserves every input payload outside eleven
reviewed resolution paths. Resolutions derive from stable standalone sources;
format algorithms are not maintained in a second copy here. Failed worktrees
are retained for inspection.

Fetch the exact commits in sources.json, then run:

    python3 legacy-v3-support/assemble.py --output /tmp/legacy-v3-check

The result is a detached local candidate. Support CPU CI checks this real
composition and selected contracts without GPU initialization or model weights.
The aggregate feature cuda-legacy forwards the three cuda-legacy-* opt-ins.
ALLOW_LEGACY alone does not activate the Mistral adapters; the future campaign
must pass the Cargo features explicitly. Candle and Cargo.lock stay common.

NVFP4/MXFP4 fixtures belong to FP4 v2. Generic backend assertions and the mixed
BF16/FP8 hardware roundtrip belong here. Obsolete integration identity metadata
is excluded. old-legacy-inventory.json records each old path's destination.

CPU success does not qualify CUDA compilation, SM61 execution, full models or
performance. The subsequent campaign must still run those gates before any
promotion. This preparation creates no legacy_v3 branch, release or GPU task.
