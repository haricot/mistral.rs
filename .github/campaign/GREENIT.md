# GreenIT: one Extended qualification, atomic Legacy promotion

## Observation (current orchestrator)
The workflow `.github/workflows/mistral-extended-sm61.yml` (trigger: manual
`workflow_dispatch` from `mirror_orch`) composes all five source candidates:

1. Legacy `allow_old_card`
2. ASD standalone
3. LFM2.5 DSpark standalone
4. Nemotron Bolt DFlash standalone
5. Extended composed binary

The hosted build step creates individual build/test archives for these exact
candidate SHAs (content-addressed; the compiler image and host builds can be
reused by validated hashes). `gpu.sh` runs **all** archives on one physical
SM61 runner in the same GitHub Actions `gpu` job, serially. This is one
runner reservation, but NOT one binary test. Do not silently drop per-component
evidence.

`campaign.py promote` verifies:
- Same run ID and final extended candidate SHA.
- Real physical GPU execution flag and target `sm_61`.
- Individual CPU/build proof and exact test archive hash for EVERY component,
  including Legacy.
- Nonzero physical executed GPU tests for EVERY component.
- Master/standalone/Extended source leases and Candle branch SHA leases.
- Atomic publication of `allow_old_card`, other component branches and
  `extended` as one Git push with exact expected old refs.

Therefore after a **successful** Extended run no separate Legacy run is needed.
The workflow `mistral-cuda-legacy-sm61.yml` remains a manual fallback for a
Legacy-only fix or when Extended cannot be qualified.

## Recommended invocation

```bash
gh workflow run mistral-extended-sm61.yml --repo haricot/mistral.rs --ref mirror_orch
```

Do not also dispatch `mistral-cuda-legacy-sm61.yml` for the same input
fingerprints. Promotion must happen only when the `promote` job succeeds.
A passed Extended-only test **cannot** substitute for an independently executed
Legacy archive of the exact Legacy candidate. Conversely, the Legacy archived
proof executed *inside* Extended's shared GPU job is sufficient.

## Evidence and carbon accounting
The actual GPU runner setup is deduplicated; each exact binary remains tested.
Operational targets: GPU runner allocations per integrated release 2 → 1;
setup overhead can fall, but GPU kernel test time is not reduced by 50%.
Measure minutes on the physical GPU, kWh from a measured power trace,
build-cache hit rate and redundant launches avoided. Do not call this an energy
or CO2 saving without measurement.

## Follow-up opportunities (not automatically enabled)
- Skip unchanged build work using the existing compiler/archive hash gates.
- Consider reusing a prior *physical* GPU proof ONLY under a strict
  content-addressed identity (source SHA, lock SHA, test archive SHA, full test
  scope, builder image digest, CUDA toolchain, features, card capability and
  driver). A mismatched or stale proof must force a real GPU execution.
- Keep an explicit override for Legacy-only recovery. Never rely on inference
  from an unrelated Extended executable to promote a distinct Legacy binary.

## Qualification
Run CPU-only contract tests first:
```bash
python3 -B -m unittest discover -v -s .github/campaign/tests
```
This staging branch has not run the physical GPU campaign.
