# ASD Runner Standalone — Haricot Mistral.rs

This branch is based on the fork's canonical `haricot/mistral.rs:master`
(`20f607a77abd4738aa5594ee93a393a9fd284605`), **not** on the external upstream
repository and not on `mirror_orch`.

## Code provenance
- Functional delta reconstructed from `haricot/mistral.rs:asd_runner` at
  `080fdd5ccf1ff8b6bcb0df9e2dbc189763770588`.
- The snapshot changes 29 files including the Pascal CUDA fallbacks and the
  model-free `asd_sm61_smoke` test. No merge of orchestration workflows.
- `candle-core`, `candle-nn`, `candle-flash-attn-v3`, and
  `candle-metal-kernels` are pinned to `haricot/candle:extended`
  commit `66c4651cc83f9a43e79a4aa09302aac9bc763d2c`.
- Candle's matching physical validation archive is
  `archive/extended/sm61-validated-38064926957-1`.

## IMPORTANT: qualification pending
This is a candidate, NOT a verified standalone release. The `Cargo.lock`
git-source SHA entries were switched mechanically to the Candle extended
commit. They require independent Cargo metadata/lock resolution against the
actual Candle source, followed by CPU/format/Clippy checks, hosted CUDA
12.9.2 sm_61 compilation and physical Pascal SM61 execution. Do not promote
or publish a binary based solely on this source reconstruction.

The original `asd_runner` branch, its proofs and any archive refs remain
unchanged. Future composition/promotion is managed by `mirror_orch`.
