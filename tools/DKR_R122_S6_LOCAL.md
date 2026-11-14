# S6 local qualification

Initial joint SM61 A/B smoke + existing typed qualification, not exhaustive
numerical coverage. No CI. Use the existing S4 worktree with resolved
lockfile; set DKR_S6_ORIGINAL_REPO to original checkout. Run:
    bash tools/dkr_r122_s6_local.sh --preflight
    DKR_APPROVE_S6=1 bash tools/dkr_r122_s6_local.sh

Preflight checks persistent objects, actual GPU capability, resolved graph
and original typed packages. Reuse typed artifacts only when tracked package
sources match the qualified revision and manifest audit passes; not
cryptographic source provenance. Native builder checks ABI before the
expensive build. One nextest archive build is reused for two runs, switching
only Candle's provider. Dedicated persistent Cargo target; original Mistral
object cache is passed unchanged, but static Candle compilation can still
be required. Never clean.

A/B covers basic native tensor ops, ten GGUF codecs and three batches, SIMT
and GGUF MoE with partial N. Tolerances provisional; report every maximum.
Not an independent CPU oracle, nor exhaustive for layouts, conv or MoE.
Existing typed tests supply numeric oracles. Evidence gate checks native
family coverage and presence of typed adapter success; not every typed
operation is guarded. Missing coverage gives REVIEW_REQUIRED, never full
qualification PASS. Outputs retained under target/dkr-r122-s6.

## S6.1 recovery: nextest metadata compatibility

Nextest 0.9.47 fails on the arm64ec-pc-windows-msvc dependency target.
Require a recent nextest (policy floor 0.9.145, not the first known compatible
version) before any PTX construction. Install a prebuilt binary from the
official nextest documentation; no Rust/CUDA rebuild required for installation.
For the already completed native PTX build at 7b3b19cf8b62, set
DKR_S6_REUSE_NATIVE_RUN to that run directory. The script checks matching
Candle revision, successful original builder log, and repeats the artifact
inventory/hash audit before using the original PTX in place. No copy or clean.
This relies on trusted local files and does not claim signed provenance.
