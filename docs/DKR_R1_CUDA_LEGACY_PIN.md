# DKR-R1 - Candle CUDA Legacy pin and reproducible lock

**Owner:** `haricot/mistral.rs` branch `dynamic-kernel-runtime` (no Candle branch rename).
**Candidate branch:** `integration/dkr-r1-cuda-legacy-pin`.
**Upstream source:** `haricot/candle:cuda_legacy` at immutable commit
`d99f6d77e507688bd9e4b654fbe20ac9e18238fc` (Candle 0.11.1).

## Purpose and boundaries

The existing Mistral F32 dynamic-provider proof was made using Candle
`9cba2e898af3d25a89257f1f2ec5acece0404dcb` (0.11.0).
That proof remains historical and valid **only for its exact original identity**.
It is not numerical, integration, or performance evidence for this new candidate.

This candidate updates only the four workspace Candle Git dependencies:
`candle-core`, `candle-nn`, `candle-flash-attn-v3`, and
`candle-metal-kernels`. Each dependency uses the **same exact SHA**, not
a floating branch reference. Newer BF16, FP8 and FP4 software paths are
provided by Candle's Legacy branch and remain opt-in as applicable.

The Mistral `dynamic-kernel-runtime` branch stays the DKR-R1 host.
`candle:cuda_legacy` supplies the kernel/backend source. Candle
`extended` is a later composition target, **not** this base.
The `candle-legacy-dispatch` format-neutral linear contract is a
separate candidate to inspect before reusing adapters; it is not merged here.
No existing ASD policy, CUDA source, PTX, benchmark or release build was changed.
The new independent typed ABI is documented in `DKR_R1_TYPED_ABI_AND_COVERAGE.md`.

## Completed lockfile gate

**Do not promote with an old Cargo.lock.** Cargo has now resolved the new Candle
workspace version 0.11.1 and its dependencies, including registry `zip` 9.1.3.
The lockfile was updated by Cargo, not by manually replacing source hashes.

In a Cargo-equipped checkout of the candidate branch, run:

```sh
cargo update -p candle-core
cargo update -p zip@9.5.0 --precise 9.1.3
cargo metadata --format-version 1 --no-deps --locked > /dev/null
git diff -- Cargo.toml Cargo.lock
```

This resolved the new lockfile **without compiling Rust or invoking NVCC**.
Review the lockfile delta before committing it. A changed Candle source
identity is expected to invalidate the corresponding build cache entry;
unchanged environments and A/B/A2 profile replay must continue to reuse
exact binaries and PTX.

The scoped CPU-only runtime test has passed on the new lock. A new SM61 CUDA
qualification remains required; do not launch multi-architecture rebuilds.
Continue to use the original F32 binary and PTX for historical A/B/A2 replay;
do not label them as evidence for the new pin.

## DKR-R1 next contract

Generalize typed kernel ABI and verified module loading independently of
ASD, while preserving the F32 compatibility adapter and strict fail-closed
tests. `candle:cuda_legacy` is a source dependency, not a mandate to move
`mistralrs-kernel-runtime` into Candle. No direct Extended merge is authorized
by this pin alone.
