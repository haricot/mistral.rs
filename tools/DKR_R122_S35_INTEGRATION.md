# DKR-R1.22/S3.5 — isolated joint integration candidate

This branch is NOT the qualified DKR branch. It pins all four direct Candle
workspace crates and dkr-kernel-registry to b231d3c4e07b04c0df788ee252979b1890430dd6. The common crate is
dependency-neutral and lives next to Candle for atomic revision pinning.

The typed manifest validator and launch admission remain in Mistral.rs;
the logical registry selector and LRU are shared with Candle. With
dkr-sm61-experimental, typed and native modules use the same context-local
content-addressed module/function cache on Candle's CudaDevice.
Logical namespaces remain independent; package directories are not merged.
Identical PTX can share a physical module only within the same device instance.
The older F32 proof-of-concept adapter still uses the historical custom loader.
The 128-entry typed preparation cache and operation-specific plan keys remain.
No global context, new CUDA kernels or graph capture changes.

The Cargo.lock file is not resolved on this branch; resolve and inspect it
before qualification. No compilation or CI performed. Both ABI paths, package
construction, existing static fallback linkage, context isolation, and parity
need the grouped SM61 qualification. The old qualified branch/pin remain intact.

Build neutral registry tests separately on CPU (no CUDA build):
    cargo test --manifest-path dkr-kernel-registry/Cargo.toml

This command is for the Candle checkout, not Mistral.rs. Do not set
CUDA_COMPUTE_CAP or clean existing persistent CUDA caches.
