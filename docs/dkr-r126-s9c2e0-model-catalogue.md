# R1.26 S9C.2E.0 — model-scoped catalogue contract

Status: source audit and offline explicit-selection prototype. No new CUDA
profiles, no runtime admission changes, no automatic model coverage claim.

## Why scope before discovery?

`KernelCatalog::discover` recursively enumerates manifests (512 limit) and
`load` admits profiles (4096 limit) before dispatch. The embedded build
also calls discovery. Filtering at resolve/launch time is too late.
A master store can retain all qualified specializations; a bounded active view
can be constructed without changing the existing runtime or the CUDA ABI.

The offline tool `tools/dkr_r126_model_catalog_scope.py` inventories trusted
master manifests beyond active limits, selects exact
(operation, implementation_id, target_cc) identities, checks device admission
and selected image hashes, and optionally writes a new root. It never mutates
or prunes the master. A partial shard retains the original profile fields;
unselected entries can remain in the copied PTX/CUBIN image. A whole manifest
is byte-identical. No symlinks are created. Production validation is still
performed by the Rust runtime / embedded build. A filtered shard reduces
profile count but does not reduce bytes of a selected image.

Example input schema (illustrative identity, not a real model allowlist):

```json
{
  "schema": "DKR-MODEL-SELECTION-V1",
  "model": "example model",
  "device_cc": 61,
  "profiles": [
    {
      "operation": "example.operation.v1",
      "implementation_id": "example.implementation.v1",
      "target_cc": 61
    }
  ]
}
```

Use the inventory to obtain actual identities. Do not select only by dtype
or model name. This prototype does not determine the full required set, does
not verify PTX signatures or CUBIN ELF targets itself, and must not be used
as deployment qualification. Master metadata has independent offline
bounds (16384 manifests, 262144 profiles); the production limits remain
512 manifests, 4096 profiles, 64 profiles/shard, 256 KiB manifest and
32 MiB per image.

## Future automatic selection (S9C.2E.1 / S9C.3)

A resolved capability descriptor must be derived after configuration and
weight metadata are known, before the first cache initialization/launch:

- Device CC, actual activation/state/cache/weight layouts and dtypes,
  quantization formats (including mixed GGUF layouts), all loaded layers.
- Model architecture and actual dimensions (MoE expert counts for every layer,
  optional bias/scale, attention geometry, GDN/SSM/ShortConv, vision/audio).
- Target/draft models, adapters, speculative mode, KV cache options,
  sampling modes, all request-time options allowed by the server.
- Shape/layout admission, dependencies of staged operations and all intended
  prefill/decode, batching and graph-capture paths.
- Required profile identities, package and image hashes, qualified implementation
  policy, provenance (configuration/weight metadata and source/catalogue revision).

Select all reachable profiles conservatively; unresolved facts require retaining
all relevant variants or failing preflight. Never omit an operation based solely
on one observed prompt or decode trace. No silent static fallback. A missing
required identity is an admission error before inference, not an invitation to
replace a fast specialization with a generic kernel. The scope descriptor must
be immutable while graph captures and prevalidated plans reference it.

Core and quant caches are process-global OnceLocks, and embedded::catalog is
also a singleton. The first prototype uses a selected filesystem root before
process start; the existing build-time embedded root can also be constructed
offline. Do not switch an environment variable under a live process. For
concurrent models, use a conservatively verified union (subject to the same
limits) or isolated execution contexts/processes. Automatic per-model
contexts would require cache/graph lifetime changes and a scope identity
in plan caches. Candle and paged-attn catalogue consumers also need explicit
integration review before claiming whole-engine scoping.

## Sampling constraints

The E.0 audit partitions the 60 declarations into 17 existing typed routes,
14 argsort, 27 sampling/routing (including one host-only workspace query),
and two graph copies. Existing typed routes need explicit closure guard
verification/qualification; no duplicate profiles should be added for them.

MoE router templates have 11 expert counts × 3 dtypes × 4 bias/scale
combinations. Keep the compile-time expert specialization (including
launch bounds), then scope by actual layer geometry. Do not merge templates
into a runtime switch merely to fit a global budget. DFlash independently
admits F32/BF16 for the hidden projection and both codebooks; select actual
dtype tuples, not just model activation dtype. Auto policy may admit both
greedy and probabilistic modes.

The cooperative ranked radix selection requires cooperative grid launch,
dynamic shared-memory admission, occupancy and residency proof; ordinary
launch can violate the software group barrier contract. Preserve vector
widths and fallback stages; no silent substitution. The host-only radix
workspace query can be replaced by a shared layout constant after proving
C++/Rust agreement. Argsort has an explicit public generic F64 path (not
in current typed ElementType) and U32 calls via topk_unsorted. Do not
declare either unreachable without evidence. The two graph copy wrappers
are driver copy operations, not CUDA compute kernels.

The audit counts uncompressed device entry scenarios, not a final profile
budget. Padding/preparation, optional output ABIs, cooperative contract
and further source/compiled reachability remain unresolved. The projected
4042 / 534 is unchanged. The selection prototype demonstrates independent
master/active bounds, not actual LFM3 or MoE model fit.
