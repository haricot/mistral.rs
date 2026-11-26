# S9C.2E.1 — capability descriptor and residual guards

## Scope of this delivery

17 historical core FFI declarations and their legacy call paths are explicitly
excluded by dkr-static-closure-probe. Existing typed routes and kernel profiles
are unchanged. 43 residual declarations remain (14 argsort, 27 sampling/routing,
2 graph copies). Source-level closure is not link or numerical qualification.

## Generic native descriptor

The Rust model_scope module defines ModelCapabilityDescriptor,
CapabilityComponent, OperationRequirement, ProfileDependency and ModelSelection.
A descriptor names all target/draft/auxiliary/adapter components, resolved facts,
their metadata hashes, allowed request modes and requirements, catalogue revision,
GPU CC and unresolved fields. No architecture-specific table is in the runtime.
The generic loader method DeviceMappedModelLoader::dkr_config_capabilities
collects actual configuration-derived per-layer attention geometry, paged KV
presence/ownership/layout, selected activation dtype and model dimensions.
Its metadata hash is for normalized configuration only. It deliberately has
empty requirements and unresolved checkpoint/quantization/device/request paths.
It is an API for loader integration, not yet invoked automatically during loading.

The resolver unions every component and request requirement, follows explicit
per-profile dependency edges to a fixed point (cycles terminate), and rejects
missing operations or implementation identities. A null implementation filter
retains every compatible implementation of that operation. Unknown *operations*
must not be treated as absent: the producer must include all reachable families,
or mark the descriptor partial. Partial descriptors cannot emit a deployment
selection through the native CLI. DeclaredComplete is an assertion by the
producer, not a qualification certificate. A metadata-only plan never admits
an image; the offline materializer and Rust verifier still run.

Example descriptor fragment (synthetic identities only):
```json
{
  "schema": "DKR-MODEL-CAPABILITIES-V1",
  "model": "example",
  "device_cc": 61,
  "catalog_revision": "example-revision",
  "components": [
    {
      "id": "target",
      "role": "target",
      "metadata_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "facts": {"expert_counts": [64,128]},
      "requirements": [
        {"operation":"example.router.v1","implementations":["example.router.e64","example.router.e128"]}
      ],
      "unresolved": []
    }
  ],
  "request_modes": ["greedy","topk"],
  "request_requirements": [
    {"operation":"example.sampling.v1","implementations":null}
  ],
  "unresolved": [],
  "coverage": "declared_complete"
}
```
Dependencies input is a JSON array, each item with
`{"profile":{"operation":"...","implementation_id":"...","target_cc":61},"requires":[{"operation":"...","implementations":null}]}`.
An empty array is only appropriate when the producer has proved there are no
additional dependencies. These illustrative identities are not deployed profiles.

## Offline use (CPU build only when deliberately requested)

1. Inventory the trusted master with dkr_r126_model_catalog_scope.py --master-root
   MASTER --inventory-out inventory.json.
2. Run dkr_model_scope_resolve DESCRIPTOR inventory.json dependencies.json
   selection.json. The native CLI writes the existing exact-selection format.
3. Run dkr_r126_model_catalog_scope.py --master-root MASTER --selection
   selection.json --materialize ACTIVE --json-out report.json.

The last step validates hashes and copies selected images / filtered manifests.
It writes dkr-active-scope.json containing profile count, device CC and canonical
scope digest. It does not prove complete model coverage or the PTX/CUBIN ABI.

## Immutable admission

The digest is SHA256 of a canonical array containing schema, exact device CC,
and all selected manifest semantic fields, sorted by
(operation, implementation_id, target_cc). It includes image hashes, optional
paired CUBIN, entry, block, shared memory and every argument field. Python and
Rust share a known-answer test fixture. It is not a signature or authentication
of an untrusted catalogue, and it does not replace existing module keys.

KernelExecutionCache::discover verifies the receipt against the verified
catalogue before publishing a cache. The embedded build verifies it and
embeds the receipt; no filesystem receipt is needed at runtime in embedded
mode. Set DKR_REQUIRE_MODEL_SCOPE=1 for scoped runtime admission to reject a
missing receipt. Process ScopeLatch rejects transitions between unscoped and
scoped mode and between different scoped identities. A scoped preparation
requires the actual CUDA CC to equal the receipt CC. No per-token selection
or environment mutation is added. Unscoped legacy behavior is preserved.

The latch covers consumers using KernelExecutionCache. It is not a guarantee
of complete Candle, paged-attn, direct PreparedKernel or graph integration.
There is no hot scope switch or cross-device heterogeneous scope. Use a union
for same-CC concurrent models or isolated processes. Later qualification must
exercise all relevant consumers, request modes and cache/graph lifetimes.

## Qualification

Python self-tests validate the offline receipt and the source audit. Rust unit
tests (dkr_model_scope_*) cover dependency cycles, conservative union, missing
variants, partial declarations, tampered receipts and process latch transitions.
They require a native CPU build and are not a substitute for CUDA qualification.
No CUDA build is required for the source gate. Existing 4042 logical / 534
physical projections are unchanged, and actual selected model fit is unproven.
