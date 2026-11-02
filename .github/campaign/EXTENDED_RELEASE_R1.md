# Extended release qualification

Standalone tests retain the campaign profile. A manual Extended campaign checks
and builds the final CLI once with the release profile and its delivery
features, including code-execution. The executable cache is bound to the exact
candidate, compiler environment and recipe. Its SHA256 is verified on reuse.

The runtime image copies that executable without compiling Rust. Before GPU
qualification, the hosted image job extracts and hashes the image payload,
checks its non-root configuration, and runs help and dependency loading with a
temporary read-only CUDA link stub. This is a loader check, not GPU execution.
The stub is never copied into the runtime image or the native archive. Matching
image archives can be reused by fingerprint after verification.

The physical runner retains its original kernel and ASD gates, then executes
Text, VL and GGUF CPU/CUDA comparisons using the delivery executable itself.
The model report binds the release manifest, binary SHA256, image proof,
candidate, profile, features, model policy, harness and original ASD report.
All three models are attempted. Numeric mismatches produce a collected report;
promotion still requires every comparison to pass the unchanged policy.

When FP8 KV qualification is enabled, the same delivered executable also runs
the pinned Text model with paged attention enabled, first with FP16 KV and then
with F8E4M3 KV. Both use 2048 cache tokens, one sequence, 128-token batching and
prefill chunks, and disabled prefix reuse. One warmup precedes three timed
requests of up to 32 generated tokens. Raw responses, actual token counts,
commands, logs and request timings are retained and hashed. Timings include
prefill and decode but exclude model loading; a throughput improvement is not
assumed. This extra Text benchmark is separate from the unchanged CPU/CUDA
Text/VL/GGUF numeric acceptance tests, which still disable paged attention.

The PR branch can be qualified before merge:

```bash
gh workflow run mistral-extended-sm61.yml --repo haricot/mistral.rs \
  --ref ci/extended-release-r1-single-build -f qualification_only=true
```

This mode cannot promote source branches or publish to GHCR. Promotion and
publication require an explicit dispatch on mirror_orch without
qualification_only, all original source/CPU/GPU/ASD gates, and accepted numeric
evidence for the exact release binary. Existing source-level model evidence is
still required as before; it cannot qualify a different release executable by
itself. Existing lease checks and atomic source pushes remain in force.

Publication loads the qualified image archive and checks its payload again;
it does not rebuild the image. The native archive contains the same executable.
The publication report records the immutable registry manifest digest as well
as the image and binary identities. Models remain external and read-only.

Physical inference uses the executable whose SHA256 was checked inside the
image, within the established runner. A complete Podman/CDI launch of the delivery
image is not claimed by the hosted loader check and remains a local integration
check. The image is not pulled onto the user's machine by the hosted jobs.
