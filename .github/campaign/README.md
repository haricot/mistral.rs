# Mistral standalone campaigns

Both SM61 workflows use one CUDA-enabled configuration. CPU execution tests run
from that configuration on the hosted builder. Physical Pascal execution uses
the exact archived test executables, without Cargo or NVCC on the GPU runner.
A distribution built without CUDA is outside this campaign's validation scope.

## Inputs and source preparation

`config.json` declares independent sources, their prerequisites, the common fork
`master`, CUDA features and pinned compiler tools. A campaign resolves every Git
head once, resolves the CUDA base image to its registry digest, preserves the
final compiler image by its Docker content ID in a verified reusable GitHub
artifact, and writes
`snapshot.json`. Candle heads must have a matching physical validation ref.

The legacy source targets Candle `cuda_legacy`. ASD, LFM2.5 DSpark and Bolt target
Candle `extended` and declare only the legacy prerequisite. They are prepared in
separate lightweight checkouts with scoped resolution caches. Other features and integration conflict resolutions are not
copied back into these sources. An unknown functional merge conflict stops
preparation. Metadata and Candle-pin conflicts have explicit, narrow handling.

`bootstrap.json`, when present, binds reviewed initial adaptations to an exact
old source and master. These temporary source commits are candidates, not
published standalones. The mapping is ignored as soon as its old source or
master changes. Normal future preparation merges the selected master and the
declared prerequisites directly.

Prepared refs are addressed by their source inputs. Commits have deterministic
timestamps, and reports carry run identities outside the source tree. Retrying
unchanged inputs therefore reuses the same prepared commits. Publication of a
standalone does not introduce a second metadata-only candidate on the next run.

## Compilation and execution

The hosted builder compiles sources sequentially in one workspace with a shared
Cargo target directory, download cache, sccache and CUDA object cache. Independent
Rust feature implementations still need independent compilation. Unchanged
dependencies and kernels can be reused; the integrated build has its own proof.

All builds use the `campaign` correctness-test profile. Rust optimization is
limited to reduce test compilation time; CUDA kernels retain their build-script
`-O3` options. These archives are not release performance binaries. Release model
benchmarks require the release profile and separate measurements.

`compiler-env.sh` fixes Rust's CPU target to baseline `x86-64`, overriding the
repository's `target-cpu=native` and inherited encoded flags. A hosted machine's
AVX-512 instructions must not become unconditional instructions on the Pascal
runner's CPU. Build proofs include the effective flags and compiler version.
Changing these flags invalidates executable reuse; source candidates and CUDA
kernel cache inputs remain independent of the host CPU instruction set.

`build.sh` runs Cargo check, builds the nextest archive, and runs the CPU test
selection from the CUDA-enabled executables. The mandatory CPU backend test
uses `Device::Cpu`; CLI argument, speculative decoding, model and quantization
tests run alongside it. GPU-only tests are selected separately.

Hosted CPU execution needs to load the CUDA-linked binaries without an NVIDIA
driver installation. `cpu-runtime.py` exposes the Toolkit link stub as
`libcuda.so.1` in a temporary loader path for that child process only, checks
that CUDA initialization returns `CUDA_ERROR_STUB_LIBRARY`, and removes the
path afterwards. The stub is created after archiving and is never included in
the executable archive or used by the physical GPU runner.

Compilation caches and per-attempt logs/archives are saved even when a CPU test
fails. A failed attempt still cannot satisfy any CPU or GPU publication gate.
Compilation phases and their logs are also streamed to the Actions console.

Each bundle contains compiled tests, nextest, JUnit evidence and `build.json`.
Proofs bind the candidate, lockfile, toolchain/image recipe and archive SHA-256.
An unchanged build bundle may be reused after these checks. Cache hits alone
never satisfy an execution gate.

The GPU runner downloads the bundle, restores each exact source revision, and
runs nextest with workspace remapping. Every component must execute the mandatory
CUDA backend test on a physical SM61 device. ASD resources are required for the
Extended profile. CUDA driver and runtime libraries must be compatible with the
pinned hosted build. Full Bolt/DSpark model inference and latency are separate
qualifications; these model-free tests do not claim them.

## Publication

After all hosted and physical reports pass, promotion checks source and Candle
leases again. A single atomic Git push updates the tested standalones and the
selected final branch, with explicit leases and archival refs. A concurrent edit
rejects the whole publication. Extended also qualifies and publishes the legacy
prerequisite with its own Candle `cuda_legacy` build.

Runner v6.4 binds the reusable Extended source manifest to the current run alias and verifies its Candle physical proof. Both Mistral profiles execute the hosted archives. The existing
four-profile runner and Candle profiles are preserved. Nextest is supplied in
the build artifact, so this path does not install a compiler on the GPU runner.

Local infrastructure tests:

```bash
python3 -B -m unittest discover -v -s .github/campaign/tests
bash -n .github/campaign/build.sh .github/campaign/gpu.sh
```
