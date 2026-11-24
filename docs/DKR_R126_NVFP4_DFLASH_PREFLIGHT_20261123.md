# DKR R1.26 candidate — NVFP4 W4A16 ModelOpt / DFlash discovery (2026-11-23)

**Status: SOURCE AUDIT ONLY. No CUDA build or GPU test was started.**
No change to the R1.25 FP8/MXFP4 GEMV/GEMM 12-manifest, 2-CUBIN numerical
and FP8 model-layer-forward PASS evidence.

## Local checkpoint candidate

The user's local Hugging Face cache inventory reported:

- `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4`;
- config `model_type=qwen3`, ModelOpt `quant_algo=W4A16_NVFP4`,
  `group_size=16`;
- NVFP4 weight format packed FP4 E2M1, group-16 FP8 E4M3 scales and an
  additional F32 global scale per row, with F16/BF16 activations.

This **is not MXFP4** (MXFP4 has group-32 E8M0 scales) and must not be
reported as covered by the R1.25 `quant-gemv` / `quant-gemm` CUBIN family
set. NVFP4 has its own `nvfp4-gemv` typed package: **4 manifest profiles**
(F16/BF16 times plain/bias), `quant.nvfp4.gemv.{mode}.b128.v1`,
with DKR runtime feature `dkr-nvfp4-gemv`.

Critical architecture caveat: official model card identifies
`architectures=["DFlashDraftModel"]`: this is a **five-layer speculative
draft model**, *not* a standalone Qwen3 chat model. It consumes target-model
hidden states and shares the target's embeddings. It is served as a DFlash
draft alongside `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`.
HF: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4

Only the config has been shown from the user's cache. Do **not** assume
`model.safetensors` is downloaded, or that the target model fits in 8 GiB.

## Verified source constraints in this DKR branch

- `mistralrs-quant/src/fp8_config.rs::model_opt_spec` recognizes exactly
  `W4A16_NVFP4` with `group_size=16` and selects `Nvfp4ActivationMode::None`.
  This covers **quantization metadata parsing**, not complete draft model
  architecture support.
- `mistralrs-quant/src/nvfp4/gemv_dynamic.rs::try_gemv` supplies a DKR
  external GEMV adapter (F16/BF16, M=1..8, packed U8 E2M1 weights,
  group-16 F8E4M3 block scales, F32 global scales, optional bias).
  Its GPU test `dkr_nvfp4_gemv_gpu` exists but **has not been qualified
  by the R1.25 pilot**.
- `mistralrs-quant/src/nvfp4/mod.rs::Nvfp4Layer::from_normalized_parts`
  calls `use_portable_kernels` when the Mistral `cuda-legacy-fp4` feature
  is not enabled. Its `portable::supported` predicate requires BF16 MMA
  (CC >= 80). **Thus the typed-only SM61 deployment can reject layer
  construction BEFORE any valid DKR NVFP4 GEMV adapter has a chance to run.**
  This is distinct from the previously fixed FP8 blockwise routing gate.
- This DKR NVFP4 adapter is decode GEMV only (<=8 rows), not a complete
  prefill GEMM replacement. Do not label full NVFP4 checkpoint inference
  qualified even after a narrow GEMV PASS.

## Next gates (without wasting NVCC compilations)

1. **Read-only inventory** of the user's snapshot: verify actual
   `model.safetensors`, `hf_quant_config.json`, `architectures`,
   `dflash_config`, ModelOpt quant settings and local
   `kernel-packages/build/cuda/sm61/nvfp4-gemv` PTX manifests. This step
   must not download any files.
2. Verify the four `nvfp4-gemv` manifest hashes, geometry/ABI and
   cache identities. **Do not** copy the R1.25 MXFP4 manifests into
   NVFP4 or edit `CUDA_COMPUTE_CAP`.
3. Introduce a narrowly gated NVFP4 typed-only SM61 **layer construction**
   path that does not require Ampere portable MMA nor silently activate
   the Mistral static legacy CUDA producer. For unsupported prefill/shape,
   fail closed or use an explicitly qualified compatible fallback.
4. Only after staged, preflighted CUBIN, run **one bounded physical NVFP4
   GEMV numeric + checkpoint-layer forward qualification**; retain the
   pre-existing `target/dkr-cuda-cache/sm61` and R1.25 paired CUBINs.
5. For a real next-token-logit comparison, use a standalone checkpoint
   with supported model architecture and identical weights across
   DKR/Extended. The DFlash draft must instead be evaluated with its
   parent model and speculative protocol.

## Read-only local inspection commands

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
python3 - <<'PY'
import json
from pathlib import Path

root = Path.home() / '.cache/huggingface/hub'
name = 'models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4'
snapshots = sorted((root / name / 'snapshots').glob('*'))
print('SNAPSHOTS', len(snapshots))
for snap in snapshots:
    cfg = snap / 'config.json'
    if not cfg.is_file():
        continue
    data = json.loads(cfg.read_text())
    quant = data.get('quantization_config', {})
    print('REV', snap.name)
    print('ARCHITECTURES', data.get('architectures'))
    print('DRAFT', data.get('dflash_config', {}))
    print('QUANT', quant.get('quant_algo'), 'GROUP', quant.get('group_size'))
    for filename in ('model.safetensors', 'hf_quant_config.json'):
        file = snap / filename
        print(filename, 'present', file.is_file(),
              'bytes', file.stat().st_size if file.is_file() else 0)

pkg = Path('/home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61/nvfp4-gemv')
print('NVFP4_PTX_ROOT', pkg, 'exists', pkg.is_dir())
print('NVFP4_MANIFESTS', len(list(pkg.glob('*/' + '*/manifest.json'))))
PY
```

This preflight is diagnostic, not a license to run unsupported DFlash
as standalone inference. Resume with actual outputs before scheduling
another expensive release build.

## 2026-11-24 follow-up — locally present weights and 4 NVFP4 manifests

The user's **read-only local inventory** returned:

```text
REVISION: 913d13c49fbbb2b477dce78552a31ab650d5b693
ARCHITECTURES: ['DFlashDraftModel']
QUANT: W4A16_NVFP4
GROUP_SIZE: 16
model.safetensors present= True bytes= 580320464
hf_quant_config.json present= False bytes= 0
NVFP4_PACKAGE_EXISTS: True
NVFP4_MANIFESTS: 4
```

This confirms the downloaded 580,320,464-byte weight artifact and four
`nvfp4-gemv` manifests, **not** their hash validity, executable CUBIN
pairing, full-model architecture support, or GPU numerical correctness.
The absent `hf_quant_config.json` is not automatically a blocker:
the embedded `config.json` already declares W4A16_NVFP4/group16, and
`mistralrs-core` supports embedded ModelOpt quantization metadata.

NVIDIA's official model card identifies this artifact as a DFlash speculative
draft (five Qwen3-style layers); the target model supplies shared embeddings
and verifies proposed tokens. The draft is **not** autonomous and must not
be run as an ordinary Qwen3 causal LM just because `model_type=qwen3`.

### Read-only validation before new architecture-specific work

The new branch `feature/dkr-r126-nvfp4-typed-sm61` forks the already
qualified R1.25 branch. Run its preflight via `git show`, without
switching the user's local R1.25 worktree or modifying Cargo.lock:

```bash
cd /home/np/dkr-worktrees/r125-candle-integration

git fetch origin feature/dkr-r126-nvfp4-typed-sm61
mkdir -p target/dkr-r126-audit

git show origin/feature/dkr-r126-nvfp4-typed-sm61:tools/dkr_r126_nvfp4_preflight.py \
  > target/dkr-r126-audit/dkr_r126_nvfp4_preflight.py
git show origin/feature/dkr-r126-nvfp4-typed-sm61:tools/test_dkr_r126_nvfp4_preflight.py \
  > target/dkr-r126-audit/test_dkr_r126_nvfp4_preflight.py

python3 -m unittest discover -s target/dkr-r126-audit \
  -p test_dkr_r126_nvfp4_preflight.py -v

python3 target/dkr-r126-audit/dkr_r126_nvfp4_preflight.py \
  --packages /home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61 \
  --json-out target/dkr-r126-audit/nvfp4-preflight.json
```

The Python implementation reads the `config.json`, Safetensors *header
only*, and the four existing manifest/PTX image pairs, checking operation
ABI/entry names, SM61 PTX target, SHA256 and optional CUBIN hashes and ELF
architecture. It never loads the GPU, calls Rust, downloads weights or
recompiles anything. CPU fixture regression tests are provided, but have
**not yet been run in the user's environment**.

Only if `DKR_R126_NVFP4_PREFLIGHT=PASS` may work advance to guarded
typed-only SM61 layer construction and a separately approved bounded
NVFP4 GPU test. A `PTX_ONLY` result is expected when CUBIN has not yet
been paired and is not a numeric CUDA qualification.

## R1.26/S0 PASS — 2026-11-24

User executed all six offline regressions (6/6 PASS, 0.016 s) and
the actual preflight (PASS). The local checkpoint header contains 170
tensors: U8=37, F8_E4M3=37, F32=74, BF16=22. The verified PTX
SHA256 is `e44c78486c280abc942dc8dad79c41adb9090b5a5f99eb537fe4ba570b7157ca`.
All four profiles share this one PTX; CUBIN status is PTX_ONLY.
The checkpoint contains extra input_scale tensors, but W4A16 metadata
selects activation=None, and the loader intentionally ignores these
calibration tensors. This is a metadata/header/PTX preflight, not
physical GPU evidence or a hash of the entire weight file.

## R1.26/S1 staged — physical qualification pending

The opt-in Mistral.rs host-side `MISTRALRS_DKR_NVFP4_TYPED_ONLY=1`
mode constructs dense rank-2 W4A16 layers without portable MMA or
legacy FP4 provider allocation. Selection is architecture-neutral;
the existing runtime resolves the device and package. A signature not
accepted by the external GEMV fails explicitly, never falls through
to static or dequantized execution. M>8, W4A4 and stacked weights
are not supported by this pilot. No Candle change and no CUDA
source/PTX modification were made.

New physical test `dkr_nvfp4_checkpoint_layer_gpu` loads a shard of
the actual `layers.0.self_attn.k_proj` via the existing ModelOpt loader
(up to 32 output rows, full K). It compares F16/BF16 CUDA output for
M=1,3,8 to CPU F32 computation rounded to the output dtype and verifies
3D shape restoration and rejection at M=9. Existing synthetic
`dkr_nvfp4_gemv_gpu` covers plain/bias F16/BF16.
These tests have NOT yet been built or executed.

The guarded runner `tools/dkr_r126_nvfp4_sm61.sh` keeps the original PTX
and existing R1.25 packages untouched. It stages a separate NVFP4-only
package tree, reuses the already-built Rust package builder with
`--pair-existing` (one PTX-to-CUBIN assembly if not cached), verifies
strict CUBIN, preserves the reconciled Cargo.lock, checks selected
features, and runs the two focused tests in a single bounded Nextest
campaign. The new log verifier requires both test passes and the
expected exact CUBIN in each test process, plus checkpoint geometry
and six numerical cases. It does not attest full-model inference or
a prefill GEMM. No fresh CUDA C++ compilation is requested; the host
Rust feature change may trigger rebuilds, and cache hits cannot be
guaranteed until the actual build. Offline Cargo metadata must
succeed before any GPU execution.

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
export MISTRALRS_CUDA_BUILD_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61
export DKR_R125_TYPED_PACKAGE_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61
DKR_APPROVE_R126_NVFP4=1 bash tools/dkr_r126_nvfp4_sm61.sh
```
Retain the locally reconciled lockfile; do not stash or reset it merely
to switch revisions. If Git reports a conflict, stop before running.

## R1.26/S1 physical PASS — 2026-11-24

User tested commit b0ce4cc7 on physical SM61, Nextest run
008d3438-bd0b-4ed0-899e-d3e18a76a990. Both tests passed in 0.363 s.
The paired CUBIN SHA256 is
`38fd99ef14140818b0485d5a70569e6402532c628ec18d4a1dfe272f7a709165`;
PTX SHA256 remains `e44c78486c280abc942dc8dad79c41adb9090b5a5f99eb537fe4ba570b7157ca`.
Both processes independently admitted that exact image in strict CUBIN
mode. All four profiles validated. Real projection:
layers.0.self_attn.k_proj, source N=1024, sampled N=32, K=2560.
F16/BF16, M=1,3,8: relative L2 and maximum absolute error were zero
at eight displayed decimal places against CPU F32 rounded to output dtype.
M=9 rejection also passed. Nextest PASS and offline evidence PASS.
The uploaded report's Nextest log SHA256 is
`19c4d3c6304987c87e830edd33a6a57f755541f49e6ce987f2e7ac80e38cf727`.
CUDA source recompiles: quant 0/49, paged-attn 0/15, core 0/18.
Host Rust build: 7m05s. The new isolated CUBIN was assembled from
existing PTX. No repeat of S1 is required to advance the GEMM family.
Scope remains dense W4A16 GEMV, not full projection coverage, prefill,
DFlash protocol, or full-model inference/performance qualification.

## R1.26/S2 staged — independent tiled NVFP4 GEMM reference

S1 physical evidence is preserved (no new run is required). S2 adds
a separate `nvfp4-gemm` package, four F16/BF16 plain/bias entries,
`quant.nvfp4.gemm.{mode}.t16.v1`, tile 16x16, F32 accumulation,
group16 E2M1/E4M3 decoding in shared memory, and final per-row F32
global multiplication plus optional bias. No globally materialized
dequantized weights. Tails in M and N are masked without skipping
block barriers. This is a reference SIMT GEMM, **not performance qualified**.

The Rust adapter shares NVFP4 tensor validation, pointer ABI and launch
plan cache with the existing GEMV, with a distinct route in the cache
key. GEMV remains M<=8; GEMM handles M>8 when explicitly enabled by
`MISTRALRS_DKR_NVFP4_GEMM=1`. Existing typed-only construction stays
architecture-neutral and fails closed if neither accepts a signature.
The S1 test retains its decode-only semantics when GEMM is disabled.
No Candle source changes or modifications to the existing GEMV CUDA
source or S1/R1.25 image files.

The Cudaforge builder now accepts `--family nvfp4-gemm` to compile just
the new CUDA translation unit. Its output is a fresh isolated package
root; no existing family is included or reassembled. The S2 runner
revalidates the previously qualified S1 CUBIN hash before the build,
checks cached static source objects and the exact typed feature set,
and launches a **single** bounded Nextest test (not the S1 tests).
Real sampled projection N=37 (or min(source N,37)), K=2560 on the
current checkpoint, M=9,17,33 for F16/BF16, compared with CPU F32
rounded to the output dtype. Four synthetic cases cover plain/bias,
signed E2M1 values, variable scale/global and nonmultiple M/N tiles
(M=17,N=19,K=48). The offline verifier checks six real numerical
cases, four synthetic markers, Nextest PASS, and exact strict GEMM
PTX/CUBIN admission. The new Rust/CUDA source and tests have **not**
been built or executed in this session.

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
export MISTRALRS_CUDA_BUILD_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61
export DKR_R125_TYPED_PACKAGE_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61
bash -n tools/dkr_r126_nvfp4_gemm_sm61.sh
DKR_APPROVE_R126_NVFP4_GEMM=1 bash tools/dkr_r126_nvfp4_gemm_sm61.sh
```

Retain the reconciled Cargo.lock. This requires one new external
PTX/CUBIN module, not a rebuild of the other external CUDA families.
Host Rust recompilation and ordinary build-script cache validation may
occur. A clean cache audit does not guarantee a hit under changed build
arguments. Full-model inference, combined decode/prefill route in a
single model, DFlash speculation, routed experts and performance
remain outside this gate.

## R1.26/S2 physical PASS — 2026-11-24

User tested 259836f3, Nextest 12c661f8-2f19-4acc-b58c-046273706300.
One test passed in 0.224 s overall. Real k_proj N=1024 sampled N=37
K=2560, F16/BF16 M=9,17,33; all six cases displayed zero relative
L2 and maximum absolute error at eight decimals against CPU F32
rounded to the output dtype. Four synthetic cases passed with both
dtypes and both bias ABIs. The uploaded attestation reports
`qualification_scope=sampled_dense_w4a16_layer_functional`,
`nvfp4_prefill_gemm_qualified=true`, full model/DFlash/performance=false.
Exact GEMM image:
- PTX: `5b71508ffcb5fe7f07ad0802bdebc0d6393dc86d1137eda34be545df494b9fde`
- CUBIN: `cefd5eb2aecfb8262eb82ed71cf0a364e4857ebad5d3d3659124917e3aeec1ef`
- Nextest log: `435802cc82a3500d7ac03d8e254036f6327c30240d8df3ae87728310e89d1102`

S1 image preserved PASS. The selective Cudaforge builder compiled
one of one new PTX source, staged one new family and paired one image.
Static CUDA rebuilds were quant 0/49, paged-attn 0/15, core 0/18.
Host Rust test compilation was 5m10s. The two individual NVFP4 family
gates are now PASS with eight manifests and two distinct CUBINs.

## R1.26/S3 staged — full projection combined catalogue and routing

The next gate copies the two exact S1/S2 families into a separate
eight-manifest catalogue and verifies the original PTX/CUBIN identities.
No external compiler, package builder or ptxas is invoked by staging.
The original S1/S2 roots and R1.25 artifacts are untouched.

A single ignored test dkr_nvfp4_checkpoint_combined_gpu uses the full
actual k_proj N=1024 K=2560, not a 32- or 37-row shard. For each
dtype F16/BF16, it creates one CPU reference and one GPU layer (CPU
reference shared across dtypes) and runs the same layer in sequence
M=17,8,9,1,17. Each result is checked against CPU F32 rounded to the
activation dtype, with the same S1/S2 tolerances. Later outputs are
also compared against the matching prefix of the initial M=17
output. Thus both boundary sides and a repeat of the first GEMM
plan are exercised. This is still a layer-level functional gate,
not actual model prefill/decode, not a performance measurement.
The verifier requires exact CUBIN/PTX admissions for both families
in one test process, full geometry, eight-manifest catalogue,
ten transition markers and eight cross-route metric records.

Run after preserving the existing resolved Cargo.lock:
```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
export MISTRALRS_CUDA_BUILD_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61
export DKR_R125_TYPED_PACKAGE_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61
bash -n tools/dkr_r126_nvfp4_combined_sm61.sh
DKR_APPROVE_R126_NVFP4_COMBINED=1 bash tools/dkr_r126_nvfp4_combined_sm61.sh
```

S3 has not been compiled or executed yet. Only host test changes and
offline tooling are new; CUDA object cache reuse is expected but not
guaranteed under changed build arguments. Do not rerun S1/S2 merely to
complete S3. No changes to Candle or existing CUDA kernel sources.

## R1.26/S3 physical PASS — 2026-11-24

User tested dbc2efef, Nextest ab15bb85-48fb-4e40-8029-f7bcc3151ff4.
Full k_proj N=1024 K=2560; F16/BF16 with M=17,8,9,1,17 on the
same layer per dtype. All ten CPU-reference comparisons and eight
cross-route comparisons displayed zero errors at eight decimals.
Both exact S1/S2 CUBIN/PTX pairs admitted in one process, eight
manifests validated. Nextest 1/1 PASS (0.450s overall); offline
combined evidence PASS. Log SHA256:
`8982c25c2586551a5177c99f63c6867b26c97131b6bd0861cffdf9cd6075d061`.
Host Rust build 5m19s; native CUDA recompilation 0/49 quant, 0/15
paged-attn, 0/18 core; no new external CUDA build. Full projection
qualified in these cases, not full model, DFlash or performance.

## R1.26/S4 staged — offline consolidation of all four quant families

The new `tools/dkr_r126_quant_catalog.py` validates the 20 expected
profile paths, operations, entries, block geometry and exact four
historically qualified PTX/CUBIN pairs. It checks actual file digests,
PTX entries and SM61 ELF64 CUDA architecture, replays the existing
R1.25 individual GEMV/GEMM and FP8 layer-forward evidence verifiers
and the S3 combined NVFP4 verifier against existing logs. Copies only
manifest and image files into a separate staging directory; rechecks
their byte identities and rejects differing existing output rather than
overwriting it. It does not call Cargo, NVCC, ptxas or the GPU.

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
python3 tools/dkr_r126_quant_catalog.py \
  --quant-root target/dkr-r125-quant-pilot/sm61 \
  --nvfp4-root target/dkr-r126-nvfp4-combined-pilot/sm61 \
  --quant-gemv-log target/dkr-r125-quant-pilot/r125-quant-gemv.log \
  --quant-gemm-log target/dkr-r125-quant-pilot/r125-quant-gemm.log \
  --fp8-forward-log target/dkr-r125-audit/forward-route.log \
  --nvfp4-combined-log target/dkr-r126-nvfp4-combined-pilot/nvfp4-combined-20261124T085058-112347.log \
  --output target/dkr-r126-quant-catalog/sm61 \
  --json-out target/dkr-r126-quant-catalog/evidence.json
```

Keep reconciled Cargo.lock and original catalogs unchanged. S4 has not
yet been executed on the user's files. The resulting quant-only root
is a candidate input to the existing `DKR_EMBED_TYPED_DIR` runtime
build contract (build.rs deduplicates blobs by hash); it is NOT the
entire DKR operation catalogue or an embedded-runtime qualification.
The next physical gate should test the embedded source in a process
with filesystem package access disabled, selecting actual operations
from all four families. No full model or Extended performance
certification, and no new Candle SM61-specific logic, is implied.

## R1.26/S4 offline PASS — 2026-11-24

The user's S4 report at commit 650ad6b9 confirms 20 profiles, four
unique CUBIN/PTX pairs, 60 manifest/image file entries, and reverified
prior numerical logs. No new compilation or GPU execution occurred.
The report retains combined_four_family_gpu_run=false,
embedded_runtime_qualified=false, full_model_qualified=false and
performance_qualified=false. Preserve this catalogue and its evidence.

## R1.26/S5 staged — embedded quant catalogue qualification

The existing runtime build.rs embeds verified PTX/CUBIN payloads
by hash. KernelExecutionCache::package_root() can use embedded::ROOT
when DKR_KERNEL_PACKAGES is absent; discover() in explicit embedded
mode selects embedded::catalog() and does not read external packages.
Its runtime trace is MISTRAL_DKR_CATALOG_EMBEDDED manifests=20.

The new runner revalidates S4 evidence against exact file digests,
stages a dedicated build-input copy, and builds once using
DKR_EMBED_TYPED_DIR. Nextest list emits binaries-only JSON metadata;
Cargo metadata is also saved before running. The runner then renames
only that dedicated copy to sm61.hidden, clears DKR_EMBED_TYPED_DIR
and DKR_KERNEL_PACKAGES, and invokes Nextest in reuse-build mode via
--binaries-metadata and --cargo-metadata. It checks that the original
build-input path is absent before and after the run and restores the
copy with an EXIT trap. Original qualified catalogues are not moved.
This tests independence of the embedded input path and explicit
embedded source selection; it is not a global filesystem sandbox.

Six existing GPU tests run in one Nextest campaign, in separate
test processes: direct quant GEMV/GEMM, FP8 production layer forward,
NVFP4 synthetic GEMV, NVFP4 checkpoint GEMM (both bias ABIs), and full
NVFP4 combined projection. Each must emit embedded catalogue admission,
strict CUBIN policy and the exact expected family image pair(s).
The verifier rejects filesystem admission and PTX fallback. One
campaign across separate processes does not mean all four families
were exercised inside a single process/model.

No new kernel source changes, external package builds, NVCC or ptxas
commands are requested. Embedding requires one Rust build and normal
build-script cache verification; native CUDA cache hits are expected
but not guaranteed. S5 has not yet been built or run.

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
export MISTRALRS_CUDA_BUILD_ROOT=/home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61
bash -n tools/dkr_r126_quant_embedded_sm61.sh
DKR_APPROVE_R126_QUANT_EMBEDDED=1 bash tools/dkr_r126_quant_embedded_sm61.sh
```

Keep Cargo.lock reconciled. If an uncatchable interruption leaves
sm61.hidden, restore only the dedicated copy after verifying that
no test is running and its original path is absent. Full-model
inference and performance remain separate gates. No Candle source
change is required for this S5 experiment.

## R1.26/S5 physical PASS — 2026-11-24

User executed fb64c508, Nextest run 8f18c786-b4b6-4831-9518-29414f25a92b.
All six tests passed (1.221s overall). Host build 5m08s. The submitted
log begins at the build completion; it does not independently enumerate
native CUDA build cache hits. No new external image was requested.
The exact embedded catalogue of 20 profiles emitted its admission
marker in all six separate processes. All four original CUBIN/PTX
pairs were admitted in strict CUBIN mode, without filesystem catalogue
admission or PTX fallback. Dedicated build input absent before and
after the tests. NVFP4 full projection and all sampled GEMM cases
again displayed zero errors at eight decimals against rounded CPU
reference; synthetic bias cases passed.

Report:
target/dkr-r126-quant-embedded/20261124T090835-123259/evidence.json
Recorded Nextest log SHA256:
`06b81d121ca8acd0c8cf9b42d1cfbbdd9eefafa1069240da34c546ca17cbdd39`.
Recorded S4 report SHA256:
`d5403531a053b59543fb6e16ee620e01af1af1237e2cb0130aed99fc00ae128a`.
embedded_runtime_qualified=true, same_process_all_four_families=false,
full_model_qualified=false, performance_qualified=false.
This is the scoped embedded quant transport and functional gate,
not proof of eliminating every static Candle CUDA kernel.

## R1.26/S6 staged — read-only full typed package coverage inventory

The current Cudaforge builder registers 26 families (25 before
the added NVFP4 GEMM). Four have the S5 embedded quant qualification.
MoE grouped GEMM, indexed-copy and paged-decode have earlier targeted
R1.25 filesystem pilot evidence; inspect the exact local package and
log before reusing them. The remaining 19 registered families have
not been covered by S5. This is a package-family count, not an
exhaustive count of all CUDA operations or backend methods.

The new tools/dkr_r126_full_catalog_inventory.py reads the builder's
registered family names and actual local manifest trees. It verifies
PTX/CUBIN digests, basic entry/target and CUDA ELF identity, reports
missing/invalid/PTX-only/partially paired/paired families, and binds S5
to S4's evidence hash. Earlier three-family log evidence is recognized
when the relevant test passed and an image of that family was observed
in the campaign; this is not a new per-profile attestation.
The inventory does not run compilers or GPU and does not claim
all ABI signatures, numerical coverage, or absence of static fallbacks.

Important remaining architectural debt: S5 uses Candle generic
cuda-legacy-bf16 and cuda-legacy-fp8 compatibility for tensor
conversions, and the broad dkr-sm61-experimental feature enables
cuda-legacy and native cuda-dkr-s31. Do not promote that broad feature
as a purely catalogue-only deployment. Future full qualification must
audit the resolved feature graph and actual production dispatch
(including native Candle and GGUF/MMQ routes), not just the 26 packages.

Run the inventory before assembling or compiling any missing families:
```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
python3 tools/dkr_r126_full_catalog_inventory.py \
  --source /home/np/tmp/mistralrs_provider/mistral.rs/kernel-packages/build/cuda/sm61 \
  --quant target/dkr-r126-quant-catalog/sm61 \
  --quant-evidence target/dkr-r126-quant-catalog/evidence.json \
  --embedded-evidence target/dkr-r126-quant-embedded/20261124T090835-123259/evidence.json \
  --real target/dkr-r125-real-pilot/sm61 \
  --real-log target/dkr-r125-real-pilot/r125-real.log \
  --json-out target/dkr-r126-full-catalog/inventory.json
```

S6 is not yet run on the local files. No change to CUDA or Rust
sources, feature flags or Candle pin for this audit.

## R1.26/S6 observed inventory and correction — 2026-11-24

User inventory at fbe535f2 reports 26 registered families, 7 paired
(28 profiles, 8 unique CUBIN images), 18 PTX-only (185 profiles,
18 unique PTX images), 1 apparently missing logits, 0 invalid.
Total 213 present profiles. Source audit found the logits package
builder writes to sparse-logits, not logits. The corrected inventory
maps the builder's family name to the actual package directory.
Do not compile logits before checking this alias; if present, expect
three more profiles and one PTX image, subject to actual verification.

## R1.26/S7 staged — isolated pairing and strict feature separation

tools/dkr_r126_full_catalog_stage.py checks a refreshed S6 inventory,
copies only unpaired families into a separate assembly root and
preserves the seven already paired families byte-for-byte. Run the
existing Cudaforge package builder with --pair-existing only on that
assembly root. The finalizer requires all 26 families, rechecks
exact PTX/profile identities, unchanged metadata except CUBIN pairing,
and rejects duplicate operation/implementation/target identities.
Original source and qualified directories are never modified. The
full staged root and its evidence are separate; no numerical or
embedded full-catalogue qualification is claimed. If logits truly is
missing, a separate --family logits --cubin build can supply
--logits-root; do not compile other families. S7 has not been run.

The pinned Candle build scripts explicitly reject CUDA_DKR_EMBEDDED
together with CUDA_DKR_S31, CUDA_LEGACY_BF16/FP8/FP4 or CUTILE.
The previous Mistral dkr-sm61-cubin-qualified feature inherited
dkr-sm61-experimental, hence enabled incompatible features.
The corrected candidate inherits a generic dkr-typed-operations
feature (including explicit FP4 dequant adapter) and qualified
native Candle catalogue features, not experimental or legacy.
Old experimental and pilot feature behaviour is unchanged.
tools/dkr_r126_strict_features.py audits actual resolved Cargo
metadata; metadata-only PASS is not a build or a GPU result.

Candle native build requires separate sealed inputs:
DKR_EMBED_NATIVE_DIR, DKR_EMBED_CUBIN_DIR, DKR_EMBED_S55_MMQ_DIR,
DKR_EMBED_S55B_RECIPES_DIR and DKR_EMBED_S55C_CUDAFORGE_DIR.
The Mistral typed catalogue does not replace these inputs.
Verify their identity/coverage, including BF16/FP8 generic conversions,
before any strict native build. No Candle source or pin changes.
Mistral core/quant/paged-attn build.rs still produce static CUDA
archives even when legacy features are off; audit source-level
dispatch and FFI closure before attempting to disable those
producers. No claim of complete elimination of static kernels.

## R1.26/S7 observed PASS — 2026-11-24

User supplied inventory-v2 and final staged evidence. All 26 registered
families, 216 profiles and 27 distinct PTX/CUBIN pairs are present;
19 newly paired families, seven preserved families (eight images).
The three logits profiles were indeed in sparse-logits. All earlier
paired image identities match the inventory. The staged report contains
648 file digests and binds the exact inventory SHA256
`35b38d90eec1639bbaf8df55daf470c921b0bf78cbb7c25ae89a81d9d093ef43`.
Submitted staged report SHA256
`19ecb4e5d930893603e8237443d2bad42f04357c899d77310d56ec6579c55d15`.
Resolved feature audit PASS (native embedded, typed, no legacy flags).
This is structural catalogue integrity and feature resolution only;
full embedded GPU and static producer elimination remain unqualified.

## R1.26/S8 staged — native Candle input preflight

At the pinned Candle revision d40ac1b4d69a209d08f76c5b5bd4e267ec63b4f0,
the native qualified feature already has the catalogue-owned FP8, FP4,
BF16 and routed recipes and the 10 MMQ codecs. Its build contract requires
five native inputs with cross-attested SHA256 provenance. Do not enable
S31/legacy to compensate for missing native inputs. The native catalogue
has 25 families, distinct from the 26 registered typed Mistral families.

The new tools/dkr_r126_candle_native_preflight.py locates the exact
Candle checkout from locked Cargo metadata, imports its existing S5.5c
offline verifier, validates a prior R1.25 P1/P2 receipt (including the
seal and derived manifests), and verifies all 648 S7 staged file
identities. It emits six build-input paths (five native plus typed)
in a JSON report without running Cargo build, NVCC, ptxas, or GPU.
Missing or inconsistent historical artifacts block rather than trigger a
rebuild. A receipt proves input provenance, not new physical execution.
If multiple distinct valid native bundles are found, select an explicit
receipt; never silently choose one. S8 preflight has not been run on
the user's files. No change to pinned Candle or CUDA sources.

Remaining closure: Mistral core/quant/paged-attn static FFI and build
producers need an independent audit; absence of legacy feature flags is
not absence of static archives. Next physical gate should use strict
native and typed embedded profiles together, check BF16/FP8 conversions
and operation dispatch, and reuse the 27 typed images. Do not regenerate
native recipes merely to combine the two catalogues.

## R1.26/S8 preflight PASS — 2026-11-24

User executed 1543df00. Both Candle R1.25 P1/P2 receipts verified
and point to the same native inputs. S8 evidence SHA256:
`c82777c8461cbff105f92e991e9ec1fbf343f112819d089b61878feefed581cc`.
Native Cudaforge catalogue SHA256:
`91ac006b963a89362c6aaaee052252b4465880c079b0c86f399c4891a5003295`.
Exact pinned Candle d40ac1b4d69a209d08f76c5b5bd4e267ec63b4f0.
All six input paths verified, 25 native families, four recipes,
10 MMQ codecs, 26 typed families, 216 profiles, 27 typed CUBINs.
No compile/GPU execution. The native integration and static fallback
elimination flags remain false. Preserve both catalogs and receipts.

## R1.26/S9A staged — static producer / FFI source inventory

Source inspection at S8 identifies three distinct producer problems:
- core/build.rs excludes selected DKR sources and defines partial
  exclusion macros for mixed translation units, but still builds
  mistralrscuda. Remaining FFI includes attention preparation,
  top-k/sampling, graph copy, DFlash context/selector, GDN recurrence
  and speculative state transitions. These are not closed by S7.
- paged-attn/build.rs still compiles src/cuda/*.cu and links
  mistralrspagedattention. backend/paged_attention.rs tries typed
  paged FP8/decode/v2 and reshape routes; residual static FFI is
  present. cache.rs, gather_kv.rs and scale_update.rs also retain
  static entrypoints. MLA, FlashInfer and sinks must be covered
  or explicitly unsupported by a strict runtime configuration.
- quant/build.rs excludes weighted_reduce.cu under dkr-reduce-norm
  but still builds mistralrsquant. GGUF fast_mmvq.rs/fast_mmq.rs
  select Mistral-owned FFI (including compact and fused paths);
  the native Candle catalog's MMQ/MMVQ coverage is not evidence
  that these paths use it. Preserve their ABI, numerical behavior
  and performance before migration. Other quant formats, LoRA,
  utility ops and fused kernels need independent closure.

New tools/dkr_r126_static_ffi_inventory.py records all tracked CUDA
translation units, producer/link sites, explicit extern C symbol
declarations, literal identifier references, qualified ffi references
and macro extern blocks. Optional nm of existing cached objects maps
literal declared symbols to historical definitions without compiling.
The scan is lexical, does not evaluate Rust cfgs or expand macros,
and neither a cached definition nor a lexical reference proves an
active linked symbol. Future S9B must classify each active route,
gate unsupported signatures explicitly, and verify strict link and
runtime evidence. Do not skip build.rs producers or link empty stubs
to obtain a misleading PASS. No source/CUDA/feature change here.

```bash
cd /home/np/dkr-worktrees/r125-candle-integration
git fetch origin feature/dkr-r126-nvfp4-typed-sm61
git switch --detach origin/feature/dkr-r126-nvfp4-typed-sm61
python3 tools/dkr_r126_static_ffi_inventory.py \
  --s8-evidence target/dkr-r126-native-preflight/evidence.json \
  --cache-root /home/np/tmp/mistralrs_provider/mistral.rs/target/dkr-cuda-cache/sm61 \
  --json-out target/dkr-r126-static-closure/inventory.json
```

This read-only inventory has not been executed against the local tree.
No new CUDA or Rust compilation is requested. Keep the reconciled lock.

## R1.26/S9A observed COMPLETE — 2026-11-24

User executed f205f586. Inventory SHA256
`c214e2e3c9e239c9aaf713a7ef496b4736015323835bd96579d3e7d66b1614e1`.
Core: 20 CUDA sources, 128 explicit FFI names, 122 matched cached definitions.
Quant: 69 sources, 317 names, 273 matched definitions, four macro extern blocks.
Paged-attn: 16 sources, 31 names, 27 matched definitions.
Total 87 cached objects, no nm errors. Not a compiled or linked call graph.
The four missing paged symbols are cfg-gated FA3 functions.

## R1.26/S9B1 staged — scoped Paged Attention catalogue-only producer

New opt-in feature `mistralrs-core/dkr-paged-catalog-only` forwards
to `mistralrs-paged-attn/dkr-paged-catalog-only`, independently of
the broad native+typed candidate. The latter is deliberately not
changed yet. The paged feature enables all six existing typed adapters
(paged decode, v2, FP8 decode, KV copy/reshape, gather, scales).
In this profile the paged build script returns before toolkit detection,
Cudaforge build and archive linkage, and the Rust FFI module is excluded.
Legacy tails of the five migrated operation paths are cfg-excluded.
Unsupported signatures return DKR_PAGED_CATALOG_ONLY errors; the
FlashInfer, MLA and sinks APIs return explicit capability errors.
No fake success or C ABI stub is used. FA3 remains disabled via its
existing build cfg. Full FP8 cache capability remains false because
scaled writes and gather are not externalized. Direct FP8 decode still
has its typed adapter. The historical CUDA feature set is unchanged
by default. Existing swap_blocks remains available.

This is a restricted pilot, NOT a drop-in full backend. It does not
establish early model capability admission or compile/link/numerical
qualification. A CPU negative-route test has been added; not executed.
S10 must verify actual build-script output, no paged static archive
linkage, existing typed numerical tests and negative admission tests
with native+typed embedded inputs. Do not promote the feature or delete
original CUDA sources yet. Core and quant producers remain.

No new device images or Rust build requested at this stage.
The next source steps should externalize missing paged signatures
or explicitly scope model capability admission, then tackle core and
quant producers while preserving Extended kernels and ASD semantics.

## R1.26/S9B1 resolved graph PASS — 2026-11-24

User executed 1908bd1a with the independent paged feature and strict
native+typed feature. The existing strict feature audit passed, with
static producer elimination NOT_QUALIFIED. No build was executed.

## R1.26/S9B2a staged — scaled FP8 VLLM cache IO

The prior catalogue covers short-context FP8 read/decode, not FP8
slot writes or scaled gather. New independently opt-in feature
`dkr-kv-fp8-io` and Cudaforge family `kv-fp8-io` add six profiles:
store and gather for F16/BF16/F32. One CUDA source and one PTX/CUBIN
pair are expected for this extension; do not overwrite S7's 216
profiles/27 images or mutate their receipts. S7/S8 remain historical
baseline gates. A later combined staging gate must account for the
new 27th family (222 profiles/28 images expected, not yet observed).
The existing S7 inventory/staging and S8 evidence checks describe
the original 26-family baseline, not this extended catalogue.

The ABI uses device-resident scalar F32 K/V scales, explicit FP8
cache representation, canonical contiguous VLLM layouts and cached
prevalidated launch plans. Writes use direct F32 division and
E4M3 SATFINITE rounding as in the historical scaled KV writer,
not intermediate half rounding from the unrelated scalar cast.
Gather decodes then multiplies by the respective scale before
rounding to output dtype. Positive finite scales, unique writable
slots and valid page metadata are caller contracts (no GPU->CPU
scalar read or synchronizing validation). Invalid indices are
bounded defensively, not reported as a new validation guarantee.
Strided variants, U32 gather metadata, FP8 long contexts, sinks,
MLA and FlashInfer are not closed by this step. Full-cache capability
flags remain deliberately unchanged.

An ignored GPU test independently checks stored raw E4M3 values and
scaled gather results for three dtypes, two different scales,
nontrivial page permutation, padding, saturation and midpoint rounding.
Not compiled or run. No new CUDA source compiled yet. Keep old
catalogue immutable; later build only kv-fp8-io in an isolated root,
then stage a verified extended catalogue and run the grouped tests.
The broad strict candidate and S9B1 remain independently opt-in;
the new feature is not automatically enabled.

Resolved graph check only:
```bash
cargo metadata --offline --locked --format-version 1 \
  --features mistralrs-core/dkr-sm61-cubin-qualified,mistralrs-core/dkr-paged-catalog-only,mistralrs-core/dkr-kv-fp8-io \
  | python3 tools/dkr_r126_strict_features.py --paged-fp8-io
```

## R1.26/S9B2a resolved graph PASS — 2026-11-24

User executed 310dc4b8. S9B2 feature audit passed with the paged-only
and FP8-IO features enabled, no legacy flags. No compilation or GPU
execution; static producer elimination remains NOT_QUALIFIED.

## R1.26/S9B2b staged — uniform strided VLLM input writes

Extend the as-yet-unbuilt kv-fp8-io image by six additional entries,
plain and scaled E4M3 strided writes for F16/BF16/F32. No change to the
existing 216 S7 profiles or 27 image identities. The new family now
has 12 profiles, hence the prospective extended catalogue contains
228 profiles, 27 families, 28 CUBINs. This is a target, not an observed
artifact count. No additional CUDA translation unit is needed.

The ABI deliberately retains the runtime's contiguous argument contract:
each input is a rank-one physical span (including padding) with
kv_cache.row_span.v1 representation, plus separate U64 K/V element
strides. The adapter verifies dense heads, uniform token stride for
rank 3 or 4, checked span arithmetic, bounds against the underlying
CudaSlice, same device and canonical contiguous cache storage. The
logical tensor itself is not described as contiguous. Nonzero offsets
are folded into pointers. No contiguous materialization or CPU
transfer of activation data. Contiguous writes still use their
previous paths. Strided cache destinations, arbitrary head strides,
and irregular batch/sequence strides remain unsupported. Valid
metadata, nonoverlapping writes and positive finite scales remain
caller contracts.

The added CPU test exercises span arithmetic; the ignored GPU test
checks independent K/V strides, nonzero offsets, rank 3 and 4, padding,
and raw cache values for all three activation dtypes with/without FP8.
Neither test has been executed or compiled in this session. Full FP8
capability remains disabled. No native Candle or legacy backend change.

## R1.26/S9B2c staged — partitioned FP8 decode

S9B2b source fetch and git diff --check succeeded on 6b39a8a7.
New family paged-fp8-v2 adds only three FP8 partition entries
(F16/BF16/F32 query). The existing paged-v2 merge consumes normalized
F32 partials and partition logsumexp unchanged, so its CUBIN is reused.
The shared Rust V2 adapter distinguishes scaled and plain launch keys;
FP8 partition is admitted only with dkr-kv-fp8-io + dkr-paged-v2,
explicit F32 K/V scales, contiguous VLLM tensors, no sinks/window/
ALiBi/softcap and context 257..32768. Short FP8 decode remains
unchanged. Existing workspace pool reused; no new graph/concurrent
stream safety claim. Valid block table and finite positive scales
remain caller contracts. No performance claim.
The ignored test covers context 256,257,513,4097, empty and partial
partitions, permuted pages, GQA, distinct K/V scales, three dtypes,
independent CPU F64 reference and repeated dispatch. Maximum 32768
is an admission bound, not a tested length yet. No code compiled/run.
S7 baseline unchanged: prospective extension 28 families, 231
profiles, 29 CUBINs (two new translation units including kv-fp8-io).
The original S7/S8 scripts still describe the original baseline;
extended staging must be supplied before physical tests.

## R1.26/S9B3a staged — paged decode options

New independent opt-in dkr-paged-options, forwarded through core.
The paged-options family provides 12 partition profiles (three query
dtypes, plain/FP8 K/V, with/without ALiBi) and three sink merge profiles.
The existing no-sink V2 merge is reused. Optional tensor parameters are
represented by separate signatures rather than null or fake tensor
arguments. Window and cap are scalar parameters. Mask is exactly the
last W tokens; ALiBi uses signed token-context+1; tanh softcap precedes
ALiBi. The sink is included once in global merge, including empty
partitions. Non-finite input values are not qualified. The reference
kernel uses accurate tanhf rather than the historical fast_tanh
approximation; numerical parity must be measured, not assumed bitwise.
Context admission 1..32768, canonical contiguous VLLM layouts and
supported head sizes; positive finite scales and valid metadata remain
caller contracts. Existing workspace stream/graph lifetime limits apply.
No change to S7 images. Prospective 29 families, 246 profiles, 30
CUBINs (three new sources). Counts are targets, not built artifacts.

Ignored dkr_paged_options_gpu exercises 108 configurations:
F16/BF16/F32, plain/FP8, contexts 256/257/513, six option combinations,
three unequal sequence lengths including empty, permuted pages, GQA,
independent CPU F64 reference, repeat dispatch. Not compiled or run.
Dense and varlen FlashAttnSinks APIs (prefill) are separate from paged
decode and still reject in catalog-only profile. MLA/FlashInfer and
other layout signatures remain open. No full backend closure claimed.

## R1.26/S9B3a observed feature audit — 2026-11-24

User fetched f08f90f6 and obtained strict feature PASS, with static
producer elimination still NOT_QUALIFIED. The earlier audit option
--paged-fp8-io checks the B2 flags, not specifically paged-options.
The new --prefill-sinks audit checks both paged-options and prefill-sinks.

## R1.26/S9B3b staged — dense/padded/packed prefill sinks

Externalize the historical tiled flash_attn_sinks.cu device algorithm,
retaining BR=8, the original head-specific BC, cooperative K/V tiling,
warp reduction, F32 online softmax and one sink per query. The new
prefill-sinks package uses device functions and typed global entry
points, no host FFI launch. 3 dtypes x 7 head dimensions x 3 layouts
x 2 sink signatures = 126 profiles, expected one PTX/CUBIN image.
No-sink profiles omit the tensor argument entirely. Separate mode and
head size operation identities prevent cross-layout plan reuse.
The opt-in dkr-prefill-sinks feature replaces the three public CUDA
APIs without affecting the default legacy profile; catalogue-only
without this feature still rejects. Same contiguous materialization as
historical APIs. No new N*N attention matrix or per-token CPU readback.
Padded output zero initialized; varlen metadata stays on GPU, with
local bounds checks for endpoints and max Q length. Well-formed
monotone cumulative metadata covering the packed inputs is a caller
contract, not fully validated on host. Entirely empty input tensors
are not admitted. Max Q length 32768, dense KV length 32768;
other geometry and allocation limits enforced by adapter/runtime.
No throughput equivalence claimed before measurements.

The ignored GPU test exercises 252 configurations (all 126 ABI
profiles with two window settings), bottom-right causal offset,
unequal Q/KV lengths including Q>KV, empty query sequence within
a batch, partial query tiles, GQA, exact padding zero, CPU F64
oracle and repeat launches. CPU negative admission test added.
Not built or run. Prospective extended total 30 families, 372
profiles, 31 images; four new translation units relative to S7.
The total remains below MAX_DISCOVERED_PACKAGES=512; actual module
sizes and embedded build limits must be checked at staging. No
changes to the 27 baseline images, Candle pin or static producer
policy. MLA/FlashInfer, noncanonical layouts and stream/graph
qualification remain open.

## R1.26/S9B3b resolved graph PASS — 2026-11-24

User executed a4193da0; S9B2, S9B3 and strict features passed.
Static producer elimination still NOT_QUALIFIED. No physical test.

## R1.26/S9B4a staged — MLA cache IO

Independent dkr-mla-cache-io feature forwards through core. Six typed
profiles (store/gather, F16/BF16/F32), one Cudaforge source. Preserve
separate CKV and KPE paged cache representations. Store takes checked
physical input row spans and U64 strides, supporting nonzero offsets
and independent dense-row padding without copies. Fixed 256-thread
block, a deliberate choice differing from the legacy variable store
block; performance not qualified. Gather retains I32 table/offset/map,
zero-initialized output and metadata materialization if needed. Both
kernels bound local indices defensively; malformed global metadata and
duplicate nonnegative write slots remain outside the caller contract.
No FP8 MLA support claimed. No CPU readback added by these IO kernels.

Existing context_attention_mla tensor orchestration now imports
the public routed gather and is enabled in catalogue-only with the
new feature. It still performs the historical host metadata reads,
causal mask construction and dense per-sequence tensor attention.
No fused MLA prefill or performance claim. MLA decode and FlashInfer
HND IO/decode remain to migrate. Legacy profile unchanged by default.

CPU span test and ignored physical IO test added: independent raw
cache and gather references, three dtypes, permuted pages, unequal
CKV/KPE widths/strides, offset and padding. Not compiled or executed.
Prospective extended catalogue: 31 families, 378 profiles, 32 images
(five new sources relative to S7). Baseline 216 profiles/27 images
remain immutable. Actual image sizes and extended staging pending.

## R1.26/S9B4a resolved graph PASS — 2026-11-24

User reports S9B2, S9B3, S9B4A and strict features PASS.
Static producer elimination still NOT_QUALIFIED. No build or GPU run.

## R1.26/S9B4b staged — FlashInfer HND cache IO

Independent opt-in dkr-flashinfer-hnd-io forwards through core.
New flashinfer-hnd-io family: six store profiles and 24 gather profiles
(three activation dtypes, plain or E4M3 cache, independent I32/U32
page-table and cumulative-offset dtypes for gather). One source;
30 entries, prospective one PTX/CUBIN pair. Canonical HND layout
[blocks,heads,page,head_dim] distinct from VLLM, no repack. Store
uses bounded physical row spans with U64 strides, dense inner heads,
rank 3 or uniformly flattenable rank 4, nonzero offsets. FP8 uses
historical direct float conversion to/from CUDA E4M3 and positive finite
host scalar K/V scales; plain requires unit scales and omits them
from the typed ABI. No new host readback or activation copies.
Gather binary-searches monotone cumulative offsets, handles empty
sequences, keeps metadata dtype without conversion and bounds local
table/offset/page accesses. Malformed global metadata, duplicate
write slots and overlapping writable allocations remain outside
the caller contract. Gather output is zero-initialized, unlike
historical uninitialized allocation; extra zeroing and binary-search
cost must be measured, as must fixed 256-thread store geometry.
Canonical contiguous caches only. Public HND IO reexports routed
with feature; historical default unchanged. No global USE_FP8
capability promotion. FlashInfer decode, MLA decode and scratch/
scheduler semantics untouched and still unavailable in catalogue-only.

Tests: checked span CPU test and ignored dkr_flashinfer_hnd_io_gpu,
12 store configurations and 48 gather configurations (3 dtypes,
plain/FP8, rank 3/4, four metadata pairs), repeat dispatch, independent
raw storage and gather references, nonzero cache/input offsets,
sentinel prefix, permuted pages, empty sequence, negative slot,
E4M3 saturation and midpoint rounding. No build/run performed.
Prospective extended catalogue 32 families, 408 profiles, 33 images
(six new sources relative to immutable S7 baseline); within current
512-profile limit, subject to actual artifact sizes and validation.
No new compilation requested. Next: MLA decode, then FlashInfer
decode with split-KV scheduler/scratch contract preserved.

## R1.26/S9B4b resolved graph PASS — 2026-11-24

User executed d1b70ae8. S9B2, S9B3, S9B4A, S9B4B and strict
features PASS. Static producer elimination still NOT_QUALIFIED.
No compilation or physical execution.

## R1.26/S9B4c staged — historical MLA decode, typed launch

Important correction to the initial split-KV expectation: current
flashinfer_mla_decode.cu invokes the generic dispatch with null tmp_v
and tmp_s, so the historical MLA API uses nonpartitioned full-context
decode. It has no scratch argument. Do not claim split-KV qualification.
New optional dkr-mla-decode forwards through core and replaces that
public reexport, with no static FFI in the selected adapter. The
legacy global entry and the new typed entry share the same vendored
device body, avoiding algorithm duplication. Retain original
512/64 dimensions, 32x8x1 block, two heads per y lane, 1/2 stages
selected at device compile target, fast-math and dynamic shared
memory from the historical dispatch formula. The new Cudaforge
recipe watches the vendored FlashInfer files, and uses extended
lambda support. Device code reconstructs Params from typed tensors
and scalar arguments; no C++ struct ABI crosses the Rust boundary.
Host fastdiv magic values use the existing unsigned algorithm and
are cached with the device-aware launch plan. No runtime CC branch
in Rust. Same public I32 planner metadata, no extra CPU reads.
Valid planner contents, nonempty contexts, aligned contiguous
Q/cache layouts and same device are caller contracts. Bounds on
packed offsets and shapes are enforced; no general metadata value
validation. No FP8 MLA, split-KV, graph-safety or performance claim.

Three profiles F16/BF16/F32. Prospective extended catalogue
33 families, 411 profiles, 34 CUBIN images, seven new sources
relative to S7. Original 27 images remain unchanged. The new
ignored GPU test covers three dtypes, three page sizes (including
nonpower-of-two), two request mappings and two scales, 36 cases,
17 heads, partial pages, distinct CKV/KPE contributions, CPU F64
reference and repeated dispatch. CPU fastdiv test exercises all
admitted page divisors. These crate tests have not been built/run.
Legacy device wrapper refactor also needs regression validation
at the grouped gate. No new CUDA build requested.
Next: ordinary FlashInfer Decode with its actual optional scratch
and partition merge semantics, then closure and grouped GPU tests.
