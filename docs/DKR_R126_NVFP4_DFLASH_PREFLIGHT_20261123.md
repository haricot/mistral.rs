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
