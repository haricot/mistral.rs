# External F32 linear provider: first mechanism proof

This standalone executable uses the exact Candle revision already pinned by
Legacy V3 (`9cba2e898af3d25a89257f1f2ec5acece0404dcb`). It requests only `cuda`.
It does not modify Candle, add a production dispatcher or connect model execution.
W8ABF16, FP8 and FP4 availability is not a prerequisite.

## What it demonstrates

1. Build the Rust executable once, without embedding the external CUDA sources/PTX.
2. Separately compile A (per-output dot product) and B (shared-memory tiled matmul).
3. Read their manifests and PTX from disk. Both intentionally use the same
   `implementation_id`; their content hashes, entry symbols and module keys differ.
4. Execute A, B, A in one process, using the same input tensors, Candle CUDA device
   and stream. The second A reuses the content-addressed cached module.
5. Compare every result with an independent F64 CPU oracle and Candle F32 matmul,
   and require exact A replay equality. The executable SHA256 must stay unchanged.

There is no Rust enum or architecture equality branch choosing A versus B. The
provided manifest path selects the implementation; the ABI determines the call.
This is explicit selection, without autotuning or a global selection policy.

## Operation contract

`mistral.linear.f32.v1`: `Y[M,N] = X[M,K] * W[N,K]^T`, no bias, no workspace.
Inputs/output, multiplication and accumulation are F32. X and W are rank two,
contiguous and zero-offset CUDA tensors on the prepared Candle device/context.

Kernel parameters in order: `const float* X`, `const float* W`, `float* Y`,
`int32 M`, `int32 N`, `int32 K`. Pointers have 64-bit addressing. Each thread owns
one output at `(blockIdx.y * blockDim.y + threadIdx.y,
blockIdx.x * blockDim.x + threadIdx.x)`. Grid X/Y use ceiling division by the
manifest block X/Y; Z is one. Tail threads must not write outside Y. The producer
must declare a block compatible with its kernel; B requires `[16,16,1]`.

The manifest binds schema/ABI, implementation identity, artifact path/hash, entry,
PTX target, admissible compute capability interval, dtype/layout, bias, launch block
and maximum M/N/K. Packages here conservatively advertise only their compiled
architecture. `CUDA_COMPUTE_CAP` selects the artifact target (61, 75, 80, etc.);
additional hardware needs a new qualified package, rather than a new host enum.

The loader verifies integrity, path confinement, target, symbol, contract, shapes
and a 64 MiB total X/W/Y budget before launch. PTX is capped at 8 MiB. CUDA code is
executable code: hashes detect content changes, not kernel correctness; install
trusted packages. Host format conversion or a new ABI still requires host support.

Nine cases cover M=1/4/5 and N/K=(17,33)/(32,64)/(31,65), including tile tails.
GPU validation also rejects offset, wrong layout/dtype/K, another Candle device
instance and CPU tensors. Output initialization uses the Candle CUDA stream.
Numerical tolerances are `atol=1e-5`, `rtol=1e-5`; nonfinite values fail. Artifact
build disables contraction and flush-to-zero. Candle TF32/reduced F32 precision
is explicitly disabled for the reference. No performance claim is made.

## Commands

From the repository root, with Rust 1.99 and Python 3.11+:

```bash
python3 legacy-provider-poc/f32/run.py cpu
```

Manifest validation and CPU tests need neither CUDA nor GPU initialization.
`validate MANIFEST` verifies a package without creating a CUDA device.

On the physical SM61 machine, with CUDA 12.9 available in PATH:

```bash
set -o pipefail
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=gcc-14 \
  python3 legacy-provider-poc/f32/run.py gpu 2>&1 | tee /tmp/provider-f32.log
```

`OUTPUT=...` identifies a new temporary directory containing the two external
packages, binary/build evidence and `evidence.json`. Successful execution ends
with `PROVIDER_F32_GPU=PASS`. The log also reports matching binary hashes.
The script never registers a runner and never reads/writes the ASD store.

To use a persistent Cargo build directory, set `CARGO_TARGET_DIR`. The executable
is a demonstration tool built in the dev profile with optimization level 1,
not the production Mistral binary. A fresh temporary target is used by default.

To compile the CUDA consumer and both artifact implementations without executing
on a GPU:

```bash
CUDA_COMPUTE_CAP=61 NVCC_CCBIN=g++ \
  python3 legacy-provider-poc/f32/run.py cuda-compile --output /tmp/provider-f32-build
```

`.github/workflows/legacy-provider-f32-poc.yml` runs CPU contracts and a CUDA
compile/PTX matrix for SM61/75/80. These jobs do not establish physical execution
or portability of one Rust binary across all GPUs. Physical validation remains
explicitly separate and emits architecture and A/B/A evidence.

The next step after physical success is one real Mistral operation integration;
this demonstration does not yet replace any Legacy V3 provider.

The shared executor now lives in `mistralrs-kernel-runtime`. See
[`../inference/README.md`](../inference/README.md) for its UnquantLinear and real
Mistral engine integration with a minimal manual ASD linear profile.
