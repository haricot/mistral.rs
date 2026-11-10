# External kernel packages (DKR generic loading proof)

The Rust runtime does not compile or embed the kernels in this directory. Each
architecture package contains a PTX artifact and a `manifest.json` with the same
versioned operation ABI and named implementation family.

```text
kernel-packages/
  cuda/
    sm61/elementwise-scale/{manifest.json,scale.ptx}
    sm80/elementwise-scale/{manifest.json,scale.ptx}
```

The architecture folders are package organisation, **not hard-coded runtime
branches**. `KernelCatalog::discover` walks the directory tree and checks hashes
and typed signatures. `KernelCatalog::resolve` selects the compatible package
with the highest PTX target capability; equal-priority compatible providers are
rejected as ambiguous. The CUDA executor checks the actual device capability.

The PTX scale examples intentionally implement exactly the same algorithm. Only
the `.target` directive differs. These are demonstration artifacts, not tuned or
numerically qualified production implementations.

## Local proof without GPU

From the repository root:

```sh
cargo test -p mistralrs-kernel-runtime --no-default-features --locked
```

Tests discover both checked-in packages, confirm automatic selection for
capabilities 61, 80 and 90, verify semantic PTX identity aside from target,
and refuse unsupported capability 60. They cannot prove device execution.

## Physical GPU proof (one host build)

From the repository root, with an already configured CUDA environment:

```sh
cargo build -p mistralrs-kernel-runtime --example dkr_scale_cuda --features cuda --locked
./target/debug/examples/dkr_scale_cuda
```

Expected result: `DKR_AUTO_SCALE_GPU=PASS` with `device_cc`, `selected_target_cc`
and the numerical error against a scalar oracle. No architecture name appears
in the executable adapter. Set `DKR_KERNEL_PACKAGES` to an alternative external
directory to test a new package without rebuilding the host executable.

Replace a package by creating a new trusted manifest with the SHA256 of its PTX;
never hand-edit a hash to bypass an integrity failure. Other GPU architecture
families can be supplied through new folders without changing Rust source,
provided they respect the same ABI. Native CUBIN and code signatures are beyond
this PTX-only R1 proof.

## DKR-R1.1: external MoE reference package

The MoE source lives in `kernel-packages/cuda/moe/moe.cu` and is not part of
`mistralrs-core/src/cuda/*.cu`. Its external package builder targets only the
requested architecture, and reuses previously compiled artifacts when the
source, NVCC version and compilation arguments are unchanged:

```sh
python3 kernel-packages/cuda/moe/build.py --cc 61
export DKR_KERNEL_PACKAGES="$PWD/kernel-packages/build/cuda"
cargo test -p mistralrs-core --features 'cuda-legacy,dkr-moe' \
  --lib dkr_moe_gpu -- --ignored --nocapture
```

`dkr-moe` opts in to the dynamically loaded provider for `moe_gemm`. The old
FFI-based MoE is retained in `moe_legacy.rs` for comparison without this
feature. The generic Rust dispatch knows only the logical operation
`moe.grouped_gemm.v1`, dtype, shapes and validated launch arguments. It has no
GPU-model-specific name or architecture switch. The caller must supply the
external catalog path via `DKR_KERNEL_PACKAGES`.

The external implementation is a scalar, numerically checkable reference for
F16 and BF16, not a promoted HFMA2/WMMA optimization. It does not use ASD and
is not numerically qualified on real hardware until the ignored GPU test passes.
Routed token indices must satisfy the same upstream invariants as the legacy
MoE: valid expert IDs and a permutation of output token indices. The current
typed ABI accepts contiguous zero-offset tensors; other layouts are rejected.
