#!/usr/bin/env bash
# Deferred qualification: never run during source-only DKR staging.
# Builds PTX for exactly sm_61, then runs one grouped cargo-nextest campaign.
set -euo pipefail

if [[ "${DKR_APPROVE_FINAL:-0}" != "1" ]]; then
  echo "DKR SM61 final campaign is deferred. Set DKR_APPROVE_FINAL=1 explicitly." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
# NEVER set, unset, or export CUDA_COMPUTE_CAP here:
# changing its value invalidates upstream build.rs caches and can rebuild hours
# of static kernels. Only the external PTX builder receives --cc 61.
echo "CUDA_COMPUTE_CAP left unchanged (${CUDA_COMPUTE_CAP-<unset>})"
if [[ -n "${MISTRALRS_CUDA_BUILD_ROOT:-}" ]]; then
  echo "DKR_PERSISTENT_CUDA_OBJECT_CACHE=$MISTRALRS_CUDA_BUILD_ROOT"
  echo "Use tools/dkr_cuda_cache_bootstrap.py --apply before the first shared-cache build."
fi

if [[ -n "${DKR_CANDLE_DIR:-}" ]]; then
  python3 tools/dkr_operation_matrix.py --candle-dir "$DKR_CANDLE_DIR" \
    --output "${DKR_OPERATION_MATRIX_OUTPUT:-/tmp/dkr-sm61-operation-matrix.json}"
else
  echo "DKR_CANDLE_DIR not set: audit will not include pinned Candle source" >&2
  python3 tools/dkr_operation_matrix.py
fi

# cudaforge owns nvcc and its incremental cache. Python only emits manifests.
cargo run --locked -p mistralrs-kernel-runtime \
  --features builder-cudaforge \
  --bin dkr_package_builder -- \
  --cc 61 --output kernel-packages/build/cuda

export DKR_KERNEL_PACKAGES="$(pwd)/kernel-packages/build/cuda/sm61"

# A single feature-unified build and grouped DKR test run.
# --test-threads=1 prevents overlapping GPU test workloads on 8 GiB devices.
cargo nextest run --locked \
  -p mistralrs-kernel-runtime \
  -p mistralrs-quant \
  -p mistralrs-paged-attn \
  -p mistralrs-core \
  --lib \
  --features 'mistralrs-core/cuda-legacy,mistralrs-core/dkr-moe,mistralrs-core/dkr-indexed-copy,mistralrs-core/dkr-gated-add,mistralrs-core/dkr-reduce-norm,mistralrs-core/dkr-add-rms-norm,mistralrs-core/dkr-causal-mask,mistralrs-core/dkr-logits,mistralrs-core/dkr-rms-variants,mistralrs-core/dkr-input-pad,mistralrs-core/dkr-ssm,mistralrs-core/dkr-rounded-rope,mistralrs-core/dkr-gemv,mistralrs-core/dkr-kv-scales,mistralrs-core/dkr-fp8-cast,mistralrs-core/dkr-kv-gather,mistralrs-core/dkr-gemm,mistralrs-core/dkr-fp4-dequant' \
  --run-ignored=all \
  --no-fail-fast \
  --test-threads=1 \
  -E 'test(dkr_) or test(prevalidated_) or test(pads_decode_input_by_aliasing_row_zero_values)'

echo "DKR_SM61_FINAL_CAMPAIGN=COMPLETED_WITHOUT_EARLY_ABORT"
