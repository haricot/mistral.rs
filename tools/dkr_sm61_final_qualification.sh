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
# Abort BEFORE even the lightweight Cargo builder if the persistent static
# cudaforge object cache is not the one bootstrapped for this SM61 workspace.
# Do not assign CUDA_COMPUTE_CAP or mutate the cache here.
CACHE_ROOT="$(pwd)/target/dkr-cuda-cache/sm61"
if [[ -z "${MISTRALRS_CUDA_BUILD_ROOT:-}" ]]; then
  echo "DKR_S9_CACHE_GUARD=FAIL MISTRALRS_CUDA_BUILD_ROOT is unset; refusing an uncached CUDA build" >&2
  exit 3
fi
if [[ "$(realpath -m -- "$MISTRALRS_CUDA_BUILD_ROOT")" != "$CACHE_ROOT" ]]; then
  echo "DKR_S9_CACHE_GUARD=FAIL unexpected cache root: $MISTRALRS_CUDA_BUILD_ROOT" >&2
  echo "Expected: $CACHE_ROOT" >&2
  exit 3
fi
for component in quant core paged-attn; do
  if [[ ! -s "$CACHE_ROOT/$component/kernels/.cudaforge_cache.json" ]]; then
    echo "DKR_S9_CACHE_GUARD=FAIL missing $component cudaforge cache index" >&2
    exit 3
  fi
done
cached_objects="$(find "$CACHE_ROOT" -type f -name '*.o' | wc -l)"
if (( cached_objects < 82 )); then
  echo "DKR_S9_CACHE_GUARD=FAIL expected at least 82 cached objects, found $cached_objects" >&2
  exit 3
fi
echo "DKR_S9_CACHE_GUARD=PASS root=$CACHE_ROOT objects=$cached_objects indices=3"
echo "Object presence is not a cache-hit guarantee; inspect compiler logs for reuse."

if [[ -n "${DKR_CANDLE_DIR:-}" ]]; then
  python3 tools/dkr_operation_matrix.py --candle-dir "$DKR_CANDLE_DIR" \
    --output "${DKR_OPERATION_MATRIX_OUTPUT:-/tmp/dkr-sm61-operation-matrix.json}"
else
  echo "DKR_CANDLE_DIR not set: audit will not include pinned Candle source" >&2
  python3 tools/dkr_operation_matrix.py
fi

# Cheap source-only checks FIRST. A failure here must not trigger a
# long static CUDA/Cargo rebuild. No CUDA_COMPUTE_CAP mutation.
python3 tools/dkr_r121_manifest_preflight.py
python3 tools/dkr_r121_v2_numeric_preflight.py
python3 tools/dkr_r121_s7_cpu_preflight.py
python3 tools/dkr_r121_s8_cpu_preflight.py
python3 tools/dkr_nvcc_flag_audit.py
python3 tools/dkr_s9_ptx_manifest_audit.py --self-test

# cudaforge owns nvcc and its incremental cache. Python only emits manifests.
cargo run --locked -p mistralrs-kernel-runtime \
  --features builder-cudaforge \
  --bin dkr_package_builder -- \
  --cc 61 --output kernel-packages/build/cuda

export DKR_KERNEL_PACKAGES="$(pwd)/kernel-packages/build/cuda/sm61"

# Real PTX gate: check actual compiled .entry symbols, architecture and SHA
# for EVERY typed manifest. Fail BEFORE cargo nextest builds the Rust crates.
python3 tools/dkr_s9_ptx_manifest_audit.py \
  --root "$DKR_KERNEL_PACKAGES" --cc 61

# A single feature-unified build and grouped DKR test run.
# --test-threads=1 prevents overlapping GPU test workloads on 8 GiB devices.
cargo nextest run --locked \
  -p mistralrs-kernel-runtime \
  -p mistralrs-quant \
  -p mistralrs-paged-attn \
  -p mistralrs-core \
  --lib \
  --features 'mistralrs-core/cuda-legacy,mistralrs-core/dkr-moe,mistralrs-core/dkr-indexed-copy,mistralrs-core/dkr-gated-add,mistralrs-core/dkr-reduce-norm,mistralrs-core/dkr-add-rms-norm,mistralrs-core/dkr-causal-mask,mistralrs-core/dkr-logits,mistralrs-core/dkr-rms-variants,mistralrs-core/dkr-input-pad,mistralrs-core/dkr-ssm,mistralrs-core/dkr-rounded-rope,mistralrs-core/dkr-gemv,mistralrs-core/dkr-kv-scales,mistralrs-core/dkr-fp8-cast,mistralrs-core/dkr-kv-gather,mistralrs-core/dkr-gemm,mistralrs-core/dkr-fp4-dequant,mistralrs-core/dkr-quant-gemv,mistralrs-core/dkr-paged-decode,mistralrs-core/dkr-nvfp4-gemv,mistralrs-core/dkr-quant-gemm,mistralrs-core/dkr-paged-fp8,mistralrs-core/dkr-paged-v2,mistralrs-core/dkr-kv-cache-ops,mistralrs-core/dkr-routed-quant' \
  --run-ignored=all \
  --no-fail-fast \
  --test-threads=1 \
  -E 'test(dkr_) or test(prevalidated_) or test(pads_decode_input_by_aliasing_row_zero_values)'

echo "DKR_SM61_FINAL_CAMPAIGN=COMPLETED_WITHOUT_EARLY_ABORT"
