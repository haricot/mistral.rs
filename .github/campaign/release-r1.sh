#!/usr/bin/env bash
# Runs inside the immutable CI compiler image; only Extended is built in release.
set -euo pipefail
plan="$1"
output="$2"
source_dir="$3"
mkdir -p "$output" "$CAMPAIGN_CACHE/cargo" "$CAMPAIGN_CACHE/sccache" "$CAMPAIGN_CACHE/target"
git config --global --add safe.directory "$source_dir"
candidate="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"])' "$plan")"
git -C "$source_dir" switch --detach --force "$candidate"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$candidate"
source "$(dirname "$0")/compiler-env.sh" --plan "$plan"
export CUDA_COMPUTE_CAP=61 ALLOW_LEGACY=bf16,fp8
export CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR="$CAMPAIGN_CACHE/target"
export CARGO_HOME="$CAMPAIGN_CACHE/cargo"
export SCCACHE_DIR="$CAMPAIGN_CACHE/sccache" SCCACHE_CACHE_SIZE=512M
export RUSTC_WRAPPER=/opt/tools/sccache
kernel_key="$(git -C "$source_dir" ls-tree -r HEAD -- mistralrs-core/build.rs mistralrs-core/src/cuda mistralrs-paged-attn/build.rs mistralrs-paged-attn/src/cuda mistralrs-quant/build.rs mistralrs-quant/kernels | sha256sum | cut -d' ' -f1)"
export MISTRALRS_CUDA_BUILD_ROOT="$CAMPAIGN_CACHE/cuda/$kernel_key"
features="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(",".join(sorted(set(p["config"]["features"]+["code-execution"]))))' "$plan")"
expected_lock="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(next(n["lock"] for n in p["matrix"] if n["component"]=="extended"))' "$plan")"
test "$(sha256sum "$source_dir/Cargo.lock" | cut -d' ' -f1)" = "$expected_lock"
cd "$source_dir"
cargo check --locked --release -p mistralrs-cli --bin mistralrs --no-default-features --features "$features"
cargo build --locked --release -p mistralrs-cli --bin mistralrs --no-default-features --features "$features"
install -m 0755 "$CARGO_TARGET_DIR/release/mistralrs" "$output/mistralrs"
mkdir -p "$output/loader-only"
install -m 0444 "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/libcuda.so" "$output/loader-only/libcuda.so.1"
python3 "$(dirname "$0")/release_r1.py" record --plan "$plan" --dir "$output"
python3 "$(dirname "$0")/cpu-runtime.py" --stub "${CUDA_HOME:-/usr/local/cuda}/lib64/stubs/libcuda.so" -- "$output/mistralrs" --help

