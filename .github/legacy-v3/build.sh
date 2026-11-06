#!/usr/bin/env bash
set -euo pipefail
prep="$1"
report="$2"
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$here/campaign.py" verify-plan --directory "$prep"
mkdir -p "$report"
candidate="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate"])' "$prep/plan.json")"
git fetch --no-tags "$prep/candidate.bundle" "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate_ref"])' "$prep/plan.json")"
git switch --detach --force "$candidate"
export CUDA_COMPUTE_CAP=61
export CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
export CARGO_HOME="$CAMPAIGN_CACHE/cargo" CARGO_TARGET_DIR="$CAMPAIGN_CACHE/target"
export SCCACHE_DIR="$CAMPAIGN_CACHE/sccache" SCCACHE_CACHE_SIZE=512M RUSTC_WRAPPER=/opt/tools/sccache
export RUSTFLAGS="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["identity"]["rustflags"])' "$prep/plan.json")"
export NVCC_CCBIN="${NVCC_CCBIN:-g++}"
mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR" "$SCCACHE_DIR"
export MISTRALRS_CUDA_BUILD_ROOT="$CAMPAIGN_CACHE/cuda/$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["key"])' "$prep/plan.json")"
{
  rustc -vV
  nvcc --version
  "$NVCC_CCBIN" --version
  printf 'RUSTFLAGS=%s\nCUDA_COMPUTE_CAP=%s\n' "$RUSTFLAGS" "$CUDA_COMPUTE_CAP"
} > "$report/compiler.txt"
cp Cargo.lock "$report/Cargo.lock"
features='mistralrs-cli/cuda-legacy,mistralrs-cli/code-execution'
cargo build --release --locked -p mistralrs-cli --no-default-features --features "$features" 2>&1 | tee "$report/build-cli.log"
cp "$CARGO_TARGET_DIR/release/mistralrs" "$report/mistralrs"
cargo nextest archive --locked --cargo-profile release \
  -p mistralrs-core -p mistralrs-quant -p mistralrs-cli -p mistralrs-server-core \
  --no-default-features --features "$features" --lib --bin mistralrs \
  --test bf16_sm61_standalone --test bf16_standalone_cpu \
  --test fp8_sm61_standalone --test fp8_standalone_cpu --test fp4_standalone_cpu \
  --test nvfp4_checkpoint_tests --test campaign_paged_fp8 --test legacy_sm61_runtime \
  --test legacy_v3_support --archive-file "$report/tests.tar.zst" \
  --config-file "$here/nextest.toml" 2>&1 | tee "$report/build-tests.log"
cp /opt/tools/cargo-nextest "$report/cargo-nextest"
python3 "$here/campaign.py" config --directory "$prep" --profile cpu --output "$report/nextest.toml"
python3 "$here/../campaign/cpu-runtime.py" -- "$report/cargo-nextest" nextest run \
  --archive-file "$report/tests.tar.zst" --config-file "$report/nextest.toml" \
  --profile cpu --no-tests=fail 2>&1 | tee "$report/cpu.log"
python3 "$here/campaign.py" build-proof --directory "$prep" --output "$report"
sccache --show-stats > "$report/cache-stats.txt" || true
