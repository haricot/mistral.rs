#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
report="$1"
plan="$2"
component="$3"
mkdir -p "$report"
python3 "$script_dir/proof.py" identity "$plan" "$component" > "$report/node.json"
candidate="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["candidate"])' "$report/node.json")"
git switch --detach --force "$candidate"
git config --global --add safe.directory "$PWD"
export CUDA_COMPUTE_CAP=61 ALLOW_LEGACY=bf16,fp8
export CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 RUSTC_WRAPPER=/opt/tools/sccache
export CARGO_TARGET_DIR="$CAMPAIGN_CACHE/target"
export SCCACHE_DIR="$CAMPAIGN_CACHE/sccache" SCCACHE_CACHE_SIZE=512M
export CARGO_HOME="$CAMPAIGN_CACHE/cargo"
mkdir -p "$CARGO_HOME" "$SCCACHE_DIR" "$CARGO_TARGET_DIR"
kernel_key="$(git ls-tree -r HEAD -- mistralrs-core/build.rs mistralrs-core/src/cuda \
  mistralrs-paged-attn/build.rs mistralrs-paged-attn/src/cuda \
  mistralrs-quant/build.rs mistralrs-quant/kernels | sha256sum | cut -d' ' -f1)"
export MISTRALRS_CUDA_BUILD_ROOT="$CAMPAIGN_CACHE/cuda/$kernel_key"
features="$(python3 -c 'import json,sys;print(",".join(json.load(open(sys.argv[1]))["config"]["features"]))' "$plan")"
test_args=(--lib --bin mistralrs)
for name in campaign_backends legacy_sm61_runtime asd_sm61_smoke extended_sm61_runtime lfm25_dspark_sm61_runtime; do
  if [[ -f "mistralrs-core/tests/$name.rs" ]]; then test_args+=(--test "$name"); fi
done
date -u +%FT%TZ > "$report/start.txt"
cargo check --locked --profile campaign -p mistralrs-cli --no-default-features \
  --features "$features" --timings > "$report/check.log" 2>&1 || { tail -n 100 "$report/check.log"; exit 1; }
cargo nextest archive --locked --cargo-profile campaign \
  -p mistralrs-core -p mistralrs-quant -p mistralrs-cli -p mistralrs-server-core --no-default-features \
  --features "$features" "${test_args[@]}" --archive-file "$report/tests.tar.zst" \
  --config-file "$script_dir/nextest.toml" > "$report/build.log" 2>&1 || { tail -n 100 "$report/build.log"; exit 1; }
cp /opt/tools/cargo-nextest "$report/cargo-nextest"
python3 "$script_dir/proof.py" config "$report/nextest.toml" cpu --junit "$report/cpu-junit.xml"
"$report/cargo-nextest" nextest run --archive-file "$report/tests.tar.zst" \
  --config-file "$report/nextest.toml" --profile cpu --no-tests=fail \
  > "$report/cpu.log" 2>&1 || { tail -n 100 "$report/cpu.log"; exit 1; }
cp Cargo.lock "$report/Cargo.lock"
find "$CARGO_TARGET_DIR/cargo-timings" -maxdepth 1 -name '*.html' -exec cp {} "$report/" \; || true
sccache --show-stats > "$report/cache-stats.txt"
python3 "$script_dir/proof.py" build "$report"
