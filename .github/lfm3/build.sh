#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
report="$1"
source "$here/../campaign/compiler-env.sh"
export CUDA_COMPUTE_CAP=61 ALLOW_LEGACY=bf16,fp8 CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR="$CAMPAIGN_CACHE/target" CARGO_HOME="$CAMPAIGN_CACHE/cargo"
export SCCACHE_DIR="$CAMPAIGN_CACHE/sccache" SCCACHE_CACHE_SIZE=512M RUSTC_WRAPPER=/opt/tools/sccache
mkdir -p "$CARGO_TARGET_DIR" "$CARGO_HOME" "$SCCACHE_DIR"
for target in legacy extended; do
  python3 "$here/qualification.py" build --repo "$PWD" --output "$report" --target "$target"
done
