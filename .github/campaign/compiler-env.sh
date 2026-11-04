#!/usr/bin/env bash
unset CARGO_ENCODED_RUSTFLAGS
campaign_cpu_args=()
if [[ "${1:-}" == --plan ]]; then campaign_cpu_args=(--plan "$2"); fi
campaign_rustflags="$(python3 "$(dirname -- "${BASH_SOURCE[0]}")/cpu_target.py" "${campaign_cpu_args[@]}" --check-host)"
export RUSTFLAGS="$campaign_rustflags"
unset campaign_cpu_args campaign_rustflags
