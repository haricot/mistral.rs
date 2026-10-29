#!/usr/bin/env bash
set -euo pipefail
rustc --version | grep -F 'rustc 1.99.0'
cargo --version | grep -F 'cargo 1.99.0'
cargo-nextest nextest --version | grep -F '0.9.138'
sccache --version | grep -F '0.12.0'
nvcc --version | grep -F 'release 12.9'
printf '__global__ void probe() {}\n' > /tmp/sm61-image-smoke.cu
nvcc -arch=sm_61 -c /tmp/sm61-image-smoke.cu -o /tmp/sm61-image-smoke.o
test -s /tmp/sm61-image-smoke.o
