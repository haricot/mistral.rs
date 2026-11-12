#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Portable SM61 dense GEMM baseline: X[M,K] * W[N,K]^T -> Y[M,N].
// Per-thread output, 16x16 shared-memory K tiles, F32 accumulation.
// This is a conservative reference, not a cuBLAS(Lt) replacement yet.
template<typename T, bool HasBias>
__device__ __forceinline__ void dkr_gemm(
    const T* __restrict__ x, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ output,
    uint32_t m, uint32_t n, uint32_t k) {
  constexpr int TILE = 16;
  const uint32_t row = blockIdx.y * TILE + threadIdx.y;
  const uint32_t col = blockIdx.x * TILE + threadIdx.x;
  __shared__ float a[TILE][TILE];
  __shared__ float b[TILE][TILE];
  float accumulator = 0.0f;
  for (uint32_t base = 0; base < k; base += TILE) {
    const uint32_t ak = base + threadIdx.x;
    const uint32_t bk = base + threadIdx.x;
    const uint32_t wrow = blockIdx.x * TILE + threadIdx.y;
    a[threadIdx.y][threadIdx.x] =
        row < m && ak < k
            ? static_cast<float>(x[static_cast<uint64_t>(row) * k + ak])
            : 0.0f;
    b[threadIdx.y][threadIdx.x] =
        wrow < n && bk < k
            ? static_cast<float>(weight[static_cast<uint64_t>(wrow) * k + bk])
            : 0.0f;
    __syncthreads();
#pragma unroll
    for (int t = 0; t < TILE; t++)
      accumulator = __fadd_rn(accumulator, __fmul_rn(
          a[threadIdx.y][t], b[threadIdx.x][t]));
    __syncthreads();
  }
  if (row < m && col < n) {
    if constexpr (HasBias) {
      accumulator += static_cast<float>(bias[col]);
    }
    output[static_cast<uint64_t>(row) * n + col] = static_cast<T>(accumulator);
  }
}

// Different operation signatures: no optional bias pointer and no variant
// check in the device kernel's hot loop.
#define DKR_GEMM_PAIR(NAME, TYPE)                                        \
extern "C" __global__ void dkr_gemm_##NAME##_plain(                     \
    const TYPE* x, const TYPE* weight, TYPE* output,                     \
    uint32_t m, uint32_t n, uint32_t k) {                               \
  dkr_gemm<TYPE, false>(x, weight, nullptr, output, m, n, k);            \
}                                                                       \
extern "C" __global__ void dkr_gemm_##NAME##_bias(                      \
    const TYPE* x, const TYPE* weight, const TYPE* bias, TYPE* output,   \
    uint32_t m, uint32_t n, uint32_t k) {                               \
  dkr_gemm<TYPE, true>(x, weight, bias, output, m, n, k);               \
}
DKR_GEMM_PAIR(f16, __half)
DKR_GEMM_PAIR(bf16, __nv_bfloat16)
DKR_GEMM_PAIR(f32, float)
