#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Simple, portable Pascal SIMT reference for small-batch dense GEMV.
// No WMMA, tensor cores, implicit repacking or optional pointer in the ABI.
template<typename T>
__device__ __forceinline__ float dkr_gemv_to_float(T x) {
  return static_cast<float>(x);
}
template<>
__device__ __forceinline__ float dkr_gemv_to_float<__half>(__half x) {
  return __half2float(x);
}
template<>
__device__ __forceinline__ float dkr_gemv_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template<typename T>
__device__ __forceinline__ T dkr_gemv_from_float(float x) {
  return static_cast<T>(x);
}
template<>
__device__ __forceinline__ __half dkr_gemv_from_float<__half>(float x) {
  return __float2half_rn(x);
}
template<>
__device__ __forceinline__ __nv_bfloat16 dkr_gemv_from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template<typename T, bool HasBias>
__device__ __forceinline__ void dkr_gemv_body(
    const T* __restrict__ weight, const T* __restrict__ input,
    const T* __restrict__ bias, T* __restrict__ output,
    uint32_t rows, uint32_t columns, uint32_t batch) {
  const uint32_t m = blockIdx.x;
  const uint32_t b = blockIdx.y;
  if (m >= rows || b >= batch) return;
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5u;
  const uint64_t w_base = static_cast<uint64_t>(m) * columns;
  const uint64_t x_base = static_cast<uint64_t>(b) * columns;
  float sum = 0.0f;
  for (uint32_t k = threadIdx.x; k < columns; k += blockDim.x) {
    const float a = dkr_gemv_to_float(weight[w_base + k]);
    const float x = dkr_gemv_to_float(input[x_base + k]);
    sum = __fadd_rn(sum, __fmul_rn(a, x));
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  }
  __shared__ float warp_sums[4];
  if (lane == 0u) warp_sums[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0u) {
    float result = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    if constexpr (HasBias) result += dkr_gemv_to_float(bias[m]);
    output[static_cast<uint64_t>(b) * rows + m] = dkr_gemv_from_float<T>(result);
  }
}

// ABI has no nullable or optional tensor: distinct symbols and manifests.
// Launch block is fixed at [128, 1, 1], grid at [rows, batch, 1].
#define DKR_DEFINE_GEMV(TYPE, NAME) \
extern "C" __global__ void dkr_gemv_##NAME##_plain( \
    const TYPE* weight, const TYPE* input, TYPE* output, \
    uint32_t rows, uint32_t columns, uint32_t batch) { \
  dkr_gemv_body<TYPE, false>(weight, input, nullptr, output, rows, columns, batch); \
} \
extern "C" __global__ void dkr_gemv_##NAME##_bias( \
    const TYPE* weight, const TYPE* input, const TYPE* bias, TYPE* output, \
    uint32_t rows, uint32_t columns, uint32_t batch) { \
  dkr_gemv_body<TYPE, true>(weight, input, bias, output, rows, columns, batch); \
}

DKR_DEFINE_GEMV(__half, f16)
DKR_DEFINE_GEMV(__nv_bfloat16, bf16)
DKR_DEFINE_GEMV(float, f32)
