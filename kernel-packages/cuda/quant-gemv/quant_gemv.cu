#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// DKR-R1.21/S1: Pascal SIMT fused quantized GEMV reference (decode).
// F32 accumulation. No intermediate dequant tensor, no tensor cores.
// Fixed 128 threads per (input row, output column), with 4 warp sums.

template <typename T>
__device__ __forceinline__ float as_float(T value) {
  return static_cast<float>(value);
}
template <>
__device__ __forceinline__ float as_float<__half>(__half value) {
  return __half2float(value);
}
template <>
__device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}
template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float fp8_decode(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}

__device__ __forceinline__ float e2m1_decode(uint8_t code) {
  // Exact values: 0, .5, 1, 1.5, 2, 3, 4, 6, with sign bit 3.
  const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                          2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = table[code & 7u];
  return code & 8u ? -magnitude : magnitude;
}
__device__ __forceinline__ float mxfp4_scale(uint8_t exponent) {
  return __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}

// Warps reduce their independently accumulated dot-products.
__device__ __forceinline__ float reduce_128(float value) {
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  for (int delta = 16; delta > 0; delta >>= 1)
    value += __shfl_down_sync(0xffffffff, value, delta);
  __shared__ float sums[4];
  if (lane == 0u) sums[warp] = value;
  __syncthreads();
  return sums[0] + sums[1] + sums[2] + sums[3];
}

template <typename T>
__device__ __forceinline__ void fp8_gemv(
    const T* input, const __nv_fp8_e4m3* weights,
    const float* scales, T* output, uint32_t m, uint32_t n,
    uint32_t k, uint32_t scale_stride, uint32_t block_y, uint32_t block_x) {
  const uint32_t row = blockIdx.y;
  const uint32_t col = blockIdx.x;
  if (row >= m || col >= n) return;
  const uint64_t x = static_cast<uint64_t>(row) * k;
  const uint64_t w = static_cast<uint64_t>(col) * k;
  const uint64_t sr = static_cast<uint64_t>(col / block_y) * scale_stride;
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const float v = as_float(input[x + i]);
    const float weight = fp8_decode(weights[w + i].__x);
    const float scale = scales[sr + i / block_x];
    sum = fmaf(v, weight * scale, sum);
  }
  const float reduced = reduce_128(sum);
  if (threadIdx.x == 0u)
    output[static_cast<uint64_t>(row) * n + col] = from_float<T>(reduced);
}

template <typename T, bool Bias>
__device__ __forceinline__ void mxfp4_gemv(
    const T* input, const uint8_t* packed,
    const uint8_t* scales, const T* bias, T* output,
    uint32_t m, uint32_t n, uint32_t k) {
  const uint32_t row = blockIdx.y;
  const uint32_t col = blockIdx.x;
  if (row >= m || col >= n) return;
  const uint64_t x = static_cast<uint64_t>(row) * k;
  const uint64_t w = static_cast<uint64_t>(col) * (k / 2u);
  const uint64_t sr = static_cast<uint64_t>(col) * (k / 32u);
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const uint8_t pair = packed[w + i / 2u];
    const uint8_t code = (i & 1u) ? pair >> 4 : pair & 15u;
    const float weight = e2m1_decode(code);
    const float scale = mxfp4_scale(scales[sr + i / 32u]);
    sum = fmaf(as_float(input[x + i]), weight * scale, sum);
  }
  float result = reduce_128(sum);
  if (threadIdx.x == 0u) {
    if constexpr (Bias) result += as_float(bias[col]);
    output[static_cast<uint64_t>(row) * n + col] = from_float<T>(result);
  }
}

#define DKR_FP8_GEMV(TAG, TYPE) \
extern "C" __global__ void dkr_fp8_block_gemv_##TAG( \
    const TYPE* input, const __nv_fp8_e4m3* weights, \
    const float* scales, TYPE* output, \
    uint32_t m, uint32_t n, uint32_t k, uint32_t scale_stride, \
    uint32_t block_y, uint32_t block_x) { \
  fp8_gemv<TYPE>(input, weights, scales, output, m, n, k, \
                 scale_stride, block_y, block_x); \
}

#define DKR_MXFP4_GEMV(TAG, TYPE, MODE, BIAS) \
extern "C" __global__ void dkr_mxfp4_gemv_##TAG##_##MODE( \
    const TYPE* input, const uint8_t* weights, \
    const uint8_t* scales, \
    DKR_MXFP4_BIAS_##BIAS \
    TYPE* output, uint32_t m, uint32_t n, uint32_t k) { \
  mxfp4_gemv<TYPE, BIAS>(input, weights, scales, DKR_MXFP4_PTR_##BIAS, \
                         output, m, n, k); \
}
#define DKR_MXFP4_BIAS_true const TYPE* bias,
#define DKR_MXFP4_BIAS_false
#define DKR_MXFP4_PTR_true bias
#define DKR_MXFP4_PTR_false nullptr

DKR_FP8_GEMV(f16, __half)
DKR_FP8_GEMV(bf16, __nv_bfloat16)
DKR_MXFP4_GEMV(f16, __half, plain, false)
DKR_MXFP4_GEMV(bf16, __nv_bfloat16, plain, false)
DKR_MXFP4_GEMV(f16, __half, bias, true)
DKR_MXFP4_GEMV(bf16, __nv_bfloat16, bias, true)
