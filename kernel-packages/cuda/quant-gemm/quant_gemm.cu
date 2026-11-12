#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// DKR-R1.21/S4: reference prefill GEMM, directly reading quantized weights.
// Each thread computes one output element; F32 accumulator, SM61 SIMT.
// Separate variants for MXFP4 with and without bias. No intermediate
// dequantized weight buffer. Do not promote without a cuBLAS/static A/B.

template <typename T>
__device__ __forceinline__ float dkr_f32(T value) {
  return static_cast<float>(value);
}
template <>
__device__ __forceinline__ float dkr_f32<__half>(__half x) {return __half2float(x);}
template <>
__device__ __forceinline__ float dkr_f32<__nv_bfloat16>(__nv_bfloat16 x) {return __bfloat162float(x);}
template <typename T>
__device__ __forceinline__ T dkr_cast(float value) {return static_cast<T>(value);}
template <>
__device__ __forceinline__ __half dkr_cast<__half>(float x) {return __float2half_rn(x);}
template <>
__device__ __forceinline__ __nv_bfloat16 dkr_cast<__nv_bfloat16>(float x) {return __float2bfloat16_rn(x);}

__device__ __forceinline__ float dkr_fp8(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}
__device__ __forceinline__ float dkr_e2m1(uint8_t code) {
  const unsigned exp = (code >> 1) & 3u;
  const unsigned mantissa = code & 1u;
  // Exponent encoding 0: 0/.5, 1: 1/1.5, 2: 2/3, 3: 4/6.
  const float magnitude =
      exp == 0u ? (mantissa ? 0.5f : 0.0f)
                : (1u << (exp - 1u)) * (1.0f + mantissa * 0.5f);
  return (code & 8u) ? -magnitude : magnitude;
}
__device__ __forceinline__ float dkr_e8m0(uint8_t exp) {
  return __uint_as_float(static_cast<unsigned>(exp) << 23);
}

template<typename T>
__device__ __forceinline__ void dkr_fp8_gemm(
    const T* input, const __nv_fp8_e4m3* weight, const float* scales,
    T* output, uint32_t m, uint32_t n, uint32_t k,
    uint32_t scale_stride, uint32_t block_y, uint32_t block_x) {
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const uint64_t xbase = static_cast<uint64_t>(row) * k;
  const uint64_t wbase = static_cast<uint64_t>(col) * k;
  const uint64_t sbase = static_cast<uint64_t>(col / block_y) * scale_stride;
  float total = 0.0f;
  for (uint32_t idx=0; idx < k; ++idx) {
    const float w = dkr_fp8(weight[wbase + idx].__x);
    const float scale = scales[sbase + idx/block_x];
    total = fmaf(dkr_f32(input[xbase + idx]), w * scale, total);
  }
  output[static_cast<uint64_t>(row) * n + col] = dkr_cast<T>(total);
}

template<typename T, bool HasBias>
__device__ __forceinline__ void dkr_mxfp4_gemm(
    const T* input, const uint8_t* weight, const uint8_t* scales,
    const T* bias, T* output, uint32_t m, uint32_t n, uint32_t k) {
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  const uint64_t xbase = static_cast<uint64_t>(row) * k;
  const uint64_t wbase = static_cast<uint64_t>(col) * (k/2);
  const uint64_t sbase = static_cast<uint64_t>(col) * (k/32);
  float total = 0.0f;
  for (uint32_t idx=0; idx<k; ++idx) {
    const uint8_t pair = weight[wbase + idx/2];
    const uint8_t code = (idx & 1u) ? pair >> 4 : pair & 15u;
    const float w = dkr_e2m1(code) * dkr_e8m0(scales[sbase + idx/32]);
    total = fmaf(dkr_f32(input[xbase + idx]), w, total);
  }
  if constexpr (HasBias) total += dkr_f32(bias[col]);
  output[static_cast<uint64_t>(row) * n + col] = dkr_cast<T>(total);
}

#define DKR_FP8_GEMM(DTYPE, NAME) \
extern "C" __global__ void dkr_fp8_block_gemm_##NAME( \
    const DTYPE* input, const __nv_fp8_e4m3* weight, const float* scales, \
    DTYPE* output, uint32_t m, uint32_t n, uint32_t k, \
    uint32_t scale_stride, uint32_t block_y, uint32_t block_x) { \
  dkr_fp8_gemm<DTYPE>(input, weight, scales, output, m, n, k, \
                       scale_stride, block_y, block_x); \
}
#define DKR_MXFP4_GEMM_PLAIN(DTYPE, NAME) \
extern "C" __global__ void dkr_mxfp4_gemm_##NAME##_plain( \
    const DTYPE* input, const uint8_t* weight, const uint8_t* scales, \
    DTYPE* output, uint32_t m, uint32_t n, uint32_t k) { \
  dkr_mxfp4_gemm<DTYPE, false>(input, weight, scales, nullptr, output, m, n, k); \
}
#define DKR_MXFP4_GEMM_BIAS(DTYPE, NAME) \
extern "C" __global__ void dkr_mxfp4_gemm_##NAME##_bias( \
    const DTYPE* input, const uint8_t* weight, const uint8_t* scales, \
    const DTYPE* bias, DTYPE* output, uint32_t m, uint32_t n, uint32_t k) { \
  dkr_mxfp4_gemm<DTYPE, true>(input, weight, scales, bias, output, m, n, k); \
}

DKR_FP8_GEMM(__half, f16)
DKR_FP8_GEMM(__nv_bfloat16, bf16)
DKR_MXFP4_GEMM_PLAIN(__half, f16)
DKR_MXFP4_GEMM_PLAIN(__nv_bfloat16, bf16)
DKR_MXFP4_GEMM_BIAS(__half, f16)
DKR_MXFP4_GEMM_BIAS(__nv_bfloat16, bf16)
