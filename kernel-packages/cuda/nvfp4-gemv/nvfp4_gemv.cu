#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// DKR-R1.21/S3: SM61 NVFP4 W4A16 decode GEMV directly from packed E2M1
// weights, E4M3 scales per group of 16, and F32 per-output-row globals.
// Separate plain/bias signatures; do not materialize dequantized weights.

__device__ __forceinline__ float e2m1(uint8_t code) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = mag[code & 7u];
  return code & 8u ? -v : v;
}
__device__ __forceinline__ float e4m3(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}
template<typename T>
__device__ __forceinline__ float to_f32(T x) { return static_cast<float>(x); }
template<>
__device__ __forceinline__ float to_f32<__half>(__half x) {
  return __half2float(x);
}
template<>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template<typename T>
__device__ __forceinline__ T from_f32(float x) { return static_cast<T>(x); }
template<>
__device__ __forceinline__ __half from_f32<__half>(float x) {
  return __float2half_rn(x);
}
template<>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template<typename T, bool HasBias>
__device__ __forceinline__ void nvfp4_gemv(
    const T* input, const uint8_t* packed,
    const __nv_fp8_e4m3* scales, const float* global,
    const T* bias, T* output,
    uint32_t m, uint32_t n, uint32_t k) {
  const uint32_t row = blockIdx.y;
  const uint32_t col = blockIdx.x;
  if (row >= m || col >= n) return;
  const uint64_t xbase = static_cast<uint64_t>(row) * k;
  const uint64_t wbase = static_cast<uint64_t>(col) * (k/2u);
  const uint64_t sbase = static_cast<uint64_t>(col) * (k/16u);
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const uint8_t packed_pair = packed[wbase + i/2u];
    const uint8_t code = i & 1u ? packed_pair >> 4 : packed_pair & 15u;
    const float weight = e2m1(code) * e4m3(scales[sbase + i/16u].__x);
    sum = fmaf(to_f32(input[xbase + i]), weight, sum);
  }
  __shared__ float scratch[4];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  for (int shift=16; shift>0; shift>>=1)
    sum += __shfl_down_sync(0xffffffff, sum, shift);
  if (lane == 0) scratch[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float result = (scratch[0]+scratch[1]+scratch[2]+scratch[3])*global[col];
    if constexpr (HasBias) result += to_f32(bias[col]);
    output[static_cast<uint64_t>(row)*n+col] = from_f32<T>(result);
  }
}

#define DKR_DEFINE_NVFP4(NAME, TYPE, MODE, HAS_BIAS) \
extern "C" __global__ void dkr_nvfp4_gemv_##NAME##_##MODE( \
    const TYPE* input, const uint8_t* packed, \
    const __nv_fp8_e4m3* scales, const float* global, \
    DKR_NVFP4_BIAS_##HAS_BIAS \
    TYPE* output, uint32_t m, uint32_t n, uint32_t k) { \
  nvfp4_gemv<TYPE, HAS_BIAS>(input, packed, scales, global, \
      DKR_NVFP4_PTR_##HAS_BIAS, output, m, n, k); \
}
#define DKR_NVFP4_BIAS_true const TYPE* bias,
#define DKR_NVFP4_BIAS_false
#define DKR_NVFP4_PTR_true bias
#define DKR_NVFP4_PTR_false nullptr

DKR_DEFINE_NVFP4(f16, __half, plain, false)
DKR_DEFINE_NVFP4(f16, __half, bias, true)
DKR_DEFINE_NVFP4(bf16, __nv_bfloat16, plain, false)
DKR_DEFINE_NVFP4(bf16, __nv_bfloat16, bias, true)
