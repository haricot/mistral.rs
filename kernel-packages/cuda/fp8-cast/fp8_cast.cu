#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Same E4M3 scalar conversion semantics as mistralrs-quant scalar_fp8.cu:
// saturation to finite ±448, intermediate half rounding, then E4M3 rounding.
template <typename T>
__device__ __forceinline__ void encode_e4m3(
    const T *__restrict__ input, __nv_fp8_e4m3 *__restrict__ output,
    uint64_t count) {
  const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  float x = static_cast<float>(input[idx]);
  if (x > 448.0f) x = 448.0f;
  if (x < -448.0f) x = -448.0f;
  const __half intermediate = __float2half(x);
  output[idx].__x =
      __nv_cvt_halfraw_to_fp8(intermediate, __NV_SATFINITE, __NV_E4M3);
}

template <typename T>
__device__ __forceinline__ void decode_e4m3(
    const __nv_fp8_e4m3 *__restrict__ input,
    T *__restrict__ output, uint64_t count) {
  const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const float value =
      __half2float(__nv_cvt_fp8_to_halfraw(input[idx].__x, __NV_E4M3));
  output[idx] = static_cast<T>(value);
}

// Separate fixed signatures for each conversion; no runtime dtype switch.
#define DKR_E4M3_PAIR(NAME, TYPE)                                     \
extern "C" __global__ void dkr_fp8_encode_##NAME(                   \
    const TYPE* input, __nv_fp8_e4m3* output, uint64_t count) {       \
  encode_e4m3<TYPE>(input, output, count);                            \
}                                                                    \
extern "C" __global__ void dkr_fp8_decode_##NAME(                   \
    const __nv_fp8_e4m3* input, TYPE* output, uint64_t count) {       \
  decode_e4m3<TYPE>(input, output, count);                            \
}

DKR_E4M3_PAIR(f16, __half)
DKR_E4M3_PAIR(bf16, __nv_bfloat16)
DKR_E4M3_PAIR(f32, float)
