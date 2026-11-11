#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ float as_float(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float as_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T> __device__ __forceinline__ T as_dtype(float v) {
  return static_cast<T>(v);
}
template <> __device__ __forceinline__ __half as_dtype<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 as_dtype<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float row_sum(float v, float* buf) {
  const unsigned t = threadIdx.x;
  buf[t] = v;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride >>= 1) {
    if (t < stride) buf[t] += buf[t + stride];
    __syncthreads();
  }
  return buf[0];
}

template <typename T, bool SCALED>
__device__ void residual_body(
    const T* x, const T* residual, const T* weight, const T* scale,
    T* output, uint32_t rows, uint32_t width, float eps) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  const uint64_t base = uint64_t(row) * width;
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
    float v = as_float(x[base + j]);
    sum += v * v;
  }
  float inv = rsqrtf(row_sum(sum, reduction) / float(width) + eps);
  float multiplier = 1.0f;
  if constexpr (SCALED) multiplier = as_float(scale[0]);
  for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
    float norm = as_float(x[base + j]) * inv * as_float(weight[j]);
    output[base + j] = as_dtype<T>(
        (as_float(residual[base + j]) + norm) * multiplier);
  }
}

template <typename T, bool SCALED>
__device__ void residual_then_body(
    const T* x, const T* residual, const T* residual_weight, const T* scale,
    const T* norm_weight, T* residual_output, T* normalized_output,
    uint32_t rows, uint32_t width, float eps1, float eps2) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float reduction[256];
  const uint64_t base = uint64_t(row) * width;
  float sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
    float v = as_float(x[base + j]);
    sum += v * v;
  }
  float inv = rsqrtf(row_sum(sum, reduction) / float(width) + eps1);
  float multiplier = 1.0f;
  if constexpr (SCALED) multiplier = as_float(scale[0]);
  float residual_sum = 0.0f;
  for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
    float norm = as_float(x[base + j]) * inv * as_float(residual_weight[j]);
    float value = (as_float(residual[base + j]) + norm) * multiplier;
    residual_output[base + j] = as_dtype<T>(value);
    residual_sum += value * value;
  }
  inv = rsqrtf(row_sum(residual_sum, reduction) / float(width) + eps2);
  for (uint32_t j = threadIdx.x; j < width; j += blockDim.x) {
    normalized_output[base + j] = as_dtype<T>(
        as_float(residual_output[base + j]) * inv * as_float(norm_weight[j]));
  }
}

#define DKR_RESIDUAL(NAME, TYPE, SCALED) \
extern "C" __global__ void dkr_rms_residual_##NAME##_plain( \
    const TYPE* x, const TYPE* residual, const TYPE* weight, TYPE* output, \
    uint32_t rows, uint32_t width, float eps) { \
  residual_body<TYPE, false>(x, residual, weight, nullptr, output, rows, width, eps); \
} \
extern "C" __global__ void dkr_rms_residual_##NAME##_scaled( \
    const TYPE* x, const TYPE* residual, const TYPE* weight, const TYPE* scale, \
    TYPE* output, uint32_t rows, uint32_t width, float eps) { \
  residual_body<TYPE, true>(x, residual, weight, scale, output, rows, width, eps); \
} \
extern "C" __global__ void dkr_rms_then_##NAME##_plain( \
    const TYPE* x, const TYPE* residual, const TYPE* rw, const TYPE* nw, \
    TYPE* residual_output, TYPE* normalized_output, \
    uint32_t rows, uint32_t width, float eps1, float eps2) { \
  residual_then_body<TYPE, false>(x, residual, rw, nullptr, nw, residual_output, \
                                  normalized_output, rows, width, eps1, eps2); \
} \
extern "C" __global__ void dkr_rms_then_##NAME##_scaled( \
    const TYPE* x, const TYPE* residual, const TYPE* rw, const TYPE* scale, \
    const TYPE* nw, TYPE* residual_output, TYPE* normalized_output, \
    uint32_t rows, uint32_t width, float eps1, float eps2) { \
  residual_then_body<TYPE, true>(x, residual, rw, scale, nw, residual_output, \
                                 normalized_output, rows, width, eps1, eps2); \
}

DKR_RESIDUAL(f32, float, 0)
DKR_RESIDUAL(f16, __half, 0)
DKR_RESIDUAL(bf16, __nv_bfloat16, 0)
