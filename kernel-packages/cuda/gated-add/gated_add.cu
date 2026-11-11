#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

extern "C" __device__ float __nv_expf(float);

__device__ __forceinline__ float rounded_add_f32(float a, float b) {
  float result;
  asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float rounded_mul_f32(float a, float b) {
  float result;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float rounded_div_f32(float a, float b) {
  float result;
  asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

__device__ __forceinline__ float rounded_float(double value) {
  float result;
  asm("cvt.rn.f32.f64 %0, %1;" : "=f"(result) : "d"(value));
  return result;
}

template <typename T> __device__ __forceinline__ T rounded_sigmoid(T value);

template <>
__device__ __forceinline__ __nv_bfloat16 rounded_sigmoid(__nv_bfloat16 value) {
  return __float2bfloat16_rn(rounded_div_f32(
      1.0f, rounded_add_f32(1.0f, __nv_expf(-__bfloat162float(value)))));
}

template <>
__device__ __forceinline__ __half rounded_sigmoid(__half value) {
  return __hdiv(__float2half_rn(1.0f),
                __hadd_rn(__float2half_rn(1.0f), hexp(__hneg(value))));
}

template <>
__device__ __forceinline__ float rounded_sigmoid(float value) {
  return rounded_float(__ddiv_rn(
      1.0, static_cast<double>(rounded_add_f32(1.0f, __nv_expf(-value)))));
}

template <typename T> __device__ __forceinline__ T rounded_mul(T a, T b) {
  return __hmul_rn(a, b);
}

template <typename T> __device__ __forceinline__ T rounded_add(T a, T b) {
  return __hadd_rn(a, b);
}

template <>
__device__ __forceinline__ float rounded_mul(float a, float b) {
  return rounded_mul_f32(a, b);
}

template <>
__device__ __forceinline__ float rounded_add(float a, float b) {
  return rounded_add_f32(a, b);
}

template <typename T>
__device__ __forceinline__ void gated_add_flat(
    const T *residual, const T *value, const T *gate, T *output,
    uint64_t elements, uint64_t columns) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const T factor = rounded_sigmoid(gate[index / columns]);
  output[index] = rounded_add(residual[index], rounded_mul(value[index], factor));
}

__device__ __forceinline__ void gated_add_rows_f32(
    const float *residual, const float *value, const float *gate,
    float *output, uint64_t columns) {
  const uint64_t tiles = (columns + blockDim.x - 1) / blockDim.x;
  const uint64_t row = blockIdx.x / tiles;
  const uint64_t column = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
  __shared__ float factor;
  if (threadIdx.x == 0) factor = rounded_sigmoid(gate[row]);
  __syncthreads();
  if (column < columns) {
    const uint64_t index = row * columns + column;
    output[index] = rounded_add(residual[index], rounded_mul(value[index], factor));
  }
}

extern "C" __global__ void dkr_gated_add_f16(
    const __half *residual, const __half *value, const __half *gate,
    __half *output, uint64_t elements, uint64_t columns) {
  gated_add_flat(residual, value, gate, output, elements, columns);
}

extern "C" __global__ void dkr_gated_add_bf16(
    const __nv_bfloat16 *residual, const __nv_bfloat16 *value,
    const __nv_bfloat16 *gate, __nv_bfloat16 *output,
    uint64_t elements, uint64_t columns) {
  gated_add_flat(residual, value, gate, output, elements, columns);
}

extern "C" __global__ void dkr_gated_add_f32_flat(
    const float *residual, const float *value, const float *gate,
    float *output, uint64_t elements, uint64_t columns) {
  gated_add_flat(residual, value, gate, output, elements, columns);
}

extern "C" __global__ void dkr_gated_add_f32_rows(
    const float *residual, const float *value, const float *gate,
    float *output, uint64_t elements, uint64_t columns) {
  (void)elements;
  gated_add_rows_f32(residual, value, gate, output, columns);
}
