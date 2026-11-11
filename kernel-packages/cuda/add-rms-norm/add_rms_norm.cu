#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// This module is the external form of the two launch bodies in
// mistralrs-core/src/cuda/sort.cu. Keep the FP32 accumulation, 16-bit
// intermediate rounding, warp reduction and vector lane order identical.
template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(
    __nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(
    float value) {
  return __float2bfloat16(value);
}

template <typename T> struct alignas(16) Vec8 {
  T data[8];
};

__device__ __forceinline__ float block_sum(float value, float *warp_sums) {
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  const int num_warps = (blockDim.x + 31) >> 5;
  value = threadIdx.x < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  if (threadIdx.x == 0) {
    warp_sums[0] = value;
  }
  __syncthreads();
  return warp_sums[0];
}

template <typename T>
__device__ __forceinline__ void scalar_body(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const uint32_t ncols, const float eps,
    float *reduce) {
  const int tid = threadIdx.x;
  const uint64_t row_offset = uint64_t(blockIdx.x) * ncols;
  float sum = 0.0f;
  for (uint32_t col = tid; col < ncols; col += blockDim.x) {
    const uint64_t idx = row_offset + col;
    const T value = from_float<T>(to_float(x[idx]) + to_float(residual[idx]));
    residual_dst[idx] = value;
    const float rounded = to_float(value);
    sum += rounded * rounded;
  }
  const float inv_rms =
      rsqrtf(block_sum(sum, reduce) / static_cast<float>(ncols) + eps);
  for (uint32_t col = tid; col < ncols; col += blockDim.x) {
    const uint64_t idx = row_offset + col;
    norm_dst[idx] = from_float<T>(
        to_float(residual_dst[idx]) * inv_rms * to_float(weight[col]));
  }
}

template <typename T>
__device__ __forceinline__ void vector_body(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const uint32_t ncols, const float eps,
    float *reduce) {
  using Vec = Vec8<T>;
  const int tid = threadIdx.x;
  const uint32_t vec_cols = ncols / 8;
  const uint64_t row_offset = uint64_t(blockIdx.x) * vec_cols;
  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ weight_vec = reinterpret_cast<const Vec *>(weight);
  Vec *__restrict__ residual_dst_vec = reinterpret_cast<Vec *>(residual_dst);
  Vec *__restrict__ norm_dst_vec = reinterpret_cast<Vec *>(norm_dst);

  float sum = 0.0f;
  for (uint32_t col = tid; col < vec_cols; col += blockDim.x) {
    const uint64_t idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out.data[i] = from_float<T>(
          to_float(x_value.data[i]) + to_float(residual_value.data[i]));
      const float value = to_float(out.data[i]);
      sum += value * value;
    }
    residual_dst_vec[idx] = out;
  }
  const float inv_rms =
      rsqrtf(block_sum(sum, reduce) / static_cast<float>(ncols) + eps);
  for (uint32_t col = tid; col < vec_cols; col += blockDim.x) {
    const uint64_t idx = row_offset + col;
    const Vec residual_value = residual_dst_vec[idx];
    const Vec weight_value = weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out.data[i] = from_float<T>(to_float(residual_value.data[i]) * inv_rms *
                                  to_float(weight_value.data[i]));
    }
    norm_dst_vec[idx] = out;
  }
}

#define DKR_ADD_RMS_SCALAR(NAME, TYPE)                                     \
  extern "C" __global__ void NAME(                                        \
      const TYPE *x, const TYPE *residual, const TYPE *weight,              \
      TYPE *residual_dst, TYPE *norm_dst, uint32_t ncols, float eps) {     \
    __shared__ float reduce[32];                                          \
    scalar_body<TYPE>(x, residual, weight, residual_dst, norm_dst, ncols,    \
                      eps, reduce);                                        \
  }

#define DKR_ADD_RMS_VEC8(NAME, TYPE)                                       \
  extern "C" __global__ void NAME(                                        \
      const TYPE *x, const TYPE *residual, const TYPE *weight,              \
      TYPE *residual_dst, TYPE *norm_dst, uint32_t ncols, float eps) {     \
    __shared__ float reduce[32];                                          \
    vector_body<TYPE>(x, residual, weight, residual_dst, norm_dst, ncols,    \
                      eps, reduce);                                        \
  }

DKR_ADD_RMS_SCALAR(dkr_add_rms_norm_f16_scalar, __half)
DKR_ADD_RMS_SCALAR(dkr_add_rms_norm_bf16_scalar, __nv_bfloat16)
DKR_ADD_RMS_SCALAR(dkr_add_rms_norm_f32_scalar, float)
DKR_ADD_RMS_VEC8(dkr_add_rms_norm_f16_vec8, __half)
DKR_ADD_RMS_VEC8(dkr_add_rms_norm_bf16_vec8, __nv_bfloat16)
