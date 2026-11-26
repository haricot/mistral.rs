#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "radix_topk.cuh"
#include <algorithm>
#include <cub/block/block_radix_sort.cuh>
#include <limits>
#include <stdint.h>
#include <type_traits>

__global__ void copy_f32_kernel(const float *__restrict__ x,
                                float *__restrict__ dst, const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = x[idx];
}

__global__ void apply_sparse_penalties_f32_kernel(
    float *__restrict__ logits, const uint32_t *__restrict__ token_ids,
    const float *__restrict__ counts, const int n, const int n_tokens,
    const float frequency_penalty, const float presence_penalty,
    const float repetition_penalty) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_tokens) {
    return;
  }

  const uint32_t token_id = token_ids[idx];
  if (token_id >= static_cast<uint32_t>(n)) {
    return;
  }

  const float count = counts[idx];
  if (count <= 0.0f) {
    return;
  }

  float value = logits[token_id];
  value -= count * frequency_penalty + presence_penalty;

  if (repetition_penalty != 1.0f) {
    value =
        value > 0.0f ? value / repetition_penalty : value * repetition_penalty;
  }

  logits[token_id] = value;
}

__global__ void apply_sparse_logits_bias_f32_kernel(
    float *__restrict__ logits, const uint32_t *__restrict__ token_ids,
    const float *__restrict__ biases, const int n, const int n_tokens) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_tokens) {
    return;
  }

  const uint32_t token_id = token_ids[idx];
  if (token_id >= static_cast<uint32_t>(n)) {
    return;
  }

  logits[token_id] += biases[idx];
}

extern "C" void
apply_sparse_penalties_f32(const void *x, void *dst, const uint32_t *token_ids,
                           const float *counts, const int n, const int n_tokens,
                           const float frequency_penalty,
                           const float presence_penalty,
                           const float repetition_penalty, int64_t stream) {
  if (n <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = 256;
  const int copy_grid = (n + block - 1) / block;
  copy_f32_kernel<<<copy_grid, block, 0, custream>>>(
      reinterpret_cast<const float *>(x), reinterpret_cast<float *>(dst), n);

  if (n_tokens <= 0) {
    return;
  }

  const int penalty_grid = (n_tokens + block - 1) / block;
  apply_sparse_penalties_f32_kernel<<<penalty_grid, block, 0, custream>>>(
      reinterpret_cast<float *>(dst), token_ids, counts, n, n_tokens,
      frequency_penalty, presence_penalty, repetition_penalty);
}

extern "C" void apply_sparse_logits_bias_f32(
    const void *x, void *dst, const uint32_t *token_ids, const float *biases,
    const int n, const int n_tokens, int64_t stream) {
  if (n <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = 256;
  const int copy_grid = (n + block - 1) / block;
  copy_f32_kernel<<<copy_grid, block, 0, custream>>>(
      reinterpret_cast<const float *>(x), reinterpret_cast<float *>(dst), n);

  if (n_tokens <= 0) {
    return;
  }

  const int bias_grid = (n_tokens + block - 1) / block;
  apply_sparse_logits_bias_f32_kernel<<<bias_grid, block, 0, custream>>>(
      reinterpret_cast<float *>(dst), token_ids, biases, n, n_tokens);
}

#ifndef MISTRALRS_DKR_CAUSAL_MASK
__global__ void apply_causal_mask_f32_kernel(float *__restrict__ scores,
                                             const int batch_heads,
                                             const int q_len,
                                             const int kv_len,
                                             const int q_offset,
                                             const int prefix_len) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_heads * q_len * kv_len;
  if (idx >= total) {
    return;
  }

  const int kv_idx = idx % kv_len;
  const int q_idx = (idx / kv_len) % q_len;
  const int q_pos = prefix_len + q_offset + q_idx;
  if (kv_idx > q_pos) {
    scores[idx] = -std::numeric_limits<float>::infinity();
  }
}

extern "C" void apply_causal_mask_f32(void *scores, const int batch_heads,
                                      const int q_len, const int kv_len,
                                      const int q_offset,
                                      const int prefix_len, int64_t stream) {
  if (batch_heads <= 0 || q_len <= 0 || kv_len <= 0) {
    return;
  }
  const cudaStream_t custream = (cudaStream_t)stream;
  const int block = 256;
  const int total = batch_heads * q_len * kv_len;
  const int grid = (total + block - 1) / block;
  apply_causal_mask_f32_kernel<<<grid, block, 0, custream>>>(
      reinterpret_cast<float *>(scores), batch_heads, q_len, kv_len, q_offset,
      prefix_len);
}
#endif  // MISTRALRS_DKR_CAUSAL_MASK

template <typename T>
__device__ __forceinline__ float rms_residual_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float rms_residual_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float
rms_residual_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T rms_residual_from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ __half rms_residual_from_float<__half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16
rms_residual_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <typename T> struct alignas(16) rms_vec8 {
  T data[8];
};

template <typename T>
__device__ __forceinline__ float rms_vec8_sum_squares(const rms_vec8<T> &vec) {
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float value = rms_residual_to_float(vec.data[i]);
    sum += value * value;
  }
  return sum;
}

template <typename T>
__host__ __forceinline__ bool rms_vec8_aligned(const void *ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(rms_vec8<T>) == 0;
}

template <typename T>
__host__ __forceinline__ bool rms_vec8_supported(const void *a, const void *b,
                                                 const void *c, const void *d,
                                                 const int ncols) {
  return ncols % 8 == 0 && rms_vec8_aligned<T>(a) && rms_vec8_aligned<T>(b) &&
         rms_vec8_aligned<T>(c) && rms_vec8_aligned<T>(d);
}

__host__ __forceinline__ int rms_vec8_block_size(const int vec_cols) {
  const int rounded = ((vec_cols + 31) / 32) * 32;
  return std::min(std::max(rounded, 32), 1024);
}

__device__ __forceinline__ float rms_block_sum(float value,
                                               float *warp_sums) {
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

#ifndef MISTRALRS_DKR_RMS_VARIANTS
template <typename T>
__global__ void rms_norm_residual_vec8_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, const T *__restrict__ scale,
    T *__restrict__ dst, const int ncols, const float eps) {
  using Vec = rms_vec8<T>;
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_cols = ncols / 8;
  const int row_offset = row * vec_cols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ weight_vec = reinterpret_cast<const Vec *>(weight);
  Vec *__restrict__ dst_vec = reinterpret_cast<Vec *>(dst);

  float sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    sum += rms_vec8_sum_squares(x_vec[row_offset + col]);
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);

  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    const Vec weight_value = weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float normed = rms_residual_to_float(x_value.data[i]) * inv_rms *
                           rms_residual_to_float(weight_value.data[i]);
      const float value =
          (rms_residual_to_float(residual_value.data[i]) + normed) *
          scale_value;
      out.data[i] = rms_residual_from_float<T>(value);
    }
    dst_vec[idx] = out;
  }
}

template <typename T>
__global__ void rms_norm_residual_kernel(const T *__restrict__ x,
                                         const T *__restrict__ residual,
                                         const T *__restrict__ weight,
                                         const T *__restrict__ scale,
                                         T *__restrict__ dst, const int ncols,
                                         const float eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * ncols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float value = rms_residual_to_float(x[row_offset + col]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float normed = rms_residual_to_float(x[row_offset + col]) * inv_rms *
                         rms_residual_to_float(weight[col]);
    const float value =
        (rms_residual_to_float(residual[row_offset + col]) + normed) *
        scale_value;
    dst[row_offset + col] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_residual(const void *x, const void *residual,
                              const void *weight, const void *scale, void *dst,
                              const int nrows, const int ncols, const float eps,
                              int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    if (rms_vec8_supported<T>(x, residual, weight, dst, ncols)) {
      const int block = rms_vec8_block_size(ncols / 8);
      rms_norm_residual_vec8_kernel<T><<<nrows, block, 0, custream>>>(
          reinterpret_cast<const T *>(x),
          reinterpret_cast<const T *>(residual),
          reinterpret_cast<const T *>(weight),
          reinterpret_cast<const T *>(scale), reinterpret_cast<T *>(dst),
          ncols, eps);
      return;
    }
  }

  const int block = ncols < 1024 ? 32 : 1024;
  rms_norm_residual_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(weight), reinterpret_cast<const T *>(scale),
      reinterpret_cast<T *>(dst), ncols, eps);
}

#endif  // MISTRALRS_DKR_RMS_VARIANTS

#ifndef MISTRALRS_DKR_ADD_RMS_NORM
template <typename T>
__global__ void add_rms_norm_vec8_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float eps) {
  using Vec = rms_vec8<T>;
  __shared__ float reduce[32];
  const int tid = threadIdx.x;
  const int vec_cols = ncols / 8;
  const int row_offset = blockIdx.x * vec_cols;

  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ weight_vec = reinterpret_cast<const Vec *>(weight);
  Vec *__restrict__ residual_dst_vec = reinterpret_cast<Vec *>(residual_dst);
  Vec *__restrict__ norm_dst_vec = reinterpret_cast<Vec *>(norm_dst);

  float sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out.data[i] = rms_residual_from_float<T>(
          rms_residual_to_float(x_value.data[i]) +
          rms_residual_to_float(residual_value.data[i]));
      const float value = rms_residual_to_float(out.data[i]);
      sum += value * value;
    }
    residual_dst_vec[idx] = out;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);

  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec residual_value = residual_dst_vec[idx];
    const Vec weight_value = weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      out.data[i] = rms_residual_from_float<T>(
          rms_residual_to_float(residual_value.data[i]) * inv_rms *
          rms_residual_to_float(weight_value.data[i]));
    }
    norm_dst_vec[idx] = out;
  }
}

template <typename T>
__global__ void add_rms_norm_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float eps) {
  __shared__ float reduce[32];
  const int tid = threadIdx.x;
  const int row_offset = blockIdx.x * ncols;

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    const T value = rms_residual_from_float<T>(
        rms_residual_to_float(x[idx]) + rms_residual_to_float(residual[idx]));
    residual_dst[idx] = value;
    const float rounded = rms_residual_to_float(value);
    sum += rounded * rounded;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) + eps);
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    norm_dst[idx] = rms_residual_from_float<T>(
        rms_residual_to_float(residual_dst[idx]) * inv_rms *
        rms_residual_to_float(weight[col]));
  }
}

template <typename T>
void launch_add_rms_norm(const void *x, const void *residual,
                         const void *weight, void *residual_dst,
                         void *norm_dst, const int nrows, const int ncols,
                         const float eps, int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    if (rms_vec8_supported<T>(x, residual, weight, residual_dst, ncols) &&
        rms_vec8_aligned<T>(norm_dst)) {
      const int block = rms_vec8_block_size(ncols / 8);
      add_rms_norm_vec8_kernel<T><<<nrows, block, 0, custream>>>(
          reinterpret_cast<const T *>(x),
          reinterpret_cast<const T *>(residual),
          reinterpret_cast<const T *>(weight),
          reinterpret_cast<T *>(residual_dst), reinterpret_cast<T *>(norm_dst),
          ncols, eps);
      return;
    }
  }

  const int block = ncols < 1024 ? 32 : 1024;
  add_rms_norm_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(weight),
      reinterpret_cast<T *>(residual_dst), reinterpret_cast<T *>(norm_dst),
      ncols, eps);
}
#endif  // MISTRALRS_DKR_ADD_RMS_NORM

#ifndef MISTRALRS_DKR_RMS_VARIANTS
template <typename T>
__global__ void rms_norm_residual_then_rms_norm_vec8_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ residual_weight, const T *__restrict__ scale,
    const T *__restrict__ norm_weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float residual_eps,
    const float norm_eps) {
  using Vec = rms_vec8<T>;
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vec_cols = ncols / 8;
  const int row_offset = row * vec_cols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  const Vec *__restrict__ x_vec = reinterpret_cast<const Vec *>(x);
  const Vec *__restrict__ residual_vec = reinterpret_cast<const Vec *>(residual);
  const Vec *__restrict__ residual_weight_vec =
      reinterpret_cast<const Vec *>(residual_weight);
  const Vec *__restrict__ norm_weight_vec =
      reinterpret_cast<const Vec *>(norm_weight);
  Vec *__restrict__ residual_dst_vec = reinterpret_cast<Vec *>(residual_dst);
  Vec *__restrict__ norm_dst_vec = reinterpret_cast<Vec *>(norm_dst);

  float sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    sum += rms_vec8_sum_squares(x_vec[row_offset + col]);
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) +
             residual_eps);

  float residual_sum = 0.0f;
  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec x_value = x_vec[idx];
    const Vec residual_value = residual_vec[idx];
    const Vec residual_weight_value = residual_weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float normed =
          rms_residual_to_float(x_value.data[i]) * inv_rms *
          rms_residual_to_float(residual_weight_value.data[i]);
      const float value =
          (rms_residual_to_float(residual_value.data[i]) + normed) *
          scale_value;
      out.data[i] = rms_residual_from_float<T>(value);
      residual_sum += value * value;
    }
    residual_dst_vec[idx] = out;
  }
  const float norm_inv_rms =
      rsqrtf(rms_block_sum(residual_sum, reduce) / static_cast<float>(ncols) +
             norm_eps);

  for (int col = tid; col < vec_cols; col += blockDim.x) {
    const int idx = row_offset + col;
    const Vec residual_value = residual_dst_vec[idx];
    const Vec norm_weight_value = norm_weight_vec[col];
    Vec out;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float value = rms_residual_to_float(residual_value.data[i]) *
                          norm_inv_rms *
                          rms_residual_to_float(norm_weight_value.data[i]);
      out.data[i] = rms_residual_from_float<T>(value);
    }
    norm_dst_vec[idx] = out;
  }
}

template <typename T>
__global__ void rms_norm_residual_then_rms_norm_kernel(
    const T *__restrict__ x, const T *__restrict__ residual,
    const T *__restrict__ residual_weight, const T *__restrict__ scale,
    const T *__restrict__ norm_weight, T *__restrict__ residual_dst,
    T *__restrict__ norm_dst, const int ncols, const float residual_eps,
    const float norm_eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int row_offset = row * ncols;
  const float scale_value =
      scale == nullptr ? 1.0f : rms_residual_to_float(scale[0]);

  float sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const float value = rms_residual_to_float(x[row_offset + col]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(ncols) +
             residual_eps);
  float residual_sum = 0.0f;
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    const float normed = rms_residual_to_float(x[idx]) * inv_rms *
                         rms_residual_to_float(residual_weight[col]);
    const float value =
        (rms_residual_to_float(residual[idx]) + normed) * scale_value;
    residual_dst[idx] = rms_residual_from_float<T>(value);
    residual_sum += value * value;
  }
  const float norm_inv_rms =
      rsqrtf(rms_block_sum(residual_sum, reduce) / static_cast<float>(ncols) +
             norm_eps);
  for (int col = tid; col < ncols; col += blockDim.x) {
    const int idx = row_offset + col;
    const float value = rms_residual_to_float(residual_dst[idx]) *
                        norm_inv_rms * rms_residual_to_float(norm_weight[col]);
    norm_dst[idx] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_residual_then_rms_norm(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  if (nrows <= 0 || ncols <= 0) {
    return;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    if (rms_vec8_supported<T>(x, residual, residual_weight, residual_dst,
                              ncols) &&
        rms_vec8_aligned<T>(norm_weight) &&
        rms_vec8_aligned<T>(norm_dst)) {
      const int block = rms_vec8_block_size(ncols / 8);
      rms_norm_residual_then_rms_norm_vec8_kernel<T>
          <<<nrows, block, 0, custream>>>(
              reinterpret_cast<const T *>(x),
              reinterpret_cast<const T *>(residual),
              reinterpret_cast<const T *>(residual_weight),
              reinterpret_cast<const T *>(scale),
              reinterpret_cast<const T *>(norm_weight),
              reinterpret_cast<T *>(residual_dst),
              reinterpret_cast<T *>(norm_dst), ncols, residual_eps, norm_eps);
      return;
    }
  }

  const int block = ncols < 1024 ? 32 : 1024;
  rms_norm_residual_then_rms_norm_kernel<T><<<nrows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(residual),
      reinterpret_cast<const T *>(residual_weight),
      reinterpret_cast<const T *>(scale),
      reinterpret_cast<const T *>(norm_weight), reinterpret_cast<T *>(residual_dst),
      reinterpret_cast<T *>(norm_dst), ncols, residual_eps, norm_eps);
}

#endif  // MISTRALRS_DKR_RMS_VARIANTS

template <typename T>
__global__ void rms_norm_strided_4d_kernel(
    const T *__restrict__ x, const T *__restrict__ weight, T *__restrict__ dst,
    const int64_t stride_b, const int64_t stride_h, const int64_t stride_s,
    const int64_t stride_d, const int batch, const int heads,
    const int seq_len, const int head_dim, const float eps) {
  __shared__ float reduce[32];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int seq = row % seq_len;
  const int tmp = row / seq_len;
  const int head = tmp % heads;
  const int batch_idx = tmp / heads;
  const int64_t src_base = static_cast<int64_t>(batch_idx) * stride_b +
                           static_cast<int64_t>(head) * stride_h +
                           static_cast<int64_t>(seq) * stride_s;
  const int64_t dst_base = static_cast<int64_t>(row) * head_dim;

  float sum = 0.0f;
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = rms_residual_to_float(x[src_base + col * stride_d]);
    sum += value * value;
  }
  const float inv_rms =
      rsqrtf(rms_block_sum(sum, reduce) / static_cast<float>(head_dim) + eps);
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float value = rms_residual_to_float(x[src_base + col * stride_d]) *
                        inv_rms * rms_residual_to_float(weight[col]);
    dst[dst_base + col] = rms_residual_from_float<T>(value);
  }
}

template <typename T>
void launch_rms_norm_strided_4d(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  if (batch <= 0 || heads <= 0 || seq_len <= 0 || head_dim <= 0) {
    return;
  }

  const int total_rows = batch * heads * seq_len;
  int block = 32;
  while (block < head_dim && block < 1024) {
    block <<= 1;
  }

  const cudaStream_t custream = (cudaStream_t)stream;
  rms_norm_strided_4d_kernel<T><<<total_rows, block, 0, custream>>>(
      reinterpret_cast<const T *>(x), reinterpret_cast<const T *>(weight),
      reinterpret_cast<T *>(dst), stride_b, stride_h, stride_s, stride_d, batch,
      heads, seq_len, head_dim, eps);
}

#ifndef MISTRALRS_DKR_RMS_VARIANTS
extern "C" void rms_norm_residual_f32(const void *x, const void *residual,
                                      const void *weight, const void *scale,
                                      void *dst, const int nrows,
                                      const int ncols, const float eps,
                                      int64_t stream) {
  launch_rms_norm_residual<float>(x, residual, weight, scale, dst, nrows, ncols,
                                  eps, stream);
}

extern "C" void rms_norm_residual_f16(const void *x, const void *residual,
                                      const void *weight, const void *scale,
                                      void *dst, const int nrows,
                                      const int ncols, const float eps,
                                      int64_t stream) {
  launch_rms_norm_residual<__half>(x, residual, weight, scale, dst, nrows,
                                   ncols, eps, stream);
}

extern "C" void rms_norm_residual_bf16(const void *x, const void *residual,
                                       const void *weight, const void *scale,
                                       void *dst, const int nrows,
                                       const int ncols, const float eps,
                                       int64_t stream) {
  launch_rms_norm_residual<__nv_bfloat16>(x, residual, weight, scale, dst,
                                          nrows, ncols, eps, stream);
}

#endif  // MISTRALRS_DKR_RMS_VARIANTS

#ifndef MISTRALRS_DKR_ADD_RMS_NORM
extern "C" void add_rms_norm_f32(const void *x, const void *residual,
                                 const void *weight, void *residual_dst,
                                 void *norm_dst, const int nrows,
                                 const int ncols, const float eps,
                                 int64_t stream) {
  launch_add_rms_norm<float>(x, residual, weight, residual_dst, norm_dst,
                             nrows, ncols, eps, stream);
}

extern "C" void add_rms_norm_f16(const void *x, const void *residual,
                                 const void *weight, void *residual_dst,
                                 void *norm_dst, const int nrows,
                                 const int ncols, const float eps,
                                 int64_t stream) {
  launch_add_rms_norm<__half>(x, residual, weight, residual_dst, norm_dst,
                              nrows, ncols, eps, stream);
}

extern "C" void add_rms_norm_bf16(const void *x, const void *residual,
                                  const void *weight, void *residual_dst,
                                  void *norm_dst, const int nrows,
                                  const int ncols, const float eps,
                                  int64_t stream) {
  launch_add_rms_norm<__nv_bfloat16>(x, residual, weight, residual_dst,
                                     norm_dst, nrows, ncols, eps, stream);
}
#endif  // MISTRALRS_DKR_ADD_RMS_NORM

#ifndef MISTRALRS_DKR_RMS_VARIANTS
extern "C" void rms_norm_residual_then_rms_norm_f32(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<float>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

extern "C" void rms_norm_residual_then_rms_norm_f16(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<__half>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

extern "C" void rms_norm_residual_then_rms_norm_bf16(
    const void *x, const void *residual, const void *residual_weight,
    const void *scale, const void *norm_weight, void *residual_dst,
    void *norm_dst, const int nrows, const int ncols,
    const float residual_eps, const float norm_eps, int64_t stream) {
  launch_rms_norm_residual_then_rms_norm<__nv_bfloat16>(
      x, residual, residual_weight, scale, norm_weight, residual_dst, norm_dst,
      nrows, ncols, residual_eps, norm_eps, stream);
}

#endif  // MISTRALRS_DKR_RMS_VARIANTS

extern "C" void rms_norm_strided_4d_f32(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<float>(x, weight, dst, stride_b, stride_h,
                                    stride_s, stride_d, batch, heads, seq_len,
                                    head_dim, eps, stream);
}

extern "C" void rms_norm_strided_4d_f16(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<__half>(x, weight, dst, stride_b, stride_h,
                                     stride_s, stride_d, batch, heads, seq_len,
                                     head_dim, eps, stream);
}

extern "C" void rms_norm_strided_4d_bf16(
    const void *x, const void *weight, void *dst, const int64_t stride_b,
    const int64_t stride_h, const int64_t stride_s, const int64_t stride_d,
    const int batch, const int heads, const int seq_len, const int head_dim,
    const float eps, int64_t stream) {
  launch_rms_norm_strided_4d<__nv_bfloat16>(x, weight, dst, stride_b, stride_h,
                                            stride_s, stride_d, batch, heads,
                                            seq_len, head_dim, eps, stream);
}

template <typename T> inline __device__ void swap(T &a, T &b) {
  T tmp = a;
  a = b;
  b = tmp;
}

template <typename T, bool ascending>
__global__ void bitonic_sort_kernel(T *arr, uint32_t *dst, int j, int k) {
  unsigned int i, ij;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ij = i ^ j;

  if (ij > i) {
    if constexpr (ascending) {
      if ((i & k) == 0) {
        if (arr[i] > arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      } else {
        if (arr[i] < arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      }
    }

    if constexpr (!ascending) {
      if ((i & k) != 0) {
        if (arr[i] > arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      } else {
        if (arr[i] < arr[ij]) {
          swap(arr[i], arr[ij]);
          swap(dst[i], dst[ij]);
        }
      }
    }
  }
  __syncthreads();
}

int next_power_of_2(int x) {
  int n = 1;
  while (n < x) {
    n *= 2;
  }
  return n;
}

#define ASORT_OP(T, RUST_NAME, ASC)                                            \
  extern "C" void RUST_NAME(void *x1, void *dst1, const int nrows,             \
                            const int ncols, bool inplace, int64_t stream) {   \
    T *x = reinterpret_cast<T *>(x1);                                          \
    uint32_t *dst = reinterpret_cast<uint32_t *>(dst1);                        \
    const cudaStream_t custream = (cudaStream_t)stream;                        \
    int ncols_pad = next_power_of_2(ncols);                                    \
    T *x_row_padded;                                                           \
    uint32_t *dst_row_padded;                                                  \
    cudaMallocAsync((void **)&x_row_padded, ncols_pad * sizeof(T), custream);  \
    cudaMallocAsync((void **)&dst_row_padded, ncols_pad * sizeof(uint32_t),    \
                    custream);                                                 \
    uint32_t *indices_padded =                                                 \
        (uint32_t *)malloc(ncols_pad * sizeof(uint32_t));                      \
    for (int i = 0; i < ncols_pad; i++) {                                      \
      indices_padded[i] = i;                                                   \
    }                                                                          \
    T *values_padded = (T *)malloc((ncols_pad - ncols) * sizeof(T));           \
    for (int i = 0; i < ncols_pad - ncols; i++) {                              \
      values_padded[i] =                                                       \
          ASC ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min(); \
    }                                                                          \
    int max_threads_per_block = 1024;                                          \
    int threads_per_block =                                                    \
        max_threads_per_block > ncols_pad ? ncols_pad : max_threads_per_block; \
    int blocks_per_row =                                                       \
        (ncols_pad + threads_per_block - 1) / threads_per_block;               \
    for (int row = 0; row < nrows; row++) {                                    \
      T *x_row = x + row * ncols;                                              \
      uint32_t *dst_row = dst + row * ncols;                                   \
      cudaMemcpyAsync(x_row_padded, x_row, ncols * sizeof(T),                  \
                      cudaMemcpyDeviceToDevice, custream);                     \
      if (ncols_pad - ncols > 0)                                               \
        cudaMemcpyAsync(x_row_padded + ncols, values_padded,                   \
                        (ncols_pad - ncols) * sizeof(T),                       \
                        cudaMemcpyHostToDevice, custream);                     \
      cudaMemcpyAsync(dst_row_padded, indices_padded,                          \
                      ncols_pad * sizeof(uint32_t), cudaMemcpyHostToDevice,    \
                      custream);                                               \
      for (int k = 2; k <= ncols_pad; k <<= 1) {                               \
        for (int j = k >> 1; j > 0; j = j >> 1) {                              \
          bitonic_sort_kernel<T, ASC>                                          \
              <<<blocks_per_row, threads_per_block, 0, custream>>>(            \
                  x_row_padded, dst_row_padded, j, k);                         \
        }                                                                      \
      }                                                                        \
      if (inplace)                                                             \
        cudaMemcpyAsync(x_row, x_row_padded, ncols * sizeof(T),                \
                        cudaMemcpyDeviceToDevice, custream);                   \
      cudaMemcpyAsync(dst_row, dst_row_padded, ncols * sizeof(uint32_t),       \
                      cudaMemcpyDeviceToDevice, custream);                     \
    }                                                                          \
    cudaFreeAsync(x_row_padded, custream);                                     \
    cudaFreeAsync(dst_row_padded, custream);                                   \
    free(indices_padded);                                                      \
    free(values_padded);                                                       \
  }

ASORT_OP(__nv_bfloat16, asort_asc_bf16, true)
ASORT_OP(__nv_bfloat16, asort_desc_bf16, false)

ASORT_OP(__half, asort_asc_f16, true)
ASORT_OP(__half, asort_desc_f16, false)

ASORT_OP(float, asort_asc_f32, true)
ASORT_OP(double, asort_asc_f64, true)
ASORT_OP(uint8_t, asort_asc_u8, true)
ASORT_OP(uint32_t, asort_asc_u32, true)
ASORT_OP(int64_t, asort_asc_i64, true)

ASORT_OP(float, asort_desc_f32, false)
ASORT_OP(double, asort_desc_f64, false)
ASORT_OP(uint8_t, asort_desc_u8, false)
ASORT_OP(uint32_t, asort_desc_u32, false)
ASORT_OP(int64_t, asort_desc_i64, false)

// ============================================================================
// Optimized parallel topk kernel for small k (MoE routing)
//
// Much faster than full sort for small k:
// - Processes all rows in parallel (one block per row)
// - Uses simple "find max k times" algorithm: O(n*k) for small k
// - Single kernel launch for all rows
// ============================================================================

#include "router_topk_device.cuh"

// Wrapper for f32 - writes to separate values and indices buffers
extern "C" void topk_f32(const float *input,
                         float *values_out,     // [nrows, k]
                         uint32_t *indices_out, // [nrows, k]
                         int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  // One block per row
  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(float) + ncols * sizeof(bool);

  topk_kernel<float><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

// Wrapper for bf16 - writes to separate values and indices buffers
extern "C" void topk_bf16(const __nv_bfloat16 *input,
                          __nv_bfloat16 *values_out, // [nrows, k]
                          uint32_t *indices_out,     // [nrows, k]
                          int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(__nv_bfloat16) + ncols * sizeof(bool);

  topk_kernel<__nv_bfloat16><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

// Wrapper for f16 - writes to separate values and indices buffers
extern "C" void topk_f16(const __half *input,
                         __half *values_out,    // [nrows, k]
                         uint32_t *indices_out, // [nrows, k]
                         int nrows, int ncols, int k, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;

  int block_size = 256;
  if (ncols <= 64)
    block_size = 64;
  else if (ncols <= 128)
    block_size = 128;
  else if (ncols <= 256)
    block_size = 256;
  else
    block_size = 512;

  size_t smem_size = ncols * sizeof(__half) + ncols * sizeof(bool);

  topk_kernel<__half><<<nrows, block_size, smem_size, custream>>>(
      input, values_out, indices_out, nrows, ncols, k);
}

template <typename T, bool has_bias, bool has_expert_scale>
void launch_moe_router_topk(const T *logits, float *weights, uint32_t *ids,
                            const float *selection_bias,
                            const float *expert_scale, const int n_rows,
                            const int n_experts, const int top_k,
                            const int score_mode, const int weight_mode,
                            const bool renormalize, const bool clamp_logits,
                            const float clamp_min, const float clamp_max,
                            const float norm_min, const float output_scale,
                            int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  dim3 block_dims(32, MOE_ROUTER_ROWS_PER_BLOCK, 1);
  dim3 grid_dims((n_rows + MOE_ROUTER_ROWS_PER_BLOCK - 1) /
                     MOE_ROUTER_ROWS_PER_BLOCK,
                 1, 1);

#define LAUNCH_MOE_ROUTER_TOPK(EXPERTS)                                        \
  moe_router_topk_kernel<T, EXPERTS, has_bias, has_expert_scale>               \
      <<<grid_dims, block_dims, 0, custream>>>(                                \
          logits, weights, ids, selection_bias, expert_scale, n_rows, top_k,   \
          score_mode, weight_mode, renormalize, clamp_logits, clamp_min,       \
          clamp_max, norm_min, output_scale);                                  \
  break

  switch (n_experts) {
  case 1:
    LAUNCH_MOE_ROUTER_TOPK(1);
  case 2:
    LAUNCH_MOE_ROUTER_TOPK(2);
  case 4:
    LAUNCH_MOE_ROUTER_TOPK(4);
  case 8:
    LAUNCH_MOE_ROUTER_TOPK(8);
  case 16:
    LAUNCH_MOE_ROUTER_TOPK(16);
  case 32:
    LAUNCH_MOE_ROUTER_TOPK(32);
  case 64:
    LAUNCH_MOE_ROUTER_TOPK(64);
  case 128:
    LAUNCH_MOE_ROUTER_TOPK(128);
  case 256:
    LAUNCH_MOE_ROUTER_TOPK(256);
  case 512:
    LAUNCH_MOE_ROUTER_TOPK(512);
  case 576:
    LAUNCH_MOE_ROUTER_TOPK(576);
  default:
    break;
  }

#undef LAUNCH_MOE_ROUTER_TOPK
}

template <typename T>
void moe_router_topk_dispatch(const T *logits, float *weights, uint32_t *ids,
                              const float *selection_bias,
                              const float *expert_scale, const int n_rows,
                              const int n_experts, const int top_k,
                              const int score_mode, const int weight_mode,
                              const bool renormalize, const bool clamp_logits,
                              const float clamp_min, const float clamp_max,
                              const float norm_min, const float output_scale,
                              int64_t stream) {
  if (selection_bias != nullptr && expert_scale != nullptr) {
    launch_moe_router_topk<T, true, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (selection_bias != nullptr) {
    launch_moe_router_topk<T, true, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else if (expert_scale != nullptr) {
    launch_moe_router_topk<T, false, true>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  } else {
    launch_moe_router_topk<T, false, false>(
        logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
        top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
        clamp_max, norm_min, output_scale, stream);
  }
}

extern "C" void moe_router_topk_f32(const float *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<float>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void
moe_router_topk_bf16(const __nv_bfloat16 *logits, float *weights, uint32_t *ids,
                     const float *selection_bias, const float *expert_scale,
                     int n_rows, int n_experts, int top_k, int score_mode,
                     int weight_mode, bool renormalize, bool clamp_logits,
                     float clamp_min, float clamp_max, float norm_min,
                     float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__nv_bfloat16>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

extern "C" void moe_router_topk_f16(const __half *logits, float *weights,
                                    uint32_t *ids, const float *selection_bias,
                                    const float *expert_scale, int n_rows,
                                    int n_experts, int top_k, int score_mode,
                                    int weight_mode, bool renormalize,
                                    bool clamp_logits, float clamp_min,
                                    float clamp_max, float norm_min,
                                    float output_scale, int64_t stream) {
  moe_router_topk_dispatch<__half>(
      logits, weights, ids, selection_bias, expert_scale, n_rows, n_experts,
      top_k, score_mode, weight_mode, renormalize, clamp_logits, clamp_min,
      clamp_max, norm_min, output_scale, stream);
}

#include "large_topk_device.cuh"

#include "sampling_device.cuh"

extern "C" void sample_ranked_topk(
    const float *packed, const float *params, uint32_t *tokens,
    const int nrows, const int packed_k, int64_t stream) {
  if (nrows <= 0 || packed_k <= 0) {
    return;
  }
  const int block = RANKED_TOPK_SAMPLE_THREADS;
  const int grid = (nrows + block - 1) / block;
  sample_ranked_topk_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      packed, params, tokens, nrows, packed_k);
}

extern "C" void topk_large_f32(const float *input, float *block_values,
                               uint32_t *block_indices, float *block_maxes,
                               float *block_sums, float *values_out,
                               uint32_t *indices_out, float *softmax_info_out,
                               int ncols, int k, int chunk_size, int nblocks,
                               float inv_temperature, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);

  topk_large_stage1<float, false, true>
      <<<nblocks, block_size, stage1_smem, custream>>>(
      input, block_values, block_indices, block_maxes, block_sums, ncols, k,
      chunk_size, nullptr, inv_temperature);
  topk_large_stage2_f32<<<1, block_size, stage2_smem, custream>>>(
      block_values, block_indices, block_maxes, block_sums, values_out,
      indices_out, softmax_info_out, nblocks, k);
}

extern "C" void topk_large_f32_packed(const float *input, float *block_values,
                                      uint32_t *block_indices,
                                      float *block_maxes, float *block_sums,
                                      float *packed_out, int ncols, int k,
                                      int chunk_size, int nblocks,
                                      float inv_temperature, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);

  topk_large_stage1<float, false, true>
      <<<nblocks, block_size, stage1_smem, custream>>>(
      input, block_values, block_indices, block_maxes, block_sums, ncols, k,
      chunk_size, nullptr, inv_temperature);
  topk_large_stage2_f32_packed<false, true>
      <<<1, block_size, stage2_smem, custream>>>(
      block_values, block_indices, block_maxes, block_sums, packed_out, nblocks,
      k);
}

template <typename T>
void launch_topk_large_packed_batched(
    const T *input, const float *inv_temperatures, float *block_values,
    uint32_t *block_indices, float *block_maxes, float *block_sums,
    float *packed_out, int nrows, int ncols, int k, int chunk_size, int nblocks,
    int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);
  const dim3 stage1_grid(nblocks, nrows);

  topk_large_stage1<T, true, true>
      <<<stage1_grid, block_size, stage1_smem, custream>>>(
          input, block_values, block_indices, block_maxes, block_sums, ncols, k,
          chunk_size, inv_temperatures, 0.0f);
  topk_large_stage2_f32_packed<true, true>
      <<<nrows, block_size, stage2_smem, custream>>>(
          block_values, block_indices, block_maxes, block_sums, packed_out,
          nblocks, k);
}

extern "C" void
topk_large_f32_packed_batched(const float *input, const float *inv_temperatures,
                              float *block_values, uint32_t *block_indices,
                              float *block_maxes, float *block_sums,
                              float *packed_out, int nrows, int ncols, int k,
                              int chunk_size, int nblocks, int64_t stream) {
  launch_topk_large_packed_batched(
      input, inv_temperatures, block_values, block_indices, block_maxes,
      block_sums, packed_out, nrows, ncols, k, chunk_size, nblocks, stream);
}

extern "C" void topk_large_bf16_packed_batched(
    const __nv_bfloat16 *input, const float *inv_temperatures,
    float *block_values, uint32_t *block_indices, float *block_maxes,
    float *block_sums, float *packed_out, int nrows, int ncols, int k,
    int chunk_size, int nblocks, int64_t stream) {
  launch_topk_large_packed_batched(
      input, inv_temperatures, block_values, block_indices, block_maxes,
      block_sums, packed_out, nrows, ncols, k, chunk_size, nblocks, stream);
}

extern "C" void
topk_large_f16_packed_batched(const __half *input,
                              const float *inv_temperatures,
                              float *block_values, uint32_t *block_indices,
                              float *block_maxes, float *block_sums,
                              float *packed_out, int nrows, int ncols, int k,
                              int chunk_size, int nblocks, int64_t stream) {
  launch_topk_large_packed_batched(
      input, inv_temperatures, block_values, block_indices, block_maxes,
      block_sums, packed_out, nrows, ncols, k, chunk_size, nblocks, stream);
}

template <typename T>
cudaError_t launch_topk_large_ranked_packed_batched(
    const T *input, float *block_values, uint32_t *block_indices,
    float *packed_out, void *radix_workspace, int nrows, int ncols, int k,
    int chunk_size, int nblocks, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  bool radix_launched = false;
  cudaError_t status = mistralrs_radix_topk::launch(
      input, packed_out, radix_workspace, nrows, ncols, k, custream,
      &radix_launched);
  if (status != cudaSuccess || radix_launched) {
    return status;
  }
  const dim3 stage1_grid(nblocks, nrows);
  const bool use_radix =
      chunk_size == RANKED_TOPK_RADIX_CAPACITY &&
      k >= RANKED_TOPK_RADIX_MIN_K && k <= RANKED_TOPK_RADIX_MAX_K &&
      nblocks * k <= RANKED_TOPK_RADIX_CAPACITY;
  if (use_radix) {
    topk_large_ranked_stage1_radix<T>
        <<<stage1_grid, RANKED_TOPK_RADIX_THREADS, 0, custream>>>(
            input, block_values, block_indices, ncols, k);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      return status;
    }
    topk_large_ranked_stage2_radix
        <<<nrows, RANKED_TOPK_RADIX_THREADS, 0, custream>>>(
            block_values, block_indices, packed_out, nblocks, k);
    return cudaGetLastError();
  }

  constexpr int block_size = 256;
  const size_t stage1_smem = static_cast<size_t>(chunk_size) * sizeof(bool);
  const size_t stage2_smem =
      static_cast<size_t>(nblocks) * static_cast<size_t>(k) * sizeof(bool);

  topk_large_stage1<T, true, false>
      <<<stage1_grid, block_size, stage1_smem, custream>>>(
          input, block_values, block_indices, nullptr, nullptr, ncols, k,
          chunk_size, nullptr, 0.0f);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }
  topk_large_stage2_f32_packed<true, false>
      <<<nrows, block_size, stage2_smem, custream>>>(
          block_values, block_indices, nullptr, nullptr, packed_out, nblocks, k);
  return cudaGetLastError();
}

extern "C" size_t topk_large_ranked_state_words_per_row() {
  return mistralrs_radix_topk::workspace_words_per_row();
}

extern "C" int topk_large_ranked_f32_packed_batched(
    const float *input, float *block_values, uint32_t *block_indices,
    float *packed_out, void *radix_workspace, int nrows, int ncols, int k,
    int chunk_size, int nblocks, int64_t stream) {
  return static_cast<int>(launch_topk_large_ranked_packed_batched(
      input, block_values, block_indices, packed_out, radix_workspace, nrows, ncols, k,
      chunk_size, nblocks, stream));
}

extern "C" int topk_large_ranked_bf16_packed_batched(
    const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices,
    float *packed_out, void *radix_workspace, int nrows, int ncols, int k,
    int chunk_size, int nblocks, int64_t stream) {
  return static_cast<int>(launch_topk_large_ranked_packed_batched(
      input, block_values, block_indices, packed_out, radix_workspace, nrows, ncols, k,
      chunk_size, nblocks, stream));
}

extern "C" int topk_large_ranked_f16_packed_batched(
    const __half *input, float *block_values, uint32_t *block_indices,
    float *packed_out, void *radix_workspace, int nrows, int ncols, int k,
    int chunk_size, int nblocks, int64_t stream) {
  return static_cast<int>(launch_topk_large_ranked_packed_batched(
      input, block_values, block_indices, packed_out, radix_workspace, nrows, ncols, k,
      chunk_size, nblocks, stream));
}

template <typename T, bool BATCHED>
void launch_top1_large_packed(const T *input, float *block_values,
                              uint32_t *block_indices, float *packed_out,
                              uint32_t *token_ids_out, int nrows, int ncols,
                              int chunk_size, int nblocks, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const dim3 stage1_grid(nblocks, BATCHED ? nrows : 1);

  top1_large_stage1<T, BATCHED, false>
      <<<stage1_grid, block_size, 0, custream>>>(
      input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
  top1_large_stage2_f32_packed<BATCHED>
      <<<BATCHED ? nrows : 1, block_size, 0, custream>>>(
      block_values, block_indices, packed_out, token_ids_out, nblocks);
}

extern "C" void top1_large_f32_packed(const float *input, float *block_values,
                                      uint32_t *block_indices,
                                      float *packed_out,
                                      uint32_t *token_ids_out, int ncols,
                                      int chunk_size, int nblocks, int64_t stream) {
  launch_top1_large_packed<float, false>(
      input, block_values, block_indices, packed_out, token_ids_out, 1, ncols,
      chunk_size, nblocks, stream);
}

extern "C" void top1_large_bf16_packed(
    const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices,
    float *packed_out, uint32_t *token_ids_out, int ncols, int chunk_size,
    int nblocks, int64_t stream) {
  launch_top1_large_packed<__nv_bfloat16, false>(
      input, block_values, block_indices, packed_out, token_ids_out, 1, ncols,
      chunk_size, nblocks, stream);
}

extern "C" void top1_large_f16_packed(
    const __half *input, float *block_values, uint32_t *block_indices,
    float *packed_out, uint32_t *token_ids_out, int ncols, int chunk_size,
    int nblocks, int64_t stream) {
  launch_top1_large_packed<__half, false>(
      input, block_values, block_indices, packed_out, token_ids_out, 1, ncols,
      chunk_size, nblocks, stream);
}

extern "C" void top1_large_f32_packed_batched(const float *input,
                                              float *block_values,
                                              uint32_t *block_indices,
                                              float *packed_out,
                                              uint32_t *token_ids_out,
                                              int nrows, int ncols,
                                              int chunk_size, int nblocks,
                                              int64_t stream) {
  launch_top1_large_packed<float, true>(
      input, block_values, block_indices, packed_out, token_ids_out, nrows,
      ncols, chunk_size, nblocks, stream);
}

extern "C" void top1_large_bf16_packed_batched(
    const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices,
    float *packed_out, uint32_t *token_ids_out, int nrows, int ncols,
    int chunk_size, int nblocks, int64_t stream) {
  launch_top1_large_packed<__nv_bfloat16, true>(
      input, block_values, block_indices, packed_out, token_ids_out, nrows,
      ncols, chunk_size, nblocks, stream);
}

extern "C" void top1_large_f16_packed_batched(
    const __half *input, float *block_values, uint32_t *block_indices,
    float *packed_out, uint32_t *token_ids_out, int nrows, int ncols,
    int chunk_size, int nblocks, int64_t stream) {
  launch_top1_large_packed<__half, true>(
      input, block_values, block_indices, packed_out, token_ids_out, nrows,
      ncols, chunk_size, nblocks, stream);
}

extern "C" void categorical_large_f32_packed_batched(
    const float *input, const float *inv_temperatures, const float *uniforms,
    float *block_values, float *block_sums, float *packed_out, int nrows,
    int ncols, int chunk_size, int nblocks, int64_t stream) {
  const cudaStream_t custream = (cudaStream_t)stream;
  constexpr int block_size = 256;
  const dim3 stage1_grid(nblocks, nrows);
  const size_t stage2_smem =
      static_cast<size_t>(chunk_size) * 2 * sizeof(float);

  top1_large_stage1<float, true, true>
      <<<stage1_grid, block_size, 0, custream>>>(
      input, block_values, nullptr, block_sums, ncols, chunk_size,
      inv_temperatures);
  categorical_large_stage2_f32_packed<<<nrows, block_size, stage2_smem,
                                        custream>>>(
      input, inv_temperatures, uniforms, block_values, block_sums, packed_out,
      ncols, chunk_size, nblocks);
}
