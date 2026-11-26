#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#ifdef DKR_GDN_NORM_IMAGE
#define GDN_NORM_FN __device__ __forceinline__
#else
#define GDN_NORM_FN __global__
#endif
constexpr int GDN_RMSNORM_FAST_HIDDEN = 128;
constexpr int GDN_RMSNORM_ROWS_PER_BLOCK = 8;
constexpr int GDN_RMSNORM_TILED_LANES_PER_ROW = 16;
constexpr int GDN_RMSNORM_TILED_ROW_PAIR_OFFSET = 2;
constexpr int GDN_RMSNORM_TILED_ROWS_PER_BLOCK = 4;
constexpr int GDN_RMSNORM_TILED_MIN_ROWS = 1024;
constexpr int GDN_RMSNORM_TILED_VALUES_PER_LANE = 8;

__device__ __forceinline__ float gdn_silu(float x) {
  if (isnan(x)) {
    return x;
  }
  if (isinf(x)) {
    return x > 0.0f ? x : 0.0f;
  }
  if (x >= 0.0f) {
    return x / (1.0f + expf(-x));
  }
  const float ex = expf(x);
  return x * ex / (1.0f + ex);
}

__device__ __forceinline__ float gdn_gate_act(float x, int sigmoid_gate) {
  return sigmoid_gate ? 1.0f / (1.0f + expf(-x)) : gdn_silu(x);
}

template <typename T>
GDN_NORM_FN void
gdn_rmsnorm_gated_kernel(const T *__restrict__ x, const T *__restrict__ gate,
                         const T *__restrict__ weight, T *__restrict__ output,
                         int rows, int hidden_dim, int outer_dim_1,
                         int outer_dim_2, int64_t x_stride_0,
                         int64_t x_stride_1, int64_t x_stride_2,
                         int64_t x_stride_3, int64_t gate_stride_0,
                         int64_t gate_stride_1, int64_t gate_stride_2,
                         int64_t gate_stride_3, float eps, int sigmoid_gate) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const size_t x_row_offset = (size_t)outer_0 * x_stride_0 +
                              (size_t)outer_1 * x_stride_1 +
                              (size_t)outer_2 * x_stride_2;
  const size_t gate_row_offset = (size_t)outer_0 * gate_stride_0 +
                                 (size_t)outer_1 * gate_stride_1 +
                                 (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * hidden_dim;

  float sum = 0.0f;
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    float x_val = (float)x[x_row_offset + (size_t)i * x_stride_3];
    sum = __fmaf_rn(x_val, x_val, sum);
  }

  __shared__ float smem[256];
  smem[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  const float inv_rms = rsqrtf(smem[0] / (float)hidden_dim + eps);
  for (int i = tid; i < hidden_dim; i += blockDim.x) {
    const float gate_val =
        (float)gate[gate_row_offset + (size_t)i * gate_stride_3];
    const float out =
        (float)x[x_row_offset + (size_t)i * x_stride_3] * inv_rms *
        (float)weight[i] * gdn_gate_act(gate_val, sigmoid_gate);
    out_row[i] = (T)out;
  }
}

template <typename T, int HIDDEN_DIM, int ROWS_PER_BLOCK>
GDN_NORM_FN void gdn_rmsnorm_gated_warp_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    int outer_dim_1, int outer_dim_2, int64_t x_stride_0,
    int64_t x_stride_1, int64_t x_stride_2, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, float eps, int sigmoid_gate) {
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int row = blockIdx.x * ROWS_PER_BLOCK + warp;
  if (row >= rows) {
    return;
  }

  const int outer_plane = outer_dim_1 * outer_dim_2;
  const int outer_0 = row / outer_plane;
  const int outer_1 = (row / outer_dim_2) % outer_dim_1;
  const int outer_2 = row % outer_dim_2;
  const T *x_row = x + (size_t)outer_0 * x_stride_0 +
                   (size_t)outer_1 * x_stride_1 +
                   (size_t)outer_2 * x_stride_2;
  const T *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                      (size_t)outer_1 * gate_stride_1 +
                      (size_t)outer_2 * gate_stride_2;
  T *out_row = output + (size_t)row * HIDDEN_DIM;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[i];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = (T)(values[i / 32] * inv_rms * (float)weight[i] *
                     gdn_gate_act((float)gate_row[i], sigmoid_gate));
  }
}

template <typename T> struct alignas(16) gdn_rmsnorm_vec8 {
  T data[GDN_RMSNORM_TILED_VALUES_PER_LANE];
};

template <typename T>
GDN_NORM_FN void gdn_rmsnorm_gated_rows4_kernel(
    const T *__restrict__ x, const T *__restrict__ gate,
    const T *__restrict__ weight, T *__restrict__ output, int rows,
    float eps, int sigmoid_gate) {
  using Vec = gdn_rmsnorm_vec8<T>;
  constexpr int vecs_per_row =
      GDN_RMSNORM_FAST_HIDDEN / GDN_RMSNORM_TILED_VALUES_PER_LANE;
  constexpr unsigned warp_mask = 0xffffffffu;

  const int lane = threadIdx.x;
  const int half_warp = lane / GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int lane_in_half = lane % GDN_RMSNORM_TILED_LANES_PER_ROW;
  const int first_row =
      blockIdx.x * GDN_RMSNORM_TILED_ROWS_PER_BLOCK + half_warp;
  const int second_row = first_row + GDN_RMSNORM_TILED_ROW_PAIR_OFFSET;
  const Vec weight_value =
      reinterpret_cast<const Vec *>(weight)[lane_in_half];

  Vec first_value{};
  Vec second_value{};
  if (first_row < rows) {
    first_value = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                   lane_in_half];
  }
  if (second_row < rows) {
    second_value = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                    lane_in_half];
  }

  float first_sum = 0.0f;
  float second_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)first_value.data[i];
    const float second = (float)second_value.data[i];
    first_sum = __fmaf_rn(first, first, first_sum);
    second_sum = __fmaf_rn(second, second, second_sum);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    first_sum += __shfl_down_sync(warp_mask, first_sum, offset,
                                  GDN_RMSNORM_TILED_LANES_PER_ROW);
    second_sum += __shfl_down_sync(warp_mask, second_sum, offset,
                                   GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float first_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, first_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);
  const float second_inv_rms =
      rsqrtf(__shfl_sync(warp_mask, second_sum, 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                 (float)GDN_RMSNORM_FAST_HIDDEN +
             eps);

  if (first_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[first_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)first_value.data[i] * first_inv_rms *
              (float)weight_value.data[i] *
              gdn_gate_act((float)gate_value.data[i], sigmoid_gate));
    }
    reinterpret_cast<Vec *>(output)[first_row * vecs_per_row + lane_in_half] =
        result;
  }
  if (second_row < rows) {
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[second_row * vecs_per_row + lane_in_half];
    Vec result;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      result.data[i] =
          (T)((float)second_value.data[i] * second_inv_rms *
              (float)weight_value.data[i] *
              gdn_gate_act((float)gate_value.data[i], sigmoid_gate));
    }
    reinterpret_cast<Vec *>(
        output)[second_row * vecs_per_row + lane_in_half] = result;
  }
}


#undef GDN_NORM_FN
