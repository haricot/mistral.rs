#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include "gdn_norm_device.cuh"
#ifdef DKR_GDN_PREPARATION_IMAGE
#define GDN_PREP_ENTRY __device__ __forceinline__
#else
#define GDN_PREP_ENTRY __global__
#endif

template <typename T>
__device__ __forceinline__ T gdn_ragged_from_float(float value);

template <>
__device__ __forceinline__ float gdn_ragged_from_float<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ __nv_bfloat16
gdn_ragged_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
GDN_PREP_ENTRY void gdn_packed_to_padded_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, size_t total_elements,
    int padded_len, int width, int64_t source_token_stride,
    float padding_value) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const size_t token = index / width;
    const int feature = index % width;
    const int row = token / padded_len;
    const int position = token % padded_len;
    const uint32_t start = cu_seqlens[row];
    const uint32_t row_len = cu_seqlens[row + 1] - start;
    output[index] = position < row_len
                        ? source[(size_t)(start + position) *
                                     source_token_stride +
                                 feature]
                        : gdn_ragged_from_float<T>(padding_value);
  }
}

template <typename T>
GDN_PREP_ENTRY void gdn_padded_to_packed_kernel(
    const T *__restrict__ source, T *__restrict__ output,
    const uint32_t *__restrict__ cu_seqlens, int width,
    int64_t source_batch_stride, int64_t source_token_stride,
    int64_t source_feature_stride, int feature_inner_width) {
  const int row = blockIdx.x;
  const uint32_t start = cu_seqlens[row];
  const uint32_t row_len = cu_seqlens[row + 1] - start;
  const size_t row_elements = (size_t)row_len * width;
  for (size_t index = (size_t)blockIdx.y * blockDim.x + threadIdx.x;
       index < row_elements;
       index += (size_t)gridDim.y * blockDim.x) {
    const int position = index / width;
    const int feature = index % width;
    const int feature_outer = feature / feature_inner_width;
    const int feature_inner = feature % feature_inner_width;
    output[(size_t)(start + position) * width + feature] =
        source[(size_t)row * source_batch_stride +
               (size_t)position * source_token_stride +
               (size_t)feature_outer * source_feature_stride + feature_inner];
  }
}

template <typename T>
GDN_PREP_ENTRY void gdn_extract_ragged_conv_state_kernel(
    const T *__restrict__ padded_input, const T *__restrict__ initial_state,
    T *__restrict__ output, const uint32_t *__restrict__ cu_seqlens,
    size_t total_elements, int padded_len, int channels, int state_width,
    int64_t input_batch_stride, int64_t input_token_stride) {
  for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       index < total_elements;
       index += (size_t)gridDim.x * blockDim.x) {
    const int state_position = index % state_width;
    const size_t channel_row = index / state_width;
    const int channel = channel_row % channels;
    const int row = channel_row / channels;
    const int row_len = cu_seqlens[row + 1] - cu_seqlens[row];
    if (row_len >= state_width) {
      const int input_position = row_len - state_width + state_position;
      output[index] =
          padded_input[(size_t)row * input_batch_stride +
                       (size_t)input_position * input_token_stride +
                       channel];
    } else {
      const int retained = state_width - row_len;
      output[index] = state_position < retained
                          ? initial_state[((size_t)row * channels + channel) *
                                              state_width +
                                          state_position + row_len]
                          : padded_input[(size_t)row * input_batch_stride +
                                         (size_t)(state_position - retained) *
                                             input_token_stride +
                                         channel];
    }
  }
}

template <typename T, typename OutT>
GDN_PREP_ENTRY void gdn_prepare_recurrence_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, OutT *__restrict__ q_out,
    OutT *__restrict__ k_out, OutT *__restrict__ v_out,
    float *__restrict__ g_out, float *__restrict__ beta_out, int batch_size,
    int seq_len, int token_stride, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int tiled_v_heads) {
  const int token_head = blockIdx.x;
  const int hv = token_head % num_v_heads;
  const int token = token_head / num_v_heads;
  const int t = token % seq_len;
  const int bidx = token / seq_len;
  const int tid = threadIdx.x;

  if (bidx >= batch_size)
    return;

  const int v_per_group = num_v_heads / num_k_heads;
  const int hk = tiled_v_heads ? hv % num_k_heads : hv / v_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const int bh = bidx * num_v_heads + hv;

  const size_t token_idx = (size_t)bidx * seq_len + t;
  const T *row = mixed_qkv + token_idx * conv_dim;
  const T *b_row = b + token_idx * num_v_heads;
  const T *a_row = a + token_idx * num_v_heads;

  __shared__ float red_q[256];
  __shared__ float red_k[256];
  __shared__ float q_mul;
  __shared__ float k_mul;

  float q_sum = 0.0f;
  float k_sum = 0.0f;
  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_sum += q_val * q_val;
    k_sum += k_val * k_val;
  }

  red_q[tid] = q_sum;
  red_k[tid] = k_sum;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red_q[tid] += red_q[tid + stride];
      red_k[tid] += red_k[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    q_mul = rsqrtf(red_q[0] + 1.0e-6f) * rsqrtf((float)head_k_dim);
    k_mul = rsqrtf(red_k[0] + 1.0e-6f);

    float b_val = (float)b_row[hv];
    float a_val = (float)a_row[hv] + dt_bias[hv];
    float softplus_val = a_val > 20.0f
                             ? a_val
                             : (a_val > 0.0f ? a_val + log1pf(expf(-a_val))
                                             : log1pf(expf(a_val)));
    beta_out[(size_t)bh * token_stride + t] = 1.0f / (1.0f + expf(-b_val));
    g_out[(size_t)bh * token_stride + t] = -expf(a_log[hv]) * softplus_val;
  }
  __syncthreads();

  OutT *q_dst = q_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *k_dst = k_out + ((size_t)bh * token_stride + t) * head_k_dim;
  OutT *v_dst = v_out + ((size_t)bh * token_stride + t) * head_v_dim;

  for (int d = tid; d < head_k_dim; d += blockDim.x) {
    float q_val = (float)row[hk * head_k_dim + d];
    float k_val = (float)row[key_dim + hk * head_k_dim + d];
    q_dst[d] = (OutT)(q_val * q_mul);
    k_dst[d] = (OutT)(k_val * k_mul);
  }

  for (int d = tid; d < head_v_dim; d += blockDim.x) {
    v_dst[d] = (OutT)(float)row[2 * key_dim + hv * head_v_dim + d];
  }
}

__device__ __forceinline__ float gdn_warp_max(float value, int width = 32) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    // rows4 may have only one live half-warp in the final block.
    const unsigned mask = width == 16 ? (0xffffu << ((threadIdx.x / 16) * 16)) : 0xffffffffu;
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset, width));
  }
  return value;
}

__device__ __forceinline__ size_t gdn_fp8_scale_offset(
    int normalized_row, int groups, int scale_stride_m) {
  const int projection_row = normalized_row / groups;
  const int group = normalized_row - projection_row * groups;
  return (size_t)group * scale_stride_m + projection_row;
}

__device__ __forceinline__ void gdn_fp8_quant_params(
    float maximum, int scale_layout, float &quant_scale,
    float &inverse_quant_scale) {
  if (scale_layout == 0) {
    inverse_quant_scale = fmaxf(maximum, 1.0e-10f) / 448.0f;
    quant_scale = 1.0f / inverse_quant_scale;
    return;
  }
  quant_scale = maximum == 0.0f ? 1.0f : 448.0f / maximum;
  inverse_quant_scale = 1.0f / quant_scale;
}

__device__ __forceinline__ gdn_fp8_e4m3 gdn_fp8_quantize(
    float value, float quant_scale, float inverse_quant_scale,
    int scale_layout) {
  const float scaled = scale_layout == 0
                           ? value / inverse_quant_scale
                           : value * quant_scale;
  return gdn_fp8_e4m3(fminf(fmaxf(scaled, -448.0f), 448.0f));
}

template <int ROWS_PER_BLOCK>
GDN_PREP_ENTRY void gdn_rmsnorm_gated_quantized_warp_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, int outer_dim_1,
    int outer_dim_2, int64_t x_stride_0, int64_t x_stride_1,
    int64_t x_stride_2, int64_t x_stride_3, int64_t gate_stride_0,
    int64_t gate_stride_1, int64_t gate_stride_2, int64_t gate_stride_3,
    float eps, int sigmoid_gate) {
  constexpr int HIDDEN_DIM = GDN_RMSNORM_FAST_HIDDEN;
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
  const __nv_bfloat16 *x_row = x + (size_t)outer_0 * x_stride_0 +
                               (size_t)outer_1 * x_stride_1 +
                               (size_t)outer_2 * x_stride_2;
  const __nv_bfloat16 *gate_row = gate + (size_t)outer_0 * gate_stride_0 +
                                  (size_t)outer_1 * gate_stride_1 +
                                  (size_t)outer_2 * gate_stride_2;

  float values[HIDDEN_DIM / 32];
  float sum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    const float value = (float)x_row[(size_t)i * x_stride_3];
    values[i / 32] = value;
    sum = __fmaf_rn(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  const float inv_rms =
      rsqrtf(__shfl_sync(0xffffffff, sum, 0) / (float)HIDDEN_DIM + eps);

  __nv_bfloat16 rounded[HIDDEN_DIM / 32];
  float maximum = 0.0f;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    rounded[i / 32] = __float2bfloat16_rn(
        values[i / 32] * inv_rms * (float)weight[i] *
        gdn_gate_act((float)gate_row[(size_t)i * gate_stride_3], sigmoid_gate));
    maximum = fmaxf(maximum, fabsf((float)rounded[i / 32]));
  }
  maximum = gdn_warp_max(maximum);
  float quant_scale;
  float inverse_quant_scale;
  gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                       inverse_quant_scale);
  if (lane == 0) {
    scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
        inverse_quant_scale;
  }
  gdn_fp8_e4m3 *out_row = output + (size_t)row * HIDDEN_DIM;
#pragma unroll
  for (int i = lane; i < HIDDEN_DIM; i += 32) {
    out_row[i] = gdn_fp8_quantize((float)rounded[i / 32], quant_scale,
                                  inverse_quant_scale, scale_layout);
  }
}

GDN_PREP_ENTRY void gdn_rmsnorm_gated_quantized_rows4_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ gate,
    const __nv_bfloat16 *__restrict__ weight,
    gdn_fp8_e4m3 *__restrict__ output, float *__restrict__ scales, int rows,
    int groups, int scale_stride_m, int scale_layout, float eps,
    int sigmoid_gate) {
  using Vec = gdn_rmsnorm_vec8<__nv_bfloat16>;
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

  Vec values[2]{};
  if (first_row < rows) {
    values[0] = reinterpret_cast<const Vec *>(x)[first_row * vecs_per_row +
                                                  lane_in_half];
  }
  if (second_row < rows) {
    values[1] = reinterpret_cast<const Vec *>(x)[second_row * vecs_per_row +
                                                  lane_in_half];
  }

  float sums[2] = {0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
    const float first = (float)values[0].data[i];
    const float second = (float)values[1].data[i];
    sums[0] = __fmaf_rn(first, first, sums[0]);
    sums[1] = __fmaf_rn(second, second, sums[1]);
  }
#pragma unroll
  for (int offset = GDN_RMSNORM_TILED_LANES_PER_ROW / 2; offset > 0;
       offset >>= 1) {
    sums[0] += __shfl_down_sync(warp_mask, sums[0], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
    sums[1] += __shfl_down_sync(warp_mask, sums[1], offset,
                                GDN_RMSNORM_TILED_LANES_PER_ROW);
  }
  const float inv_rms[2] = {
      rsqrtf(__shfl_sync(warp_mask, sums[0], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps),
      rsqrtf(__shfl_sync(warp_mask, sums[1], 0,
                         GDN_RMSNORM_TILED_LANES_PER_ROW) /
                     (float)GDN_RMSNORM_FAST_HIDDEN +
                 eps)};

#pragma unroll
  for (int row_index = 0; row_index < 2; ++row_index) {
    const int row = row_index == 0 ? first_row : second_row;
    if (row >= rows) {
      continue;
    }
    const Vec gate_value = reinterpret_cast<const Vec *>(
        gate)[row * vecs_per_row + lane_in_half];
    __nv_bfloat16 rounded[GDN_RMSNORM_TILED_VALUES_PER_LANE];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      rounded[i] = __float2bfloat16_rn(
          (float)values[row_index].data[i] * inv_rms[row_index] *
          (float)weight_value.data[i] * gdn_gate_act((float)gate_value.data[i], sigmoid_gate));
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
    }
    maximum = gdn_warp_max(maximum, GDN_RMSNORM_TILED_LANES_PER_ROW);
    float quant_scale;
    float inverse_quant_scale;
    gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                         inverse_quant_scale);
    if (lane_in_half == 0) {
      scales[gdn_fp8_scale_offset(row, groups, scale_stride_m)] =
          inverse_quant_scale;
    }
    gdn_fp8_e4m3 *out = output + (size_t)row * GDN_RMSNORM_FAST_HIDDEN +
                          lane_in_half * GDN_RMSNORM_TILED_VALUES_PER_LANE;
#pragma unroll
    for (int i = 0; i < GDN_RMSNORM_TILED_VALUES_PER_LANE; ++i) {
      out[i] = gdn_fp8_quantize((float)rounded[i], quant_scale,
                                inverse_quant_scale, scale_layout);
    }
  }
}


template <typename T>
GDN_PREP_ENTRY void
fused_gdn_gating_kernel(const T *__restrict__ b,           // [total]
                        const T *__restrict__ a,           // [total]
                        const float *__restrict__ a_log,   // [num_heads]
                        const float *__restrict__ dt_bias, // [num_heads]
                        T *__restrict__ beta_out,          // [total]
                        T *__restrict__ g_out,             // [total]
                        int total_elements, int num_heads) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements)
    return;

  // Head index: elements are laid out as [..., num_heads]
  int head_idx = idx % num_heads;

  // beta = sigmoid(b)
  float b_val = (float)b[idx];
  float beta = 1.0f / (1.0f + expf(-b_val));

  // g = -exp(a_log) * softplus(a + dt_bias)
  float a_val = (float)a[idx];
  float a_log_val = a_log[head_idx];
  float dt_bias_val = dt_bias[head_idx];

  float sp_input = a_val + dt_bias_val;
  float softplus_val = logf(1.0f + expf(sp_input));
  float g_val = -expf(a_log_val) * softplus_val;

  beta_out[idx] = (T)beta;
  g_out[idx] = (T)g_val;
}

#undef GDN_PREP_ENTRY
