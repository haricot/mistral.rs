#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef DKR_ROUTER_TOPK_EXTERNAL
#define DKR_ROUTER_TOPK_ENTRY __device__ __forceinline__
#define DKR_ROUTER_MOE_ENTRY __device__ __forceinline__
#else
#define DKR_ROUTER_TOPK_ENTRY __global__
#define DKR_ROUTER_MOE_ENTRY __launch_bounds__(MOE_ROUTER_ROWS_PER_BLOCK * 32, 1) __global__
#endif

template <typename T>
__device__ __forceinline__ T warp_reduce_max_with_idx(T val, int idx,
                                                      int &max_idx) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other_val = __shfl_down_sync(0xffffffff, val, offset);
    int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val ||
        (other_val == val && other_idx >= 0 && (idx < 0 || other_idx < idx))) {
      val = other_val;
      idx = other_idx;
    }
  }
  max_idx = idx;
  return val;
}

// One block per row, finds top-k elements
// For n <= 1024 (typical MoE expert count), single block is sufficient
// Writes values and indices to SEPARATE buffers (no post-processing needed)
template <typename T>
DKR_ROUTER_TOPK_ENTRY void topk_kernel(const T *__restrict__ input, // [nrows, ncols]
                            T *__restrict__ values_out,  // [nrows, k]
                            uint32_t *__restrict__ indices_out, // [nrows, k]
                            const int nrows, const int ncols, const int k) {
  const int row = blockIdx.x;
  if (row >= nrows)
    return;

  const T *row_in = input + row * ncols;
  T *row_values = values_out + row * k;
  uint32_t *row_indices = indices_out + row * k;

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  // Shared memory for this row's data and mask
  extern __shared__ char smem[];
  T *s_data = (T *)smem;
  bool *s_used = (bool *)(s_data + ncols);

  // Load data into shared memory
  for (int i = tid; i < ncols; i += block_size) {
    s_data[i] = row_in[i];
    s_used[i] = false;
  }
  __syncthreads();

  // Find top-k elements
  for (int ki = 0; ki < k; ki++) {
    // Find max among unused elements
    T local_max = (T)(-INFINITY);
    int local_idx = -1;

    for (int i = tid; i < ncols; i += block_size) {
      float candidate = (float)s_data[i];
      if (!s_used[i] && candidate == candidate &&
          candidate > (float)local_max) {
        local_max = s_data[i];
        local_idx = i;
      }
    }

    // Warp reduction to find max
    int warp_max_idx;
    T warp_max = warp_reduce_max_with_idx(local_max, local_idx, warp_max_idx);

    // Block reduction (if more than 1 warp)
    __shared__ T warp_maxes[32];
    __shared__ int warp_indices[32];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (block_size + 31) / 32;

    if (lane_id == 0) {
      warp_maxes[warp_id] = warp_max;
      warp_indices[warp_id] = warp_max_idx;
    }
    __syncthreads();

    // Final reduction in first warp
    if (tid < 32) {
      T val = (tid < num_warps) ? warp_maxes[tid] : (T)(-INFINITY);
      int idx = (tid < num_warps) ? warp_indices[tid] : -1;
      int final_idx;
      T final_max = warp_reduce_max_with_idx(val, idx, final_idx);

      if (tid == 0) {
        if (final_idx < 0) {
          final_idx = 0;
          final_max = (T)0;
        }
        row_values[ki] = final_max;
        row_indices[ki] = (uint32_t)final_idx;
        s_used[final_idx] = true;
      }
    }
    __syncthreads();
  }
}

constexpr int MOE_ROUTER_SCORE_RAW = 0;
constexpr int MOE_ROUTER_SCORE_SOFTMAX = 1;
constexpr int MOE_ROUTER_SCORE_SIGMOID = 2;
constexpr int MOE_ROUTER_WEIGHT_SCORE = 0;
constexpr int MOE_ROUTER_WEIGHT_SOFTMAX = 1;
constexpr int MOE_ROUTER_WEIGHT_SIGMOID = 2;
constexpr int MOE_ROUTER_ROWS_PER_BLOCK = 4;

template <typename T> __device__ __forceinline__ float router_to_float(T x) {
  return static_cast<float>(x);
}

template <> __device__ __forceinline__ float router_to_float<__half>(__half x) {
  return __half2float(x);
}

template <>
__device__ __forceinline__ float
router_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ float router_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  }
  return val;
}

__device__ __forceinline__ float router_warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
  }
  return val;
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_softmax_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
  float max_val = -INFINITY;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      max_val = max(max_val, vals[i]);
    }
  }

  max_val = router_warp_reduce_max(max_val);
  float sum = 0.0f;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] = expf(vals[i] - max_val);
      sum += vals[i];
    } else {
      vals[i] = 0.0f;
    }
  }

  sum = router_warp_reduce_sum(sum);
  const float inv_sum = 1.0f / sum;

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (!use_limit || idx < limit) {
      vals[i] *= inv_sum;
    }
  }
}

template <int values_per_thread, bool use_limit>
__device__ __forceinline__ void
router_sigmoid_warp(float (&vals)[values_per_thread], const int limit,
                    const int lane) {
#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    vals[i] = (!use_limit || idx < limit) ? 1.0f / (1.0f + expf(-vals[i]))
                                          : -INFINITY;
  }
}

template <typename T, int n_experts, bool has_bias, bool has_expert_scale>
DKR_ROUTER_MOE_ENTRY
    void moe_router_topk_kernel(
        const T *__restrict__ logits, float *__restrict__ weights,
        uint32_t *__restrict__ ids, const float *__restrict__ selection_bias,
        const float *__restrict__ expert_scale, const int n_rows,
        const int top_k, const int score_mode, const int weight_mode,
        const bool renormalize, const bool clamp_logits, const float clamp_min,
        const float clamp_max, const float norm_min, const float output_scale) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= n_rows) {
    return;
  }

  logits += row * n_experts;
  weights += row * top_k;
  ids += row * top_k;

  constexpr int values_per_thread = n_experts > 32 ? n_experts / 32 : 1;

  float raw[values_per_thread];
  float score[values_per_thread];
  float selection[values_per_thread];
  float output_weights[values_per_thread];
  uint32_t output_ids[values_per_thread];

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    float value = (n_experts % 32 == 0 || expert < n_experts)
                      ? router_to_float(logits[expert])
                      : -INFINITY;
    if (clamp_logits && expert < n_experts) {
      value = fminf(fmaxf(value, clamp_min), clamp_max);
    }
    if (value != value) {
      value = -INFINITY;
    }
    raw[i] = value;
    score[i] = value;
    selection[i] = value;
    output_weights[i] = 0.0f;
    output_ids[i] = 0;
  }

  if (score_mode == MOE_ROUTER_SCORE_SOFTMAX) {
    router_softmax_warp<values_per_thread, false>(score, n_experts, lane);
  } else if (score_mode == MOE_ROUTER_SCORE_SIGMOID) {
    router_sigmoid_warp<values_per_thread, false>(score, n_experts, lane);
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int expert = lane + i * 32;
    // Padded warp lanes must never compete with real experts, notably
    // sigmoid/softmax scores plus negative selection bias. Otherwise an
    // out-of-range expert id can index expert_scale below.
    selection[i] = expert < n_experts ? score[i] : -INFINITY;
    if constexpr (has_bias) {
      if (expert < n_experts) {
        selection[i] += selection_bias[expert];
      }
    }
    if (selection[i] != selection[i]) {
      selection[i] = -INFINITY;
    }
  }

  for (int k_idx = 0; k_idx < top_k; k_idx++) {
    float best_selection = selection[0];
    float best_score = score[0];
    float best_raw = raw[0];
    int best_expert = lane;

#pragma unroll
    for (int i = 1; i < values_per_thread; i++) {
      const int expert = lane + i * 32;
      if ((n_experts % 32 == 0 || expert < n_experts) &&
          selection[i] > best_selection) {
        best_selection = selection[i];
        best_score = score[i];
        best_raw = raw[i];
        best_expert = expert;
      }
    }

#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      const float other_selection =
          __shfl_xor_sync(0xffffffff, best_selection, mask, 32);
      const float other_score =
          __shfl_xor_sync(0xffffffff, best_score, mask, 32);
      const float other_raw = __shfl_xor_sync(0xffffffff, best_raw, mask, 32);
      const int other_expert =
          __shfl_xor_sync(0xffffffff, best_expert, mask, 32);
      if (other_selection > best_selection ||
          (other_selection == best_selection && other_expert < best_expert)) {
        best_selection = other_selection;
        best_score = other_score;
        best_raw = other_raw;
        best_expert = other_expert;
      }
    }

    float out = best_score;
    if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
      out = best_raw;
    } else if (weight_mode == MOE_ROUTER_WEIGHT_SIGMOID) {
      out = 1.0f / (1.0f + expf(-best_raw));
    }

    if ((k_idx & 31) == lane) {
      output_weights[k_idx / 32] = out;
      output_ids[k_idx / 32] = static_cast<uint32_t>(best_expert);
    }

    if ((best_expert & 31) == lane) {
      selection[best_expert / 32] = -INFINITY;
    }
  }

  if (weight_mode == MOE_ROUTER_WEIGHT_SOFTMAX) {
    router_softmax_warp<values_per_thread, true>(output_weights, top_k, lane);
  }

  if (renormalize) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      const int idx = lane + i * 32;
      if (idx < top_k) {
        sum += output_weights[i];
      }
    }
    sum = router_warp_reduce_sum(sum);
    sum = fmaxf(sum, norm_min);
    const float inv_sum = 1.0f / sum;
#pragma unroll
    for (int i = 0; i < values_per_thread; i++) {
      output_weights[i] *= inv_sum;
    }
  }

#pragma unroll
  for (int i = 0; i < values_per_thread; i++) {
    const int idx = lane + i * 32;
    if (idx < top_k) {
      float scale = output_scale;
      if constexpr (has_expert_scale) {
        scale *= expert_scale[output_ids[i]];
      }
      weights[idx] = output_weights[i] * scale;
      ids[idx] = output_ids[i];
    }
  }
}

#undef DKR_ROUTER_TOPK_ENTRY
#undef DKR_ROUTER_MOE_ENTRY
