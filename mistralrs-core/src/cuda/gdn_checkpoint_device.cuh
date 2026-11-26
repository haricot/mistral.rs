#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#ifdef DKR_GDN_CHECKPOINT_IMAGE
#define GDN_CHECKPOINT_ENTRY __device__ __forceinline__
#else
#define GDN_CHECKPOINT_ENTRY __global__
#endif
constexpr int GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH = 16;
constexpr uint32_t GDN_SPEC_CHECKPOINT_PAD_SLOT = 0xffffffffu;

constexpr int GDN_SPEC_COMMIT_WARPS = 4;
constexpr int GDN_SPEC_COMMIT_MAX_K = 256;

template <typename T>
GDN_CHECKPOINT_ENTRY void gdn_speculative_conv_state_commit_kernel(
    const T *__restrict__ x, const T *__restrict__ initial_state,
    T *__restrict__ state_pool, const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int conv_dim, int kernel_size, int capacity) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (channel >= conv_dim || batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows <= 0 || rows > seq_len || slot_indices[batch_idx] == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (capacity > 0 && slot_indices[batch_idx] >= (uint32_t)capacity)) {
    return;
  }

  const T *prior = initial_state +
                   ((size_t)batch_idx * conv_dim + channel) * kernel_size;
  T *destination =
      state_pool +
      ((size_t)slot_indices[batch_idx] * conv_dim + channel) * kernel_size;
  const int pad = kernel_size - rows;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      destination[i] = prior[i + rows];
    } else {
      const int position = rows - kernel_size + i;
      destination[i] =
          x[((size_t)batch_idx * seq_len + position) * conv_dim + channel];
    }
  }
}

template <typename T, typename StateT, bool VALUE_MAJOR>
GDN_CHECKPOINT_ENTRY void gdn_speculative_recurrent_state_commit_kernel(
    const T *__restrict__ convolved_qkv, const T *__restrict__ b,
    const T *__restrict__ a,
    const StateT *__restrict__ initial_recurrent_state,
    const float *__restrict__ a_log, const float *__restrict__ dt_bias,
    StateT *__restrict__ recurrent_state_pool,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ slot_indices, int batch_size, int seq_len,
    int num_k_heads, int num_v_heads, int head_k_dim, int head_v_dim,
    int tiled_v_heads, int capacity) {
  constexpr int WARP_SIZE = 32;
  constexpr int K_PER_LANE = GDN_SPEC_COMMIT_MAX_K / WARP_SIZE;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_idx = blockIdx.x * GDN_SPEC_COMMIT_WARPS + warp;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  if (rows <= 0 || rows > seq_len || slot_indices[batch_idx] == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (capacity > 0 && slot_indices[batch_idx] >= (uint32_t)capacity)) {
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head =
      tiled_v_heads ? value_head % num_k_heads : value_head / values_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t initial_head =
      ((size_t)batch_idx * num_v_heads + value_head) * head_k_dim *
      head_v_dim;
  const size_t destination_head =
      ((size_t)slot_indices[batch_idx] * num_v_heads + value_head) *
      head_k_dim * head_v_dim;
  const int thread_idx = warp * WARP_SIZE + lane;

  __shared__ float key_buffer[GDN_SPEC_COMMIT_MAX_K];
  __shared__ float norm_buffer[WARP_SIZE * GDN_SPEC_COMMIT_WARPS];
  __shared__ float key_multiplier;
  __shared__ float beta;
  __shared__ float decay;
  __shared__ float values[GDN_SPEC_COMMIT_WARPS];

  float state[K_PER_LANE];
#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      state[r] = initial_recurrent_state[initial_head + offset];
    }
  }

  for (int position = 0; position < rows; position++) {
    float key_norm_partial = 0.0f;
    for (int key_idx = thread_idx; key_idx < head_k_dim;
         key_idx += WARP_SIZE * GDN_SPEC_COMMIT_WARPS) {
      const int channel = key_dim + key_head * head_k_dim + key_idx;
      const float value =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
      key_buffer[key_idx] = value;
      key_norm_partial = __fmaf_rn(value, value, key_norm_partial);
    }
    norm_buffer[thread_idx] = key_norm_partial;
    __syncthreads();
    for (int stride = (WARP_SIZE * GDN_SPEC_COMMIT_WARPS) / 2; stride > 0;
         stride >>= 1) {
      if (thread_idx < stride) {
        norm_buffer[thread_idx] += norm_buffer[thread_idx + stride];
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      key_multiplier = rsqrtf(norm_buffer[0] + 1.0e-6f);
      const size_t gate_offset =
          ((size_t)batch_idx * seq_len + position) * num_v_heads +
          value_head;
      const float b_value = (float)b[gate_offset];
      const float a_value = (float)a[gate_offset] + dt_bias[value_head];
      const float softplus =
          a_value > 20.0f
              ? a_value
              : (a_value > 0.0f ? a_value + log1pf(expf(-a_value))
                                : log1pf(expf(a_value)));
      beta = 1.0f / (1.0f + expf(-b_value));
      decay = expf(-expf(a_log[value_head]) * softplus);
    }
    if (lane == 0) {
      const int channel = 2 * key_dim + value_head * head_v_dim + value_idx;
      values[warp] =
          (float)convolved_qkv[((size_t)batch_idx * seq_len + position) *
                                   conv_dim +
                               channel];
    }
    __syncthreads();

    float state_dot_key = 0.0f;
#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state_dot_key = __fmaf_rn(state[r], key, state_dot_key);
      }
    }
    state_dot_key = gdn_warp_sum<WARP_SIZE>(state_dot_key);
    const float delta = (values[warp] - decay * state_dot_key) * beta;

#pragma unroll
    for (int r = 0; r < K_PER_LANE; r++) {
      const int key_idx = r * WARP_SIZE + lane;
      if (key_idx < head_k_dim) {
        const float key = key_buffer[key_idx] * key_multiplier;
        state[r] = __fmaf_rn(key, delta, decay * state[r]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < K_PER_LANE; r++) {
    const int key_idx = r * WARP_SIZE + lane;
    if (key_idx < head_k_dim) {
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      recurrent_state_pool[destination_head + offset] = state[r];
    }
  }
}

__device__ __forceinline__ size_t
gdn_spec_checkpoint_base(uint32_t active_slot, int checkpoint_lanes) {
  return ((size_t)active_slot / checkpoint_lanes) * checkpoint_lanes;
}

template <typename T>
GDN_CHECKPOINT_ENTRY void gdn_speculative_conv_checkpoints_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int kernel_size, int checkpoint_lanes,
    bool write_checkpoints, T *pending_conv_input,
    const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c, int capacity) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (capacity > 0 && active_slot >= (uint32_t)capacity)) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  T state[GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH];
  T *source = channel_valid
                  ? state_pool +
                        ((size_t)active_slot * conv_dim + channel) * kernel_size
                  : nullptr;
  if (channel_valid) {
    for (int i = 0; i < kernel_size; i++) {
      state[i] = source[i];
    }
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
      for (int i = 0; i < kernel_size - 1; i++) {
        state[i] = state[i + 1];
      }
      state[kernel_size - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    for (int i = 0; i < kernel_size; i++) {
      source[i] = state[i];
    }
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const T *channel_weight = weight + (size_t)channel * kernel_size;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
    for (int i = 0; i < kernel_size - 1; i++) {
      state[i] = state[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state[kernel_size - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
      acc = __fmaf_rn((float)state[i], (float)channel_weight[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;

    if (write_checkpoints) {
      T *destination =
          state_pool +
          (((base_slot + position) * conv_dim + channel) * kernel_size);
      for (int i = 0; i < kernel_size; i++) {
        destination[i] = state[i];
      }
    }
  }
}

template <typename T>
GDN_CHECKPOINT_ENTRY void gdn_speculative_conv_checkpoints_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ state_pool, T *__restrict__ output,
    const uint32_t *__restrict__ active_slots, int batch_size, int seq_len,
    int conv_dim, int checkpoint_lanes, bool write_checkpoints,
    T *pending_conv_input, const uint32_t *__restrict__ pending_keep_rows,
    const uint32_t *__restrict__ pending_epochs,
    uint32_t *__restrict__ conv_applied_epochs, int max_pending_rows,
    int pending_capacity, int conv_blocks, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c, int capacity) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_idx = blockIdx.y;
  if (batch_idx >= batch_size) {
    return;
  }
  const bool channel_valid = channel < conv_dim;

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (capacity > 0 && active_slot >= (uint32_t)capacity)) {
    for (int position = 0; channel_valid && position < seq_len; position++) {
      output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
          (T)0.0f;
    }
    return;
  }

  auto *states = reinterpret_cast<GdnConvWidth4<T> *>(state_pool);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  GdnConvWidth4<T> state;
  if (channel_valid) {
    state = states[(size_t)active_slot * conv_dim + channel];
  }
  const int conv_block = blockIdx.x;
  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  if (pending_epochs != nullptr && active_slot < pending_capacity) {
    pending_epoch = pending_epochs[active_slot];
    pending_rows = (int)pending_keep_rows[active_slot];
    apply_pending = pending_epoch != 0 && pending_rows > 0 &&
                    pending_rows <= max_pending_rows &&
                    conv_applied_epochs[(size_t)active_slot * conv_blocks +
                                        conv_block] != pending_epoch;
  }
  if (channel_valid && apply_pending) {
    for (int position = 0; position < pending_rows; position++) {
#pragma unroll
      for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
        state.values[i] = state.values[i + 1];
      }
      state.values[GDN_PACKED_CONV_WIDTH - 1] =
          pending_conv_input[((size_t)active_slot * max_pending_rows +
                              position) *
                                 conv_dim +
                             channel];
    }
    states[(size_t)active_slot * conv_dim + channel] = state;
  }
  if (pending_epochs != nullptr) {
    __syncthreads();
    if (threadIdx.x == 0 && apply_pending) {
      conv_applied_epochs[(size_t)active_slot * conv_blocks + conv_block] =
          pending_epoch;
    }
  }
  if (!channel_valid) {
    return;
  }
  const GdnConvWidth4<T> channel_weight = weights[channel];
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  for (int position = 0; position < seq_len; position++) {
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
      state.values[i] = state.values[i + 1];
    }
    const T input =
        x[(size_t)batch_idx * x_stride_b + (size_t)position * x_stride_s +
          (size_t)channel * x_stride_c];
    state.values[GDN_PACKED_CONV_WIDTH - 1] = input;
    if (!write_checkpoints && pending_conv_input != nullptr) {
      pending_conv_input[((size_t)active_slot * max_pending_rows + position) *
                             conv_dim +
                         channel] = input;
    }

    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
      acc = __fmaf_rn((float)state.values[i],
                      (float)channel_weight.values[i], acc);
    }
    const float result = acc / (1.0f + expf(-acc));
    output[((size_t)batch_idx * seq_len + position) * conv_dim + channel] =
        (T)result;
    if (write_checkpoints) {
      states[(base_slot + position) * conv_dim + channel] = state;
    }
  }
}

#undef GDN_CHECKPOINT_ENTRY
