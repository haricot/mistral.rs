#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#ifdef DKR_GDN_SPEC_IMAGE
#define GDN_SPEC_ENTRY __device__ __forceinline__
#define GDN_SPEC_LAUNCH(N,M) __device__ __forceinline__
#define GDN_SPEC_WARP(N) __device__ __forceinline__
#else
#define GDN_SPEC_ENTRY __global__
#define GDN_SPEC_LAUNCH(N,M) __global__ __launch_bounds__(N,M)
#define GDN_SPEC_WARP(N) __global__ __launch_bounds__(N)
#endif

constexpr int GDN_SPEC_CHECKPOINT_MAX_K = 256;
constexpr int GDN_SPEC_CHECKPOINT_VALUE_TILE = 32;
constexpr int GDN_SPEC_CHECKPOINT_WARPS = 4;
constexpr int GDN_SPEC_CHECKPOINT_VALUES_PER_WARP =
    GDN_SPEC_CHECKPOINT_VALUE_TILE / GDN_SPEC_CHECKPOINT_WARPS;
constexpr int GDN_SPEC_FUSED_MAX_TOKENS = 8;
constexpr int GDN_SPEC_FUSED_THREADS = 256;
constexpr int GDN_SPEC_FUSED_WARPS = GDN_SPEC_FUSED_THREADS / 32;
constexpr int GDN_SPEC_FUSED_VALUE_CHUNK = 32;
constexpr int GDN_SPEC_FUSED_VALUE_CHUNKS =
    GDN_DECODE_VALUE_MAJOR_V / GDN_SPEC_FUSED_VALUE_CHUNK;
constexpr int GDN_SPEC_FUSED_VALUES_PER_WARP =
    GDN_SPEC_FUSED_VALUE_CHUNK / GDN_SPEC_FUSED_WARPS;
// Paired reductions favor 16-bit states and underfilled or saturated F32 grids; serial sustains mid-grid bandwidth.
constexpr int GDN_SPEC_FUSED_PAIR_LOW_GRID_MAX = 144;
constexpr int GDN_SPEC_FUSED_PAIR_HIGH_GRID_MIN = 768;
constexpr int GDN_DEFERRED_STATE_DEPTH = 4;

__device__ __forceinline__ float gdn_spec_softplus(float value) {
  return value > 20.0f
             ? value
             : (value > 0.0f ? value + log1pf(expf(-value))
                             : log1pf(expf(value)));
}

__device__ __forceinline__ float2 gdn_spec_warp_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset);
    y += __shfl_xor_sync(0xffffffff, y, offset);
  }
  return make_float2(x, y);
}

template <typename StateT>
__device__ __forceinline__ void gdn_spec_copy_state_chunk(
    StateT *shared_state, const StateT *state, int chunk, int thread) {
  constexpr int ELEMENTS_PER_COPY = 16 / sizeof(StateT);
  constexpr int ELEMENTS_PER_CHUNK =
      GDN_SPEC_FUSED_VALUE_CHUNK * GDN_DECODE_VALUE_MAJOR_K;
  constexpr int COPIES_PER_CHUNK = ELEMENTS_PER_CHUNK / ELEMENTS_PER_COPY;
  for (int copy = thread; copy < COPIES_PER_CHUNK;
       copy += GDN_SPEC_FUSED_THREADS) {
    const int element = copy * ELEMENTS_PER_COPY;
    gdn_cp_async_cg_16(shared_state + element,
                       state + chunk * ELEMENTS_PER_CHUNK + element);
  }
  gdn_cp_async_commit();
}

// Adapted from vLLM revision c8438a3d40168ce1d9eade0dc15ccbe5d27adb68.
// Copyright contributors to the vLLM project; Apache-2.0. See third_party/flashinfer_gdn.
template <typename T, typename StateT, bool PAIRED_REDUCTIONS,
          bool DIRECT_TRANSITIONS>
GDN_SPEC_LAUNCH(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_speculative_recurrence_rmsnorm_gate_value_major_128_kernel(
        const T *__restrict__ mixed_qkv, const T *__restrict__ b,
        const T *__restrict__ a, const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
        T *__restrict__ output, gdn_fp8_e4m3 *__restrict__ quantized_output,
        float *__restrict__ output_scales, int scale_stride_m,
        int scale_layout, const uint32_t *__restrict__ active_slots,
        const T *__restrict__ gate, const T *__restrict__ norm_weight,
        float *__restrict__ transition_key,
        float *__restrict__ transition_delta,
        float *__restrict__ transition_decay,
        float *pending_key_banks,
        const uint32_t *__restrict__ pending_key_bank, float *pending_delta,
        float *pending_decay,
        const uint32_t *__restrict__ pending_keep_rows,
        const uint32_t *__restrict__ pending_epochs,
        uint32_t *__restrict__ recurrent_applied_epochs,
        int max_pending_rows, int pending_capacity,
        int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
        int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
        int64_t gate_stride_b, int64_t gate_stride_s,
        int64_t gate_stride_h, int64_t gate_stride_v, int batch_size,
        int seq_len, int num_k_heads, int num_v_heads,
        int checkpoint_lanes, int tiled_v_heads, float norm_eps,
        int sigmoid_gate, int pool_capacity) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (pool_capacity > 0 && active_slot >= (uint32_t)pool_capacity)) {
    for (int linear = tid; linear < seq_len * V;
         linear += GDN_SPEC_FUSED_THREADS) {
      const int position = linear / V;
      const int value = linear - position * V;
      const size_t offset =
          (((size_t)batch_idx * seq_len + position) * num_v_heads +
           value_head) *
              V +
          value;
      if (quantized_output != nullptr) {
        quantized_output[offset] = gdn_fp8_e4m3(0.0f);
      } else {
        output[offset] = (T)0.0f;
      }
    }
    if (quantized_output != nullptr) {
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(0.0f, scale_layout, quant_scale,
                           inverse_quant_scale);
      (void)quant_scale;
      for (int position = tid; position < seq_len;
           position += GDN_SPEC_FUSED_THREADS) {
        const int normalized_row =
            ((batch_idx * seq_len + position) * num_v_heads) + value_head;
        output_scales[gdn_fp8_scale_offset(normalized_row, num_v_heads,
                                           scale_stride_m)] =
            inverse_quant_scale;
      }
    }
    if constexpr (!DIRECT_TRANSITIONS) {
      if (transition_delta == nullptr) {
        return;
      }
      for (int linear = tid; linear < seq_len * V;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / V;
        const int value = linear - position * V;
        transition_delta[(((size_t)batch_idx * seq_len + position) *
                              num_v_heads +
                          value_head) *
                             V +
                         value] = 0.0f;
      }
      for (int position = tid; position < seq_len;
           position += GDN_SPEC_FUSED_THREADS) {
        transition_decay[((size_t)batch_idx * seq_len + position) *
                             num_v_heads +
                         value_head] = 0.0f;
      }
      const bool key_leader =
          tiled_v_heads ? value_head < num_k_heads
                        : value_head % (num_v_heads / num_k_heads) == 0;
      if (key_leader) {
        const int key_head = tiled_v_heads
                                 ? value_head % num_k_heads
                                 : value_head / (num_v_heads / num_k_heads);
        for (int linear = tid; linear < seq_len * K;
             linear += GDN_SPEC_FUSED_THREADS) {
          const int position = linear / K;
          const int key = linear - position * K;
          transition_key[(((size_t)batch_idx * seq_len + position) *
                              num_k_heads +
                          key_head) *
                             K +
                         key] = 0.0f;
        }
      }
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)V * K;
  StateT *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) *
                       state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);
  const size_t transition_row_base = (size_t)batch_idx * seq_len;

  __shared__ __align__(16) StateT shared_state[GDN_SPEC_FUSED_VALUE_CHUNK][K];
  __shared__ float shared_query[GDN_SPEC_FUSED_MAX_TOKENS][K];
  __shared__ float shared_key[GDN_SPEC_FUSED_MAX_TOKENS][K];
  __shared__ T shared_value[GDN_SPEC_FUSED_MAX_TOKENS][V];
  __shared__ T shared_output[GDN_SPEC_FUSED_MAX_TOKENS][V];
  __shared__ float shared_decay[GDN_SPEC_FUSED_MAX_TOKENS];
  __shared__ float shared_beta[GDN_SPEC_FUSED_MAX_TOKENS];
  __shared__ float shared_pending_key
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1]
      [DIRECT_TRANSITIONS ? K : 1];
  __shared__ float shared_pending_delta
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1]
      [DIRECT_TRANSITIONS ? V : 1];
  __shared__ float shared_pending_decay
      [DIRECT_TRANSITIONS ? GDN_SPEC_FUSED_MAX_TOKENS : 1];

  uint32_t pending_epoch = 0;
  int pending_rows = 0;
  bool apply_pending = false;
  uint32_t published_key_bank = 0;
  float *published_key = nullptr;
  float *candidate_key = nullptr;
  if constexpr (DIRECT_TRANSITIONS) {
    if (active_slot < pending_capacity) {
      pending_epoch = pending_epochs[active_slot];
      pending_rows = (int)pending_keep_rows[active_slot];
      apply_pending =
          pending_epoch != 0 && pending_rows > 0 &&
          pending_rows <= max_pending_rows &&
          recurrent_applied_epochs[(size_t)active_slot * num_v_heads +
                                   value_head] != pending_epoch;
      published_key_bank =
          pending_key_bank[active_slot] & GDN_PENDING_KEY_BANK_MASK;
      const size_t key_bank_stride =
          (size_t)pending_capacity * max_pending_rows * num_k_heads * K;
      published_key = pending_key_banks + published_key_bank * key_bank_stride;
      candidate_key =
          pending_key_banks + (published_key_bank ^ 1u) * key_bank_stride;
    }
  }

  gdn_spec_copy_state_chunk(&shared_state[0][0], source, 0, tid);

  if constexpr (DIRECT_TRANSITIONS) {
    if (apply_pending) {
      for (int linear = tid; linear < pending_rows * K;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / K;
        const int key = linear - position * K;
        shared_pending_key[position][key] =
            published_key[(((size_t)active_slot * max_pending_rows + position) *
                               num_k_heads +
                           key_head) *
                              K +
                          key];
      }
      for (int linear = tid; linear < pending_rows * V;
           linear += GDN_SPEC_FUSED_THREADS) {
        const int position = linear / V;
        const int value = linear - position * V;
        shared_pending_delta[position][value] =
            pending_delta[(((size_t)active_slot * max_pending_rows + position) *
                               num_v_heads +
                           value_head) *
                              V +
                          value];
      }
      for (int position = tid; position < pending_rows;
           position += GDN_SPEC_FUSED_THREADS) {
        shared_pending_decay[position] =
            pending_decay[((size_t)active_slot * max_pending_rows + position) *
                              num_v_heads +
                          value_head];
      }
    }
  }

  if (warp < seq_len) {
    const int position = warp;
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float4 query =
        gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm = key.x * key.x + key.y * key.y + key.z * key.z +
                     key.w * key.w;
    if constexpr (PAIRED_REDUCTIONS) {
      const float2 qk_norm = gdn_spec_warp_sum_pair(query_norm, key_norm);
      query_norm = qk_norm.x;
      key_norm = qk_norm.y;
    } else {
      query_norm = gdn_warp_sum<32>(query_norm);
      key_norm = gdn_warp_sum<32>(key_norm);
    }
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const float query_values[4] = {query.x, query.y, query.z, query.w};
    const float key_values[4] = {key.x, key.y, key.z, key.w};

#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key = lane * 4 + i;
      const int value = lane + i * 32;
      shared_query[position][key] = query_values[i] * query_multiplier;
      shared_key[position][key] = key_values[i] * key_multiplier;
      shared_value[position][value] =
          row[2 * key_dim + value_head * V + value];
    }
    if (lane == 0) {
      const size_t b_offset = (size_t)batch_idx * b_stride_b +
                              (size_t)position * b_stride_s +
                              (size_t)value_head * b_stride_h;
      const size_t a_offset = (size_t)batch_idx * a_stride_b +
                              (size_t)position * a_stride_s +
                              (size_t)value_head * a_stride_h;
      shared_beta[position] =
          1.0f / (1.0f + expf(-(float)b[b_offset]));
      const float biased_a = (float)a[a_offset] + dt_bias[value_head];
      shared_decay[position] =
          expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));
    }
  }
  __syncthreads();

  if constexpr (DIRECT_TRANSITIONS) {
    const bool key_leader =
        tiled_v_heads ? value_head < num_k_heads
                      : value_head % values_per_group == 0;
    if (key_leader && warp < seq_len) {
      const int position = warp;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int key = lane * 4 + i;
        candidate_key[(((size_t)active_slot * max_pending_rows + position) *
                           num_k_heads +
                       key_head) *
                          K +
                      key] = shared_key[position][key];
      }
    }
    if (warp < seq_len && lane == 0) {
      const int position = warp;
      pending_decay[((size_t)active_slot * max_pending_rows + position) *
                        num_v_heads +
                    value_head] = shared_decay[position];
    }
  } else if (transition_key != nullptr) {
    const bool key_leader =
        tiled_v_heads ? value_head < num_k_heads
                      : value_head % values_per_group == 0;
    if (key_leader && warp < seq_len) {
      const int position = warp;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int key = lane * 4 + i;
        transition_key[((transition_row_base + position) * num_k_heads +
                        key_head) *
                           K +
                       key] = shared_key[position][key];
      }
    }
    if (warp < seq_len && lane == 0) {
      const int position = warp;
      transition_decay[(transition_row_base + position) * num_v_heads +
                       value_head] = shared_decay[position];
    }
  }

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row = 0; row < VALUES_PER_WARP; row++) {
    value_rows[row] = warp + row * GDN_SPEC_FUSED_WARPS;
  }

#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    gdn_cp_async_wait();
    __syncthreads();

    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row = 0; row < VALUES_PER_WARP; row++) {
      state[row] = gdn_load_state_x4(
          &shared_state[value_rows[row]][key_base]);
    }
    __syncthreads();
    if (chunk + 1 < GDN_SPEC_FUSED_VALUE_CHUNKS) {
      gdn_spec_copy_state_chunk(&shared_state[0][0], source, chunk + 1, tid);
    }

    if constexpr (DIRECT_TRANSITIONS) {
      if (apply_pending) {
        for (int position = 0; position < pending_rows; position++) {
          const float4 key = *reinterpret_cast<const float4 *>(
              &shared_pending_key[position][key_base]);
          const float decay = shared_pending_decay[position];
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                              value_rows[row];
            const float delta = shared_pending_delta[position][value];
            state[row].x = __fmaf_rn(key.x, delta, state[row].x * decay);
            state[row].y = __fmaf_rn(key.y, delta, state[row].y * decay);
            state[row].z = __fmaf_rn(key.z, delta, state[row].z * decay);
            state[row].w = __fmaf_rn(key.w, delta, state[row].w * decay);
          }
        }
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          const int value =
              chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
          gdn_store_state_x4(source + (size_t)value * K + key_base,
                             state[row]);
          if constexpr (sizeof(StateT) < sizeof(float)) {
            state[row].x = (float)(StateT)state[row].x;
            state[row].y = (float)(StateT)state[row].y;
            state[row].z = (float)(StateT)state[row].z;
            state[row].w = (float)(StateT)state[row].w;
          }
        }
      }
    }

    for (int position = 0; position < seq_len; position++) {
      const float4 query = *reinterpret_cast<const float4 *>(
          &shared_query[position][key_base]);
      const float4 key = *reinterpret_cast<const float4 *>(
          &shared_key[position][key_base]);
      float state_dot_key[VALUES_PER_WARP];
#pragma unroll
      for (int row = 0; row < VALUES_PER_WARP; row++) {
        state[row].x *= shared_decay[position];
        state[row].y *= shared_decay[position];
        state[row].z *= shared_decay[position];
        state[row].w *= shared_decay[position];
        float dot = state[row].x * key.x;
        dot = __fmaf_rn(state[row].y, key.y, dot);
        dot = __fmaf_rn(state[row].z, key.z, dot);
        state_dot_key[row] = __fmaf_rn(state[row].w, key.w, dot);
      }
      if constexpr (PAIRED_REDUCTIONS) {
        const float2 state_dot_key01 =
            gdn_spec_warp_sum_pair(state_dot_key[0], state_dot_key[1]);
        const float2 state_dot_key23 =
            gdn_spec_warp_sum_pair(state_dot_key[2], state_dot_key[3]);
        state_dot_key[0] = state_dot_key01.x;
        state_dot_key[1] = state_dot_key01.y;
        state_dot_key[2] = state_dot_key23.x;
        state_dot_key[3] = state_dot_key23.y;

        float state_dot_query[VALUES_PER_WARP];
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          const int value =
              chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
          const float delta =
              ((float)shared_value[position][value] - state_dot_key[row]) *
              shared_beta[position];
          state[row].x = __fmaf_rn(key.x, delta, state[row].x);
          state[row].y = __fmaf_rn(key.y, delta, state[row].y);
          state[row].z = __fmaf_rn(key.z, delta, state[row].z);
          state[row].w = __fmaf_rn(key.w, delta, state[row].w);
          if (lane == 0) {
            if constexpr (DIRECT_TRANSITIONS) {
              pending_delta[(((size_t)active_slot * max_pending_rows +
                              position) *
                                 num_v_heads +
                             value_head) *
                                V +
                            value] = delta;
            } else if (transition_delta != nullptr) {
              transition_delta[((transition_row_base + position) *
                                    num_v_heads +
                                value_head) *
                                   V +
                               value] = delta;
            }
          }
          float dot = state[row].x * query.x;
          dot = __fmaf_rn(state[row].y, query.y, dot);
          dot = __fmaf_rn(state[row].z, query.z, dot);
          state_dot_query[row] = __fmaf_rn(state[row].w, query.w, dot);
        }
        const float2 state_dot_query01 =
            gdn_spec_warp_sum_pair(state_dot_query[0], state_dot_query[1]);
        const float2 state_dot_query23 =
            gdn_spec_warp_sum_pair(state_dot_query[2], state_dot_query[3]);
        state_dot_query[0] = state_dot_query01.x;
        state_dot_query[1] = state_dot_query01.y;
        state_dot_query[2] = state_dot_query23.x;
        state_dot_query[3] = state_dot_query23.y;
        if (lane == 0) {
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value =
                chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row];
            shared_output[position][value] = (T)state_dot_query[row];
          }
        }
      } else {
#pragma unroll
        for (int row = 0; row < VALUES_PER_WARP; row++) {
          state_dot_key[row] = gdn_warp_sum<32>(state_dot_key[row]);
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row];
          const float delta =
              ((float)shared_value[position][value] - state_dot_key[row]) *
              shared_beta[position];
          state[row].x = __fmaf_rn(key.x, delta, state[row].x);
          state[row].y = __fmaf_rn(key.y, delta, state[row].y);
          state[row].z = __fmaf_rn(key.z, delta, state[row].z);
          state[row].w = __fmaf_rn(key.w, delta, state[row].w);
          if (lane == 0) {
            if constexpr (DIRECT_TRANSITIONS) {
              pending_delta[(((size_t)active_slot * max_pending_rows +
                              position) *
                                 num_v_heads +
                             value_head) *
                                V +
                            value] = delta;
            } else if (transition_delta != nullptr) {
              transition_delta[((transition_row_base + position) *
                                    num_v_heads +
                                value_head) *
                                   V +
                               value] = delta;
            }
          }
          float dot = state[row].x * query.x;
          dot = __fmaf_rn(state[row].y, query.y, dot);
          dot = __fmaf_rn(state[row].z, query.z, dot);
          const float state_dot_query =
              gdn_warp_sum<32>(__fmaf_rn(state[row].w, query.w, dot));
          if (lane == 0) {
            shared_output[position][value] = (T)state_dot_query;
          }
        }
      }

      if constexpr (!DIRECT_TRANSITIONS) {
        if (transition_delta == nullptr) {
          StateT *destination =
              state_pool +
              (((base_slot + position) * num_v_heads + value_head) *
               state_head_elements);
#pragma unroll
          for (int row = 0; row < VALUES_PER_WARP; row++) {
            const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                              value_rows[row];
            gdn_store_state_x4(destination + (size_t)value * K + key_base,
                               state[row]);
          }
        }
      }
    }
  }
  __syncthreads();

  if constexpr (DIRECT_TRANSITIONS) {
    if (tid == 0 && apply_pending) {
      recurrent_applied_epochs[(size_t)active_slot * num_v_heads +
                               value_head] = pending_epoch;
    }
  }

  if (warp < seq_len) {
    const int position = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      output_values[i] = (float)shared_output[position][value];
      sum_square =
          __fmaf_rn(output_values[i], output_values[i], sum_square);
    }
    sum_square = gdn_warp_sum<32>(sum_square);
    const float rstd = rsqrtf(sum_square / (float)V + norm_eps);
    T rounded[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      const size_t gate_offset =
          (size_t)batch_idx * gate_stride_b +
          (size_t)position * gate_stride_s +
          (size_t)value_head * gate_stride_h +
          (size_t)value * gate_stride_v;
      const float gate_value = (float)gate[gate_offset];
      const float gate_act = gdn_gate_act(gate_value, sigmoid_gate);
      const size_t output_offset =
          (((size_t)batch_idx * seq_len + position) * num_v_heads +
           value_head) *
              V +
          value;
      rounded[i] = (T)(output_values[i] * rstd * (float)norm_weight[value] *
                       gate_act);
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
      if (quantized_output == nullptr) {
        output[output_offset] = rounded[i];
      }
    }
    if (quantized_output != nullptr) {
      maximum = gdn_warp_max(maximum);
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                           inverse_quant_scale);
      if (lane == 0) {
        const int normalized_row =
            ((batch_idx * seq_len + position) * num_v_heads) + value_head;
        output_scales[gdn_fp8_scale_offset(normalized_row, num_v_heads,
                                           scale_stride_m)] =
            inverse_quant_scale;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        const size_t output_offset =
            (((size_t)batch_idx * seq_len + position) * num_v_heads +
             value_head) *
                V +
            value;
        quantized_output[output_offset] =
            gdn_fp8_quantize((float)rounded[i], quant_scale,
                             inverse_quant_scale, scale_layout);
      }
    }
  }
}

GDN_SPEC_LAUNCH(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_deferred_recurrence_rmsnorm_gate_value_major_128_kernel(
        const __nv_bfloat16 *__restrict__ mixed_qkv,
        const __nv_bfloat16 *__restrict__ b,
        const __nv_bfloat16 *__restrict__ a,
        const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, float *__restrict__ state_pool,
        __nv_bfloat16 *__restrict__ output,
        gdn_fp8_e4m3 *__restrict__ quantized_output,
        float *__restrict__ output_scales, int scale_stride_m,
        int scale_layout,
        const uint32_t *__restrict__ active_slots,
        float *__restrict__ deferred_key,
        float *__restrict__ deferred_delta,
        float *__restrict__ deferred_decay,
        const uint32_t *__restrict__ deferred_cursor,
        const __nv_bfloat16 *__restrict__ gate,
        const __nv_bfloat16 *__restrict__ norm_weight,
        int64_t b_stride_b, int64_t b_stride_h, int64_t a_stride_b,
        int64_t a_stride_h, int64_t gate_stride_b,
        int64_t gate_stride_h, int64_t gate_stride_v, int batch_size,
        int capacity, int num_k_heads, int num_v_heads,
        int tiled_v_heads, float norm_eps, int sigmoid_gate) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  const bool valid_slot = active_slot != GDN_SPEC_CHECKPOINT_PAD_SLOT && active_slot < (uint32_t)capacity;
  const bool invalid_cursor = valid_slot && deferred_cursor[active_slot] >= GDN_DEFERRED_STATE_DEPTH;
  if (!valid_slot || invalid_cursor) {
    const float invalid_value = invalid_cursor ? NAN : 0.0f;
    for (int value = tid; value < V; value += GDN_SPEC_FUSED_THREADS) {
      const size_t offset =
          ((size_t)batch_idx * num_v_heads + value_head) * V + value;
      if (quantized_output != nullptr) {
        quantized_output[offset] = gdn_fp8_e4m3(invalid_value);
      } else {
        output[offset] = (__nv_bfloat16)invalid_value;
      }
    }
    if (tid == 0 && quantized_output != nullptr) {
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(0.0f, scale_layout, quant_scale,
                           inverse_quant_scale);
      (void)quant_scale;
      output_scales[(size_t)value_head * scale_stride_m + batch_idx] =
          invalid_cursor ? NAN : inverse_quant_scale;
    }
    return;
  }

  const uint32_t deferred_rows = deferred_cursor[active_slot];
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const bool key_leader = tiled_v_heads
                              ? value_head < num_k_heads
                              : value_head % values_per_group == 0;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const __nv_bfloat16 *row = mixed_qkv + (size_t)batch_idx * conv_dim;
  float *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) * V * K;

  __shared__ float shared_query[K];
  __shared__ float shared_key[K];
  __shared__ float shared_pending_key[GDN_DEFERRED_STATE_DEPTH][K];
  __shared__ float shared_pending_decay[GDN_DEFERRED_STATE_DEPTH];
  __shared__ __nv_bfloat16 shared_output[V];
  __shared__ float shared_beta;
  __shared__ float shared_decay;

  for (int linear = tid; linear < (int)deferred_rows * K;
       linear += GDN_SPEC_FUSED_THREADS) {
    const int position = linear / K;
    const int key = linear - position * K;
    shared_pending_key[position][key] =
        deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                       position) *
                          num_k_heads +
                      key_head) *
                         K +
                     key];
  }
  for (int position = tid; position < (int)deferred_rows;
       position += GDN_SPEC_FUSED_THREADS) {
    shared_pending_decay[position] =
        deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                        position) *
                           num_v_heads +
                       value_head];
  }

  if (warp == 0) {
    float4 query = gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm =
        key.x * key.x + key.y * key.y + key.z * key.z + key.w * key.w;
    query_norm = gdn_warp_sum<32>(query_norm);
    key_norm = gdn_warp_sum<32>(key_norm);
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const float query_values[4] = {query.x, query.y, query.z, query.w};
    const float key_values[4] = {key.x, key.y, key.z, key.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int key_idx = lane * 4 + i;
      shared_query[key_idx] = query_values[i] * query_multiplier;
      shared_key[key_idx] = key_values[i] * key_multiplier;
    }
    if (lane == 0) {
      const float b_value =
          (float)b[(size_t)batch_idx * b_stride_b +
                   value_head * b_stride_h];
      const float a_value =
          (float)a[(size_t)batch_idx * a_stride_b +
                   value_head * a_stride_h] +
          dt_bias[value_head];
      shared_beta = 1.0f / (1.0f + expf(-b_value));
      shared_decay =
          expf(-expf(a_log[value_head]) * gdn_spec_softplus(a_value));
    }
  }
  __syncthreads();

  if (key_leader) {
    for (int key = tid; key < K; key += GDN_SPEC_FUSED_THREADS) {
      deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                     deferred_rows) *
                        num_k_heads +
                    key_head) *
                       K +
                   key] = shared_key[key];
    }
  }
  if (tid == 0) {
    deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                    deferred_rows) *
                       num_v_heads +
                   value_head] = shared_decay;
  }

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
    value_rows[row_idx] = warp + row_idx * GDN_SPEC_FUSED_WARPS;
  }

#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      state[row_idx] =
          gdn_load_state_x4(source + (size_t)value * K + key_base);
    }

#pragma unroll
    for (int position = 0; position < GDN_DEFERRED_STATE_DEPTH - 1;
         position++) {
      if (position < deferred_rows) {
        const float4 key = *reinterpret_cast<const float4 *>(
            &shared_pending_key[position][key_base]);
        const float decay = shared_pending_decay[position];
#pragma unroll
        for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row_idx];
          float delta = lane == 0
                            ? deferred_delta
                                  [(((size_t)active_slot *
                                         GDN_DEFERRED_STATE_DEPTH +
                                     position) *
                                        num_v_heads +
                                    value_head) *
                                       V +
                                   value]
                            : 0.0f;
          delta = __shfl_sync(0xffffffff, delta, 0);
          state[row_idx].x =
              __fmaf_rn(key.x, delta, state[row_idx].x * decay);
          state[row_idx].y =
              __fmaf_rn(key.y, delta, state[row_idx].y * decay);
          state[row_idx].z =
              __fmaf_rn(key.z, delta, state[row_idx].z * decay);
          state[row_idx].w =
              __fmaf_rn(key.w, delta, state[row_idx].w * decay);
        }
      }
    }

    const float4 query =
        *reinterpret_cast<const float4 *>(&shared_query[key_base]);
    const float4 key =
        *reinterpret_cast<const float4 *>(&shared_key[key_base]);
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      state[row_idx].x *= shared_decay;
      state[row_idx].y *= shared_decay;
      state[row_idx].z *= shared_decay;
      state[row_idx].w *= shared_decay;
      float state_dot_key = state[row_idx].x * key.x;
      state_dot_key =
          __fmaf_rn(state[row_idx].y, key.y, state_dot_key);
      state_dot_key =
          __fmaf_rn(state[row_idx].z, key.z, state_dot_key);
      state_dot_key = gdn_warp_sum<32>(
          __fmaf_rn(state[row_idx].w, key.w, state_dot_key));
      float value_input =
          lane == 0
              ? (float)row[2 * key_dim + value_head * V + value]
              : 0.0f;
      value_input = __shfl_sync(0xffffffff, value_input, 0);
      const float delta = (value_input - state_dot_key) * shared_beta;
      state[row_idx].x = __fmaf_rn(key.x, delta, state[row_idx].x);
      state[row_idx].y = __fmaf_rn(key.y, delta, state[row_idx].y);
      state[row_idx].z = __fmaf_rn(key.z, delta, state[row_idx].z);
      state[row_idx].w = __fmaf_rn(key.w, delta, state[row_idx].w);
      if (lane == 0) {
        deferred_delta[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                         deferred_rows) *
                            num_v_heads +
                        value_head) *
                           V +
                       value] = delta;
      }
      float state_dot_query = state[row_idx].x * query.x;
      state_dot_query =
          __fmaf_rn(state[row_idx].y, query.y, state_dot_query);
      state_dot_query =
          __fmaf_rn(state[row_idx].z, query.z, state_dot_query);
      state_dot_query = gdn_warp_sum<32>(
          __fmaf_rn(state[row_idx].w, query.w, state_dot_query));
      if (lane == 0) {
        shared_output[value] = (__nv_bfloat16)state_dot_query;
      }
    }

    if (deferred_rows == GDN_DEFERRED_STATE_DEPTH - 1) {
#pragma unroll
      for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
        const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                          value_rows[row_idx];
        gdn_store_state_x4(source + (size_t)value * K + key_base,
                           state[row_idx]);
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      output_values[i] = (float)shared_output[value];
      sum_square =
          __fmaf_rn(output_values[i], output_values[i], sum_square);
    }
    sum_square = gdn_warp_sum<32>(sum_square);
    const float rstd = rsqrtf(sum_square / (float)V + norm_eps);
    __nv_bfloat16 rounded[4];
    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int value = lane + i * 32;
      const size_t gate_offset =
          (size_t)batch_idx * gate_stride_b +
          (size_t)value_head * gate_stride_h +
          (size_t)value * gate_stride_v;
      rounded[i] = __float2bfloat16_rn(
          output_values[i] * rstd * (float)norm_weight[value] *
          gdn_gate_act((float)gate[gate_offset], sigmoid_gate));
      maximum = fmaxf(maximum, fabsf((float)rounded[i]));
    }
    if (quantized_output != nullptr) {
      maximum = gdn_warp_max(maximum);
      float quant_scale;
      float inverse_quant_scale;
      gdn_fp8_quant_params(maximum, scale_layout, quant_scale,
                           inverse_quant_scale);
      if (lane == 0) {
        output_scales[(size_t)value_head * scale_stride_m + batch_idx] =
            inverse_quant_scale;
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        quantized_output[((size_t)batch_idx * num_v_heads + value_head) * V +
                         value] =
            gdn_fp8_quantize((float)rounded[i], quant_scale,
                             inverse_quant_scale, scale_layout);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int value = lane + i * 32;
        output[((size_t)batch_idx * num_v_heads + value_head) * V + value] =
            rounded[i];
      }
    }
  }
}

GDN_SPEC_ENTRY void gdn_deferred_cursor_advance_kernel(
    uint32_t *__restrict__ deferred_cursor,
    const uint32_t *__restrict__ active_slots, int batch_size,
    int capacity) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || active_slot >= capacity) {
    return;
  }
  const uint32_t cursor = deferred_cursor[active_slot];
  if (cursor >= GDN_DEFERRED_STATE_DEPTH) return;
  deferred_cursor[active_slot] =
      cursor == GDN_DEFERRED_STATE_DEPTH - 1 ? 0 : cursor + 1;
}

GDN_SPEC_LAUNCH(GDN_SPEC_FUSED_THREADS, 2)
    void gdn_flush_deferred_state_value_major_128_kernel(
        float *__restrict__ state_pool,
        const uint32_t *__restrict__ active_slots,
        const float *__restrict__ deferred_key,
        const float *__restrict__ deferred_delta,
        const float *__restrict__ deferred_decay,
        const uint32_t *__restrict__ deferred_cursor, int batch_size,
        int capacity, int num_k_heads, int num_v_heads,
        int tiled_v_heads) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_FUSED_VALUES_PER_WARP;
  const int batch_idx = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT || active_slot >= capacity) {
    return;
  }
  const uint32_t deferred_rows = deferred_cursor[active_slot];
  if (deferred_rows == 0 || deferred_rows >= GDN_DEFERRED_STATE_DEPTH) {
    return;
  }
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  float *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) * V * K;
  __shared__ __align__(16) float shared_state[GDN_SPEC_FUSED_VALUE_CHUNK][K];
  __shared__ float shared_pending_key[GDN_DEFERRED_STATE_DEPTH][K];
  __shared__ float shared_pending_decay[GDN_DEFERRED_STATE_DEPTH];

  gdn_spec_copy_state_chunk(&shared_state[0][0], source, 0, tid);
  for (int linear = tid; linear < (int)deferred_rows * K;
       linear += GDN_SPEC_FUSED_THREADS) {
    const int position = linear / K;
    const int key = linear - position * K;
    shared_pending_key[position][key] =
        deferred_key[(((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                       position) *
                          num_k_heads +
                      key_head) *
                         K +
                     key];
  }
  for (int position = tid; position < (int)deferred_rows;
       position += GDN_SPEC_FUSED_THREADS) {
    shared_pending_decay[position] =
        deferred_decay[((size_t)active_slot * GDN_DEFERRED_STATE_DEPTH +
                        position) *
                           num_v_heads +
                       value_head];
  }
  __syncthreads();

  const int key_base = lane * 4;
  int value_rows[VALUES_PER_WARP];
#pragma unroll
  for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
    value_rows[row_idx] = warp + row_idx * GDN_SPEC_FUSED_WARPS;
  }
#pragma unroll
  for (int chunk = 0; chunk < GDN_SPEC_FUSED_VALUE_CHUNKS; chunk++) {
    gdn_cp_async_wait();
    __syncthreads();
    float4 state[VALUES_PER_WARP];
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      state[row_idx] =
          gdn_load_state_x4(&shared_state[value_rows[row_idx]][key_base]);
    }
    __syncthreads();
    if (chunk + 1 < GDN_SPEC_FUSED_VALUE_CHUNKS) {
      gdn_spec_copy_state_chunk(&shared_state[0][0], source, chunk + 1, tid);
    }
#pragma unroll
    for (int position = 0; position < GDN_DEFERRED_STATE_DEPTH; position++) {
      if (position < deferred_rows) {
        const float4 key = *reinterpret_cast<const float4 *>(
            &shared_pending_key[position][key_base]);
        const float decay = shared_pending_decay[position];
#pragma unroll
        for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
          const int value = chunk * GDN_SPEC_FUSED_VALUE_CHUNK +
                            value_rows[row_idx];
          float delta = lane == 0
                            ? deferred_delta
                                  [(((size_t)active_slot *
                                         GDN_DEFERRED_STATE_DEPTH +
                                     position) *
                                        num_v_heads +
                                    value_head) *
                                       V +
                                   value]
                            : 0.0f;
          delta = __shfl_sync(0xffffffff, delta, 0);
          state[row_idx].x =
              __fmaf_rn(key.x, delta, state[row_idx].x * decay);
          state[row_idx].y =
              __fmaf_rn(key.y, delta, state[row_idx].y * decay);
          state[row_idx].z =
              __fmaf_rn(key.z, delta, state[row_idx].z * decay);
          state[row_idx].w =
              __fmaf_rn(key.w, delta, state[row_idx].w * decay);
        }
      }
    }
#pragma unroll
    for (int row_idx = 0; row_idx < VALUES_PER_WARP; row_idx++) {
      const int value =
          chunk * GDN_SPEC_FUSED_VALUE_CHUNK + value_rows[row_idx];
      gdn_store_state_x4(source + (size_t)value * K + key_base,
                         state[row_idx]);
    }
  }
}

GDN_SPEC_ENTRY void gdn_deferred_cursor_clear_kernel(
    uint32_t *__restrict__ deferred_cursor,
    const uint32_t *__restrict__ active_slots, int batch_size,
    int capacity) {
  const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot != GDN_SPEC_CHECKPOINT_PAD_SLOT && active_slot < capacity &&
      deferred_cursor[active_slot] < GDN_DEFERRED_STATE_DEPTH) {
    deferred_cursor[active_slot] = 0;
  }
}

template <typename T, typename StateT>
GDN_SPEC_WARP(32 * GDN_SPEC_CHECKPOINT_WARPS)
    void gdn_speculative_recurrence_checkpoints_value_major_128_kernel(
        const T *__restrict__ mixed_qkv, const T *__restrict__ b,
        const T *__restrict__ a, const float *__restrict__ a_log,
        const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
        T *__restrict__ output, const uint32_t *__restrict__ active_slots,
        int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
        int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
        int batch_size, int seq_len, int num_k_heads, int num_v_heads,
        int checkpoint_lanes, int tiled_v_heads, int pool_capacity) {
  constexpr int K = GDN_DECODE_VALUE_MAJOR_K;
  constexpr int V = GDN_DECODE_VALUE_MAJOR_V;
  constexpr int VALUES_PER_WARP = GDN_SPEC_CHECKPOINT_VALUES_PER_WARP;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_base = blockIdx.x * GDN_SPEC_CHECKPOINT_VALUE_TILE +
                         warp * VALUES_PER_WARP;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (pool_capacity > 0 && active_slot >= (uint32_t)pool_capacity)) {
    if (lane < VALUES_PER_WARP) {
      const int value_idx = value_base + lane;
      for (int position = 0; position < seq_len; position++) {
        output[((size_t)batch_head * seq_len + position) * V + value_idx] =
            (T)0.0f;
      }
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * K;
  const int value_dim = num_v_heads * V;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)V * K;
  const StateT *source =
      state_pool + ((size_t)active_slot * num_v_heads + value_head) *
                       state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);

  float4 state[VALUES_PER_WARP];
#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
    state[value] = gdn_load_state_x4(
        source + (size_t)(value_base + value) * K + lane * 4);
  }

  for (int position = 0; position < seq_len; position++) {
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float4 query =
        gdn_load_state_x4(row + key_head * K + lane * 4);
    float4 key =
        gdn_load_state_x4(row + key_dim + key_head * K + lane * 4);
    float query_norm = query.x * query.x + query.y * query.y +
                       query.z * query.z + query.w * query.w;
    float key_norm =
        key.x * key.x + key.y * key.y + key.z * key.z + key.w * key.w;
    query_norm = gdn_warp_sum<32>(query_norm);
    key_norm = gdn_warp_sum<32>(key_norm);
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)K);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    query = make_float4(query.x * query_multiplier,
                        query.y * query_multiplier,
                        query.z * query_multiplier,
                        query.w * query_multiplier);
    key = make_float4(key.x * key_multiplier, key.y * key_multiplier,
                      key.z * key_multiplier, key.w * key_multiplier);

    float beta = 0.0f;
    float decay = 0.0f;
    if (lane == 0) {
      const size_t b_offset = (size_t)batch_idx * b_stride_b +
                              (size_t)position * b_stride_s +
                              (size_t)value_head * b_stride_h;
      const size_t a_offset = (size_t)batch_idx * a_stride_b +
                              (size_t)position * a_stride_s +
                              (size_t)value_head * a_stride_h;
      beta = 1.0f / (1.0f + expf(-(float)b[b_offset]));
      const float biased_a = (float)a[a_offset] + dt_bias[value_head];
      decay = expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));
    }
    beta = __shfl_sync(0xffffffff, beta, 0);
    decay = __shfl_sync(0xffffffff, decay, 0);

#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      float4 next = make_float4(state[value].x * decay,
                                state[value].y * decay,
                                state[value].z * decay,
                                state[value].w * decay);
      float state_dot_key = next.x * key.x;
      state_dot_key = __fmaf_rn(next.y, key.y, state_dot_key);
      state_dot_key = __fmaf_rn(next.z, key.z, state_dot_key);
      state_dot_key = __fmaf_rn(next.w, key.w, state_dot_key);
      state_dot_key = gdn_warp_sum<32>(state_dot_key);
      float value_input =
          lane == 0
              ? (float)row[2 * key_dim + value_head * V + value_base + value]
              : 0.0f;
      value_input = __shfl_sync(0xffffffff, value_input, 0);
      const float delta = (value_input - state_dot_key) * beta;
      next.x = __fmaf_rn(key.x, delta, next.x);
      next.y = __fmaf_rn(key.y, delta, next.y);
      next.z = __fmaf_rn(key.z, delta, next.z);
      next.w = __fmaf_rn(key.w, delta, next.w);
      state[value] = next;

      float state_dot_query = next.x * query.x;
      state_dot_query = __fmaf_rn(next.y, query.y, state_dot_query);
      state_dot_query = __fmaf_rn(next.z, query.z, state_dot_query);
      state_dot_query = __fmaf_rn(next.w, query.w, state_dot_query);
      state_dot_query = gdn_warp_sum<32>(state_dot_query);
      if (lane == 0) {
        output[((size_t)batch_head * seq_len + position) * V + value_base +
               value] = (T)state_dot_query;
      }
    }

    StateT *destination =
        state_pool +
        (((base_slot + position) * num_v_heads + value_head) *
         state_head_elements);
#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      gdn_store_state_x4(
          destination + (size_t)(value_base + value) * K + lane * 4,
          state[value]);
    }
  }
}

template <typename T, typename StateT, bool VALUE_MAJOR>
GDN_SPEC_ENTRY void gdn_speculative_recurrence_checkpoints_fallback_kernel(
    const T *__restrict__ mixed_qkv, const T *__restrict__ b,
    const T *__restrict__ a, const float *__restrict__ a_log,
    const float *__restrict__ dt_bias, StateT *__restrict__ state_pool,
    T *__restrict__ output, const uint32_t *__restrict__ active_slots,
    int64_t b_stride_b, int64_t b_stride_s, int64_t b_stride_h,
    int64_t a_stride_b, int64_t a_stride_s, int64_t a_stride_h,
    int batch_size, int seq_len, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int checkpoint_lanes,
    int tiled_v_heads, int pool_capacity) {
  const int value_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_head = blockIdx.y;
  const int batch_idx = batch_head / num_v_heads;
  const int value_head = batch_head - batch_idx * num_v_heads;
  if (batch_idx >= batch_size || value_idx >= head_v_dim) {
    return;
  }

  const uint32_t active_slot = active_slots[batch_idx];
  if (active_slot == GDN_SPEC_CHECKPOINT_PAD_SLOT ||
      (pool_capacity > 0 && active_slot >= (uint32_t)pool_capacity)) {
    for (int position = 0; position < seq_len; position++) {
      output[((size_t)batch_head * seq_len + position) * head_v_dim +
             value_idx] = (T)0.0f;
    }
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads
                           ? value_head % num_k_heads
                           : value_head / values_per_group;
  const int key_dim = num_k_heads * head_k_dim;
  const int value_dim = num_v_heads * head_v_dim;
  const int conv_dim = 2 * key_dim + value_dim;
  const size_t state_head_elements = (size_t)head_k_dim * head_v_dim;
  const size_t source_head =
      ((size_t)active_slot * num_v_heads + value_head) * state_head_elements;
  const size_t base_slot =
      gdn_spec_checkpoint_base(active_slot, checkpoint_lanes);
  float state[GDN_SPEC_CHECKPOINT_MAX_K];
  for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
    const size_t offset = VALUE_MAJOR
                              ? (size_t)value_idx * head_k_dim + key_idx
                              : (size_t)key_idx * head_v_dim + value_idx;
    state[key_idx] = state_pool[source_head + offset];
  }

  for (int position = 0; position < seq_len; position++) {
    const T *row =
        mixed_qkv + ((size_t)batch_idx * seq_len + position) * conv_dim;
    float query_norm = 0.0f;
    float key_norm = 0.0f;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      const float query = (float)row[key_head * head_k_dim + key_idx];
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx];
      query_norm = __fmaf_rn(query, query, query_norm);
      key_norm = __fmaf_rn(key, key, key_norm);
    }
    const float query_multiplier =
        rsqrtf(query_norm + 1.0e-6f) * rsqrtf((float)head_k_dim);
    const float key_multiplier = rsqrtf(key_norm + 1.0e-6f);
    const size_t b_offset = (size_t)batch_idx * b_stride_b +
                            (size_t)position * b_stride_s +
                            (size_t)value_head * b_stride_h;
    const size_t a_offset = (size_t)batch_idx * a_stride_b +
                            (size_t)position * a_stride_s +
                            (size_t)value_head * a_stride_h;
    const float beta = 1.0f / (1.0f + expf(-(float)b[b_offset]));
    const float biased_a = (float)a[a_offset] + dt_bias[value_head];
    const float decay =
        expf(-expf(a_log[value_head]) * gdn_spec_softplus(biased_a));

    float state_dot_key = 0.0f;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      state[key_idx] *= decay;
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx] *
          key_multiplier;
      state_dot_key = __fmaf_rn(state[key_idx], key, state_dot_key);
    }
    const float value_input =
        (float)row[2 * key_dim + value_head * head_v_dim + value_idx];
    const float delta = (value_input - state_dot_key) * beta;
    float state_dot_query = 0.0f;
    const size_t destination_head =
        ((base_slot + position) * num_v_heads + value_head) *
        state_head_elements;
    for (int key_idx = 0; key_idx < head_k_dim; key_idx++) {
      const float key =
          (float)row[key_dim + key_head * head_k_dim + key_idx] *
          key_multiplier;
      state[key_idx] = __fmaf_rn(key, delta, state[key_idx]);
      const float query =
          (float)row[key_head * head_k_dim + key_idx] * query_multiplier;
      state_dot_query = __fmaf_rn(state[key_idx], query, state_dot_query);
      const size_t offset = VALUE_MAJOR
                                ? (size_t)value_idx * head_k_dim + key_idx
                                : (size_t)key_idx * head_v_dim + value_idx;
      state_pool[destination_head + offset] = state[key_idx];
    }
    output[((size_t)batch_head * seq_len + position) * head_v_dim +
           value_idx] = (T)state_dot_query;
  }
}

#undef GDN_SPEC_ENTRY
#undef GDN_SPEC_LAUNCH
#undef GDN_SPEC_WARP
