#pragma once
#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#ifndef DKR_GDN_ENTRY
#define DKR_GDN_ENTRY __global__
#define DKR_GDN_WARP_ENTRY(N) __global__ __launch_bounds__(32 * N, 2)
#endif

constexpr int GDN_PACKED_CONV_WIDTH = 4;

__device__ __forceinline__ bool
gdn_is_padding_row(const int32_t *__restrict__ slot_indices, int bidx) {
  return slot_indices && slot_indices[bidx] < 0;
}

__device__ __forceinline__ size_t gdn_state_row(const int32_t *__restrict__ slot_indices,
                                                int bidx, int h, int num_heads) {
  const size_t slot = slot_indices ? (size_t)slot_indices[bidx] : (size_t)bidx;
  return slot * num_heads + h;
}

template <typename StateT, int BK, int BV>
DKR_GDN_ENTRY void gated_delta_rule_recurrence_kernel_tiled(
    const float *__restrict__ q,    // [BH, S, K]
    const float *__restrict__ k,    // [BH, S, K]
    const float *__restrict__ v,    // [BH, S, V]
    const float *__restrict__ g,    // [BH, S]
    const float *__restrict__ beta, // [BH, S]
    StateT *__restrict__ state,     // [BH, K, V] or the pool with slot_indices
    float *__restrict__ output,     // [BH, S, V]
    int seq_len, int v_dim, const int32_t *__restrict__ slot_indices,
    int num_heads) {

  const int v_tile = blockIdx.x;       // which V-tile
  const int bh = blockIdx.y;           // batch*head index
  const int tid = threadIdx.x;         // thread within tile [0, BV)
  const int v_idx = v_tile * BV + tid; // global V index

  // All threads must participate in cooperative loads and block barriers.
  const bool active = v_idx < v_dim;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      if (active) out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Shared memory: k_buf[BK] + q_buf[BK]
  __shared__ float k_buf[BK];
  __shared__ float q_buf[BK];

  // Load state column into registers — BK is compile-time, so this is
  // a true register array (not spilled to local memory)
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    s[j] = active ? (float)state_bh[j * v_dim + v_idx] : 0.0f;
  }

  for (int t = 0; t < seq_len; t++) {
// Collaboratively load k_t into shared memory
// BK / BV loads per thread (e.g. 128/64 = 2)
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      k_buf[j] = k_bh[t * BK + j];
    }
    __syncthreads();

    // Load scalars for this timestep
    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = active ? v_bh[t * v_dim + v_idx] : 0.0f;

    // Fused pass 1: decay state + compute kv_mem
    float kv_mem = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    // Delta rule
    float delta = (v_t - kv_mem) * beta_t;

// Collaboratively load q_t into shared memory
#pragma unroll
    for (int j = tid; j < BK; j += BV) {
      q_buf[j] = q_bh[t * BK + j];
    }
    __syncthreads();

    // Fused pass 2: update state + compute output
    float y_t = 0.0f;
#pragma unroll
    for (int j = 0; j < BK; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    if (active) out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

// Write state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if (active) state_bh[j * v_dim + v_idx] = s[j];
  }
}

template <typename StateT, int BV, int MAX_K>
DKR_GDN_ENTRY void gated_delta_rule_recurrence_kernel_fallback(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len, int k_dim, int v_dim,
    const int32_t *__restrict__ slot_indices, int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  // All threads must participate in cooperative loads and block barriers.
  const bool active = v_idx < v_dim;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      if (active) out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * k_dim;
  const float *k_bh = k + (size_t)bh * seq_len * k_dim;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * k_dim * v_dim;

  extern __shared__ float shared[];
  float *k_buf = shared;
  float *q_buf = shared + k_dim;

  float s[MAX_K];
  for (int j = 0; j < k_dim; j++) {
    s[j] = active ? (float)state_bh[j * v_dim + v_idx] : 0.0f;
  }

  for (int t = 0; t < seq_len; t++) {
    for (int j = tid; j < k_dim; j += BV) {
      k_buf[j] = k_bh[t * k_dim + j];
    }
    __syncthreads();

    float decay = expf(g_bh[t]);
    float beta_t = beta_bh[t];
    float v_t = active ? v_bh[t * v_dim + v_idx] : 0.0f;

    float kv_mem = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] *= decay;
      kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);
    }

    float delta = (v_t - kv_mem) * beta_t;

    for (int j = tid; j < k_dim; j += BV) {
      q_buf[j] = q_bh[t * k_dim + j];
    }
    __syncthreads();

    float y_t = 0.0f;
    for (int j = 0; j < k_dim; j++) {
      s[j] = __fmaf_rn(k_buf[j], delta, s[j]);
      y_t = __fmaf_rn(s[j], q_buf[j], y_t);
    }

    if (active) out_bh[t * v_dim + v_idx] = y_t;

    __syncthreads();
  }

  for (int j = 0; j < k_dim; j++) {
    if (active) state_bh[j * v_dim + v_idx] = s[j];
  }
}

template <int WARP_SIZE>
__device__ __forceinline__ float gdn_warp_sum(float x) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffff, x, offset, WARP_SIZE);
  }
  return __shfl_sync(0xffffffff, x, 0, WARP_SIZE);
}

template <typename StateT, int BK, int NUM_WARPS, bool VALUE_MAJOR = false>
DKR_GDN_WARP_ENTRY(NUM_WARPS) void gated_delta_rule_recurrence_kernel_warp(const float *__restrict__ q,
                                                    const float *__restrict__ k,
                                                    const float *__restrict__ v,
                                                    const float *__restrict__ g,
                                                    const float
                                                        *__restrict__ beta,
                                                    StateT *__restrict__ state,
                                                    float *__restrict__ output,
                                                    int seq_len, int v_dim,
                                                    const int32_t *__restrict__ slot_indices,
                                                    int num_heads) {

  constexpr int WARP_SIZE = 32;
  static_assert(BK % WARP_SIZE == 0, "BK must be a multiple of warp size");
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int v_idx = blockIdx.x * NUM_WARPS + warp;
  const int bh = blockIdx.y;

  if (v_idx >= v_dim) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
      for (int t = 0; t < seq_len; t++) {
        out_bh[t * v_dim + v_idx] = 0.0f;
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  float s[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      s[r] = state_bh[v_idx * BK + row];
    } else {
      s[r] = state_bh[row * v_dim + v_idx];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;

    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      const int row = r * WARP_SIZE + lane;
      const float k_val = k_t[row];
      k_reg[r] = k_val;
      q_reg[r] = q_t[row];
      kv_partial = __fmaf_rn(s[r], k_val, kv_partial);
    }

    const float decay = expf(g_bh[t]);
    const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial);
    const float delta = (v_bh[t * v_dim + v_idx] - decay * kv_col) * beta_bh[t];

    float y_partial = 0.0f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; r++) {
      s[r] = __fmaf_rn(k_reg[r], delta, decay * s[r]);
      y_partial = __fmaf_rn(s[r], q_reg[r], y_partial);
    }

    const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
    if (lane == 0) {
      out_bh[t * v_dim + v_idx] = y_col;
    }
  }

#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; r++) {
    const int row = r * WARP_SIZE + lane;
    if constexpr (VALUE_MAJOR) {
      state_bh[v_idx * BK + row] = s[r];
    } else {
      state_bh[row * v_dim + v_idx] = s[r];
    }
  }
}

template <typename StateT, int VALUES_PER_WARP, int NUM_WARPS>
DKR_GDN_WARP_ENTRY(NUM_WARPS) void
gated_delta_rule_recurrence_kernel_vmajor_grouped(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, const float *__restrict__ g,
    const float *__restrict__ beta, StateT *__restrict__ state,
    float *__restrict__ output, int seq_len,
    const int32_t *__restrict__ slot_indices, int num_heads) {
  constexpr int WARP_SIZE = 32;
  constexpr int BK = 128;
  constexpr int V_DIM = 128;
  constexpr int ROWS_PER_LANE = BK / WARP_SIZE;

  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const int value_group = blockIdx.x * NUM_WARPS + warp;
  const int first_value = value_group * VALUES_PER_WARP;
  const int bh = blockIdx.y;

  if (first_value >= V_DIM) {
    return;
  }

  float *out_bh = output + (size_t)bh * seq_len * V_DIM;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    if (lane == 0) {
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        for (int t = 0; t < seq_len; t++) {
          out_bh[t * V_DIM + first_value + value] = 0.0f;
        }
      }
    }
    return;
  }

  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * V_DIM;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads,
                            num_heads) *
                  BK * V_DIM;

  float s[VALUES_PER_WARP][ROWS_PER_LANE];
#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      s[value][row] = state_bh[(first_value + value) * BK + key];
    }
  }

  for (int t = 0; t < seq_len; t++) {
    const float *q_t = q_bh + t * BK;
    const float *k_t = k_bh + t * BK;
    float k_reg[ROWS_PER_LANE];
    float q_reg[ROWS_PER_LANE];
    float kv_partial[VALUES_PER_WARP] = {};

#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      const float k_value = k_t[key];
      k_reg[row] = k_value;
      q_reg[row] = q_t[key];
#pragma unroll
      for (int value = 0; value < VALUES_PER_WARP; value++) {
        kv_partial[value] =
            __fmaf_rn(s[value][row], k_value, kv_partial[value]);
      }
    }

    const float decay = expf(g_bh[t]);
#pragma unroll
    for (int value = 0; value < VALUES_PER_WARP; value++) {
      const int value_idx = first_value + value;
      const float kv_col = gdn_warp_sum<WARP_SIZE>(kv_partial[value]);
      const float delta =
          (v_bh[t * V_DIM + value_idx] - decay * kv_col) * beta_bh[t];
      float y_partial = 0.0f;
#pragma unroll
      for (int row = 0; row < ROWS_PER_LANE; row++) {
        s[value][row] =
            __fmaf_rn(k_reg[row], delta, decay * s[value][row]);
        y_partial = __fmaf_rn(s[value][row], q_reg[row], y_partial);
      }
      const float y_col = gdn_warp_sum<WARP_SIZE>(y_partial);
      if (lane == 0) {
        out_bh[t * V_DIM + value_idx] = y_col;
      }
    }
  }

#pragma unroll
  for (int value = 0; value < VALUES_PER_WARP; value++) {
#pragma unroll
    for (int row = 0; row < ROWS_PER_LANE; row++) {
      const int key = row * WARP_SIZE + lane;
      state_bh[(first_value + value) * BK + key] = s[value][row];
    }
  }
}

template <typename StateT, int BT, int BK, int BV, bool VALUE_MAJOR = false>
DKR_GDN_ENTRY void
chunked_gated_delta_rule_kernel(const float *__restrict__ q,    // [BH, S, K]
                                const float *__restrict__ k,    // [BH, S, K]
                                const float *__restrict__ v,    // [BH, S, V]
                                const float *__restrict__ g,    // [BH, S]
                                const float *__restrict__ beta, // [BH, S]
                                StateT *__restrict__ state,     // key-major/value-major or pool
                                float *__restrict__ output,     // [BH, S, V]
                                int seq_len, int v_dim,
                                const int32_t *__restrict__ slot_indices,
                                int num_heads) {

  const int v_tile = blockIdx.x;
  const int bh = blockIdx.y;
  const int tid = threadIdx.x;
  const int v_idx = v_tile * BV + tid;

  // All threads must participate in cooperative loads and block barriers.
  const bool active = v_idx < v_dim;

  float *out_bh = output + (size_t)bh * seq_len * v_dim;
  if (gdn_is_padding_row(slot_indices, bh / num_heads)) {
    for (int t = 0; t < seq_len; t++) {
      if (active) out_bh[t * v_dim + v_idx] = 0.0f;
    }
    return;
  }

  const int num_chunks = (seq_len + BT - 1) / BT;

  // Pointers for this (batch, head)
  const float *q_bh = q + (size_t)bh * seq_len * BK;
  const float *k_bh = k + (size_t)bh * seq_len * BK;
  const float *v_bh = v + (size_t)bh * seq_len * v_dim;
  const float *g_bh = g + (size_t)bh * seq_len;
  const float *beta_bh = beta + (size_t)bh * seq_len;
  StateT *state_bh =
      state + gdn_state_row(slot_indices, bh / num_heads, bh % num_heads, num_heads) * BK * v_dim;

  // Dynamic shared memory layout
  extern __shared__ float smem[];
  float *k_chunk = smem;                  // [BT * BK]
  float *kk_dot = smem + BT * BK;         // [BT * BT]
  float *gcum = smem + BT * BK + BT * BT; // [BT]
  float *beta_s = gcum + BT;              // [BT]
  float *q_buf = beta_s + BT;             // [BK]

  // Load state column into registers
  float s[BK];
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      s[j] = active ? (float)state_bh[v_idx * BK + j] : 0.0f;
    } else {
      s[j] = active ? (float)state_bh[j * v_dim + v_idx] : 0.0f;
    }
  }

  // Per-thread register array for corrected deltas
  float delta[BT];

  for (int c = 0; c < num_chunks; c++) {
    const int chunk_start = c * BT;
    const int chunk_len = min(BT, seq_len - chunk_start);

    // === Phase 1: Cooperative load of k, beta, g into shared memory ===
    for (int t = 0; t < chunk_len; t++) {
      for (int j = tid; j < BK; j += BV) {
        k_chunk[t * BK + j] = k_bh[(chunk_start + t) * BK + j];
      }
    }
    if (tid < chunk_len) {
      beta_s[tid] = beta_bh[chunk_start + tid];
      gcum[tid] = g_bh[chunk_start + tid];
    }
    __syncthreads();

    // === Phase 1b: Parallel prefix sum of g (Hillis-Steele) ===
    for (int stride = 1; stride < BT; stride <<= 1) {
      float prev = 0.0f;
      if (tid < chunk_len && (int)tid >= stride)
        prev = gcum[tid - stride];
      __syncthreads();
      if (tid < chunk_len && (int)tid >= stride)
        gcum[tid] += prev;
      __syncthreads();
    }

    // === Phase 2: Compute kk_dot[i][j] = dot(k[i], k[j]) for j < i ===
    // Only lower-triangular entries needed (strictly lower)
    for (int idx = tid; idx < chunk_len * chunk_len; idx += BV) {
      int i = idx / chunk_len;
      int j = idx % chunk_len;
      if (j < i) {
        float dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          dot = __fmaf_rn(k_chunk[i * BK + d], k_chunk[j * BK + d], dot);
        }
        kk_dot[i * BT + j] = dot;
      }
    }
    __syncthreads();

    // === Phase 3: Forward substitution (per V-column, in registers) ===
    // Computes corrected delta values via triangular solve
    for (int i = 0; i < chunk_len; i++) {
      float v_i = active ? v_bh[(chunk_start + i) * v_dim + v_idx] : 0.0f;
      float decay_i = expf(gcum[i]);
      float beta_i = beta_s[i];

      // Inter-chunk contribution: state @ k[i] with decay
      float kv_mem = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        kv_mem = __fmaf_rn(s[d] * decay_i, k_chunk[i * BK + d], kv_mem);
      }

      float rhs = beta_i * (v_i - kv_mem);

      // Subtract lower-triangular contributions (intra-chunk)
      for (int j = 0; j < i; j++) {
        float a_ij = beta_i * kk_dot[i * BT + j] * expf(gcum[i] - gcum[j]);
        rhs -= a_ij * delta[j];
      }
      delta[i] = rhs;
    }

    // === Phase 4: Output computation (per V-column) ===
    for (int i = 0; i < chunk_len; i++) {
      // Cooperatively load q[i] into shared
      for (int j = tid; j < BK; j += BV) {
        q_buf[j] = q_bh[(chunk_start + i) * BK + j];
      }
      __syncthreads();

      float decay_i = expf(gcum[i]);

      // Inter-chunk contribution: q[i] @ (state * decay)
      float o_val = 0.0f;
#pragma unroll
      for (int d = 0; d < BK; d++) {
        o_val = __fmaf_rn(q_buf[d], s[d] * decay_i, o_val);
      }

      // Intra-chunk contribution: sum_{j<=i} dot(q[i], k[j]) * delta[j] *
      // exp(gcum[i] - gcum[j])
      for (int j = 0; j <= i; j++) {
        float qk_dot = 0.0f;
        for (int d = 0; d < BK; d++) {
          qk_dot = __fmaf_rn(q_buf[d], k_chunk[j * BK + d], qk_dot);
        }
        o_val += qk_dot * delta[j] * expf(gcum[i] - gcum[j]);
      }

      if (active) out_bh[(chunk_start + i) * v_dim + v_idx] = o_val;
      __syncthreads();
    }

    // === Phase 5: State update for next chunk ===
    float g_total = gcum[chunk_len - 1];
#pragma unroll
    for (int d = 0; d < BK; d++) {
      float s_new = s[d] * expf(g_total);
      for (int t = 0; t < chunk_len; t++) {
        s_new += k_chunk[t * BK + d] * delta[t] * expf(g_total - gcum[t]);
      }
      s[d] = s_new;
    }

    __syncthreads();
  }

  // Write final state back
#pragma unroll
  for (int j = 0; j < BK; j++) {
    if constexpr (VALUE_MAJOR) {
      if (active) state_bh[v_idx * BK + j] = s[j];
    } else {
      if (active) state_bh[j * v_dim + v_idx] = s[j];
    }
  }
}

template <typename T>
DKR_GDN_ENTRY void causal_conv1d_update_kernel(
    const T *__restrict__ x,      // [B, 1, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    T *__restrict__ conv_state,   // [B, conv_dim, kernel_size]
    T *__restrict__ output,       // [B, 1, conv_dim]
    int batch_size, int conv_dim, int kernel_size, int64_t x_stride_b,
    int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    output[(size_t)b * conv_dim + ch] = (T)0.0f;
    return;
  }

  // Pointer to this batch/channel's conv state
  T *cs = conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;
  const T *w = weight + ch * kernel_size;

  // Shift state left by 1
  for (int i = 0; i < kernel_size - 1; i++) {
    cs[i] = cs[i + 1];
  }
  // Insert new value
  cs[kernel_size - 1] =
      x[(size_t)b * x_stride_b + (size_t)ch * x_stride_c];

  // Dot product with weight
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    acc += (float)cs[i] * (float)w[i];
  }

  // SiLU activation: x * sigmoid(x)
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[b * conv_dim + ch] = (T)result;
}

template <typename T> struct alignas(8) GdnConvWidth4 {
  T values[GDN_PACKED_CONV_WIDTH];
};

template <typename T>
__device__ __forceinline__ T gdn_conv_width4_update(
    T input, const GdnConvWidth4<T> &weights, GdnConvWidth4<T> *state) {
  GdnConvWidth4<T> values = *state;

#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH - 1; i++) {
    values.values[i] = values.values[i + 1];
  }
  values.values[GDN_PACKED_CONV_WIDTH - 1] = input;
  *state = values;

  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < GDN_PACKED_CONV_WIDTH; i++) {
    acc += (float)values.values[i] * (float)weights.values[i];
  }
  const float sig = 1.0f / (1.0f + expf(-acc));
  return (T)(acc * sig);
}

template <typename T>
DKR_GDN_ENTRY void causal_conv1d_update_width4_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  const size_t input_idx = (size_t)b * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[input_idx] = (T)0.0f;
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const size_t state_idx = state_row * conv_dim + ch;
  const size_t x_idx = (size_t)b * x_stride_b + (size_t)ch * x_stride_c;
  auto *state = reinterpret_cast<GdnConvWidth4<T> *>(conv_state);
  const auto *weights = reinterpret_cast<const GdnConvWidth4<T> *>(weight);
  output[input_idx] = gdn_conv_width4_update(
      x[x_idx], weights[ch], &state[state_idx]);
}

template <typename T>
DKR_GDN_ENTRY void causal_conv1d_full_kernel(
    const T *__restrict__ x,      // [B, S, conv_dim]
    const T *__restrict__ weight, // [conv_dim, kernel_size]
    const T *__restrict__ conv_state,
    T *__restrict__ output, // [B, S, conv_dim]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  const size_t plane = (size_t)conv_dim * seq_len;

  if (idx >= plane || b >= batch_size)
    return;

  const int pos = (int)(idx / conv_dim);
  const int ch = (int)(idx % conv_dim);

  const size_t output_idx = ((size_t)b * seq_len + pos) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    output[output_idx] = (T)0.0f;
    return;
  }

  const T *w = weight + (size_t)ch * kernel_size;
  const T *cs =
      conv_state + (gdn_state_row(slot_indices, b, 0, 1) * conv_dim + ch) * kernel_size;

  float acc = 0.0f;
  for (int i = 0; i < kernel_size; i++) {
    int src_pos = pos - (kernel_size - 1) + i;
    float x_val = src_pos >= 0
                      ? (float)x[(size_t)b * x_stride_b +
                                 (size_t)src_pos * x_stride_s +
                                 (size_t)ch * x_stride_c]
                      : (float)cs[kernel_size + src_pos];
    acc += x_val * (float)w[i];
  }

  // SiLU
  float sig = 1.0f / (1.0f + expf(-acc));
  float result = acc * sig;

  output[output_idx] = (T)result;
}

template <typename T>
__device__ __forceinline__ float causal_conv1d_width4_load(
    const T *__restrict__ x, const T *__restrict__ state, int pos,
    size_t x_batch_offset, int64_t x_stride_s, int64_t x_stride_c, int ch) {
  if (pos >= 0) {
    return (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                    (size_t)ch * x_stride_c];
  }
  return (float)state[GDN_PACKED_CONV_WIDTH + pos];
}

template <typename T, int TOKEN_TILE>
DKR_GDN_ENTRY void causal_conv1d_full_width4_tiled_kernel(
    const T *__restrict__ x, const T *__restrict__ weight,
    const T *__restrict__ conv_state, T *__restrict__ output, int batch_size,
    int conv_dim, int seq_len, int64_t x_stride_b, int64_t x_stride_s,
    int64_t x_stride_c, const int32_t *__restrict__ slot_indices) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int start = blockIdx.y * TOKEN_TILE;
  const int b = blockIdx.z;
  if (ch >= conv_dim || start >= seq_len || b >= batch_size) {
    return;
  }

  const int end = min(start + TOKEN_TILE, seq_len);
  T *out = output + ((size_t)b * seq_len + start) * conv_dim + ch;
  if (gdn_is_padding_row(slot_indices, b)) {
    for (int pos = start; pos < end; ++pos) {
      *out = (T)0.0f;
      out += conv_dim;
    }
    return;
  }

  const size_t state_row = gdn_state_row(slot_indices, b, 0, 1);
  const T *state = conv_state +
                   (state_row * conv_dim + ch) * GDN_PACKED_CONV_WIDTH;
  const T *w = weight + (size_t)ch * GDN_PACKED_CONV_WIDTH;
  const size_t x_batch_offset = (size_t)b * x_stride_b;
  float x0 = causal_conv1d_width4_load(
      x, state, start - 3, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x1 = causal_conv1d_width4_load(
      x, state, start - 2, x_batch_offset, x_stride_s, x_stride_c, ch);
  float x2 = causal_conv1d_width4_load(
      x, state, start - 1, x_batch_offset, x_stride_s, x_stride_c, ch);
  const float w0 = (float)w[0];
  const float w1 = (float)w[1];
  const float w2 = (float)w[2];
  const float w3 = (float)w[3];

  for (int pos = start; pos < end; ++pos) {
    const float x3 = (float)x[x_batch_offset + (size_t)pos * x_stride_s +
                              (size_t)ch * x_stride_c];
    const float acc = __fmaf_rn(x0, w0, __fmaf_rn(x1, w1, __fmaf_rn(x2, w2, x3 * w3)));
    const float result = acc / (1.0f + expf(-acc));
    *out = (T)result;
    out += conv_dim;
    x0 = x1;
    x1 = x2;
    x2 = x3;
  }
}

template <typename T>
DKR_GDN_ENTRY void save_conv_state_kernel(
    const T *__restrict__ x, // [B, S, conv_dim]
    // May alias conv_state_out (pooled in-place update): every read is ahead of the write position
    const T *conv_state_in,
    T *conv_state_out, // [B, conv_dim, kernel_size]
    int batch_size, int conv_dim, int seq_len, int kernel_size,
    int64_t x_stride_b, int64_t x_stride_s, int64_t x_stride_c,
    const int32_t *__restrict__ slot_indices) {

  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;

  if (ch >= conv_dim || b >= batch_size)
    return;

  if (gdn_is_padding_row(slot_indices, b)) {
    return;
  }

  const size_t row = gdn_state_row(slot_indices, b, 0, 1);
  const T *prior = conv_state_in + (row * conv_dim + ch) * kernel_size;
  T *cs = conv_state_out + (row * conv_dim + ch) * kernel_size;

  int pad = kernel_size - seq_len;
  for (int i = 0; i < kernel_size; i++) {
    if (i < pad) {
      cs[i] = prior[i + seq_len];
    } else {
      const int pos = seq_len - kernel_size + i;
      cs[i] = x[(size_t)b * x_stride_b + (size_t)pos * x_stride_s +
                (size_t)ch * x_stride_c];
    }
  }
}

#undef DKR_GDN_ENTRY
#undef DKR_GDN_WARP_ENTRY
