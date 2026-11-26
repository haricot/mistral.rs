#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#ifdef DKR_GDN_TRANSACTION_IMAGE
#define GDN_TX_ENTRY __device__ __forceinline__
#else
#define GDN_TX_ENTRY __global__
#endif
// Contract shared by the existing producer and the typed producer.
constexpr int GDN_TX_MAX_ROWS = 8;
constexpr int GDN_TX_MAX_CONV_WIDTH = 16;
constexpr int GDN_TX_KEY_DIM = 128;
constexpr int GDN_TX_VALUE_DIM = 128;
constexpr uint32_t GDN_TX_PAD_SLOT = 0xffffffffu;
constexpr uint32_t GDN_TX_BANK_MASK = 1u;

constexpr int GDN_TRANSITION_CONV_INPUT = 0;
constexpr int GDN_TRANSITION_KEY = 1;
constexpr int GDN_TRANSITION_DELTA = 2;
constexpr int GDN_TRANSITION_DECAY = 3;
constexpr int GDN_TRANSITION_CONV_STATE = 4;
constexpr int GDN_TRANSITION_RECURRENT_STATE = 5;

template <typename T, typename StateT>
GDN_TX_ENTRY void gdn_speculative_transition_commit_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ active_slots, int layer_count, int batch_size,
    int seq_len, int num_k_heads, int num_v_heads, int head_k_dim,
    int head_v_dim, int conv_dim, int conv_width, int tiled_v_heads,
    int value_major, int capacity) {
  const int conv_blocks = (conv_dim + blockDim.x - 1) / blockDim.x;
  const int state_elements = head_k_dim * head_v_dim;
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  const int batch_block = blockIdx.x;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const int rows = (int)keep_rows[batch_idx];
  const uint32_t slot = active_slots[batch_idx];
  if (rows <= 0 || rows > seq_len || rows > GDN_TX_MAX_ROWS || slot == GDN_TX_PAD_SLOT ||
      (capacity > 0 && slot >= (uint32_t)capacity)) {
    return;
  }

  if (batch_block < conv_blocks) {
    const int channel = batch_block * blockDim.x + threadIdx.x;
    if (channel >= conv_dim) {
      return;
    }
    const auto *input = reinterpret_cast<const T *>(
        pointer_table[GDN_TRANSITION_CONV_INPUT * layer_count + layer]);
    auto *conv_state = reinterpret_cast<T *>(
        pointer_table[GDN_TRANSITION_CONV_STATE * layer_count + layer]);
    T state[GDN_TX_MAX_CONV_WIDTH];
    T *destination =
        conv_state + ((size_t)slot * conv_dim + channel) * conv_width;
    for (int i = 0; i < conv_width; i++) {
      state[i] = destination[i];
    }
    for (int position = 0; position < rows; position++) {
      for (int i = 0; i < conv_width - 1; i++) {
        state[i] = state[i + 1];
      }
      state[conv_width - 1] =
          input[((size_t)batch_idx * seq_len + position) * conv_dim +
                channel];
    }
    for (int i = 0; i < conv_width; i++) {
      destination[i] = state[i];
    }
    return;
  }

  const int recurrence_block = batch_block - conv_blocks;
  const int value_head = recurrence_block;
  if (value_head >= num_v_heads) {
    return;
  }
  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads ? value_head % num_k_heads
                                     : value_head / values_per_group;
  const auto *transition_key = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_KEY * layer_count + layer]);
  const auto *transition_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_DELTA * layer_count + layer]);
  const auto *transition_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_DECAY * layer_count + layer]);
  auto *recurrent_state = reinterpret_cast<StateT *>(
      pointer_table[GDN_TRANSITION_RECURRENT_STATE * layer_count + layer]);
  const size_t state_head =
      ((size_t)slot * num_v_heads + value_head) * state_elements;
  __shared__ float shared_key[GDN_TX_MAX_ROWS]
                             [GDN_TX_KEY_DIM];
  __shared__ float shared_delta[GDN_TX_MAX_ROWS]
                               [GDN_TX_VALUE_DIM];
  __shared__ float shared_decay[GDN_TX_MAX_ROWS];
  for (int linear = threadIdx.x; linear < rows * head_k_dim;
       linear += blockDim.x) {
    const int position = linear / head_k_dim;
    const int key = linear - position * head_k_dim;
    shared_key[position][key] =
        transition_key[(((size_t)batch_idx * seq_len + position) *
                            num_k_heads +
                        key_head) *
                           head_k_dim +
                       key];
  }
  for (int linear = threadIdx.x; linear < rows * head_v_dim;
       linear += blockDim.x) {
    const int position = linear / head_v_dim;
    const int value = linear - position * head_v_dim;
    shared_delta[position][value] =
        transition_delta[(((size_t)batch_idx * seq_len + position) *
                              num_v_heads +
                          value_head) *
                             head_v_dim +
                         value];
  }
  for (int position = threadIdx.x; position < rows;
       position += blockDim.x) {
    shared_decay[position] =
        transition_decay[((size_t)batch_idx * seq_len + position) *
                             num_v_heads +
                         value_head];
  }
  __syncthreads();
  for (int element = threadIdx.x; element < state_elements;
       element += blockDim.x) {
    const int value = element / head_k_dim;
    const int key = element - value * head_k_dim;
    const size_t state_offset = value_major
                                    ? (size_t)value * head_k_dim + key
                                    : (size_t)key * head_v_dim + value;
    float state = (float)recurrent_state[state_head + state_offset];
    for (int position = 0; position < rows; position++) {
      state = __fmaf_rn(shared_key[position][key],
                        shared_delta[position][value],
                        __fmul_rn(shared_decay[position], state));
    }
    recurrent_state[state_head + state_offset] = (StateT)state;
  }
}

constexpr int GDN_TRANSITION_STAGE_SRC_CONV = 0;
constexpr int GDN_TRANSITION_STAGE_SRC_KEY = 1;
constexpr int GDN_TRANSITION_STAGE_SRC_DELTA = 2;
constexpr int GDN_TRANSITION_STAGE_SRC_DECAY = 3;
constexpr int GDN_TRANSITION_STAGE_DST_CONV = 4;
constexpr int GDN_TRANSITION_STAGE_DST_KEY = 5;
constexpr int GDN_TRANSITION_STAGE_DST_DELTA = 6;
constexpr int GDN_TRANSITION_STAGE_DST_DECAY = 7;
constexpr int GDN_TRANSITION_STAGE_DST_KEEP = 8;
constexpr int GDN_TRANSITION_STAGE_DST_EPOCH = 9;
constexpr int GDN_TRANSITION_STAGE_COPY_BLOCKS = 4;

template <typename T>
GDN_TX_ENTRY void gdn_speculative_transition_stage_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ destination_slots, int layer_count,
    int batch_size,
    int seq_len, int max_rows, int destination_capacity, int num_k_heads,
    int num_v_heads, int head_k_dim, int head_v_dim, int conv_dim) {
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const uint32_t slot = destination_slots[batch_idx];
  if (slot == GDN_TX_PAD_SLOT || slot >= destination_capacity) {
    return;
  }
  const int rows = (int)keep_rows[batch_idx];
  auto *pending_keep = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_KEEP * layer_count + layer]);
  auto *pending_epoch = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_EPOCH * layer_count + layer]);
  if (rows <= 0 || rows > seq_len || rows > max_rows) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      pending_keep[slot] = 0;
      const uint32_t next_epoch = pending_epoch[slot] + 1;
      pending_epoch[slot] = next_epoch == 0 ? 1 : next_epoch;
    }
    return;
  }

  const size_t conv_elements = (size_t)rows * conv_dim;
  const size_t key_row_elements = (size_t)num_k_heads * head_k_dim;
  const size_t key_elements = (size_t)rows * key_row_elements;
  const size_t delta_row_elements = (size_t)num_v_heads * head_v_dim;
  const size_t delta_elements = (size_t)rows * delta_row_elements;
  const size_t decay_row_elements = num_v_heads;
  const size_t decay_elements = (size_t)rows * decay_row_elements;
  const size_t total_elements =
      conv_elements + key_elements + delta_elements + decay_elements;
  const auto *source_conv = reinterpret_cast<const T *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_CONV * layer_count + layer]);
  const auto *source_key = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_KEY * layer_count + layer]);
  const auto *source_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_DELTA * layer_count + layer]);
  const auto *source_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_STAGE_SRC_DECAY * layer_count + layer]);
  auto *destination_conv = reinterpret_cast<T *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_CONV * layer_count + layer]);
  auto *destination_key = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_KEY * layer_count + layer]);
  auto *destination_delta = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_DELTA * layer_count + layer]);
  auto *destination_decay = reinterpret_cast<float *>(
      pointer_table[GDN_TRANSITION_STAGE_DST_DECAY * layer_count + layer]);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t linear = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements; linear += stride) {
    if (linear < conv_elements) {
      destination_conv[((size_t)slot * max_rows * conv_dim) + linear] =
          source_conv[((size_t)batch_idx * seq_len * conv_dim) + linear];
    } else if (linear < conv_elements + key_elements) {
      const size_t offset = linear - conv_elements;
      destination_key[((size_t)slot * max_rows * key_row_elements) + offset] =
          source_key[((size_t)batch_idx * seq_len * key_row_elements) + offset];
    } else if (linear < conv_elements + key_elements + delta_elements) {
      const size_t offset = linear - conv_elements - key_elements;
      destination_delta[((size_t)slot * max_rows * delta_row_elements) +
                        offset] =
          source_delta[((size_t)batch_idx * seq_len * delta_row_elements) +
                       offset];
    } else {
      const size_t offset =
          linear - conv_elements - key_elements - delta_elements;
      destination_decay[((size_t)slot * max_rows * decay_row_elements) +
                        offset] =
          source_decay[((size_t)batch_idx * seq_len * decay_row_elements) +
                       offset];
    }
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    pending_keep[slot] = (uint32_t)rows;
    const uint32_t next_epoch = pending_epoch[slot] + 1;
    pending_epoch[slot] = next_epoch == 0 ? 1 : next_epoch;
  }
}

constexpr int GDN_TRANSITION_PUBLISH_KEEP = 0;
constexpr int GDN_TRANSITION_PUBLISH_EPOCH = 1;
constexpr int GDN_TRANSITION_PUBLISH_KEY_BANK = 2;

GDN_TX_ENTRY void gdn_pending_transition_publish_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ keep_rows,
    const uint32_t *__restrict__ destination_slots, int layer_count,
    int batch_size, int max_rows, int destination_capacity) {
  const int batch_idx = blockIdx.x;
  const int layer = blockIdx.y;
  if (batch_idx >= batch_size || layer >= layer_count || threadIdx.x != 0) {
    return;
  }
  const uint32_t slot = destination_slots[batch_idx];
  if (slot == GDN_TX_PAD_SLOT || slot >= destination_capacity) {
    return;
  }
  auto *pending_keep = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_KEEP * layer_count + layer]);
  auto *pending_epoch = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_EPOCH * layer_count + layer]);
  auto *pending_key_bank = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_PUBLISH_KEY_BANK * layer_count + layer]);
  const uint32_t rows = keep_rows[batch_idx];
  uint32_t next_epoch = pending_epoch[slot] + 1;
  if (next_epoch == 0) {
    next_epoch = 1;
  }
  pending_keep[slot] = rows <= (uint32_t)max_rows ? rows : 0;
  pending_key_bank[slot] =
      (pending_key_bank[slot] ^ GDN_TX_BANK_MASK) &
      GDN_TX_BANK_MASK;
  pending_epoch[slot] = next_epoch;
}

constexpr int GDN_TRANSITION_APPLY_PENDING_CONV = 0;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANKS = 1;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEY_BANK = 2;
constexpr int GDN_TRANSITION_APPLY_PENDING_DELTA = 3;
constexpr int GDN_TRANSITION_APPLY_PENDING_DECAY = 4;
constexpr int GDN_TRANSITION_APPLY_PENDING_KEEP = 5;
constexpr int GDN_TRANSITION_APPLY_PENDING_EPOCH = 6;
constexpr int GDN_TRANSITION_APPLY_CONV_EPOCH = 7;
constexpr int GDN_TRANSITION_APPLY_RECURRENT_EPOCH = 8;
constexpr int GDN_TRANSITION_APPLY_CONV_STATE = 9;
constexpr int GDN_TRANSITION_APPLY_RECURRENT_STATE = 10;

template <typename T, typename StateT>
GDN_TX_ENTRY void gdn_pending_transition_apply_batched_kernel(
    const uint64_t *__restrict__ pointer_table,
    const uint32_t *__restrict__ active_slots, int layer_count, int batch_size,
    int max_rows, int capacity, int num_k_heads, int num_v_heads,
    int head_k_dim, int head_v_dim, int conv_dim, int conv_width,
    int tiled_v_heads, int conv_blocks) {
  const int layer = blockIdx.z;
  const int batch_idx = blockIdx.y;
  const int batch_block = blockIdx.x;
  if (layer >= layer_count || batch_idx >= batch_size) {
    return;
  }

  const uint32_t slot = active_slots[batch_idx];
  if (slot == GDN_TX_PAD_SLOT || slot >= capacity) {
    return;
  }
  const auto *pending_keep = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEEP * layer_count + layer]);
  const auto *pending_epoch = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_EPOCH * layer_count + layer]);
  const int rows = (int)pending_keep[slot];
  const uint32_t epoch = pending_epoch[slot];
  if (epoch == 0 || rows <= 0 || rows > max_rows) {
    return;
  }

  if (batch_block < conv_blocks) {
    auto *applied_epochs = reinterpret_cast<uint32_t *>(
        pointer_table[GDN_TRANSITION_APPLY_CONV_EPOCH * layer_count + layer]);
    const size_t applied_offset = (size_t)slot * conv_blocks + batch_block;
    if (applied_epochs[applied_offset] == epoch) {
      return;
    }
    const int channel = batch_block * blockDim.x + threadIdx.x;
    if (channel < conv_dim) {
      const auto *pending_conv = reinterpret_cast<const T *>(
          pointer_table[GDN_TRANSITION_APPLY_PENDING_CONV * layer_count +
                        layer]);
      auto *conv_state = reinterpret_cast<T *>(
          pointer_table[GDN_TRANSITION_APPLY_CONV_STATE * layer_count + layer]);
      T state[GDN_TX_MAX_CONV_WIDTH];
      T *destination =
          conv_state + ((size_t)slot * conv_dim + channel) * conv_width;
      for (int i = 0; i < conv_width; i++) {
        state[i] = destination[i];
      }
      for (int position = 0; position < rows; position++) {
        for (int i = 0; i < conv_width - 1; i++) {
          state[i] = state[i + 1];
        }
        state[conv_width - 1] =
            pending_conv[((size_t)slot * max_rows + position) * conv_dim +
                         channel];
      }
      for (int i = 0; i < conv_width; i++) {
        destination[i] = state[i];
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      applied_epochs[applied_offset] = epoch;
    }
    return;
  }

  const int value_head = batch_block - conv_blocks;
  if (value_head >= num_v_heads) {
    return;
  }
  auto *applied_epochs = reinterpret_cast<uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_RECURRENT_EPOCH * layer_count +
                    layer]);
  const size_t applied_offset = (size_t)slot * num_v_heads + value_head;
  if (applied_epochs[applied_offset] == epoch) {
    return;
  }

  const int values_per_group = num_v_heads / num_k_heads;
  const int key_head = tiled_v_heads ? value_head % num_k_heads
                                     : value_head / values_per_group;
  const auto *pending_key_banks = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEY_BANKS * layer_count +
                    layer]);
  const auto *pending_key_bank = reinterpret_cast<const uint32_t *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_KEY_BANK * layer_count +
                    layer]);
  const size_t key_bank_stride =
      (size_t)capacity * max_rows * num_k_heads * head_k_dim;
  const auto *pending_key =
      pending_key_banks +
      (pending_key_bank[slot] & GDN_TX_BANK_MASK) * key_bank_stride;
  const auto *pending_delta = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_DELTA * layer_count + layer]);
  const auto *pending_decay = reinterpret_cast<const float *>(
      pointer_table[GDN_TRANSITION_APPLY_PENDING_DECAY * layer_count + layer]);
  auto *recurrent_state = reinterpret_cast<StateT *>(
      pointer_table[GDN_TRANSITION_APPLY_RECURRENT_STATE * layer_count +
                    layer]);
  __shared__ float shared_key[GDN_TX_MAX_ROWS]
                             [GDN_TX_KEY_DIM];
  __shared__ float shared_delta[GDN_TX_MAX_ROWS]
                               [GDN_TX_VALUE_DIM];
  __shared__ float shared_decay[GDN_TX_MAX_ROWS];
  for (int linear = threadIdx.x; linear < rows * head_k_dim;
       linear += blockDim.x) {
    const int position = linear / head_k_dim;
    const int key = linear - position * head_k_dim;
    shared_key[position][key] =
        pending_key[(((size_t)slot * max_rows + position) * num_k_heads +
                     key_head) *
                        head_k_dim +
                    key];
  }
  for (int linear = threadIdx.x; linear < rows * head_v_dim;
       linear += blockDim.x) {
    const int position = linear / head_v_dim;
    const int value = linear - position * head_v_dim;
    shared_delta[position][value] =
        pending_delta[(((size_t)slot * max_rows + position) * num_v_heads +
                       value_head) *
                          head_v_dim +
                      value];
  }
  for (int position = threadIdx.x; position < rows;
       position += blockDim.x) {
    shared_decay[position] =
        pending_decay[((size_t)slot * max_rows + position) * num_v_heads +
                      value_head];
  }
  __syncthreads();

  const int state_elements = head_k_dim * head_v_dim;
  const size_t state_head =
      ((size_t)slot * num_v_heads + value_head) * state_elements;
  for (int element = threadIdx.x; element < state_elements;
       element += blockDim.x) {
    const int value = element / head_k_dim;
    const int key = element - value * head_k_dim;
    float state = (float)recurrent_state[state_head + element];
    for (int position = 0; position < rows; position++) {
      state = __fmaf_rn(shared_key[position][key],
                        shared_delta[position][value],
                        __fmul_rn(shared_decay[position], state));
    }
    recurrent_state[state_head + element] = (StateT)state;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    applied_epochs[applied_offset] = epoch;
  }
}

#undef GDN_TX_ENTRY
