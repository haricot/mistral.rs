#pragma once
#include "large_topk_device.cuh"
#ifdef DKR_SAMPLING_EXTERNAL
#define DKR_SAMPLING_ENTRY __device__ __forceinline__
#else
#define DKR_SAMPLING_ENTRY __global__
#endif

DKR_SAMPLING_ENTRY void sample_ranked_topk_kernel(
    const float *__restrict__ packed, const float *__restrict__ params,
    uint32_t *__restrict__ tokens, const int nrows, const int packed_k) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) {
    return;
  }

  const float *row_params =
      params + row * static_cast<size_t>(RANKED_TOPK_SAMPLE_PARAM_WIDTH);
  const float inv_temperature = row_params[0];
  const int row_k = min(static_cast<int>(row_params[1]), packed_k);
  const float top_p = row_params[2];
  const float min_p = row_params[3];
  const float uniform = row_params[4];
  if (row_k <= 0 || !(inv_temperature > 0.0f) || !isfinite(inv_temperature) ||
      !isfinite(top_p) || !isfinite(min_p) || !(uniform >= 0.0f) ||
      !(uniform < 1.0f)) {
    tokens[row] = UINT32_MAX;
    return;
  }

  const float *row_packed = packed + row * static_cast<size_t>(2 * packed_k);
  const float scaled_max = row_packed[0] * inv_temperature;
  if (!isfinite(scaled_max)) {
    tokens[row] = UINT32_MAX;
    return;
  }

  float denominator = 0.0f;
  for (int rank = 0; rank < row_k; ++rank) {
    denominator += expf(row_packed[rank] * inv_temperature - scaled_max);
  }
  if (!(denominator > 0.0f) || !isfinite(denominator)) {
    tokens[row] = UINT32_MAX;
    return;
  }

  float normalized_sum = 0.0f;
  for (int rank = 0; rank < row_k; ++rank) {
    normalized_sum +=
        expf(row_packed[rank] * inv_temperature - scaled_max) / denominator;
  }
  const bool filter_top_p = top_p > 0.0f && top_p < 1.0f;
  const float cutoff = filter_top_p ? top_p * normalized_sum : 0.0f;
  float nucleus_sum = 0.0f;
  int retained = row_k;
  if (filter_top_p) {
    for (int rank = 0; rank < row_k; ++rank) {
      if (nucleus_sum >= cutoff) {
        retained = rank;
        break;
      }
      nucleus_sum +=
          expf(row_packed[rank] * inv_temperature - scaled_max) / denominator;
    }
  }

  const bool filter_min_p = min_p > 0.0f && min_p < 1.0f;
  const float first_probability = 1.0f / denominator;
  const float min_probability =
      filter_min_p ? first_probability * min_p : -1.0f;
  float total = 0.0f;
  for (int rank = 0; rank < retained; ++rank) {
    const float probability =
        expf(row_packed[rank] * inv_temperature - scaled_max) / denominator;
    if (!filter_min_p || !(min_probability >= probability)) {
      total += probability;
    }
  }
  if (!(total > 0.0f) || !isfinite(total)) {
    tokens[row] = UINT32_MAX;
    return;
  }

  const float chosen = uniform * total;
  float cumulative = 0.0f;
  int selected = -1;
  int last_positive = -1;
  for (int rank = 0; rank < retained; ++rank) {
    const float probability =
        expf(row_packed[rank] * inv_temperature - scaled_max) / denominator;
    if (filter_min_p && min_probability >= probability) {
      continue;
    }
    cumulative += probability;
    if (probability > 0.0f) {
      last_positive = rank;
    }
    if (cumulative > chosen) {
      selected = rank;
      break;
    }
  }
  if (selected < 0) {
    selected = last_positive;
  }
  const float token = selected >= 0 ? row_packed[packed_k + selected] : -1.0f;
  tokens[row] = token >= 0.0f && isfinite(token) ? static_cast<uint32_t>(token)
                                                 : UINT32_MAX;
}

template <typename T, bool BATCHED, bool COMPUTE_SUMS>
DKR_SAMPLING_ENTRY void top1_large_stage1(
    const T *__restrict__ input, float *__restrict__ block_values,
    uint32_t *__restrict__ block_indices, float *__restrict__ block_sums,
    const int ncols, const int chunk_size,
    const float *__restrict__ inv_temperatures) {
  const size_t row = BATCHED ? blockIdx.y : 0;
  const int chunk = blockIdx.x;
  if constexpr (BATCHED) {
    const size_t block_stride = gridDim.x;
    input += row * static_cast<size_t>(ncols);
    block_values += row * block_stride;
    if constexpr (!COMPUTE_SUMS) {
      block_indices += row * block_stride;
    }
    if constexpr (COMPUTE_SUMS) {
      block_sums += row * block_stride;
    }
  }
  const int start = chunk * chunk_size;
  const int end = min(start + chunk_size, ncols);
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;

  float local_max = -INFINITY;
  int local_idx = -1;
  bool local_has_nan = false;
  for (int idx = start + tid; idx < end; idx += block_size) {
    const float candidate = topk_to_float(input[idx]);
    if (candidate != candidate) {
      local_has_nan = true;
    } else if (candidate > local_max) {
      local_max = candidate;
      local_idx = idx;
    }
  }

  int warp_max_idx;
  float warp_max =
      warp_reduce_max_with_idx<float>(local_max, local_idx, warp_max_idx);

  __shared__ float warp_maxes[32];
  __shared__ int warp_indices[32];

  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_max;
    warp_indices[warp_id] = warp_max_idx;
  }
  const bool block_has_nan = __syncthreads_or(local_has_nan);

  if (tid < 32) {
    float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
    int idx = (tid < num_warps) ? warp_indices[tid] : -1;
    int final_idx;
    float final_max = warp_reduce_max_with_idx<float>(val, idx, final_idx);

    if (tid == 0) {
      block_values[chunk] = block_has_nan ? NAN : final_max;
      if constexpr (!COMPUTE_SUMS) {
        block_indices[chunk] = block_has_nan || final_idx < 0
                                   ? 0
                                   : static_cast<uint32_t>(final_idx);
      }
    }
  }
  if constexpr (COMPUTE_SUMS) {
    __syncthreads();
    const float inv_temperature = inv_temperatures[row];
    const float block_max = block_values[chunk] * inv_temperature;
    float local_sum = 0.0f;
    for (int idx = start + tid; idx < end; idx += block_size) {
      const float candidate = topk_to_float(input[idx]);
      if (candidate != candidate) {
        local_sum = NAN;
      } else if (block_max != -INFINITY) {
        local_sum += expf(candidate * inv_temperature - block_max);
      }
    }
    const float block_sum = block_reduce_sum_f32(local_sum);
    if (tid == 0) {
      block_sums[chunk] = block_sum;
    }
  }
}

DKR_SAMPLING_ENTRY void categorical_large_stage2_f32_packed(
    const float *__restrict__ input, const float *__restrict__ inv_temperatures,
    const float *__restrict__ uniforms, const float *__restrict__ block_values,
    const float *__restrict__ block_sums, float *__restrict__ packed_out,
    const int ncols, const int chunk_size, const int nblocks) {
  const size_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  input += row * static_cast<size_t>(ncols);
  block_values += row * static_cast<size_t>(nblocks);
  block_sums += row * static_cast<size_t>(nblocks);
  constexpr int packed_width = 2;
  packed_out += row * packed_width;

  float local_global_max = -INFINITY;
  for (int block = tid; block < nblocks; block += block_size) {
    local_global_max = fmaxf(local_global_max, block_values[block]);
  }

  int unused_idx;
  float warp_global_max =
      warp_reduce_max_with_idx<float>(local_global_max, tid, unused_idx);
  __shared__ float warp_maxes[32];
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;
  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_global_max;
  }
  __syncthreads();

  __shared__ float s_global_max;
  __shared__ float s_denom;
  __shared__ float s_chunk_target;
  __shared__ int s_selected_chunk;
  __shared__ int s_selected_token;
  if (tid < 32) {
    const float value = tid < num_warps ? warp_maxes[tid] : -INFINITY;
    int final_idx;
    const float final_max =
        warp_reduce_max_with_idx<float>(value, tid, final_idx);
    if (tid == 0) {
      s_global_max = final_max * inv_temperatures[row];
    }
  }
  __syncthreads();

  if (tid == 0) {
    const float inv_temperature = inv_temperatures[row];
    float denom = 0.0f;
    for (int block = 0; block < nblocks; ++block) {
      denom += block_sums[block] *
               expf(block_values[block] * inv_temperature - s_global_max);
    }
    s_denom = denom;
    s_selected_chunk = -1;
    s_chunk_target = NAN;

    const float uniform = uniforms[row];
    if (inv_temperature > 0.0f && isfinite(inv_temperature) &&
        uniform >= 0.0f && uniform < 1.0f && isfinite(uniform) &&
        isfinite(s_global_max) && denom > 0.0f && isfinite(denom)) {
      const float target = fminf(uniform * denom, nextafterf(denom, -INFINITY));
      float cumulative = 0.0f;
      for (int block = 0; block < nblocks; ++block) {
        const float mass =
            block_sums[block] *
            expf(block_values[block] * inv_temperature - s_global_max);
        const float next = cumulative + mass;
        if (target < next) {
          s_selected_chunk = block;
          s_chunk_target = target - cumulative;
          break;
        }
        cumulative = next;
      }
    }

    if (s_selected_chunk < 0) {
      packed_out[0] = NAN;
      packed_out[1] = NAN;
    }
    s_selected_token = ncols;
  }
  __syncthreads();

  if (s_selected_chunk < 0) {
    return;
  }

  extern __shared__ char smem[];
  float *weights = reinterpret_cast<float *>(smem);
  float *prefix = weights + chunk_size;
  const int start = s_selected_chunk * chunk_size;
  const int width = min(chunk_size, ncols - start);
  const float inv_temperature = inv_temperatures[row];
  for (int local = tid; local < chunk_size; local += block_size) {
    const float weight =
        local < width
            ? expf(input[start + local] * inv_temperature - s_global_max)
            : 0.0f;
    weights[local] = weight;
    prefix[local] = weight;
  }
  __syncthreads();

  for (int stride = 1; stride < chunk_size; stride <<= 1) {
    const int step = stride << 1;
    for (int index = (tid + 1) * step - 1; index < chunk_size;
         index += block_size * step) {
      prefix[index] += prefix[index - stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    prefix[chunk_size - 1] = 0.0f;
  }
  __syncthreads();
  for (int stride = chunk_size >> 1; stride > 0; stride >>= 1) {
    const int step = stride << 1;
    for (int index = (tid + 1) * step - 1; index < chunk_size;
         index += block_size * step) {
      const float left = prefix[index - stride];
      prefix[index - stride] = prefix[index];
      prefix[index] += left;
    }
    __syncthreads();
  }

  for (int local = tid; local < width; local += block_size) {
    if (weights[local] > 0.0f &&
        prefix[local] + weights[local] > s_chunk_target) {
      atomicMin(&s_selected_token, start + local);
    }
  }
  __syncthreads();

  if (tid == 0) {
    if (s_selected_token == ncols) {
      for (int local = width - 1; local >= 0; --local) {
        if (weights[local] > 0.0f) {
          s_selected_token = start + local;
          break;
        }
      }
    }
    if (s_selected_token == ncols) {
      packed_out[0] = NAN;
      packed_out[1] = NAN;
      return;
    }
    packed_out[0] = static_cast<float>(s_selected_token);
    packed_out[1] = input[s_selected_token] * inv_temperature - s_global_max -
                    logf(s_denom);
  }
}

template <bool BATCHED>
DKR_SAMPLING_ENTRY void
top1_large_stage2_f32_packed(const float *__restrict__ block_values,
                             const uint32_t *__restrict__ block_indices,
                             float *__restrict__ packed_out,
                             uint32_t *__restrict__ token_ids_out,
                             const int nblocks) {
  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  if constexpr (BATCHED) {
    const size_t row = blockIdx.x;
    block_values += row * static_cast<size_t>(nblocks);
    block_indices += row * static_cast<size_t>(nblocks);
    if (packed_out != nullptr) {
      packed_out += row * 2;
    }
    if (token_ids_out != nullptr) {
      token_ids_out += row;
    }
  }

  float local_max = -INFINITY;
  int local_pos = -1;
  bool local_has_nan = false;
  for (int pos = tid; pos < nblocks; pos += block_size) {
    const float candidate = block_values[pos];
    if (candidate != candidate) {
      local_has_nan = true;
    } else if (candidate > local_max) {
      local_max = candidate;
      local_pos = pos;
    }
  }

  int warp_max_pos;
  float warp_max =
      warp_reduce_max_with_idx<float>(local_max, local_pos, warp_max_pos);

  __shared__ float warp_maxes[32];
  __shared__ int warp_indices[32];

  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  const int num_warps = (block_size + 31) / 32;

  if (lane_id == 0) {
    warp_maxes[warp_id] = warp_max;
    warp_indices[warp_id] = warp_max_pos;
  }
  const bool row_has_nan = __syncthreads_or(local_has_nan);

  if (tid < 32) {
    float val = (tid < num_warps) ? warp_maxes[tid] : -INFINITY;
    int pos = (tid < num_warps) ? warp_indices[tid] : -1;
    int final_pos;
    float final_max = warp_reduce_max_with_idx<float>(val, pos, final_pos);

    if (tid == 0) {
      const uint32_t token_id = row_has_nan
                                    ? UINT32_MAX
                                    : final_pos >= 0
                                          ? block_indices[final_pos]
                                          : 0;
      if (packed_out != nullptr) {
        packed_out[0] = row_has_nan ? NAN : final_max;
        packed_out[1] = row_has_nan ? NAN : static_cast<float>(token_id);
      }
      if (token_ids_out != nullptr) {
        token_ids_out[0] = token_id;
      }
    }
  }
}


#undef DKR_SAMPLING_ENTRY
