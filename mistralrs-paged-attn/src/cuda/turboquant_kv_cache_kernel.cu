#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#include "cuda_compat.h"

#include <algorithm>
#include <math_constants.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(err);                                                               \
    }                                                                          \
  } while (0)

namespace vllm {

static constexpr int TQ_MSE_BITS = 3;
static constexpr int TQ_NORM_BYTES = 4;
static constexpr float TQ_CLIP = 4.0f;

__device__ __forceinline__ uint32_t read_bits(const uint8_t *data,
                                              int bit_offset, int bits) {
  uint32_t value = 0;
  for (int bit = 0; bit < bits; ++bit) {
    const int pos = bit_offset + bit;
    value |= static_cast<uint32_t>((data[pos / 8] >> (pos % 8)) & 1) << bit;
  }
  return value;
}

__device__ __forceinline__ void write_bits(uint8_t *data, int bit_offset,
                                           uint32_t value, int bits) {
  for (int bit = 0; bit < bits; ++bit) {
    if (((value >> bit) & 1u) != 0u) {
      const int pos = bit_offset + bit;
      data[pos / 8] |= static_cast<uint8_t>(1u << (pos % 8));
    }
  }
}

__device__ __forceinline__ void write_f32(uint8_t *dst, float value) {
  union {
    float f;
    uint8_t bytes[4];
  } encoded;
  encoded.f = value;
  dst[0] = encoded.bytes[0];
  dst[1] = encoded.bytes[1];
  dst[2] = encoded.bytes[2];
  dst[3] = encoded.bytes[3];
}

__device__ __forceinline__ float read_f32(const uint8_t *src) {
  union {
    float f;
    uint8_t bytes[4];
  } encoded;
  encoded.bytes[0] = src[0];
  encoded.bytes[1] = src[1];
  encoded.bytes[2] = src[2];
  encoded.bytes[3] = src[3];
  return encoded.f;
}

__device__ __forceinline__ uint32_t quantize_scalar(float value, int bits) {
  const uint32_t levels = (1u << bits) - 1u;
  value = fminf(fmaxf(value, -TQ_CLIP), TQ_CLIP);
  return static_cast<uint32_t>(
      roundf(((value + TQ_CLIP) / (2.0f * TQ_CLIP)) *
             static_cast<float>(levels)));
}

__device__ __forceinline__ float dequantize_scalar(uint32_t index, int bits) {
  const uint32_t levels = (1u << bits) - 1u;
  return -TQ_CLIP + (static_cast<float>(index) / static_cast<float>(levels)) *
                        (2.0f * TQ_CLIP);
}

__device__ __forceinline__ float deterministic_sign(uint64_t index) {
  uint64_t x = index;
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return ((x ^ (x >> 31)) & 1ULL) == 0ULL ? 1.0f : -1.0f;
}

__device__ __forceinline__ float qjl_projection_sign(int row, int col) {
  const uint64_t index =
      static_cast<uint64_t>(row) * 0x9E3779B1ULL +
      static_cast<uint64_t>(col) * 0x85EBCA77ULL;
  return deterministic_sign(index);
}

template <typename in_t>
__device__ __forceinline__ float load_input(const in_t *ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_input<uint16_t>(const uint16_t *ptr) {
  return __half2float(__ushort_as_half(*ptr));
}

template <>
__device__ __forceinline__ float load_input<__nv_bfloat16>(
    const __nv_bfloat16 *ptr) {
  return __bfloat162float(*ptr);
}

__device__ __forceinline__ float block_sum(float value, float *scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

__device__ __forceinline__ void fwht_shared(float *values, int head_dim) {
  for (int h = 1; h < head_dim; h <<= 1) {
    const int pairs = head_dim >> 1;
    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
      const int base = (pair / h) * (h << 1);
      const int j = base + (pair % h);
      const float x = values[j];
      const float y = values[j + h];
      values[j] = x + y;
      values[j + h] = x - y;
    }
    __syncthreads();
  }
}

template <typename out_t>
__device__ __forceinline__ void store_output(out_t *out, int64_t index,
                                             float value) {
  out[index] = static_cast<out_t>(value);
}

template <>
__device__ __forceinline__ void store_output<uint16_t>(uint16_t *out,
                                                       int64_t index,
                                                       float value) {
  out[index] = __half_as_ushort(__float2half_rn(value));
}

template <>
__device__ __forceinline__ void store_output<__nv_bfloat16>(
    __nv_bfloat16 *out, int64_t index, float value) {
  out[index] = __float2bfloat16(value);
}

template <typename in_t>
__global__ void turboquant_reshape_and_cache_kernel(
    const in_t *__restrict__ key,
    const in_t *__restrict__ value,
    uint8_t *__restrict__ key_cache,
    uint8_t *__restrict__ value_cache,
    const int64_t *__restrict__ slot_mapping,
    const int32_t num_tokens,
    const int32_t num_heads,
    const int32_t k_head_dim,
    const int32_t v_head_dim,
    const int32_t block_size,
    const int32_t num_blocks,
    const int32_t k_row_bytes,
    const int32_t v_row_bytes,
    const int32_t key_stride,
    const int32_t value_stride) {
  const int32_t token_id = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const bool is_value = blockIdx.z != 0;
  if (token_id >= num_tokens || head_idx >= num_heads) {
    return;
  }

  const int64_t slot = slot_mapping[token_id];
  if (slot < 0) {
    return;
  }
  const int32_t block_id = static_cast<int32_t>(slot / block_size);
  const int32_t block_offset = static_cast<int32_t>(slot % block_size);
  if (block_id < 0 || block_id >= num_blocks) {
    return;
  }

  const int32_t head_dim = is_value ? v_head_dim : k_head_dim;
  const int32_t row_bytes = is_value ? v_row_bytes : k_row_bytes;
  const in_t *src = is_value ? value : key;
  const int32_t src_stride = is_value ? value_stride : key_stride;
  uint8_t *cache = is_value ? value_cache : key_cache;
  uint8_t *row =
      cache + (((static_cast<int64_t>(block_id) * num_heads + head_idx) *
                    block_size +
                block_offset) *
               row_bytes);

  extern __shared__ float shared[];
  const int32_t max_head_dim = k_head_dim > v_head_dim ? k_head_dim : v_head_dim;
  float *original = shared;
  float *values = shared + max_head_dim;
  float *scratch = shared + 2 * max_head_dim;

  for (int byte = threadIdx.x; byte < row_bytes; byte += blockDim.x) {
    row[byte] = 0;
  }
  __syncthreads();

  float local_norm = 0.0f;
  const int64_t src_base =
      static_cast<int64_t>(token_id) * src_stride +
      static_cast<int64_t>(head_idx) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float x = load_input(src + src_base + d);
    original[d] = x;
    local_norm += x * x;
  }
  const float norm = sqrtf(block_sum(local_norm, scratch));
  if (threadIdx.x == 0) {
    write_f32(row, norm);
  }
  __syncthreads();

  if (norm == 0.0f || !isfinite(norm)) {
    return;
  }

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    values[d] = deterministic_sign(static_cast<uint64_t>(d)) * original[d] /
                norm;
  }
  __syncthreads();
  fwht_shared(values, head_dim);

  const int32_t mse_bytes = (head_dim * TQ_MSE_BITS + 7) / 8;
  uint8_t *mse_bits = row + 2 * TQ_NORM_BYTES;
  uint8_t *qjl_bits = mse_bits + mse_bytes;
  if (threadIdx.x == 0) {
    for (int d = 0; d < head_dim; ++d) {
      const uint32_t quantized = quantize_scalar(values[d], TQ_MSE_BITS);
      write_bits(mse_bits, d * TQ_MSE_BITS, quantized, TQ_MSE_BITS);
      values[d] = dequantize_scalar(quantized, TQ_MSE_BITS);
    }
  }
  __syncthreads();

  fwht_shared(values, head_dim);
  const float inv_dim = 1.0f / static_cast<float>(head_dim);
  float local_residual_norm = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float reconstructed =
        deterministic_sign(static_cast<uint64_t>(d)) * values[d] * inv_dim *
        norm;
    const float residual = original[d] - reconstructed;
    original[d] = residual;
    local_residual_norm += residual * residual;
  }
  const float residual_norm = sqrtf(block_sum(local_residual_norm, scratch));
  if (threadIdx.x == 0) {
    write_f32(row + TQ_NORM_BYTES, residual_norm);
  }
  __syncthreads();

  if (residual_norm == 0.0f || !isfinite(residual_norm)) {
    return;
  }

  for (int projection_row = threadIdx.x; projection_row < head_dim;
       projection_row += blockDim.x) {
    float projected = 0.0f;
    for (int col = 0; col < head_dim; ++col) {
      projected += qjl_projection_sign(projection_row, col) * original[col];
    }
    values[projection_row] = projected >= 0.0f ? 1.0f : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int projection_row = 0; projection_row < head_dim; ++projection_row) {
      write_bits(qjl_bits, projection_row,
                 values[projection_row] > 0.0f ? 1u : 0u, 1);
    }
  }
}

template <typename out_t>
__global__ void turboquant_gather_kv_cache_kernel(
    const uint8_t *__restrict__ key_cache,
    const uint8_t *__restrict__ value_cache,
    out_t *__restrict__ k_out,
    out_t *__restrict__ v_out,
    const int32_t *__restrict__ block_table,
    const int32_t *__restrict__ cu_seq_lens,
    const int32_t num_tokens,
    const int32_t num_seqs,
    const int32_t block_size,
    const int32_t block_table_stride,
    const int32_t num_kv_heads,
    const int32_t k_head_dim,
    const int32_t v_head_dim,
    const int32_t k_row_bytes,
    const int32_t v_row_bytes) {
  const int32_t token_id = blockIdx.x;
  const int32_t head_idx = blockIdx.y;
  const bool is_value = blockIdx.z != 0;
  if (token_id >= num_tokens || head_idx >= num_kv_heads) {
    return;
  }

  int32_t lo = 0;
  int32_t hi = num_seqs;
  while (lo < hi) {
    const int32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= token_id) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int32_t batch_id = lo;
  const int32_t batch_offset = token_id - cu_seq_lens[batch_id];
  const int32_t block_table_id = batch_offset / block_size;
  const int32_t slot = batch_offset % block_size;
  const int32_t block_id =
      block_table[batch_id * block_table_stride + block_table_id];

  const int32_t head_dim = is_value ? v_head_dim : k_head_dim;
  const int32_t row_bytes = is_value ? v_row_bytes : k_row_bytes;
  out_t *out = is_value ? v_out : k_out;
  if (block_id < 0) {
    const int64_t out_base =
        (static_cast<int64_t>(token_id) * num_kv_heads + head_idx) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      store_output(out, out_base + d, 0.0f);
    }
    return;
  }

  const uint8_t *cache = is_value ? value_cache : key_cache;
  const int64_t row_index =
      ((static_cast<int64_t>(block_id) * num_kv_heads + head_idx) *
           block_size +
       slot) *
      row_bytes;
  const uint8_t *row = cache + row_index;

  extern __shared__ float shared[];
  float *values = shared;
  float *qjl_signs = shared + (k_head_dim > v_head_dim ? k_head_dim : v_head_dim);

  const float norm = read_f32(row);
  const float residual_norm = read_f32(row + TQ_NORM_BYTES);
  const int32_t mse_bytes = (head_dim * TQ_MSE_BITS + 7) / 8;
  const uint8_t *mse_bits = row + 2 * TQ_NORM_BYTES;
  const uint8_t *qjl_bits = mse_bits + mse_bytes;

  if (norm == 0.0f || !isfinite(norm)) {
    const int64_t out_base =
        (static_cast<int64_t>(token_id) * num_kv_heads + head_idx) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
      store_output(out, out_base + d, 0.0f);
    }
    return;
  }

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    values[d] = dequantize_scalar(read_bits(mse_bits, d * TQ_MSE_BITS,
                                            TQ_MSE_BITS),
                                  TQ_MSE_BITS);
  }
  __syncthreads();

  fwht_shared(values, head_dim);

  const float inv_dim = 1.0f / static_cast<float>(head_dim);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    values[d] = deterministic_sign(static_cast<uint64_t>(d)) * values[d] *
                inv_dim * norm;
  }
  __syncthreads();

  if (residual_norm != 0.0f && isfinite(residual_norm)) {
    for (int r = threadIdx.x; r < head_dim; r += blockDim.x) {
      qjl_signs[r] = read_bits(qjl_bits, r, 1) == 1 ? 1.0f : -1.0f;
    }
    __syncthreads();

    const float residual_scale =
        sqrtf(CUDART_PI_F * 0.5f) / static_cast<float>(head_dim);
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      float projected = 0.0f;
      for (int row_idx = 0; row_idx < head_dim; ++row_idx) {
        projected += qjl_signs[row_idx] * qjl_projection_sign(row_idx, col);
      }
      values[col] += residual_norm * residual_scale * projected;
    }
    __syncthreads();
  }

  const int64_t out_base =
      (static_cast<int64_t>(token_id) * num_kv_heads + head_idx) * head_dim;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    store_output(out, out_base + d, values[d]);
  }
}

} // namespace vllm

#define CALL_TURBOQUANT_RESHAPE(IN_T)                                         \
  vllm::turboquant_reshape_and_cache_kernel<IN_T>                             \
      <<<grid, block, shared_mem_bytes, stream>>>(                             \
          reinterpret_cast<const IN_T *>(key),                                 \
          reinterpret_cast<const IN_T *>(value),                               \
          reinterpret_cast<uint8_t *>(key_cache),                              \
          reinterpret_cast<uint8_t *>(value_cache), slot_mapping, num_tokens,  \
          num_heads, k_head_dim, v_head_dim, block_size, num_blocks,           \
          k_row_bytes, v_row_bytes, key_stride, value_stride);

extern "C" void turboquant_reshape_and_cache(
    void *key,
    void *value,
    void *key_cache,
    void *value_cache,
    const int64_t *slot_mapping,
    int32_t num_tokens,
    int32_t num_heads,
    int32_t k_head_dim,
    int32_t v_head_dim,
    int32_t block_size,
    int32_t num_blocks,
    int32_t k_row_bytes,
    int32_t v_row_bytes,
    int32_t key_stride,
    int32_t value_stride,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_tokens <= 0) {
    return;
  }

  const int32_t max_head_dim = std::max(k_head_dim, v_head_dim);
  dim3 grid(num_tokens, num_heads, 2);
  dim3 block(std::min(max_head_dim, 512));
  const size_t shared_mem_bytes =
      (2 * max_head_dim + static_cast<int32_t>(block.x)) * sizeof(float);

  if (dtype == 0) {
    CALL_TURBOQUANT_RESHAPE(uint16_t);
  } else if (dtype == 1) {
    CALL_TURBOQUANT_RESHAPE(__nv_bfloat16);
  } else if (dtype == 2) {
    CALL_TURBOQUANT_RESHAPE(float);
  }
  CUDA_CHECK(cudaGetLastError());
}

#define CALL_TURBOQUANT_GATHER(OUT_T)                                         \
  vllm::turboquant_gather_kv_cache_kernel<OUT_T>                              \
      <<<grid, block, shared_mem_bytes, stream>>>(                             \
          reinterpret_cast<const uint8_t *>(key_cache),                        \
          reinterpret_cast<const uint8_t *>(value_cache),                      \
          reinterpret_cast<OUT_T *>(k_out), reinterpret_cast<OUT_T *>(v_out),  \
          block_table, cu_seq_lens, num_tokens, num_seqs, block_size,          \
          block_table_stride, num_kv_heads, k_head_dim, v_head_dim,            \
          k_row_bytes, v_row_bytes);

extern "C" void turboquant_gather_kv_cache(
    void *key_cache,
    void *value_cache,
    void *k_out,
    void *v_out,
    const int32_t *block_table,
    const int32_t *cu_seq_lens,
    int32_t num_tokens,
    int32_t num_seqs,
    int32_t block_size,
    int32_t block_table_stride,
    int32_t num_kv_heads,
    int32_t k_head_dim,
    int32_t v_head_dim,
    int32_t k_row_bytes,
    int32_t v_row_bytes,
    cudaStream_t stream,
    uint32_t out_dtype) {
  if (num_tokens <= 0) {
    return;
  }

  const int32_t max_head_dim = std::max(k_head_dim, v_head_dim);
  dim3 grid(num_tokens, num_kv_heads, 2);
  dim3 block(std::min(max_head_dim, 512));
  const size_t shared_mem_bytes = 2 * max_head_dim * sizeof(float);

  if (out_dtype == 0) {
    CALL_TURBOQUANT_GATHER(uint16_t);
  } else if (out_dtype == 1) {
    CALL_TURBOQUANT_GATHER(__nv_bfloat16);
  } else if (out_dtype == 2) {
    CALL_TURBOQUANT_GATHER(float);
  }
  CUDA_CHECK(cudaGetLastError());
}
