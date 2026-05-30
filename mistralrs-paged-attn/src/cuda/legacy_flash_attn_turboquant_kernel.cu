// legacy_flash_attn_turboquant_kernel.cu
//
// Experimental SM61/Pascal-friendly decode-only attention over TurboQuant KV cache.
// This is NOT FlashAttention v2/v3. It is an online-softmax kernel that reads
// TurboQuant rows directly and does not materialize dense K/V globally.
//
// DType codes:
//   0 => f16, 1 => bf16, 2 => f32

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define LEGACY_TQ_WARP_SIZE 32
#define LEGACY_TQ_FULL_MASK 0xffffffffu

#define LEGACY_TQ_CUDA_CHECK(call)                                                   \
  do {                                                                               \
    cudaError_t err = (call);                                                        \
    if (err != cudaSuccess) {                                                        \
      fprintf(stderr, "legacy_turboquant CUDA error at %s:%d: %s\n", __FILE__,     \
              __LINE__, cudaGetErrorString(err));                                    \
    }                                                                                \
  } while (0)

namespace legacy_tq_attn {

static constexpr int TQ_MSE_BITS = 3;
static constexpr int TQ_NORM_BYTES = 4;
static constexpr float TQ_CLIP = 4.0f;

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}

__device__ __forceinline__ uint32_t read_bits(const uint8_t *data, int bit_offset, int bits) {
  uint32_t value = 0;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) {
    if (bit < bits) {
      const int pos = bit_offset + bit;
      value |= static_cast<uint32_t>((data[pos / 8] >> (pos % 8)) & 1) << bit;
    }
  }
  return value;
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

__device__ __forceinline__ float dequantize_scalar(uint32_t index, int bits) {
  const uint32_t levels = (1u << bits) - 1u;
  return -TQ_CLIP + (static_cast<float>(index) / static_cast<float>(levels)) *
                        (2.0f * TQ_CLIP);
}

__device__ __forceinline__ float deterministic_sign(uint32_t index) {
  uint32_t x = index;
  x += 0x9E3779B9U;
  x = (x ^ (x >> 15)) * 0x85EBCA6BU;
  x = (x ^ (x >> 13)) * 0xC2B2AE35U;
  return ((x ^ (x >> 16)) & 1U) == 0U ? 1.0f : -1.0f;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = LEGACY_TQ_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(LEGACY_TQ_FULL_MASK, v, offset);
  }
  return v;
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

// Decode one TurboQuant row into `values[0..head_dim]` in shared memory.
// This mirrors turboquant_gather_kv_cache_kernel math, but leaves the decoded
// row in shared memory for immediate QK/PV use.
__device__ __forceinline__ void decode_turboquant_row_to_shared(
    const uint8_t *__restrict__ row,
    int head_dim,
    int row_bytes,
    float *__restrict__ values,
    float *__restrict__ qjl_signs,
    uint8_t *__restrict__ shared_row) {
  for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) {
    shared_row[i] = row[i];
  }
  __syncthreads();

  const float norm = read_f32(shared_row);
  const float residual_norm = read_f32(shared_row + TQ_NORM_BYTES);
  const int mse_bytes = (head_dim * TQ_MSE_BITS + 7) / 8;
  const uint8_t *mse_bits = shared_row + 2 * TQ_NORM_BYTES;
  const uint8_t *qjl_bits = mse_bits + mse_bytes;

  if (norm == 0.0f || !isfinite(norm)) {
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) values[d] = 0.0f;
    __syncthreads();
    return;
  }

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    values[d] = dequantize_scalar(
        read_bits(mse_bits, d * TQ_MSE_BITS, TQ_MSE_BITS), TQ_MSE_BITS);
  }
  __syncthreads();

  fwht_shared(values, head_dim);

  const float inv_dim = 1.0f / static_cast<float>(head_dim);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    values[d] = deterministic_sign(static_cast<uint32_t>(d)) * values[d] * inv_dim * norm;
  }
  __syncthreads();

  if (residual_norm != 0.0f && isfinite(residual_norm)) {
    for (int r = threadIdx.x; r < head_dim; r += blockDim.x) {
      const float sign = read_bits(qjl_bits, r, 1) == 1 ? 1.0f : -1.0f;
      qjl_signs[r] = sign * deterministic_sign(static_cast<uint32_t>(r));
    }
    __syncthreads();

    fwht_shared(qjl_signs, head_dim);

    const float residual_scale = sqrtf(CUDART_PI_F * 0.5f) / static_cast<float>(head_dim);
    for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
      const float projected =
          deterministic_sign(static_cast<uint32_t>(col)) * qjl_signs[col];
      values[col] += residual_norm * residual_scale * projected;
    }
    __syncthreads();
  }
}

template <typename scalar_t, int HEAD_DIM>
__global__ void legacy_flash_attn_decode_turboquant_kernel(
    const scalar_t *__restrict__ Q,        // [num_seqs, Hq, D] or [num_seqs,Hq,1,D]
    const uint8_t *__restrict__ K_cache,   // [num_blocks, Hkv, block_size, k_row_bytes]
    const uint8_t *__restrict__ V_cache,   // [num_blocks, Hkv, block_size, v_row_bytes]
    const int *__restrict__ block_tables,  // [num_seqs, block_table_stride]
    const int *__restrict__ cu_seq_lens,   // [num_seqs + 1]
    scalar_t *__restrict__ O,              // [num_seqs, Hq, D] or [num_seqs,Hq,1,D]
    int num_seqs,
    int block_size,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int k_row_bytes,
    int v_row_bytes,
    float scale,
    int window_size) {
  constexpr int D_PAD = ((HEAD_DIM + LEGACY_TQ_WARP_SIZE - 1) / LEGACY_TQ_WARP_SIZE) * LEGACY_TQ_WARP_SIZE;
  constexpr int EPT = D_PAD / LEGACY_TQ_WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int lane = threadIdx.x;

  if (head_idx >= num_heads || seq_idx >= num_seqs) return;

  const int seq_start = cu_seq_lens[seq_idx];
  const int seq_end = cu_seq_lens[seq_idx + 1];
  int seq_len = seq_end - seq_start;
  if (seq_len <= 0) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_TQ_WARP_SIZE + lane;
      if (d < HEAD_DIM) O[(seq_idx * num_heads + head_idx) * HEAD_DIM + d] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int q_base = (seq_idx * num_heads + head_idx) * HEAD_DIM;
  const int out_base = q_base;

  extern __shared__ unsigned char smem[];
  float *values = reinterpret_cast<float *>(smem);
  float *qjl_signs = values + HEAD_DIM;
  uint8_t *shared_row = reinterpret_cast<uint8_t *>(qjl_signs + HEAD_DIM);

  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_TQ_WARP_SIZE + lane;
    q_reg[i] = (d < HEAD_DIM) ? to_float(Q[q_base + d]) * scale : 0.0f;
  }

  float acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) acc[i] = 0.0f;

  float m = -FLT_MAX;
  float l = 0.0f;

  int start = 0;
  if (window_size > 0 && seq_len > window_size) start = seq_len - window_size;

  for (int token_idx = start; token_idx < seq_len; ++token_idx) {
    const int logical_block = token_idx / block_size;
    const int block_offset = token_idx - logical_block * block_size;
    const int physical_block = block_tables[seq_idx * block_table_stride + logical_block];
    if (physical_block < 0) continue;

    const int64_t k_row_index =
        ((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) * block_size + block_offset) *
        static_cast<int64_t>(k_row_bytes);
    const uint8_t *k_row = K_cache + k_row_index;

    decode_turboquant_row_to_shared(k_row, HEAD_DIM, k_row_bytes, values, qjl_signs, shared_row);

    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_TQ_WARP_SIZE + lane;
      if (d < HEAD_DIM) dot_local += q_reg[i] * values[d];
    }
    const float score = warp_sum(dot_local);

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

    const int64_t v_row_index =
        ((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) * block_size + block_offset) *
        static_cast<int64_t>(v_row_bytes);
    const uint8_t *v_row = V_cache + v_row_index;

    decode_turboquant_row_to_shared(v_row, HEAD_DIM, v_row_bytes, values, qjl_signs, shared_row);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_TQ_WARP_SIZE + lane;
      const float vv = (d < HEAD_DIM) ? values[d] : 0.0f;
      acc[i] = acc[i] * alpha + beta * vv;
    }

    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_TQ_WARP_SIZE + lane;
    if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(acc[i] * inv_l);
  }
}

template <typename scalar_t>
void legacy_flash_attn_decode_turboquant_launch(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_size,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_row_bytes,
    int v_row_bytes,
    float scale,
    int window_size,
    cudaStream_t stream) {
  const dim3 grid(num_heads, num_seqs);
  const dim3 block(LEGACY_TQ_WARP_SIZE);
  const int max_row_bytes = k_row_bytes > v_row_bytes ? k_row_bytes : v_row_bytes;
  const size_t shared_mem_bytes = 2 * head_dim * sizeof(float) + max_row_bytes;

#define LEGACY_TQ_LAUNCH(D)                                                         \
  legacy_flash_attn_decode_turboquant_kernel<scalar_t, D><<<grid, block, shared_mem_bytes, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                        \
      reinterpret_cast<const uint8_t *>(K_cache),                                   \
      reinterpret_cast<const uint8_t *>(V_cache),                                   \
      block_tables, cu_seq_lens, reinterpret_cast<scalar_t *>(O),                   \
      num_seqs, block_size, block_table_stride, num_heads, num_kv_heads,            \
      k_row_bytes, v_row_bytes, scale, window_size)

  switch (head_dim) {
    case 32:  LEGACY_TQ_LAUNCH(32); break;
    case 64:  LEGACY_TQ_LAUNCH(64); break;
    case 80:  LEGACY_TQ_LAUNCH(80); break;
    case 96:  LEGACY_TQ_LAUNCH(96); break;
    case 112: LEGACY_TQ_LAUNCH(112); break;
    case 128: LEGACY_TQ_LAUNCH(128); break;
    case 160: LEGACY_TQ_LAUNCH(160); break;
    case 192: LEGACY_TQ_LAUNCH(192); break;
    case 256: LEGACY_TQ_LAUNCH(256); break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_turboquant: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef LEGACY_TQ_LAUNCH
  LEGACY_TQ_CUDA_CHECK(cudaGetLastError());
}

} // namespace legacy_tq_attn

extern "C" void legacy_flash_attn_decode_turboquant(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_size,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_row_bytes,
    int v_row_bytes,
    float scale,
    int window_size,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_seqs <= 0) return;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  if ((num_heads % num_kv_heads) != 0) {
    fprintf(stderr, "legacy_flash_attn_decode_turboquant: num_heads must be divisible by num_kv_heads\n");
    return;
  }

  switch (dtype) {
    case 0:
      legacy_tq_attn::legacy_flash_attn_decode_turboquant_launch<__half>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_size, block_table_stride, num_heads, num_kv_heads, head_dim,
          k_row_bytes, v_row_bytes, scale, window_size, stream);
      break;
    case 1:
      legacy_tq_attn::legacy_flash_attn_decode_turboquant_launch<__nv_bfloat16>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_size, block_table_stride, num_heads, num_kv_heads, head_dim,
          k_row_bytes, v_row_bytes, scale, window_size, stream);
      break;
    case 2:
      legacy_tq_attn::legacy_flash_attn_decode_turboquant_launch<float>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_size, block_table_stride, num_heads, num_kv_heads, head_dim,
          k_row_bytes, v_row_bytes, scale, window_size, stream);
      break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_turboquant: unsupported dtype=%u\n", dtype);
      break;
  }
}
