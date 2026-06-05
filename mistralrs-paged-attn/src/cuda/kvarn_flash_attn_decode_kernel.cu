// Decode-only online-softmax attention over KVarN paged KV cache.
//
// Two entry points are exported:
//   - kvarn_flash_attn_decode: warp-level reduction for normal CUDA paths.
//   - kvarn_flash_attn_decode_cc61: shared-memory reductions for Pascal/cc61.
//
// Cache records match mistralrs-core/src/paged_attention/kvarn_cache.rs:
//   K: [status, q[D, G/2], s_col[D], zp[D], s_row[G]]
//   V: [status, q[G, D/4], s_col[D], s_row[G], zp[G]]
// with G=128, K4/V2, little-endian fp16 scales.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define KVARN_WARP_SIZE 32
#define KVARN_FULL_MASK 0xffffffffu

#define KVARN_CUDA_CHECK(call)                                                   \
  do {                                                                           \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "kvarn CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(err));                                          \
    }                                                                            \
  } while (0)

namespace kvarn_attn {

static constexpr int KVARN_GROUP = 128;
static constexpr int KVARN_KEY_BITS = 4;
static constexpr int KVARN_VALUE_BITS = 2;
static constexpr uint8_t KVARN_STATUS_QUANTIZED = 1;

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

__device__ __forceinline__ float read_f16(const uint8_t *src) {
  union {
    uint16_t u;
    __half h;
  } encoded;
  encoded.u = static_cast<uint16_t>(src[0]) |
              (static_cast<uint16_t>(src[1]) << 8);
  return __half2float(encoded.h);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = KVARN_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(KVARN_FULL_MASK, v, offset);
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_sum(float v) {
  __shared__ float scratch[BLOCK_SIZE];
  const int tid = threadIdx.x;
  scratch[tid] = v;
  __syncthreads();
#pragma unroll
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
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

template <int HEAD_DIM>
__device__ __forceinline__ void decode_kvarn_key_row_to_shared(
    const uint8_t *__restrict__ record,
    int block_offset,
    float *__restrict__ values) {
  if (record[0] != KVARN_STATUS_QUANTIZED) {
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) values[d] = 0.0f;
    __syncthreads();
    return;
  }

  constexpr int K_PACK = 8 / KVARN_KEY_BITS;
  constexpr int K_ROW_PACKED_BYTES = KVARN_GROUP / K_PACK;
  constexpr int PACKED_OFFSET = 1;
  constexpr int S_COL_OFFSET = PACKED_OFFSET + HEAD_DIM * K_ROW_PACKED_BYTES;
  constexpr int ZP_OFFSET = S_COL_OFFSET + HEAD_DIM * 2;
  constexpr int S_ROW_OFFSET = ZP_OFFSET + HEAD_DIM * 2;

  const float s_row = read_f16(record + S_ROW_OFFSET + block_offset * 2);
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    const int byte_idx = PACKED_OFFSET + d * K_ROW_PACKED_BYTES + block_offset / K_PACK;
    const uint8_t q = (record[byte_idx] >> ((block_offset % K_PACK) * KVARN_KEY_BITS)) & 0x0f;
    const float s_col = read_f16(record + S_COL_OFFSET + d * 2);
    const float zp = read_f16(record + ZP_OFFSET + d * 2);
    values[d] = (static_cast<float>(q) * s_col + zp) * s_row;
  }
  __syncthreads();

  fwht_shared(values, HEAD_DIM);

  const float inv_sqrt_dim = rsqrtf(static_cast<float>(HEAD_DIM));
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    values[d] *= inv_sqrt_dim;
  }
  __syncthreads();
}

template <int HEAD_DIM>
__device__ __forceinline__ void decode_kvarn_value_row_to_shared(
    const uint8_t *__restrict__ record,
    int block_offset,
    float *__restrict__ values) {
  if (record[0] != KVARN_STATUS_QUANTIZED) {
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) values[d] = 0.0f;
    __syncthreads();
    return;
  }

  constexpr int V_PACK = 8 / KVARN_VALUE_BITS;
  constexpr int V_ROW_PACKED_BYTES = HEAD_DIM / V_PACK;
  constexpr int PACKED_OFFSET = 1;
  constexpr int S_COL_OFFSET = PACKED_OFFSET + KVARN_GROUP * V_ROW_PACKED_BYTES;
  constexpr int S_ROW_OFFSET = S_COL_OFFSET + HEAD_DIM * 2;
  constexpr int ZP_OFFSET = S_ROW_OFFSET + KVARN_GROUP * 2;

  const float s_row = read_f16(record + S_ROW_OFFSET + block_offset * 2);
  const float zp = read_f16(record + ZP_OFFSET + block_offset * 2);
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    const int byte_idx = PACKED_OFFSET + block_offset * V_ROW_PACKED_BYTES + d / V_PACK;
    const uint8_t q = (record[byte_idx] >> ((d % V_PACK) * KVARN_VALUE_BITS)) & 0x03;
    const float s_col = read_f16(record + S_COL_OFFSET + d * 2);
    values[d] = (static_cast<float>(q) * s_row + zp) * s_col;
  }
  __syncthreads();

  fwht_shared(values, HEAD_DIM);

  const float inv_sqrt_dim = rsqrtf(static_cast<float>(HEAD_DIM));
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    values[d] *= inv_sqrt_dim;
  }
  __syncthreads();
}

template <typename scalar_t, int HEAD_DIM>
__global__ void kvarn_flash_attn_decode_kernel(
    const scalar_t *__restrict__ Q,
    const uint8_t *__restrict__ K_cache,
    const uint8_t *__restrict__ V_cache,
    const int *__restrict__ block_tables,
    const int *__restrict__ cu_seq_lens,
    scalar_t *__restrict__ O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int k_record_bytes,
    int v_record_bytes,
    float scale) {
  constexpr int D_PAD = ((HEAD_DIM + KVARN_WARP_SIZE - 1) / KVARN_WARP_SIZE) * KVARN_WARP_SIZE;
  constexpr int EPT = D_PAD / KVARN_WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int lane = threadIdx.x;
  if (head_idx >= num_heads || seq_idx >= num_seqs) return;

  const int seq_start = cu_seq_lens[seq_idx];
  const int seq_end = cu_seq_lens[seq_idx + 1];
  const int seq_len = seq_end - seq_start;
  const int out_base = (seq_idx * num_heads + head_idx) * HEAD_DIM;
  if (seq_len <= 0) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * KVARN_WARP_SIZE + lane;
      if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int q_base = out_base;

  extern __shared__ unsigned char smem[];
  float *values = reinterpret_cast<float *>(smem);

  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * KVARN_WARP_SIZE + lane;
    q_reg[i] = (d < HEAD_DIM) ? to_float(Q[q_base + d]) * scale : 0.0f;
  }

  float acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) acc[i] = 0.0f;

  float m = -FLT_MAX;
  float l = 0.0f;

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int logical_block = token_idx / KVARN_GROUP;
    const int block_offset = token_idx - logical_block * KVARN_GROUP;
    const int physical_block = block_tables[seq_idx * block_table_stride + logical_block];
    if (physical_block < 0) continue;

    const int64_t k_record_index =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
        static_cast<int64_t>(k_record_bytes);
    decode_kvarn_key_row_to_shared<HEAD_DIM>(
        K_cache + k_record_index, block_offset, values);

    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * KVARN_WARP_SIZE + lane;
      if (d < HEAD_DIM) dot_local += q_reg[i] * values[d];
    }
    const float score = warp_sum(dot_local);

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

    const int64_t v_record_index =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
        static_cast<int64_t>(v_record_bytes);
    decode_kvarn_value_row_to_shared<HEAD_DIM>(
        V_cache + v_record_index, block_offset, values);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * KVARN_WARP_SIZE + lane;
      const float vv = (d < HEAD_DIM) ? values[d] : 0.0f;
      acc[i] = acc[i] * alpha + beta * vv;
    }
    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * KVARN_WARP_SIZE + lane;
    if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(acc[i] * inv_l);
  }
}

template <typename scalar_t, int HEAD_DIM, int BLOCK>
__global__ void kvarn_flash_attn_decode_cc61_kernel(
    const scalar_t *__restrict__ Q,
    const uint8_t *__restrict__ K_cache,
    const uint8_t *__restrict__ V_cache,
    const int *__restrict__ block_tables,
    const int *__restrict__ cu_seq_lens,
    scalar_t *__restrict__ O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int k_record_bytes,
    int v_record_bytes,
    float scale) {
  constexpr int EPT = (HEAD_DIM + BLOCK - 1) / BLOCK;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int tid = threadIdx.x;
  if (head_idx >= num_heads || seq_idx >= num_seqs) return;

  const int seq_start = cu_seq_lens[seq_idx];
  const int seq_end = cu_seq_lens[seq_idx + 1];
  const int seq_len = seq_end - seq_start;
  const int out_base = (seq_idx * num_heads + head_idx) * HEAD_DIM;
  if (seq_len <= 0) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = tid + i * BLOCK;
      if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int q_base = out_base;

  extern __shared__ unsigned char smem[];
  float *values = reinterpret_cast<float *>(smem);

  float q_reg[EPT];
  float acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = tid + i * BLOCK;
    q_reg[i] = (d < HEAD_DIM) ? to_float(Q[q_base + d]) * scale : 0.0f;
    acc[i] = 0.0f;
  }

  float m = -FLT_MAX;
  float l = 0.0f;

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int logical_block = token_idx / KVARN_GROUP;
    const int block_offset = token_idx - logical_block * KVARN_GROUP;
    const int physical_block = block_tables[seq_idx * block_table_stride + logical_block];
    if (physical_block < 0) continue;

    const int64_t k_record_index =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
        static_cast<int64_t>(k_record_bytes);
    decode_kvarn_key_row_to_shared<HEAD_DIM>(
        K_cache + k_record_index, block_offset, values);

    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = tid + i * BLOCK;
      if (d < HEAD_DIM) dot_local += q_reg[i] * values[d];
    }
    const float score = block_sum<BLOCK>(dot_local);

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

    const int64_t v_record_index =
        (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
        static_cast<int64_t>(v_record_bytes);
    decode_kvarn_value_row_to_shared<HEAD_DIM>(
        V_cache + v_record_index, block_offset, values);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = tid + i * BLOCK;
      const float vv = (d < HEAD_DIM) ? values[d] : 0.0f;
      acc[i] = acc[i] * alpha + beta * vv;
    }
    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = tid + i * BLOCK;
    if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(acc[i] * inv_l);
  }
}

template <typename scalar_t>
void launch_kvarn_decode(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream) {
  const dim3 grid(num_heads, num_seqs);
  const dim3 block(KVARN_WARP_SIZE);
  const size_t shared_mem_bytes = head_dim * sizeof(float);

#define KVARN_LAUNCH(D)                                                            \
  kvarn_flash_attn_decode_kernel<scalar_t, D><<<grid, block, shared_mem_bytes, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                        \
      reinterpret_cast<const uint8_t *>(K_cache),                                   \
      reinterpret_cast<const uint8_t *>(V_cache),                                   \
      block_tables, cu_seq_lens, reinterpret_cast<scalar_t *>(O),                   \
      num_seqs, block_table_stride, num_heads, num_kv_heads,                        \
      k_record_bytes, v_record_bytes, scale)

  switch (head_dim) {
    case 32:  KVARN_LAUNCH(32); break;
    case 64:  KVARN_LAUNCH(64); break;
    case 128: KVARN_LAUNCH(128); break;
    case 256: KVARN_LAUNCH(256); break;
    case 512: KVARN_LAUNCH(512); break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef KVARN_LAUNCH
  KVARN_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void launch_kvarn_decode_cc61(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream) {
  constexpr int BLOCK = 128;
  const dim3 grid(num_heads, num_seqs);
  const dim3 block(BLOCK);
  const size_t shared_mem_bytes = head_dim * sizeof(float);

#define KVARN_CC61_LAUNCH(D)                                                       \
  kvarn_flash_attn_decode_cc61_kernel<scalar_t, D, BLOCK><<<grid, block, shared_mem_bytes, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                        \
      reinterpret_cast<const uint8_t *>(K_cache),                                   \
      reinterpret_cast<const uint8_t *>(V_cache),                                   \
      block_tables, cu_seq_lens, reinterpret_cast<scalar_t *>(O),                   \
      num_seqs, block_table_stride, num_heads, num_kv_heads,                        \
      k_record_bytes, v_record_bytes, scale)

  switch (head_dim) {
    case 32:  KVARN_CC61_LAUNCH(32); break;
    case 64:  KVARN_CC61_LAUNCH(64); break;
    case 128: KVARN_CC61_LAUNCH(128); break;
    case 256: KVARN_CC61_LAUNCH(256); break;
    case 512: KVARN_CC61_LAUNCH(512); break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_cc61: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef KVARN_CC61_LAUNCH
  KVARN_CUDA_CHECK(cudaGetLastError());
}

} // namespace kvarn_attn

extern "C" void kvarn_flash_attn_decode(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_seqs <= 0) return;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  if ((num_heads % num_kv_heads) != 0) {
    fprintf(stderr, "kvarn_flash_attn_decode: num_heads must be divisible by num_kv_heads\n");
    return;
  }

  switch (dtype) {
    case 0:
      kvarn_attn::launch_kvarn_decode<__half>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode<__nv_bfloat16>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode<float>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode: unsupported dtype=%u\n", dtype);
      break;
  }
}

extern "C" void kvarn_flash_attn_decode_cc61(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_seqs,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_seqs <= 0) return;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  if ((num_heads % num_kv_heads) != 0) {
    fprintf(stderr, "kvarn_flash_attn_decode_cc61: num_heads must be divisible by num_kv_heads\n");
    return;
  }

  switch (dtype) {
    case 0:
      kvarn_attn::launch_kvarn_decode_cc61<__half>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode_cc61<__nv_bfloat16>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode_cc61<float>(
          Q, K_cache, V_cache, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_cc61: unsupported dtype=%u\n", dtype);
      break;
  }
}
