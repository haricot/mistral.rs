// Decode-only online-softmax attention over KVarN paged KV cache.
//
// Two entry points are exported:
//   - kvarn_flash_attn_decode: warp-level reduction for normal CUDA paths.
//   - kvarn_flash_attn_decode_cc61: shared-memory reductions for Pascal/cc61.
//   - kvarn_flash_attn_decode_mtp: fused multi-query decode for MTP verification.
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
static constexpr int KVARN_MTP_MAX_Q = 8;
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

template <int BLOCK_SIZE>
__device__ __forceinline__ float decode_sum(float v) {
  return block_sum<BLOCK_SIZE>(v);
}

template <>
__device__ __forceinline__ float decode_sum<KVARN_WARP_SIZE>(float v) {
  return warp_sum(v);
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
__device__ __forceinline__ void load_kvarn_tail_row_to_shared(
    const void *__restrict__ tail_pool,
    const int *__restrict__ block_to_tail_slot,
    int physical_block,
    int kv_head_idx,
    int num_kv_heads,
    int block_offset,
    float *__restrict__ values) {
  if (tail_pool == nullptr || block_to_tail_slot == nullptr) {
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) values[d] = 0.0f;
    __syncthreads();
    return;
  }

  const int tail_slot = block_to_tail_slot[physical_block];
  if (tail_slot < 0) {
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) values[d] = 0.0f;
    __syncthreads();
    return;
  }

  const scalar_t *raw = reinterpret_cast<const scalar_t *>(tail_pool);
  const int64_t base =
      (((static_cast<int64_t>(tail_slot) * num_kv_heads + kv_head_idx) *
            KVARN_GROUP +
        block_offset) *
       HEAD_DIM);
  for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
    values[d] = to_float(raw[base + d]);
  }
  __syncthreads();
}

template <typename scalar_t, int HEAD_DIM>
__global__ void kvarn_flash_attn_decode_kernel(
    const scalar_t *__restrict__ Q,
    const uint8_t *__restrict__ K_cache,
    const uint8_t *__restrict__ V_cache,
    const void *__restrict__ K_tail_pool,
    const void *__restrict__ V_tail_pool,
    const int *__restrict__ K_tail_slots,
    const int *__restrict__ V_tail_slots,
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
    const uint8_t *k_record = K_cache + k_record_index;
    if (k_record[0] == KVARN_STATUS_QUANTIZED) {
      decode_kvarn_key_row_to_shared<HEAD_DIM>(k_record, block_offset, values);
    } else {
      load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
          K_tail_pool, K_tail_slots, physical_block, kv_head_idx, num_kv_heads,
          block_offset, values);
    }

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
    const uint8_t *v_record = V_cache + v_record_index;
    if (v_record[0] == KVARN_STATUS_QUANTIZED) {
      decode_kvarn_value_row_to_shared<HEAD_DIM>(v_record, block_offset, values);
    } else {
      load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
          V_tail_pool, V_tail_slots, physical_block, kv_head_idx, num_kv_heads,
          block_offset, values);
    }

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
__global__ void kvarn_flash_attn_decode_mtp_kernel(
    const scalar_t *__restrict__ Q,
    const uint8_t *__restrict__ K_cache,
    const uint8_t *__restrict__ V_cache,
    const void *__restrict__ K_tail_pool,
    const void *__restrict__ V_tail_pool,
    const int *__restrict__ K_tail_slots,
    const int *__restrict__ V_tail_slots,
    const int *__restrict__ block_tables,
    const int *__restrict__ cu_seq_lens,
    scalar_t *__restrict__ O,
    int num_queries,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int k_record_bytes,
    int v_record_bytes,
    float scale) {
  constexpr int EPT = (HEAD_DIM + BLOCK - 1) / BLOCK;
  const int head_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (head_idx >= num_heads || num_queries <= 0 || num_queries > KVARN_MTP_MAX_Q) return;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  extern __shared__ unsigned char smem[];
  float *values = reinterpret_cast<float *>(smem);

  int seq_lens[KVARN_MTP_MAX_Q];
  int max_seq_len = 0;
  float m[KVARN_MTP_MAX_Q];
  float l[KVARN_MTP_MAX_Q];
  float q_reg[KVARN_MTP_MAX_Q][EPT];
  float acc[KVARN_MTP_MAX_Q][EPT];

#pragma unroll
  for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
    if (q < num_queries) {
      seq_lens[q] = cu_seq_lens[q + 1] - cu_seq_lens[q];
      if (seq_lens[q] > max_seq_len) max_seq_len = seq_lens[q];
    } else {
      seq_lens[q] = 0;
    }
    m[q] = -FLT_MAX;
    l[q] = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * BLOCK + tid;
      const int q_base = (q * num_heads + head_idx) * HEAD_DIM;
      q_reg[q][i] = (q < num_queries && d < HEAD_DIM) ? to_float(Q[q_base + d]) * scale : 0.0f;
      acc[q][i] = 0.0f;
    }
  }

  for (int token_idx = 0; token_idx < max_seq_len; ++token_idx) {
    const int logical_block = token_idx / KVARN_GROUP;
    const int block_offset = token_idx - logical_block * KVARN_GROUP;

    bool any_active = false;
    bool common_block = true;
    int shared_physical_block = -1;
#pragma unroll
    for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
      if (q >= num_queries || token_idx >= seq_lens[q]) continue;
      const int physical_block = block_tables[q * block_table_stride + logical_block];
      if (physical_block < 0) continue;
      if (!any_active) {
        shared_physical_block = physical_block;
        any_active = true;
      } else if (physical_block != shared_physical_block) {
        common_block = false;
      }
    }
    if (!any_active) continue;

    if (common_block) {
      const int64_t k_record_index =
          (static_cast<int64_t>(shared_physical_block) * num_kv_heads + kv_head_idx) *
          static_cast<int64_t>(k_record_bytes);
      const uint8_t *k_record = K_cache + k_record_index;
      if (k_record[0] == KVARN_STATUS_QUANTIZED) {
        decode_kvarn_key_row_to_shared<HEAD_DIM>(k_record, block_offset, values);
      } else {
        load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
            K_tail_pool, K_tail_slots, shared_physical_block, kv_head_idx,
            num_kv_heads, block_offset, values);
      }

      float beta[KVARN_MTP_MAX_Q];
      float alpha[KVARN_MTP_MAX_Q];
#pragma unroll
      for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
        beta[q] = 0.0f;
        alpha[q] = 1.0f;
        if (q >= num_queries || token_idx >= seq_lens[q]) continue;

        float dot_local = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int d = i * BLOCK + tid;
          if (d < HEAD_DIM) dot_local += q_reg[q][i] * values[d];
        }
        const float score = decode_sum<BLOCK>(dot_local);
        const float m_new = fmaxf(m[q], score);
        alpha[q] = expf(m[q] - m_new);
        beta[q] = expf(score - m_new);
        m[q] = m_new;
      }

      const int64_t v_record_index =
          (static_cast<int64_t>(shared_physical_block) * num_kv_heads + kv_head_idx) *
          static_cast<int64_t>(v_record_bytes);
      const uint8_t *v_record = V_cache + v_record_index;
      if (v_record[0] == KVARN_STATUS_QUANTIZED) {
        decode_kvarn_value_row_to_shared<HEAD_DIM>(v_record, block_offset, values);
      } else {
        load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
            V_tail_pool, V_tail_slots, shared_physical_block, kv_head_idx,
            num_kv_heads, block_offset, values);
      }

#pragma unroll
      for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
        if (q >= num_queries || token_idx >= seq_lens[q]) continue;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int d = i * BLOCK + tid;
          const float vv = (d < HEAD_DIM) ? values[d] : 0.0f;
          acc[q][i] = acc[q][i] * alpha[q] + beta[q] * vv;
        }
        l[q] = l[q] * alpha[q] + beta[q];
      }
    } else {
#pragma unroll
      for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
        if (q >= num_queries || token_idx >= seq_lens[q]) continue;
        const int physical_block = block_tables[q * block_table_stride + logical_block];
        if (physical_block < 0) continue;

        const int64_t k_record_index =
            (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
            static_cast<int64_t>(k_record_bytes);
        const uint8_t *k_record = K_cache + k_record_index;
        if (k_record[0] == KVARN_STATUS_QUANTIZED) {
          decode_kvarn_key_row_to_shared<HEAD_DIM>(k_record, block_offset, values);
        } else {
          load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
              K_tail_pool, K_tail_slots, physical_block, kv_head_idx,
              num_kv_heads, block_offset, values);
        }

        float dot_local = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int d = i * BLOCK + tid;
          if (d < HEAD_DIM) dot_local += q_reg[q][i] * values[d];
        }
        const float score = decode_sum<BLOCK>(dot_local);
        const float m_new = fmaxf(m[q], score);
        const float alpha = expf(m[q] - m_new);
        const float beta = expf(score - m_new);

        const int64_t v_record_index =
            (static_cast<int64_t>(physical_block) * num_kv_heads + kv_head_idx) *
            static_cast<int64_t>(v_record_bytes);
        const uint8_t *v_record = V_cache + v_record_index;
        if (v_record[0] == KVARN_STATUS_QUANTIZED) {
          decode_kvarn_value_row_to_shared<HEAD_DIM>(v_record, block_offset, values);
        } else {
          load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
              V_tail_pool, V_tail_slots, physical_block, kv_head_idx,
              num_kv_heads, block_offset, values);
        }

#pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int d = i * BLOCK + tid;
          const float vv = (d < HEAD_DIM) ? values[d] : 0.0f;
          acc[q][i] = acc[q][i] * alpha + beta * vv;
        }
        l[q] = l[q] * alpha + beta;
        m[q] = m_new;
      }
    }
  }

#pragma unroll
  for (int q = 0; q < KVARN_MTP_MAX_Q; ++q) {
    if (q >= num_queries) continue;
    const float inv_l = (l[q] > 0.0f) ? (1.0f / l[q]) : 0.0f;
    const int out_base = (q * num_heads + head_idx) * HEAD_DIM;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * BLOCK + tid;
      if (d < HEAD_DIM) O[out_base + d] = from_float<scalar_t>(acc[q][i] * inv_l);
    }
  }
}

template <typename scalar_t, int HEAD_DIM, int BLOCK>
__global__ void kvarn_flash_attn_decode_cc61_kernel(
    const scalar_t *__restrict__ Q,
    const uint8_t *__restrict__ K_cache,
    const uint8_t *__restrict__ V_cache,
    const void *__restrict__ K_tail_pool,
    const void *__restrict__ V_tail_pool,
    const int *__restrict__ K_tail_slots,
    const int *__restrict__ V_tail_slots,
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
    const uint8_t *k_record = K_cache + k_record_index;
    if (k_record[0] == KVARN_STATUS_QUANTIZED) {
      decode_kvarn_key_row_to_shared<HEAD_DIM>(k_record, block_offset, values);
    } else {
      load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
          K_tail_pool, K_tail_slots, physical_block, kv_head_idx, num_kv_heads,
          block_offset, values);
    }

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
    const uint8_t *v_record = V_cache + v_record_index;
    if (v_record[0] == KVARN_STATUS_QUANTIZED) {
      decode_kvarn_value_row_to_shared<HEAD_DIM>(v_record, block_offset, values);
    } else {
      load_kvarn_tail_row_to_shared<scalar_t, HEAD_DIM>(
          V_tail_pool, V_tail_slots, physical_block, kv_head_idx, num_kv_heads,
          block_offset, values);
    }

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
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
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
      K_tail_pool, V_tail_pool, K_tail_slots, V_tail_slots,                         \
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
void launch_kvarn_decode_mtp(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_queries,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream) {
  const dim3 grid(num_heads);
  const dim3 block(KVARN_WARP_SIZE);
  const size_t shared_mem_bytes = head_dim * sizeof(float);

#define KVARN_MTP_LAUNCH(D)                                                        \
  kvarn_flash_attn_decode_mtp_kernel<scalar_t, D, KVARN_WARP_SIZE><<<grid, block, shared_mem_bytes, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                        \
      reinterpret_cast<const uint8_t *>(K_cache),                                   \
      reinterpret_cast<const uint8_t *>(V_cache),                                   \
      K_tail_pool, V_tail_pool, K_tail_slots, V_tail_slots,                         \
      block_tables, cu_seq_lens, reinterpret_cast<scalar_t *>(O),                   \
      num_queries, block_table_stride, num_heads, num_kv_heads,                     \
      k_record_bytes, v_record_bytes, scale)

  switch (head_dim) {
    case 32:  KVARN_MTP_LAUNCH(32); break;
    case 64:  KVARN_MTP_LAUNCH(64); break;
    case 128: KVARN_MTP_LAUNCH(128); break;
    case 256: KVARN_MTP_LAUNCH(256); break;
    case 512: KVARN_MTP_LAUNCH(512); break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_mtp: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef KVARN_MTP_LAUNCH
  KVARN_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void launch_kvarn_decode_mtp_cc61(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_queries,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream) {
  constexpr int BLOCK = 128;
  const dim3 grid(num_heads);
  const dim3 block(BLOCK);
  const size_t shared_mem_bytes = head_dim * sizeof(float);

#define KVARN_MTP_CC61_LAUNCH(D)                                                   \
  kvarn_flash_attn_decode_mtp_kernel<scalar_t, D, BLOCK><<<grid, block, shared_mem_bytes, stream>>>( \
      reinterpret_cast<const scalar_t *>(Q),                                        \
      reinterpret_cast<const uint8_t *>(K_cache),                                   \
      reinterpret_cast<const uint8_t *>(V_cache),                                   \
      K_tail_pool, V_tail_pool, K_tail_slots, V_tail_slots,                         \
      block_tables, cu_seq_lens, reinterpret_cast<scalar_t *>(O),                   \
      num_queries, block_table_stride, num_heads, num_kv_heads,                     \
      k_record_bytes, v_record_bytes, scale)

  switch (head_dim) {
    case 32:  KVARN_MTP_CC61_LAUNCH(32); break;
    case 64:  KVARN_MTP_CC61_LAUNCH(64); break;
    case 128: KVARN_MTP_CC61_LAUNCH(128); break;
    case 256: KVARN_MTP_CC61_LAUNCH(256); break;
    case 512: KVARN_MTP_CC61_LAUNCH(512); break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_mtp_cc61: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef KVARN_MTP_CC61_LAUNCH
  KVARN_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void launch_kvarn_decode_cc61(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
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
      K_tail_pool, V_tail_pool, K_tail_slots, V_tail_slots,                         \
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

namespace kvarn_store {

template <typename T>
__device__ __forceinline__ T identity(T v) {
  return v;
}

template <typename scalar_t>
__global__ void kvarn_store_tail_kernel(
    const scalar_t *__restrict__ key,
    const scalar_t *__restrict__ value,
    uint8_t *__restrict__ key_cache,
    uint8_t *__restrict__ value_cache,
    scalar_t *__restrict__ key_tail_pool,
    scalar_t *__restrict__ value_tail_pool,
    const int64_t *__restrict__ slot_mapping,
    const int *__restrict__ key_tail_slots,
    const int *__restrict__ value_tail_slots,
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int num_blocks,
    int key_record_bytes,
    int value_record_bytes,
    int num_tail_slots,
    int key_stride,
    int value_stride) {
  const int token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const bool is_value = blockIdx.z != 0;
  if (token_idx >= num_tokens || head_idx >= num_heads) return;

  const int tail_slot =
      is_value ? value_tail_slots[token_idx] : key_tail_slots[token_idx];
  if (tail_slot < 0 || tail_slot >= num_tail_slots) return;

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;
  const int block_id = static_cast<int>(slot / block_size);
  const int block_offset = static_cast<int>(slot % block_size);
  if (block_id < 0 || block_id >= num_blocks) return;
  if (block_offset < 0 || block_offset >= block_size) return;

  const scalar_t *src = is_value ? value : key;
  scalar_t *dst = is_value ? value_tail_pool : key_tail_pool;
  uint8_t *cache = is_value ? value_cache : key_cache;
  const int record_bytes = is_value ? value_record_bytes : key_record_bytes;
  const int stride = is_value ? value_stride : key_stride;
  if (threadIdx.x == 0) {
    const int64_t record =
        (static_cast<int64_t>(block_id) * num_heads + head_idx) * record_bytes;
    cache[record] = 0;
  }
  const int64_t src_base =
      static_cast<int64_t>(token_idx) * stride +
      static_cast<int64_t>(head_idx) * head_dim;
  const int64_t dst_base =
      (((static_cast<int64_t>(tail_slot) * num_heads + head_idx) * block_size +
        block_offset) *
       head_dim);

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    dst[dst_base + d] = identity(src[src_base + d]);
  }
}

template <typename scalar_t>
void launch_kvarn_store_tail(
    const void *key,
    const void *value,
    void *key_cache,
    void *value_cache,
    void *key_tail_pool,
    void *value_tail_pool,
    const int64_t *slot_mapping,
    const int *key_tail_slots,
    const int *value_tail_slots,
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int num_blocks,
    int key_record_bytes,
    int value_record_bytes,
    int num_tail_slots,
    int key_stride,
    int value_stride,
    cudaStream_t stream) {
  if (num_tokens <= 0 || num_heads <= 0 || head_dim <= 0) return;
  const dim3 grid(num_tokens, num_heads, 2);
  const dim3 block(256);
  kvarn_store_tail_kernel<scalar_t><<<grid, block, 0, stream>>>(
      reinterpret_cast<const scalar_t *>(key),
      reinterpret_cast<const scalar_t *>(value),
      reinterpret_cast<uint8_t *>(key_cache),
      reinterpret_cast<uint8_t *>(value_cache),
      reinterpret_cast<scalar_t *>(key_tail_pool),
      reinterpret_cast<scalar_t *>(value_tail_pool),
      slot_mapping, key_tail_slots, value_tail_slots, num_tokens, num_heads,
      head_dim, block_size, num_blocks, key_record_bytes, value_record_bytes,
      num_tail_slots, key_stride, value_stride);
  KVARN_CUDA_CHECK(cudaGetLastError());
}

} // namespace kvarn_store

extern "C" void kvarn_flash_attn_decode(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
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
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode<__nv_bfloat16>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode<float>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
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
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
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
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode_cc61<__nv_bfloat16>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode_cc61<float>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_seqs,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_cc61: unsupported dtype=%u\n", dtype);
      break;
  }
}

extern "C" void kvarn_flash_attn_decode_mtp(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_queries,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_queries <= 0 || num_queries > kvarn_attn::KVARN_MTP_MAX_Q) return;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  if ((num_heads % num_kv_heads) != 0) {
    fprintf(stderr, "kvarn_flash_attn_decode_mtp: num_heads must be divisible by num_kv_heads\n");
    return;
  }

  switch (dtype) {
    case 0:
      kvarn_attn::launch_kvarn_decode_mtp<__half>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode_mtp<__nv_bfloat16>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode_mtp<float>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_mtp: unsupported dtype=%u\n", dtype);
      break;
  }
}

extern "C" void kvarn_flash_attn_decode_mtp_cc61(
    const void *Q,
    const void *K_cache,
    const void *V_cache,
    const void *K_tail_pool,
    const void *V_tail_pool,
    const int *K_tail_slots,
    const int *V_tail_slots,
    const int *block_tables,
    const int *cu_seq_lens,
    void *O,
    int num_queries,
    int block_table_stride,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int k_record_bytes,
    int v_record_bytes,
    float scale,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_queries <= 0 || num_queries > kvarn_attn::KVARN_MTP_MAX_Q) return;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
  if ((num_heads % num_kv_heads) != 0) {
    fprintf(stderr, "kvarn_flash_attn_decode_mtp_cc61: num_heads must be divisible by num_kv_heads\n");
    return;
  }

  switch (dtype) {
    case 0:
      kvarn_attn::launch_kvarn_decode_mtp_cc61<__half>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 1:
      kvarn_attn::launch_kvarn_decode_mtp_cc61<__nv_bfloat16>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    case 2:
      kvarn_attn::launch_kvarn_decode_mtp_cc61<float>(
          Q, K_cache, V_cache, K_tail_pool, V_tail_pool, K_tail_slots,
          V_tail_slots, block_tables, cu_seq_lens, O, num_queries,
          block_table_stride, num_heads, num_kv_heads, head_dim,
          k_record_bytes, v_record_bytes, scale, stream);
      break;
    default:
      fprintf(stderr, "kvarn_flash_attn_decode_mtp_cc61: unsupported dtype=%u\n", dtype);
      break;
  }
}

extern "C" void kvarn_store_tail(
    const void *key,
    const void *value,
    void *key_cache,
    void *value_cache,
    void *key_tail_pool,
    void *value_tail_pool,
    const int64_t *slot_mapping,
    const int *key_tail_slots,
    const int *value_tail_slots,
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int num_blocks,
    int key_record_bytes,
    int value_record_bytes,
    int num_tail_slots,
    int key_stride,
    int value_stride,
    cudaStream_t stream,
    uint32_t dtype) {
  if (num_tokens <= 0) return;
  switch (dtype) {
    case 0:
      kvarn_store::launch_kvarn_store_tail<__half>(
          key, value, key_cache, value_cache, key_tail_pool, value_tail_pool, slot_mapping,
          key_tail_slots, value_tail_slots,
          num_tokens, num_heads, head_dim, block_size, num_blocks,
          key_record_bytes, value_record_bytes, num_tail_slots,
          key_stride, value_stride, stream);
      break;
    case 1:
      kvarn_store::launch_kvarn_store_tail<__nv_bfloat16>(
          key, value, key_cache, value_cache, key_tail_pool, value_tail_pool, slot_mapping,
          key_tail_slots, value_tail_slots,
          num_tokens, num_heads, head_dim, block_size, num_blocks,
          key_record_bytes, value_record_bytes, num_tail_slots,
          key_stride, value_stride, stream);
      break;
    case 2:
      kvarn_store::launch_kvarn_store_tail<float>(
          key, value, key_cache, value_cache, key_tail_pool, value_tail_pool, slot_mapping,
          key_tail_slots, value_tail_slots,
          num_tokens, num_heads, head_dim, block_size, num_blocks,
          key_record_bytes, value_record_bytes, num_tail_slots,
          key_stride, value_stride, stream);
      break;
    default:
      fprintf(stderr, "kvarn_store_tail: unsupported dtype=%u\n", dtype);
      break;
  }
}
