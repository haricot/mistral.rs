// legacy_flash_attn_kernel.cu
//
// SM61/Pascal-friendly legacy streaming attention kernels for mistralrs-paged-attn.
// This is NOT FlashAttention v2/v3. It is a decode-only online-softmax backend:
// - one warp per (sequence/batch item, query head)
// - q_len == 1
// - f32 accumulation
// - no Tensor Cores
// - dense KV path and paged KV path
//
// DType codes match the existing crate convention:
//   0 => f16, 1 => bf16, 2 => f32

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define LEGACY_FA_WARP_SIZE 32
#define LEGACY_FA_FULL_MASK 0xffffffffu

#define LEGACY_FA_CUDA_CHECK(call)                                                    \
  do {                                                                                \
    cudaError_t err = (call);                                                         \
    if (err != cudaSuccess) {                                                         \
      fprintf(stderr, "legacy_flash_attn CUDA error at %s:%d: %s\n", __FILE__,      \
              __LINE__, cudaGetErrorString(err));                                     \
    }                                                                                 \
  } while (0)

template <typename T>
__device__ __forceinline__ float legacy_to_float(T v);

template <>
__device__ __forceinline__ float legacy_to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float legacy_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float legacy_to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T legacy_from_float(float v);

template <>
__device__ __forceinline__ __half legacy_from_float<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 legacy_from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float legacy_from_float<float>(float v) {
  return v;
}

__device__ __forceinline__ float legacy_warp_sum(float v) {
#pragma unroll
  for (int offset = LEGACY_FA_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(LEGACY_FA_FULL_MASK, v, offset);
  }
  return v;
}

template <typename scalar_t, int HEAD_DIM>
__global__ void legacy_flash_attn_decode_dense_kernel(
    const scalar_t *__restrict__ Q,      // [B, Hq, 1, D]
    const scalar_t *__restrict__ K,      // [B, Hkv, S, D]
    const scalar_t *__restrict__ V,      // [B, Hkv, S, D]
    scalar_t *__restrict__ O,            // [B, Hq, 1, D]
    int batch_size,
    int kv_len,
    int num_heads,
    int num_kv_heads,
    float scale,
    int window_size) {
  constexpr int D_PAD = ((HEAD_DIM + LEGACY_FA_WARP_SIZE - 1) / LEGACY_FA_WARP_SIZE) * LEGACY_FA_WARP_SIZE;
  constexpr int EPT = D_PAD / LEGACY_FA_WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int lane = threadIdx.x;

  if (head_idx >= num_heads || batch_idx >= batch_size) return;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  const int q_base = ((batch_idx * num_heads + head_idx) * 1) * HEAD_DIM;
  const int kv_base = (batch_idx * num_kv_heads + kv_head_idx) * kv_len * HEAD_DIM;
  const int out_base = q_base;

  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_FA_WARP_SIZE + lane;
    q_reg[i] = (d < HEAD_DIM) ? legacy_to_float(Q[q_base + d]) * scale : 0.0f;
  }

  float acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) acc[i] = 0.0f;

  float m = -FLT_MAX;
  float l = 0.0f;

  int start = 0;
  if (window_size > 0 && kv_len > window_size) start = kv_len - window_size;

  for (int t = start; t < kv_len; ++t) {
    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_FA_WARP_SIZE + lane;
      if (d < HEAD_DIM) {
        dot_local += q_reg[i] * legacy_to_float(__ldg(&K[kv_base + t * HEAD_DIM + d]));
      }
    }
    const float score = legacy_warp_sum(dot_local);

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_FA_WARP_SIZE + lane;
      const float vv = (d < HEAD_DIM) ? legacy_to_float(__ldg(&V[kv_base + t * HEAD_DIM + d])) : 0.0f;
      acc[i] = acc[i] * alpha + beta * vv;
    }

    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_FA_WARP_SIZE + lane;
    if (d < HEAD_DIM) {
      O[out_base + d] = legacy_from_float<scalar_t>(acc[i] * inv_l);
    }
  }
}

template <typename scalar_t, int HEAD_DIM>
__global__ void legacy_flash_attn_decode_paged_kernel(
    const scalar_t *__restrict__ Q,      // [num_seqs, Hq, D]
    const scalar_t *__restrict__ K,      // [num_blocks, Hkv, D/x, block_size, x]
    const scalar_t *__restrict__ V,      // [num_blocks, Hkv, D, block_size]
    const int *__restrict__ block_tables,// [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens,// [num_seqs]
    scalar_t *__restrict__ O,            // [num_seqs, Hq, D]
    int num_seqs,
    int max_context_len,
    int block_size,
    int max_num_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    int x,
    int k_block_stride,
    int k_head_stride,
    int v_block_stride,
    int v_head_stride,
    float scale,
    int window_size) {
  constexpr int D_PAD = ((HEAD_DIM + LEGACY_FA_WARP_SIZE - 1) / LEGACY_FA_WARP_SIZE) * LEGACY_FA_WARP_SIZE;
  constexpr int EPT = D_PAD / LEGACY_FA_WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int lane = threadIdx.x;

  if (head_idx >= num_heads || seq_idx >= num_seqs) return;

  int seq_len = context_lens[seq_idx];
  seq_len = min(seq_len, max_context_len);
  if (seq_len <= 0) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_FA_WARP_SIZE + lane;
      if (d < HEAD_DIM) O[(seq_idx * num_heads + head_idx) * HEAD_DIM + d] = legacy_from_float<scalar_t>(0.0f);
    }
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int q_base = (seq_idx * num_heads + head_idx) * HEAD_DIM;
  const int out_base = q_base;

  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_FA_WARP_SIZE + lane;
    q_reg[i] = (d < HEAD_DIM) ? legacy_to_float(Q[q_base + d]) * scale : 0.0f;
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
    const int physical_block = block_tables[seq_idx * max_num_blocks_per_seq + logical_block];

    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_FA_WARP_SIZE + lane;
      if (d < HEAD_DIM) {
        const int k_offset = physical_block * k_block_stride
                           + kv_head_idx * k_head_stride
                           + (d / x) * block_size * x
                           + block_offset * x
                           + (d % x);
        dot_local += q_reg[i] * legacy_to_float(__ldg(&K[k_offset]));
      }
    }
    const float score = legacy_warp_sum(dot_local);

    const float m_new = fmaxf(m, score);
    const float alpha = expf(m - m_new);
    const float beta = expf(score - m_new);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * LEGACY_FA_WARP_SIZE + lane;
      float vv = 0.0f;
      if (d < HEAD_DIM) {
        const int v_offset = physical_block * v_block_stride
                           + kv_head_idx * v_head_stride
                           + d * block_size
                           + block_offset;
        vv = legacy_to_float(__ldg(&V[v_offset]));
      }
      acc[i] = acc[i] * alpha + beta * vv;
    }

    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * LEGACY_FA_WARP_SIZE + lane;
    if (d < HEAD_DIM) {
      O[out_base + d] = legacy_from_float<scalar_t>(acc[i] * inv_l);
    }
  }
}

template <typename scalar_t>
void legacy_flash_attn_decode_dense_launch(
    const void *Q,
    const void *K,
    const void *V,
    void *O,
    int batch_size,
    int kv_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int window_size,
    cudaStream_t stream) {
  const dim3 grid(num_heads, batch_size);
  const dim3 block(LEGACY_FA_WARP_SIZE);

#define LEGACY_FA_LAUNCH_DENSE(D)                                                       \
  legacy_flash_attn_decode_dense_kernel<scalar_t, D><<<grid, block, 0, stream>>>(        \
      reinterpret_cast<const scalar_t *>(Q),                                             \
      reinterpret_cast<const scalar_t *>(K),                                             \
      reinterpret_cast<const scalar_t *>(V),                                             \
      reinterpret_cast<scalar_t *>(O),                                                   \
      batch_size, kv_len, num_heads, num_kv_heads, scale, window_size)

  switch (head_dim) {
    case 32:  LEGACY_FA_LAUNCH_DENSE(32); break;
    case 64:  LEGACY_FA_LAUNCH_DENSE(64); break;
    case 80:  LEGACY_FA_LAUNCH_DENSE(80); break;
    case 96:  LEGACY_FA_LAUNCH_DENSE(96); break;
    case 112: LEGACY_FA_LAUNCH_DENSE(112); break;
    case 128: LEGACY_FA_LAUNCH_DENSE(128); break;
    case 160: LEGACY_FA_LAUNCH_DENSE(160); break;
    case 192: LEGACY_FA_LAUNCH_DENSE(192); break;
    case 256: LEGACY_FA_LAUNCH_DENSE(256); break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_dense: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef LEGACY_FA_LAUNCH_DENSE
  LEGACY_FA_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t>
void legacy_flash_attn_decode_paged_launch(
    const void *Q,
    const void *K,
    const void *V,
    const int *block_tables,
    const int *context_lens,
    void *O,
    int num_seqs,
    int max_context_len,
    int block_size,
    int max_num_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int x,
    int k_block_stride,
    int k_head_stride,
    int v_block_stride,
    int v_head_stride,
    float scale,
    int window_size,
    cudaStream_t stream) {
  const dim3 grid(num_heads, num_seqs);
  const dim3 block(LEGACY_FA_WARP_SIZE);

#define LEGACY_FA_LAUNCH_PAGED(D)                                                       \
  legacy_flash_attn_decode_paged_kernel<scalar_t, D><<<grid, block, 0, stream>>>(        \
      reinterpret_cast<const scalar_t *>(Q),                                             \
      reinterpret_cast<const scalar_t *>(K),                                             \
      reinterpret_cast<const scalar_t *>(V),                                             \
      block_tables, context_lens, reinterpret_cast<scalar_t *>(O),                       \
      num_seqs, max_context_len, block_size, max_num_blocks_per_seq,                     \
      num_heads, num_kv_heads, x, k_block_stride, k_head_stride,                         \
      v_block_stride, v_head_stride, scale, window_size)

  switch (head_dim) {
    case 32:  LEGACY_FA_LAUNCH_PAGED(32); break;
    case 64:  LEGACY_FA_LAUNCH_PAGED(64); break;
    case 80:  LEGACY_FA_LAUNCH_PAGED(80); break;
    case 96:  LEGACY_FA_LAUNCH_PAGED(96); break;
    case 112: LEGACY_FA_LAUNCH_PAGED(112); break;
    case 128: LEGACY_FA_LAUNCH_PAGED(128); break;
    case 160: LEGACY_FA_LAUNCH_PAGED(160); break;
    case 192: LEGACY_FA_LAUNCH_PAGED(192); break;
    case 256: LEGACY_FA_LAUNCH_PAGED(256); break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_paged: unsupported head_dim=%d\n", head_dim);
      break;
  }
#undef LEGACY_FA_LAUNCH_PAGED
  LEGACY_FA_CUDA_CHECK(cudaGetLastError());
}

extern "C" void legacy_flash_attn_decode_dense(
    const void *Q,
    const void *K,
    const void *V,
    void *O,
    int batch_size,
    int kv_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int window_size,
    cudaStream_t stream,
    uint32_t dtype) {
  switch (dtype) {
    case 0:
      legacy_flash_attn_decode_dense_launch<__half>(Q, K, V, O, batch_size, kv_len,
                                                    num_heads, num_kv_heads, head_dim,
                                                    scale, window_size, stream);
      break;
    case 1:
      legacy_flash_attn_decode_dense_launch<__nv_bfloat16>(Q, K, V, O, batch_size, kv_len,
                                                           num_heads, num_kv_heads, head_dim,
                                                           scale, window_size, stream);
      break;
    case 2:
      legacy_flash_attn_decode_dense_launch<float>(Q, K, V, O, batch_size, kv_len,
                                                   num_heads, num_kv_heads, head_dim,
                                                   scale, window_size, stream);
      break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_dense: unsupported dtype=%u\n", dtype);
      break;
  }
}

extern "C" void legacy_flash_attn_decode_paged(
    const void *Q,
    const void *K,
    const void *V,
    const int *block_tables,
    const int *context_lens,
    void *O,
    int num_seqs,
    int max_context_len,
    int block_size,
    int max_num_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int x,
    int k_block_stride,
    int k_head_stride,
    int v_block_stride,
    int v_head_stride,
    float scale,
    int window_size,
    cudaStream_t stream,
    uint32_t dtype) {
  switch (dtype) {
    case 0:
      legacy_flash_attn_decode_paged_launch<__half>(Q, K, V, block_tables, context_lens, O,
                                                    num_seqs, max_context_len, block_size,
                                                    max_num_blocks_per_seq, num_heads,
                                                    num_kv_heads, head_dim, x,
                                                    k_block_stride, k_head_stride,
                                                    v_block_stride, v_head_stride,
                                                    scale, window_size, stream);
      break;
    case 1:
      legacy_flash_attn_decode_paged_launch<__nv_bfloat16>(Q, K, V, block_tables, context_lens, O,
                                                           num_seqs, max_context_len, block_size,
                                                           max_num_blocks_per_seq, num_heads,
                                                           num_kv_heads, head_dim, x,
                                                           k_block_stride, k_head_stride,
                                                           v_block_stride, v_head_stride,
                                                           scale, window_size, stream);
      break;
    case 2:
      legacy_flash_attn_decode_paged_launch<float>(Q, K, V, block_tables, context_lens, O,
                                                   num_seqs, max_context_len, block_size,
                                                   max_num_blocks_per_seq, num_heads,
                                                   num_kv_heads, head_dim, x,
                                                   k_block_stride, k_head_stride,
                                                   v_block_stride, v_head_stride,
                                                   scale, window_size, stream);
      break;
    default:
      fprintf(stderr, "legacy_flash_attn_decode_paged: unsupported dtype=%u\n", dtype);
      break;
  }
}
