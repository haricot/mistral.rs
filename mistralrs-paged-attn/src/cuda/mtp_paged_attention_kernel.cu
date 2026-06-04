// MTP paged-attention verification kernel.
//
// The standard paged-attention decode kernel treats staged MTP verification as
// `batch * q_len` independent decode rows. That is correct, but it rereads the
// same paged KV prefix for every verification token. This kernel groups the
// verification window in one CUDA block per (batch, query head), while still
// honoring the per-query block table and context length used for masking.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#ifdef USE_ROCM
#include "quantization/fp8/amd/quant_utils.cuh"
#else
#include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#define MTP_MAX_Q_LEN 8
#define MTP_BLOCK_THREADS 256

#define MTP_CUDA_CHECK(call)                                                    \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                    \
      fprintf(stderr, "mtp_paged_attention CUDA error at %s:%d: %s\n",          \
              __FILE__, __LINE__, cudaGetErrorString(err));                     \
    }                                                                           \
  } while (0)

template <typename T>
__device__ __forceinline__ float mtp_to_float(T v);

template <>
__device__ __forceinline__ float mtp_to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float mtp_to_float<__nv_bfloat16>(
    __nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float mtp_to_float<float>(float v) {
  return v;
}

template <typename T>
__device__ __forceinline__ T mtp_from_float(float v);

template <>
__device__ __forceinline__ __half mtp_from_float<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 mtp_from_float<__nv_bfloat16>(
    float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ float mtp_from_float<float>(float v) {
  return v;
}

template <typename cache_t, vllm::Fp8KVCacheDataType KV_DT>
__device__ __forceinline__ float mtp_cache_to_float(cache_t v,
                                                     const float *scale) {
  if constexpr (KV_DT == vllm::Fp8KVCacheDataType::kAuto) {
    return mtp_to_float(v);
  } else {
    const float s = scale == nullptr ? 1.0f : *scale;
    return vllm::fp8::scaled_convert<float, cache_t, KV_DT>(v, s);
  }
}

__device__ __forceinline__ float mtp_softcap(float v, float softcapping) {
  if (softcapping != 1.0f) {
    v = tanhf(v / softcapping) * softcapping;
  }
  return v;
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DT, int HEAD_DIM>
__global__ void mtp_paged_attention_kernel(
    const scalar_t *__restrict__ query,      // [B * Q, Hq, D]
    const cache_t *__restrict__ key_cache,   // [blocks, Hkv, D/x, BS, x]
    const cache_t *__restrict__ value_cache, // [blocks, Hkv, D, BS]
    const float *__restrict__ k_scale,
    const float *__restrict__ v_scale,
    const int *__restrict__ block_tables, // [B * Q, max_blocks]
    const int *__restrict__ context_lens, // [B * Q]
    scalar_t *__restrict__ out,           // [B * Q, Hq, D]
    int batch_size,
    int q_len,
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
    float softcapping) {
  constexpr int D_PAD =
      ((HEAD_DIM + MTP_BLOCK_THREADS - 1) / MTP_BLOCK_THREADS) *
      MTP_BLOCK_THREADS;
  constexpr int EPT = D_PAD / MTP_BLOCK_THREADS;

  __shared__ float reduce[MTP_MAX_Q_LEN * MTP_BLOCK_THREADS];

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int tid = threadIdx.x;

  if (head_idx >= num_heads || batch_idx >= batch_size || q_len <= 0 ||
      q_len > MTP_MAX_Q_LEN) {
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  int seq_lens[MTP_MAX_Q_LEN];
  int max_seq_len = 0;
#pragma unroll
  for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
    int len = 0;
    if (qr < q_len) {
      const int row = batch_idx * q_len + qr;
      len = context_lens[row];
      len = max(0, min(len, max_context_len));
      max_seq_len = max(max_seq_len, len);
    }
    seq_lens[qr] = len;
  }

  float q_reg[MTP_MAX_Q_LEN][EPT];
  float acc[MTP_MAX_Q_LEN][EPT];
  float m[MTP_MAX_Q_LEN];
  float l[MTP_MAX_Q_LEN];

#pragma unroll
  for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      q_reg[qr][i] = 0.0f;
      acc[qr][i] = 0.0f;
    }
    m[qr] = -FLT_MAX;
    l[qr] = 0.0f;
  }

#pragma unroll
  for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
    if (qr >= q_len) {
      continue;
    }
    const int row = batch_idx * q_len + qr;
    const int q_base = (row * num_heads + head_idx) * HEAD_DIM;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * MTP_BLOCK_THREADS + tid;
      q_reg[qr][i] =
          d < HEAD_DIM ? mtp_to_float(query[q_base + d]) * scale : 0.0f;
    }
  }

  for (int token_idx = 0; token_idx < max_seq_len; ++token_idx) {
    const int logical_block = token_idx / block_size;
    const int block_offset = token_idx - logical_block * block_size;

    bool active[MTP_MAX_Q_LEN];
    int physical_blocks[MTP_MAX_Q_LEN];
    int rep_block = -1;

#pragma unroll
    for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
      active[qr] = qr < q_len && token_idx < seq_lens[qr];
      int block = -1;
      if (active[qr]) {
        const int row = batch_idx * q_len + qr;
        block = block_tables[row * max_num_blocks_per_seq + logical_block];
        if (rep_block < 0) {
          rep_block = block;
        }
      }
      physical_blocks[qr] = block;
    }

    float dot[MTP_MAX_Q_LEN];
#pragma unroll
    for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
      dot[qr] = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * MTP_BLOCK_THREADS + tid;
      if (d >= HEAD_DIM || rep_block < 0) {
        continue;
      }

      const int64_t rep_k_offset =
          static_cast<int64_t>(rep_block) * k_block_stride +
          static_cast<int64_t>(kv_head_idx) * k_head_stride +
          static_cast<int64_t>(d / x) * block_size * x +
          block_offset * x + (d % x);
      const float rep_k =
          mtp_cache_to_float<cache_t, KV_DT>(__ldg(&key_cache[rep_k_offset]),
                                             k_scale);

#pragma unroll
      for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
        if (!active[qr]) {
          continue;
        }
        float k_val = rep_k;
        if (physical_blocks[qr] != rep_block) {
          const int64_t k_offset =
              static_cast<int64_t>(physical_blocks[qr]) * k_block_stride +
              static_cast<int64_t>(kv_head_idx) * k_head_stride +
              static_cast<int64_t>(d / x) * block_size * x +
              block_offset * x + (d % x);
          k_val = mtp_cache_to_float<cache_t, KV_DT>(
              __ldg(&key_cache[k_offset]), k_scale);
        }
        dot[qr] += q_reg[qr][i] * k_val;
      }
    }

#pragma unroll
    for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
      reduce[qr * MTP_BLOCK_THREADS + tid] = dot[qr];
    }
    __syncthreads();

    for (int stride = MTP_BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
#pragma unroll
        for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
          reduce[qr * MTP_BLOCK_THREADS + tid] +=
              reduce[qr * MTP_BLOCK_THREADS + tid + stride];
        }
      }
      __syncthreads();
    }

    float scores[MTP_MAX_Q_LEN];
#pragma unroll
    for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
      scores[qr] = mtp_softcap(reduce[qr * MTP_BLOCK_THREADS], softcapping);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * MTP_BLOCK_THREADS + tid;
      if (d >= HEAD_DIM || rep_block < 0) {
        continue;
      }

      const int64_t rep_v_offset =
          static_cast<int64_t>(rep_block) * v_block_stride +
          static_cast<int64_t>(kv_head_idx) * v_head_stride +
          static_cast<int64_t>(d) * block_size + block_offset;
      const float rep_v =
          mtp_cache_to_float<cache_t, KV_DT>(__ldg(&value_cache[rep_v_offset]),
                                             v_scale);

#pragma unroll
      for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
        if (!active[qr]) {
          continue;
        }
        const float score = scores[qr];
        const float m_new = fmaxf(m[qr], score);
        const float alpha = __expf(m[qr] - m_new);
        const float beta = __expf(score - m_new);

        float v_val = rep_v;
        if (physical_blocks[qr] != rep_block) {
          const int64_t v_offset =
              static_cast<int64_t>(physical_blocks[qr]) * v_block_stride +
              static_cast<int64_t>(kv_head_idx) * v_head_stride +
              static_cast<int64_t>(d) * block_size + block_offset;
          v_val = mtp_cache_to_float<cache_t, KV_DT>(
              __ldg(&value_cache[v_offset]), v_scale);
        }

        acc[qr][i] = acc[qr][i] * alpha + beta * v_val;
      }
    }

#pragma unroll
    for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
      if (!active[qr]) {
        continue;
      }
      const float score = scores[qr];
      const float m_new = fmaxf(m[qr], score);
      const float alpha = __expf(m[qr] - m_new);
      const float beta = __expf(score - m_new);
      l[qr] = l[qr] * alpha + beta;
      m[qr] = m_new;
    }
  }

#pragma unroll
  for (int qr = 0; qr < MTP_MAX_Q_LEN; ++qr) {
    if (qr >= q_len) {
      continue;
    }
    const int row = batch_idx * q_len + qr;
    const int out_base = (row * num_heads + head_idx) * HEAD_DIM;
    const float inv_l = l[qr] > 0.0f ? 1.0f / (l[qr] + 1e-6f) : 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * MTP_BLOCK_THREADS + tid;
      if (d < HEAD_DIM) {
        out[out_base + d] = mtp_from_float<scalar_t>(acc[qr][i] * inv_l);
      }
    }
  }
}

__device__ __forceinline__ float mtp_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DT, int HEAD_DIM>
__global__ void mtp_paged_attention_warp_kernel(
    const scalar_t *__restrict__ query,      // [B * Q, Hq, D]
    const cache_t *__restrict__ key_cache,   // [blocks, Hkv, D/x, BS, x]
    const cache_t *__restrict__ value_cache, // [blocks, Hkv, D, BS]
    const float *__restrict__ k_scale,
    const float *__restrict__ v_scale,
    const int *__restrict__ block_tables, // [B * Q, max_blocks]
    const int *__restrict__ context_lens, // [B * Q]
    scalar_t *__restrict__ out,           // [B * Q, Hq, D]
    int batch_size,
    int q_len,
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
    float softcapping) {
  constexpr int WARP_SIZE = 32;
  constexpr int D_PAD =
      ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int qr = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x & (WARP_SIZE - 1);

  if (head_idx >= num_heads || batch_idx >= batch_size || qr >= q_len ||
      q_len <= 0 || q_len > MTP_MAX_Q_LEN) {
    return;
  }

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;
  const int row = batch_idx * q_len + qr;
  int seq_len = context_lens[row];
  seq_len = max(0, min(seq_len, max_context_len));

  const int q_base = (row * num_heads + head_idx) * HEAD_DIM;
  const int out_base = q_base;
  const int *block_table = block_tables + row * max_num_blocks_per_seq;

  float q_reg[EPT];
  float acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * WARP_SIZE + lane;
    q_reg[i] = d < HEAD_DIM ? mtp_to_float(query[q_base + d]) * scale : 0.0f;
    acc[i] = 0.0f;
  }

  float m = -FLT_MAX;
  float l = 0.0f;

  for (int token_idx = 0; token_idx < seq_len; ++token_idx) {
    const int logical_block = token_idx / block_size;
    const int block_offset = token_idx - logical_block * block_size;
    const int physical_block = block_table[logical_block];

    float dot_local = 0.0f;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * WARP_SIZE + lane;
      if (d < HEAD_DIM) {
        const int64_t k_offset =
            static_cast<int64_t>(physical_block) * k_block_stride +
            static_cast<int64_t>(kv_head_idx) * k_head_stride +
            static_cast<int64_t>(d / x) * block_size * x + block_offset * x +
            (d % x);
        const float k_val = mtp_cache_to_float<cache_t, KV_DT>(
            __ldg(&key_cache[k_offset]), k_scale);
        dot_local += q_reg[i] * k_val;
      }
    }

    float score = mtp_warp_sum(dot_local);
    score = mtp_softcap(score, softcapping);

    const float m_new = fmaxf(m, score);
    const float alpha = __expf(m - m_new);
    const float beta = __expf(score - m_new);

#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int d = i * WARP_SIZE + lane;
      float v_val = 0.0f;
      if (d < HEAD_DIM) {
        const int64_t v_offset =
            static_cast<int64_t>(physical_block) * v_block_stride +
            static_cast<int64_t>(kv_head_idx) * v_head_stride +
            static_cast<int64_t>(d) * block_size + block_offset;
        v_val = mtp_cache_to_float<cache_t, KV_DT>(
            __ldg(&value_cache[v_offset]), v_scale);
      }
      acc[i] = acc[i] * alpha + beta * v_val;
    }

    l = l * alpha + beta;
    m = m_new;
  }

  const float inv_l = l > 0.0f ? 1.0f / (l + 1e-6f) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int d = i * WARP_SIZE + lane;
    if (d < HEAD_DIM) {
      out[out_base + d] = mtp_from_float<scalar_t>(acc[i] * inv_l);
    }
  }
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DT>
void mtp_paged_attention_launch(
    const void *query,
    const void *key_cache,
    const void *value_cache,
    const float *k_scale,
    const float *v_scale,
    const int *block_tables,
    const int *context_lens,
    void *out,
    int batch_size,
    int q_len,
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
    float softcapping,
    cudaStream_t stream) {
  if (q_len <= 0 || batch_size <= 0) {
    return;
  }
  if (q_len > MTP_MAX_Q_LEN) {
    fprintf(stderr, "mtp_paged_attention: q_len=%d exceeds max %d\n", q_len,
            MTP_MAX_Q_LEN);
    return;
  }
  if (num_heads % num_kv_heads != 0) {
    fprintf(stderr,
            "mtp_paged_attention: num_heads=%d must be divisible by "
            "num_kv_heads=%d\n",
            num_heads, num_kv_heads);
    return;
  }

  const dim3 grid(num_heads, batch_size);
  const dim3 block(MTP_MAX_Q_LEN * 32);

#define MTP_LAUNCH_HEAD_DIM(D)                                                  \
  mtp_paged_attention_warp_kernel<scalar_t, cache_t, KV_DT, D><<<grid, block,   \
                                                                 0, stream>>>(  \
      reinterpret_cast<const scalar_t *>(query),                                \
      reinterpret_cast<const cache_t *>(key_cache),                             \
      reinterpret_cast<const cache_t *>(value_cache), k_scale, v_scale,         \
      block_tables, context_lens, reinterpret_cast<scalar_t *>(out),            \
      batch_size, q_len, max_context_len, block_size,                           \
      max_num_blocks_per_seq, num_heads, num_kv_heads, x, k_block_stride,       \
      k_head_stride, v_block_stride, v_head_stride, scale, softcapping)

  switch (head_dim) {
    case 32:
      MTP_LAUNCH_HEAD_DIM(32);
      break;
    case 64:
      MTP_LAUNCH_HEAD_DIM(64);
      break;
    case 80:
      MTP_LAUNCH_HEAD_DIM(80);
      break;
    case 96:
      MTP_LAUNCH_HEAD_DIM(96);
      break;
    case 112:
      MTP_LAUNCH_HEAD_DIM(112);
      break;
    case 128:
      MTP_LAUNCH_HEAD_DIM(128);
      break;
    case 160:
      MTP_LAUNCH_HEAD_DIM(160);
      break;
    case 192:
      MTP_LAUNCH_HEAD_DIM(192);
      break;
    case 256:
      MTP_LAUNCH_HEAD_DIM(256);
      break;
    case 512:
      MTP_LAUNCH_HEAD_DIM(512);
      break;
    default:
      fprintf(stderr, "mtp_paged_attention: unsupported head_dim=%d\n",
              head_dim);
      break;
  }

#undef MTP_LAUNCH_HEAD_DIM
}

extern "C" void mtp_paged_attention(
    const void *query,
    const void *key_cache,
    const void *value_cache,
    const float *k_scale,
    const float *v_scale,
    const int *block_tables,
    const int *context_lens,
    void *out,
    int batch_size,
    int q_len,
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
    float softcapping,
    cudaStream_t stream,
    uint32_t dtype,
    uint32_t cache_dtype) {
  if (cache_dtype == 3) {
    if (dtype == 0) {
      mtp_paged_attention_launch<__half, uint8_t,
                                 vllm::Fp8KVCacheDataType::kFp8E4M3>(
          query, key_cache, value_cache, k_scale, v_scale, block_tables,
          context_lens, out, batch_size, q_len, max_context_len, block_size,
          max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
          k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
          softcapping, stream);
    } else if (dtype == 1) {
      mtp_paged_attention_launch<__nv_bfloat16, uint8_t,
                                 vllm::Fp8KVCacheDataType::kFp8E4M3>(
          query, key_cache, value_cache, k_scale, v_scale, block_tables,
          context_lens, out, batch_size, q_len, max_context_len, block_size,
          max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
          k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
          softcapping, stream);
    } else if (dtype == 2) {
      mtp_paged_attention_launch<float, uint8_t,
                                 vllm::Fp8KVCacheDataType::kFp8E4M3>(
          query, key_cache, value_cache, k_scale, v_scale, block_tables,
          context_lens, out, batch_size, q_len, max_context_len, block_size,
          max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
          k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
          softcapping, stream);
    }
  } else if (dtype == 0 && cache_dtype == 0) {
    mtp_paged_attention_launch<__half, __half,
                               vllm::Fp8KVCacheDataType::kAuto>(
        query, key_cache, value_cache, k_scale, v_scale, block_tables,
        context_lens, out, batch_size, q_len, max_context_len, block_size,
        max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
        k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
        softcapping, stream);
  } else if (dtype == 1 && cache_dtype == 1) {
    mtp_paged_attention_launch<__nv_bfloat16, __nv_bfloat16,
                               vllm::Fp8KVCacheDataType::kAuto>(
        query, key_cache, value_cache, k_scale, v_scale, block_tables,
        context_lens, out, batch_size, q_len, max_context_len, block_size,
        max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
        k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
        softcapping, stream);
  } else if (dtype == 2 && cache_dtype == 2) {
    mtp_paged_attention_launch<float, float,
                               vllm::Fp8KVCacheDataType::kAuto>(
        query, key_cache, value_cache, k_scale, v_scale, block_tables,
        context_lens, out, batch_size, q_len, max_context_len, block_size,
        max_num_blocks_per_seq, num_heads, num_kv_heads, head_dim, x,
        k_block_stride, k_head_stride, v_block_stride, v_head_stride, scale,
        softcapping, stream);
  } else {
    fprintf(stderr,
            "mtp_paged_attention: unsupported dtype/cache_dtype combination "
            "%u/%u\n",
            dtype, cache_dtype);
  }

  MTP_CUDA_CHECK(cudaGetLastError());
}
