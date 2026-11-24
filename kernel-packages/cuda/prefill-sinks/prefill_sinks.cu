// Externalized historical flash_attn_sinks.cu device algorithms; no host launch or FFI.
//
// Tiled flash attention with per-head attention sinks for prefill.
// Uses FlashAttention-2 online softmax with shared-memory K/V tiling:
//   - Never materializes the N x N attention matrix
//   - BR warps per block, each handling one Q row
//   - K/V tiles loaded cooperatively into shared memory (float32)
//   - Two-pass per tile: compute scores, then one rescale + V accumulation
//   - Sinks integrated once after all tiles
//
// Layout: Q [B, num_heads, S, D], K [B, num_kv_heads, S, D], V same as K
// GQA: kv_head = q_head / (num_heads / num_kv_heads)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu



// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T val);

template <> __device__ __forceinline__ float to_float<__half>(__half val) {
  return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <> __device__ __forceinline__ float to_float<float>(float val) {
  return val;
}

template <typename T> __device__ __forceinline__ T from_float(float val);

template <> __device__ __forceinline__ __half from_float<__half>(float val) {
  return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
  return __float2bfloat16(val);
}

template <> __device__ __forceinline__ float from_float<float>(float val) {
  return val;
}

// ---------------------------------------------------------------------------
// Warp-level reductions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}

// ---------------------------------------------------------------------------
// Tiled flash attention kernel with sinks
//
// Template params:
//   scalar_t : __half, __nv_bfloat16, or float
//   HEAD_DIM : actual head dimension (64, 80, 96, 112, 128, 192, 256)
//   BR       : number of warps per block (= number of Q rows per block)
//   BC       : KV tile size (number of KV positions per shared-memory tile)
//
// Grid : (num_heads, batch_size, cdiv(q_len, BR))
// Block: (BR * WARP_SIZE)
//
// Shared memory (float32 to avoid bank conflicts):
//   k_smem[BC][D_PAD]  — K tile
//   v_smem[BC][D_PAD]  — V tile
//
// Each warp handles one Q row. Within each tile:
//   Pass 1: compute BC dot-product scores using k_smem
//   Pass 2: one rescale of accumulators, then accumulate V from v_smem
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM, int BR, int BC>
__device__ __forceinline__ void flash_attn_sinks_kernel(
    const scalar_t *__restrict__ Q,  // [B, num_heads, q_len, D]
    const scalar_t *__restrict__ K,  // [B, num_kv_heads, kv_len, D]
    const scalar_t *__restrict__ V,  // [B, num_kv_heads, kv_len, D]
    scalar_t *__restrict__ O,        // [B, num_heads, q_len, D]
    const float *__restrict__ sinks, // [num_heads] or nullptr
    const float scale, const int q_len, const int kv_len, const int num_heads,
    const int num_kv_heads,
    const int window_size // 0 = no window (full)
) {
  // Padded head dim for clean warp division (round up to multiple of 32)
  constexpr int D_PAD = ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE; // elements per thread
  constexpr int BLOCK_SIZE = BR * WARP_SIZE;

  // Shared memory for K and V tiles (float32 avoids bank conflicts)
  __shared__ float k_smem[BC * D_PAD];
  __shared__ float v_smem[BC * D_PAD];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Which query row this warp handles
  const int q_row = q_tile_idx * BR + warp_id;

  // Offset between Q local row index and absolute KV position
  // (bottom-right aligned causal masking, matching Metal kernel)
  const int kv_offset = kv_len - q_len;

  // Offsets into contiguous [B, H, S, D] layout
  const int q_offset =
      ((batch_idx * num_heads + head_idx) * q_len + q_row) * HEAD_DIM;
  const int kv_base =
      (batch_idx * num_kv_heads + kv_head_idx) * kv_len * HEAD_DIM;

  // Load Q row into registers (pre-scaled)
  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    q_reg[i] = (q_row < q_len && d < HEAD_DIM)
                   ? to_float(Q[q_offset + d]) * scale
                   : 0.0f;
  }

  // Output accumulator in registers
  float o_acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -FLT_MAX; // running max
  float l_i = 0.0f;     // running sum of exp

  // Block-level KV bounds (union of all warps' causal windows)
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end =
      min(kv_len, block_q_end + kv_offset); // max causal bound across warps

  // Per-warp causal bounds
  const int my_kv_start = (window_size > 0 && q_row < q_len)
                              ? max(0, q_row + kv_offset - window_size + 1)
                              : 0;
  const int my_kv_end = (q_row < q_len) ? (q_row + kv_offset + 1) : 0;

  // Score buffer for current tile (in registers)
  float scores[BC];

  // Tile loop over K/V
  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile into shared memory ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] =
          (kj < tile_len && kd < HEAD_DIM)
              ? to_float(__ldg(&K[kv_base + (tile_start + kj) * HEAD_DIM + kd]))
              : 0.0f;
    }

    // --- Cooperatively load V tile into shared memory ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] =
          (vj < tile_len && vd < HEAD_DIM)
              ? to_float(__ldg(&V[kv_base + (tile_start + vj) * HEAD_DIM + vd]))
              : 0.0f;
    }
    __syncthreads();

    if (q_row < q_len) {
      // --- Pass 1: Compute scores for this tile ---
      float tile_max = -FLT_MAX;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        // Causal: no valid positions beyond q_row
        if (kv_pos >= my_kv_end) {
          // All remaining positions in tile are also beyond causal bound
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -FLT_MAX;
          break;
        }

        // Sliding window check
        if (kv_pos < my_kv_start) {
          scores[j] = -FLT_MAX;
          continue;
        }

        // Dot product: q_reg . K[j] from shared memory
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * WARP_SIZE + lane_id];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = fmaxf(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -FLT_MAX) {
        const float m_new = fmaxf(m_i, tile_max);
        const float rescale = expf(m_i - m_new);

        // Rescale old accumulators (ONCE per tile, not per position)
#pragma unroll
        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        // Accumulate V contributions
        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -FLT_MAX)
            continue; // masked position

          const float p = expf(scores[j] - m_i);
          l_i += p;

#pragma unroll
          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * WARP_SIZE + lane_id];
          }
        }
      }
    }
    __syncthreads();
  }

  if (q_row >= q_len)
    return;

  // Integrate per-head sinks into the softmax denominator.
  // The sink adds exp(sink_val) to the denominator but does NOT
  // contribute any value to the output (virtual token).
  if (sinks != nullptr) {
    const float sink_val = sinks[head_idx];
    const float m_new = fmaxf(m_i, sink_val);
    const float rescale = expf(m_i - m_new);
#pragma unroll
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + expf(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    if (d < HEAD_DIM) {
      O[q_offset + d] = from_float<scalar_t>(o_acc[i] * inv_l);
    }
  }
}

// ---------------------------------------------------------------------------
// Varlen tiled flash attention kernel with sinks
//
// Q is padded: [B, num_heads, max_q_len, D]
// K/V are packed: [total_kv, num_kv_heads, D]
// O is padded: [B, num_heads, max_q_len, D]
//
// cu_seqlens_q[B+1]: cumulative Q lengths
// cu_seqlens_k[B+1]: cumulative KV lengths
//
// Grid : (num_heads, batch_size, cdiv(max_q_len, BR))
// Block: (BR * WARP_SIZE)
// ---------------------------------------------------------------------------

template <typename scalar_t, int HEAD_DIM, int BR, int BC>
__device__ __forceinline__ void flash_attn_sinks_varlen_kernel(
    const scalar_t *__restrict__ Q,  // [B, num_heads, max_q_len, D]
    const scalar_t *__restrict__ K,  // [total_kv, num_kv_heads, D]
    const scalar_t *__restrict__ V,  // [total_kv, num_kv_heads, D]
    scalar_t *__restrict__ O,        // [B, num_heads, max_q_len, D]
    const float *__restrict__ sinks, // [num_heads] or nullptr
    const unsigned int *__restrict__ cu_seqlens_q, // [B+1]
    const unsigned int *__restrict__ cu_seqlens_k, // [B+1]
    const float scale, const int max_q_len, const int num_heads,
    const int num_kv_heads, const int window_size, const bool packed_q, const uint32_t total_q, const uint32_t total_kv) {
  constexpr int D_PAD = ((HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  constexpr int EPT = D_PAD / WARP_SIZE;
  constexpr int BLOCK_SIZE = BR * WARP_SIZE;

  __shared__ float k_smem[BC * D_PAD];
  __shared__ float v_smem[BC * D_PAD];

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int batch_idx = blockIdx.y;
  const int q_tile_idx = blockIdx.z;

  const int gqa_ratio = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / gqa_ratio;

  // Uniform metadata bounds before any per-row work or barriers.
  const uint32_t qb=cu_seqlens_q[batch_idx], qe=cu_seqlens_q[batch_idx+1];
  const uint32_t kb=cu_seqlens_k[batch_idx], ke=cu_seqlens_k[batch_idx+1];
  if (qe<qb || ke<kb || qe>total_q || ke>total_kv || qe-qb>static_cast<uint32_t>(max_q_len)) return;
  // Per-batch-item lengths (derived from cumulative arrays)
  const int my_q_len =
      (int)(cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx]);
  const int kv_start = (int)cu_seqlens_k[batch_idx];
  const int my_kv_len =
      (int)(cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx]);

  if (q_tile_idx * BR >= my_q_len)
    return;

  const int q_row = q_tile_idx * BR + warp_id;

  // Out-of-bounds warps must NOT return early: they participate in cooperative
  // K/V loads and __syncthreads barriers inside the tile loop. Returning early
  // causes barrier divergence (undefined behavior / hangs on partially-filled
  // tiles). Instead, we guard per-warp score computation with
  // `if (q_row < my_q_len)` inside the loop, matching the non-varlen kernel.
  const bool q_row_valid = (q_row < my_q_len);

  // Bottom-right aligned causal offset (safe even for out-of-bounds q_row;
  // only used inside the q_row_valid guard)
  const int kv_offset = my_kv_len - my_q_len;

  const int q_offset =
      packed_q
          ? (((int)cu_seqlens_q[batch_idx] + q_row) * num_heads + head_idx) *
                HEAD_DIM
          : ((batch_idx * num_heads + head_idx) * max_q_len + q_row) * HEAD_DIM;

  // Load Q row into registers (pre-scaled); out-of-bounds rows get zeros
  float q_reg[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    q_reg[i] = (q_row_valid && d < HEAD_DIM) ? to_float(Q[q_offset + d]) * scale
                                             : 0.0f;
  }

  float o_acc[EPT];
#pragma unroll
  for (int i = 0; i < EPT; i++)
    o_acc[i] = 0.0f;

  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  // Block-level KV bounds (shared across all warps in the block)
  const int block_q_start = q_tile_idx * BR;
  const int block_q_end = min(block_q_start + BR, my_q_len);
  const int block_kv_start =
      (window_size > 0) ? max(0, block_q_start + kv_offset - window_size + 1)
                        : 0;
  const int block_kv_end = min(my_kv_len, block_q_end + kv_offset);

  // Per-warp causal bounds (only meaningful when q_row_valid)
  const int my_kv_start_w =
      (window_size > 0) ? max(0, q_row + kv_offset - window_size + 1) : 0;
  const int my_kv_end = q_row + kv_offset + 1;

  float scores[BC];

  for (int tile_start = block_kv_start; tile_start < block_kv_end;
       tile_start += BC) {
    const int tile_end = min(tile_start + BC, block_kv_end);
    const int tile_len = tile_end - tile_start;

    // --- Cooperatively load K tile from packed layout ---
    // K: [total_kv, num_kv_heads, D]
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int kj = idx / D_PAD;
      const int kd = idx % D_PAD;
      k_smem[idx] =
          (kj < tile_len && kd < HEAD_DIM)
              ? to_float(__ldg(&K[((kv_start + tile_start + kj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  kd]))
              : 0.0f;
    }

    // --- Cooperatively load V tile from packed layout ---
    for (int idx = threadIdx.x; idx < BC * D_PAD; idx += BLOCK_SIZE) {
      const int vj = idx / D_PAD;
      const int vd = idx % D_PAD;
      v_smem[idx] =
          (vj < tile_len && vd < HEAD_DIM)
              ? to_float(__ldg(&V[((kv_start + tile_start + vj) * num_kv_heads +
                                   kv_head_idx) *
                                      HEAD_DIM +
                                  vd]))
              : 0.0f;
    }
    __syncthreads();

    if (q_row_valid) {
      // --- Pass 1: Compute scores ---
      float tile_max = -FLT_MAX;

      for (int j = 0; j < tile_len; j++) {
        const int kv_pos = tile_start + j;

        if (kv_pos >= my_kv_end) {
          for (int jj = j; jj < tile_len; jj++)
            scores[jj] = -FLT_MAX;
          break;
        }

        if (kv_pos < my_kv_start_w) {
          scores[j] = -FLT_MAX;
          continue;
        }

        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < EPT; i++) {
          dot += q_reg[i] * k_smem[j * D_PAD + i * WARP_SIZE + lane_id];
        }
        dot = warp_reduce_sum(dot);

        scores[j] = dot;
        tile_max = fmaxf(tile_max, dot);
      }

      // --- Pass 2: Online softmax update + V accumulation ---
      if (tile_max > -FLT_MAX) {
        const float m_new = fmaxf(m_i, tile_max);
        const float rescale = expf(m_i - m_new);

#pragma unroll
        for (int i = 0; i < EPT; i++)
          o_acc[i] *= rescale;
        l_i *= rescale;
        m_i = m_new;

        for (int j = 0; j < tile_len; j++) {
          if (scores[j] <= -FLT_MAX)
            continue;

          const float p = expf(scores[j] - m_i);
          l_i += p;

#pragma unroll
          for (int i = 0; i < EPT; i++) {
            o_acc[i] += p * v_smem[j * D_PAD + i * WARP_SIZE + lane_id];
          }
        }
      }
    }
    __syncthreads();
  }

  // Out-of-bounds warps exit after participating in all barriers
  if (!q_row_valid)
    return;

  // Integrate sinks
  if (sinks != nullptr) {
    const float sink_val = sinks[head_idx];
    const float m_new = fmaxf(m_i, sink_val);
    const float rescale = expf(m_i - m_new);
#pragma unroll
    for (int i = 0; i < EPT; i++)
      o_acc[i] *= rescale;
    l_i = l_i * rescale + expf(sink_val - m_new);
    m_i = m_new;
  }

  // Normalize and write output
  const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
#pragma unroll
  for (int i = 0; i < EPT; i++) {
    const int d = i * WARP_SIZE + lane_id;
    if (d < HEAD_DIM) {
      O[q_offset + d] = from_float<scalar_t>(o_acc[i] * inv_l);
    }
  }
}

// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,64,8,64>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,64,8,64>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d64(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,80,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,80,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d80(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,96,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,96,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d96(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,112,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,112,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d112(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,128,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,128,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d128(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,192,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,192,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d192(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,256,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__half,256,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f16_d256(
 const __half* q, const __half* k, const __half* v, __half* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__half,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d64(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d80(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d96(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d112(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d128(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d192(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_bf16_d256(
 const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<__nv_bfloat16,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d64(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,64,8,64>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d64(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,64,8,64>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d64(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d64(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d64(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,64,8,64>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d64(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,64,8,64>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d80(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,80,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d80(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,80,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d80(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d80(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d80(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,80,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d80(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,80,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d96(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,96,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d96(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,96,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d96(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d96(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d96(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,96,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d96(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,96,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d112(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,112,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d112(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,112,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d112(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d112(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d112(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,112,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d112(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,112,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d128(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,128,8,32>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d128(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,128,8,32>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d128(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d128(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d128(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,128,8,32>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d128(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,128,8,32>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d192(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,192,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d192(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,192,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d192(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d192(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d192(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,192,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d192(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,192,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_plain_f32_d256(
 const float* q, const float* k, const float* v, float* o, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,256,8,16>(q,k,v,o,nullptr,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_dense_sink_f32_d256(
 const float* q, const float* k, const float* v, float* o, const float* sinks, float scale, uint32_t q_len, uint32_t kv_len, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_kernel<float,256,8,16>(q,k,v,o,sinks,scale,q_len,kv_len,heads,kv_heads,window);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_plain_f32_d256(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_varlen_sink_f32_d256(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,false,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_plain_f32_d256(
 const float* q, const float* k, const float* v, float* o, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,256,8,16>(q,k,v,o,nullptr,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}

extern "C" __global__ __launch_bounds__(256) void dkr_prefill_sinks_packed_sink_f32_d256(
 const float* q, const float* k, const float* v, float* o, const float* sinks, const uint32_t* qlens, const uint32_t* klens, float scale, uint32_t q_len, uint32_t total_q, uint32_t total_kv, uint32_t heads, uint32_t kv_heads, uint32_t window) {
 flash_attn_sinks_varlen_kernel<float,256,8,16>(q,k,v,o,sinks,qlens,klens,scale,q_len,heads,kv_heads,window,true,total_q,total_kv);
}
