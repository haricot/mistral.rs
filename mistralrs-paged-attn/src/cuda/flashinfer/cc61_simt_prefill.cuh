/*
 * Minimal CUDA SIMT fallback for FlashAttention-style single prefill on Pascal / sm_61.
 *
 * Properties:
 * - no cooperative_groups
 * - no cp.async
 * - no ldmatrix
 * - no mma / Tensor Cores
 * - accumulates QK and O in float
 *
 * Intended as a correctness/compatibility path, not a high-performance FlashAttention replacement.
 */
#ifndef FLASHINFER_CC61_SIMT_PREFILL_CUH_
#define FLASHINFER_CC61_SIMT_PREFILL_CUH_

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace flashinfer {
namespace cc61_simt {

template <typename T>
__device__ __forceinline__ float load_as_float(const T* p) {
  return static_cast<float>(*p);
}

template <>
__device__ __forceinline__ float load_as_float<half>(const half* p) {
  return __half2float(*p);
}

template <typename T>
__device__ __forceinline__ T cast_from_float(float x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half cast_from_float<half>(float x) {
  return __float2half_rn(x);
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float x) {
  __shared__ float smem[BLOCK_SIZE];
  const int tid = threadIdx.x;
  smem[tid] = x;
  __syncthreads();

#pragma unroll
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }
  return smem[0];
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float x) {
  __shared__ float smem[BLOCK_SIZE];
  const int tid = threadIdx.x;
  smem[tid] = x;
  __syncthreads();

#pragma unroll
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
    __syncthreads();
  }
  return smem[0];
}

template <typename TQ, typename TKV, int HEAD_DIM_QK>
__device__ __forceinline__ float qk_dot(const TQ* __restrict__ q_ptr,
                                        const TKV* __restrict__ k_ptr) {
  float acc = 0.f;
#pragma unroll
  for (int d = 0; d < HEAD_DIM_QK; ++d) {
    acc = fmaf(load_as_float(q_ptr + d), load_as_float(k_ptr + d), acc);
  }
  return acc;
}

/*
 * Raw contiguous single-prefill kernel.
 *
 * Tensor layout expected by strides, in elements, not bytes:
 *   q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d]
 *   k[kv_idx * k_stride_n + kv_head_idx * k_stride_h + d]
 *   v[kv_idx * v_stride_n + kv_head_idx * v_stride_h + d]
 *   o[q_idx * o_stride_n + qo_head_idx * o_stride_h + d]
 *
 * Grid:
 *   dim3 grid(qo_len, num_qo_heads)
 * Block:
 *   BLOCK_SIZE threads, 128 or 256 recommended on sm_61.
 */
template <typename TQ, typename TKV, typename TO, int HEAD_DIM_QK, int HEAD_DIM_VO, bool CAUSAL,
          int BLOCK_SIZE = 256>
__global__ __launch_bounds__(BLOCK_SIZE) void single_prefill_kernel(
    const TQ* __restrict__ q, const TKV* __restrict__ k, const TKV* __restrict__ v,
    TO* __restrict__ o, float* __restrict__ lse, int qo_len, int kv_len, int num_qo_heads,
    int num_kv_heads, int q_stride_n, int q_stride_h, int k_stride_n, int k_stride_h,
    int v_stride_n, int v_stride_h, int o_stride_n, int o_stride_h, float sm_scale) {
  static_assert(BLOCK_SIZE == 128 || BLOCK_SIZE == 256,
                "cc61_simt::single_prefill_kernel expects BLOCK_SIZE 128 or 256");
  static_assert(HEAD_DIM_QK > 0 && HEAD_DIM_VO > 0, "invalid head dim");

  __shared__ float weights[BLOCK_SIZE];

  const int tid = threadIdx.x;
  const int q_idx = blockIdx.x;
  const int qo_head_idx = blockIdx.y;
  const int group_size = num_qo_heads / num_kv_heads;
  const int kv_head_idx = qo_head_idx / group_size;

  const TQ* q_row = q + q_idx * q_stride_n + qo_head_idx * q_stride_h;

  int kv_end = kv_len;
  if constexpr (CAUSAL) {
    // FlashAttention prefill alignment: when kv_len >= qo_len, query 0 attends up to
    // key index kv_len - qo_len, query qo_len-1 attends up to kv_len-1.
    kv_end = kv_len - qo_len + q_idx + 1;
    if (kv_end < 0) kv_end = 0;
    if (kv_end > kv_len) kv_end = kv_len;
  }

  if (q_idx >= qo_len || qo_head_idx >= num_qo_heads || kv_head_idx >= num_kv_heads ||
      kv_end <= 0) {
    for (int d = tid; d < HEAD_DIM_VO; d += BLOCK_SIZE) {
      o[q_idx * o_stride_n + qo_head_idx * o_stride_h + d] = cast_from_float<TO>(0.f);
    }
    if (tid == 0 && lse != nullptr) {
      lse[q_idx * num_qo_heads + qo_head_idx] = -CUDART_INF_F;
    }
    return;
  }

  // Pass 1: row maximum of scaled QK.
  float local_max = -CUDART_INF_F;
  for (int kv_idx = tid; kv_idx < kv_end; kv_idx += BLOCK_SIZE) {
    const TKV* k_row = k + kv_idx * k_stride_n + kv_head_idx * k_stride_h;
    const float score = qk_dot<TQ, TKV, HEAD_DIM_QK>(q_row, k_row) * sm_scale;
    local_max = fmaxf(local_max, score);
  }
  const float row_max = block_reduce_max<BLOCK_SIZE>(local_max);

  // Pass 2: denominator.
  float local_denom = 0.f;
  for (int kv_idx = tid; kv_idx < kv_end; kv_idx += BLOCK_SIZE) {
    const TKV* k_row = k + kv_idx * k_stride_n + kv_head_idx * k_stride_h;
    const float score = qk_dot<TQ, TKV, HEAD_DIM_QK>(q_row, k_row) * sm_scale;
    local_denom += expf(score - row_max);
  }
  const float denom = block_reduce_sum<BLOCK_SIZE>(local_denom);
  const float inv_denom = 1.f / denom;

  if (tid == 0 && lse != nullptr) {
    lse[q_idx * num_qo_heads + qo_head_idx] = logf(denom) + row_max;
  }

  // Pass 3: reuse one tile of unnormalized softmax weights from shared memory.
  // Threads with tid < HEAD_DIM_VO own one output dimension; for HEAD_DIM_VO > BLOCK_SIZE,
  // each thread handles multiple dimensions and recomputes the tiled weights for each one.
  for (int out_d = tid; out_d < HEAD_DIM_VO; out_d += BLOCK_SIZE) {
    float acc = 0.f;

    for (int tile = 0; tile < kv_end; tile += BLOCK_SIZE) {
      const int kv_idx = tile + tid;
      if (kv_idx < kv_end) {
        const TKV* k_row = k + kv_idx * k_stride_n + kv_head_idx * k_stride_h;
        const float score = qk_dot<TQ, TKV, HEAD_DIM_QK>(q_row, k_row) * sm_scale;
        weights[tid] = expf(score - row_max);
      } else {
        weights[tid] = 0.f;
      }
      __syncthreads();

      const int tile_size = min(BLOCK_SIZE, kv_end - tile);
#pragma unroll 1
      for (int j = 0; j < tile_size; ++j) {
        const int kv_j = tile + j;
        const TKV* v_row = v + kv_j * v_stride_n + kv_head_idx * v_stride_h;
        acc = fmaf(weights[j], load_as_float(v_row + out_d), acc);
      }
      __syncthreads();
    }

    o[q_idx * o_stride_n + qo_head_idx * o_stride_h + out_d] = cast_from_float<TO>(acc * inv_denom);
  }
}

template <typename TQ, typename TKV, typename TO, int HEAD_DIM_QK, int HEAD_DIM_VO, bool CAUSAL,
          int BLOCK_SIZE = 256>
inline cudaError_t launch_single_prefill(const TQ* q, const TKV* k, const TKV* v, TO* o,
                                         float* lse, int qo_len, int kv_len, int num_qo_heads,
                                         int num_kv_heads, int q_stride_n, int q_stride_h,
                                         int k_stride_n, int k_stride_h, int v_stride_n,
                                         int v_stride_h, int o_stride_n, int o_stride_h,
                                         float sm_scale, cudaStream_t stream) {
  dim3 grid(qo_len, num_qo_heads, 1);
  dim3 block(BLOCK_SIZE, 1, 1);
  single_prefill_kernel<TQ, TKV, TO, HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL, BLOCK_SIZE>
      <<<grid, block, 0, stream>>>(q, k, v, o, lse, qo_len, kv_len, num_qo_heads, num_kv_heads,
                                  q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n,
                                  v_stride_h, o_stride_n, o_stride_h, sm_scale);
  return cudaGetLastError();
}

/*
 * FlashInfer-Params adapter for single prefill.
 * Requires params fields used by FlashInfer prefill.cuh:
 *   q, k, v, o, lse, qo_len, kv_len, num_qo_heads, num_kv_heads,
 *   q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h.
 * It intentionally ignores partition_kv and custom variants.
 */
template <int HEAD_DIM_QK, int HEAD_DIM_VO, bool CAUSAL, typename Params, int BLOCK_SIZE = 256>
inline cudaError_t launch_single_prefill_from_params(Params params, float sm_scale,
                                                     cudaStream_t stream) {
  using TQ = typename Params::DTypeQ;
  using TKV = typename Params::DTypeKV;
  using TO = typename Params::DTypeO;

  params.partition_kv = false;
  const int o_stride_h = HEAD_DIM_VO;
  const int o_stride_n = params.num_qo_heads * HEAD_DIM_VO;

  return launch_single_prefill<TQ, TKV, TO, HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL, BLOCK_SIZE>(
      params.q, params.k, params.v, params.o, params.lse, params.qo_len, params.kv_len,
      params.num_qo_heads, params.num_kv_heads, params.q_stride_n, params.q_stride_h,
      params.k_stride_n, params.k_stride_h, params.v_stride_n, params.v_stride_h, o_stride_n,
      o_stride_h, sm_scale, stream);
}

}  // namespace cc61_simt
}  // namespace flashinfer

#endif  // FLASHINFER_CC61_SIMT_PREFILL_CUH_
