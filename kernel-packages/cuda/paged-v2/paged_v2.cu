#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

// DKR-R1.21/S6: Pascal SIMT partitioned paged-attention v2.
// Exact input layout: Q[seq,query_head,head_dim],
// K[block,kv_head,head_dim/x,block_size,x],
// V[block,kv_head,head_dim,block_size].
// All inputs contiguous and the same nonquantized F16/BF16/F32 dtype.
// Scratch [seq,head,part,head_dim] and LSE [seq,head,part] use F32.
// Two prevalidated launches, no intermediate P-by-token tensor and
// no extra buffer allocation once the shared static-v2 pool is resident.

constexpr uint32_t DKR_V2_THREADS = 128u;
constexpr uint32_t DKR_V2_PARTITION = 256u;

template <typename T> __device__ __forceinline__ float v2_to_float(T value) {
  return static_cast<float>(value);
}
template <> __device__ __forceinline__ float v2_to_float<__half>(__half value) {
  return __half2float(value);
}
template <> __device__ __forceinline__ float v2_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
template <typename T> __device__ __forceinline__ T v2_from_float(float value) {
  return static_cast<T>(value);
}
template <> __device__ __forceinline__ __half v2_from_float<__half>(float value) {
  return __float2half_rn(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 v2_from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ void paged_v2_partition(
    const T* __restrict__ query,
    const T* __restrict__ key_cache,
    const T* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ context_lens,
    float* __restrict__ partial,
    float* __restrict__ logsumexp,
    uint32_t seqs, uint32_t heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t x,
    uint32_t table_stride, uint32_t max_context, float scale) {
  const uint32_t head = blockIdx.x;
  const uint32_t seq = blockIdx.y;
  const uint32_t part = blockIdx.z;
  if (seq >= seqs || head >= heads) return;
  const uint32_t parts = gridDim.z;
  const uint32_t tid = threadIdx.x;
  const uint64_t scratch_index =
      (static_cast<uint64_t>(seq) * heads + head) * parts + part;
  const uint64_t partial_base = scratch_index * head_size;
  const uint32_t context = min(context_lens[seq], max_context);
  const uint32_t start = part * DKR_V2_PARTITION;
  const uint32_t end = min(context, start + DKR_V2_PARTITION);
  if (start >= end) {
    if (tid == 0) logsumexp[scratch_index] = -CUDART_INF_F;
    for (uint32_t d = tid; d < head_size; d += blockDim.x) {
      partial[partial_base + d] = 0.0f;
    }
    return;
  }
  const uint32_t kvhead = head / (heads / kv_heads);
  const uint64_t query_base =
      (static_cast<uint64_t>(seq) * heads + head) * head_size;
  const uint64_t page_elements =
      static_cast<uint64_t>(kv_heads) * head_size * block_size;
  const uint64_t head_offset =
      static_cast<uint64_t>(kvhead) * head_size * block_size;

  float query_value[4] = {};
  float accumulator[4] = {};
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t d = tid + j * DKR_V2_THREADS;
    if (d < head_size) query_value[j] = v2_to_float(query[query_base + d]);
  }
  __shared__ float dot_sums[DKR_V2_THREADS];
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0f;
  for (uint32_t token = start; token < end; ++token) {
    const uint32_t page = block_table[
        static_cast<uint64_t>(seq) * table_stride + token / block_size];
    const uint32_t slot = token % block_size;
    const uint64_t kbase = static_cast<uint64_t>(page) * page_elements + head_offset;
    const uint64_t vbase = kbase;
    float dot = 0.0f;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t d = tid + j * DKR_V2_THREADS;
      if (d < head_size) {
        const uint64_t address = kbase
            + static_cast<uint64_t>(d / x) * block_size * x
            + static_cast<uint64_t>(slot) * x + d % x;
        dot = fmaf(query_value[j], v2_to_float(key_cache[address]), dot);
      }
    }
    dot_sums[tid] = dot;
    __syncthreads();
    for (uint32_t stride = DKR_V2_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) dot_sums[tid] += dot_sums[tid + stride];
      __syncthreads();
    }
    const float logit = dot_sums[0] * scale;
    const float next_max = fmaxf(running_max, logit);
    const float old_weight = expf(running_max - next_max);
    const float token_weight = expf(logit - next_max);
    running_sum = running_sum * old_weight + token_weight;
    running_max = next_max;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t d = tid + j * DKR_V2_THREADS;
      if (d < head_size) {
        const uint64_t address =
            vbase + static_cast<uint64_t>(d) * block_size + slot;
        accumulator[j] = accumulator[j] * old_weight
            + token_weight * v2_to_float(value_cache[address]);
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    logsumexp[scratch_index] =
        running_sum > 0.0f ? running_max + logf(running_sum) : -CUDART_INF_F;
  }
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t d = tid + j * DKR_V2_THREADS;
    if (d < head_size) {
      partial[partial_base + d] =
          running_sum > 0.0f ? accumulator[j] / running_sum : 0.0f;
    }
  }
}

template <typename T>
__device__ __forceinline__ void paged_v2_merge(
    const float* __restrict__ partial,
    const float* __restrict__ logsumexp,
    T* __restrict__ output,
    uint32_t seqs, uint32_t heads, uint32_t head_size, uint32_t partitions) {
  const uint32_t head = blockIdx.x;
  const uint32_t seq = blockIdx.y;
  if (head >= heads || seq >= seqs) return;
  const uint32_t d = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(seq) * heads + head;
  const uint64_t lse_base = row * partitions;
  const uint64_t out_base = row * head_size;
  float max_log = -CUDART_INF_F;
  for (uint32_t p = 0; p < partitions; ++p) {
    const float lse = logsumexp[lse_base + p];
    max_log = fmaxf(max_log, lse);
  }
  const bool any = max_log != -CUDART_INF_F;
  float denominator = 0.0f;
  float numerator[4] = {};
  for (uint32_t p = 0; p < partitions; ++p) {
    const float log_value = logsumexp[lse_base + p];
    const float weight = any && log_value != -CUDART_INF_F
        ? expf(log_value - max_log) : 0.0f;
    denominator += weight;
    const uint64_t partial_base = (lse_base + p) * head_size;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t dim = d + j * DKR_V2_THREADS;
      if (dim < head_size) {
        numerator[j] = fmaf(weight, partial[partial_base + dim], numerator[j]);
      }
    }
  }
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t dim = d + j * DKR_V2_THREADS;
    if (dim < head_size) {
      output[out_base + dim] = v2_from_float<T>(
          denominator > 0.0f ? numerator[j] / denominator : 0.0f);
    }
  }
}

#define DKR_PAGED_V2(NAME, TYPE) \
extern "C" __global__ void dkr_paged_v2_partition_##NAME( \
    const TYPE* q, const TYPE* k, const TYPE* v, \
    const uint32_t* table, const uint32_t* lens, \
    float* partial, float* lse, \
    uint32_t seqs, uint32_t heads, uint32_t kv_heads, \
    uint32_t head_size, uint32_t block_size, uint32_t x, \
    uint32_t table_stride, uint32_t max_context, float scale) { \
  paged_v2_partition<TYPE>(q,k,v,table,lens,partial,lse,seqs,heads,kv_heads, \
                           head_size,block_size,x,table_stride,max_context,scale); \
} \
extern "C" __global__ void dkr_paged_v2_merge_##NAME( \
    const float* partial, const float* lse, TYPE* output, \
    uint32_t seqs, uint32_t heads, uint32_t head_size, uint32_t partitions) { \
  paged_v2_merge<TYPE>(partial,lse,output,seqs,heads,head_size,partitions); \
}

DKR_PAGED_V2(f16, __half)
DKR_PAGED_V2(bf16, __nv_bfloat16)
DKR_PAGED_V2(f32, float)
