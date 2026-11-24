#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <math.h>
#include <stdint.h>

#include <cuda_fp8.h>

// FP8 E4M3 partitioned decode, F32 partials and log-normalizers.
// The existing paged-v2 merge ABI is unchanged.
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

template <typename C, bool Scaled>
__device__ __forceinline__ float options_value(C value, const float* scale) {
  if constexpr (Scaled) return __half2float(__nv_cvt_fp8_to_halfraw(value.__x, __NV_E4M3))*scale[0];
  else return v2_to_float(value);
}

template <typename T, typename C, bool Scaled, bool Alibi>
__device__ __forceinline__ void paged_options_partition(
    const T* __restrict__ query,
    const C* __restrict__ key_cache,
    const C* __restrict__ value_cache,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ context_lens,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const float* __restrict__ slopes,
    float* __restrict__ partial,
    float* __restrict__ logsumexp,
    uint32_t seqs, uint32_t heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t x,
    uint32_t table_stride, uint32_t max_context, uint32_t blocks, float scale, uint32_t window, float cap) {
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
  const uint32_t begin = part * DKR_V2_PARTITION;
  const uint32_t start = max(begin, window && context > window ? context-window : 0u);
  const uint32_t end = min(context, begin + DKR_V2_PARTITION);
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
    if (page >= blocks) continue; // uniform across the block; valid metadata is a caller contract
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
        dot = fmaf(query_value[j], options_value<C,Scaled>(key_cache[address], k_scale), dot);
      }
    }
    dot_sums[tid] = dot;
    __syncthreads();
    for (uint32_t stride = DKR_V2_THREADS / 2; stride > 0; stride >>= 1) {
      if (tid < stride) dot_sums[tid] += dot_sums[tid + stride];
      __syncthreads();
    }
    float logit = dot_sums[0] * scale;
    if (cap != 1.0f) logit = tanhf(logit/cap)*cap;
    if constexpr (Alibi) logit += slopes[head]*(static_cast<int>(token)-static_cast<int>(context)+1);
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
            + token_weight * options_value<C,Scaled>(value_cache[address], v_scale);
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
__device__ __forceinline__ void paged_sink_merge(
    const float* __restrict__ partial,
    const float* __restrict__ logsumexp,
    const float* __restrict__ sinks,
    T* __restrict__ output,
    uint32_t seqs, uint32_t heads, uint32_t head_size, uint32_t partitions) {
  const uint32_t head = blockIdx.x;
  const uint32_t seq = blockIdx.y;
  if (head >= heads || seq >= seqs) return;
  const uint32_t d = threadIdx.x;
  const uint64_t row = static_cast<uint64_t>(seq) * heads + head;
  const uint64_t lse_base = row * partitions;
  const uint64_t out_base = row * head_size;
  float max_log = sinks[head];
  for (uint32_t p = 0; p < partitions; ++p) {
    const float lse = logsumexp[lse_base + p];
    max_log = fmaxf(max_log, lse);
  }
  const bool any = max_log != -CUDART_INF_F;
  float denominator = any && sinks[head] != -CUDART_INF_F ? expf(sinks[head] - max_log) : 0.0f;
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


extern "C" __global__ void dkr_paged_options_plain_noalibi_f16(const __half* q,const __half* k,const __half* v,const uint32_t* table,const uint32_t* lens,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__half,__half,false,false>(q,k,v,table,lens,nullptr,nullptr,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_plain_alibi_f16(const __half* q,const __half* k,const __half* v,const uint32_t* table,const uint32_t* lens,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__half,__half,false,true>(q,k,v,table,lens,nullptr,nullptr,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_noalibi_f16(const __half* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__half,__nv_fp8_e4m3,true,false>(q,k,v,table,lens,ks,vs,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_alibi_f16(const __half* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__half,__nv_fp8_e4m3,true,true>(q,k,v,table,lens,ks,vs,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_sink_merge_f16(const float* p,const float* l,const float* sinks,__half* out,uint32_t seqs,uint32_t heads,uint32_t hd,uint32_t parts) { paged_sink_merge<__half>(p,l,sinks,out,seqs,heads,hd,parts); }

extern "C" __global__ void dkr_paged_options_plain_noalibi_bf16(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const uint32_t* table,const uint32_t* lens,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__nv_bfloat16,__nv_bfloat16,false,false>(q,k,v,table,lens,nullptr,nullptr,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_plain_alibi_bf16(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,const uint32_t* table,const uint32_t* lens,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__nv_bfloat16,__nv_bfloat16,false,true>(q,k,v,table,lens,nullptr,nullptr,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_noalibi_bf16(const __nv_bfloat16* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__nv_bfloat16,__nv_fp8_e4m3,true,false>(q,k,v,table,lens,ks,vs,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_alibi_bf16(const __nv_bfloat16* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<__nv_bfloat16,__nv_fp8_e4m3,true,true>(q,k,v,table,lens,ks,vs,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_sink_merge_bf16(const float* p,const float* l,const float* sinks,__nv_bfloat16* out,uint32_t seqs,uint32_t heads,uint32_t hd,uint32_t parts) { paged_sink_merge<__nv_bfloat16>(p,l,sinks,out,seqs,heads,hd,parts); }

extern "C" __global__ void dkr_paged_options_plain_noalibi_f32(const float* q,const float* k,const float* v,const uint32_t* table,const uint32_t* lens,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<float,float,false,false>(q,k,v,table,lens,nullptr,nullptr,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_plain_alibi_f32(const float* q,const float* k,const float* v,const uint32_t* table,const uint32_t* lens,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<float,float,false,true>(q,k,v,table,lens,nullptr,nullptr,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_noalibi_f32(const float* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<float,__nv_fp8_e4m3,true,false>(q,k,v,table,lens,ks,vs,nullptr,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_fp8_alibi_f32(const float* q,const __nv_fp8_e4m3* k,const __nv_fp8_e4m3* v,const uint32_t* table,const uint32_t* lens,const float* ks,const float* vs,const float* slopes,float* partial,float* lse,uint32_t seqs,uint32_t heads,uint32_t kv_heads,uint32_t hd,uint32_t page,uint32_t x,uint32_t stride,uint32_t max_ctx,uint32_t blocks,float scale,uint32_t window,float cap) {
 paged_options_partition<float,__nv_fp8_e4m3,true,true>(q,k,v,table,lens,ks,vs,slopes,partial,lse,seqs,heads,kv_heads,hd,page,x,stride,max_ctx,blocks,scale,window,cap);
}

extern "C" __global__ void dkr_paged_options_sink_merge_f32(const float* p,const float* l,const float* sinks,float* out,uint32_t seqs,uint32_t heads,uint32_t hd,uint32_t parts) { paged_sink_merge<float>(p,l,sinks,out,seqs,heads,hd,parts); }
