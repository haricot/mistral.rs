#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Canonical contiguous VLLM layouts, device-resident positive scalar scales.
// Direct float -> E4M3 SATFINITE (not the half-rounded scalar-cast ABI).
// Valid page metadata and unique nonnegative write slots are caller contracts.
template<typename T>
__device__ __forceinline__ void kv_fp8_store(
    const T* __restrict__ key,
    const T* __restrict__ value,
    uint8_t* __restrict__ key_cache,
    uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slots,
    const float* k_scale, const float* v_scale,
    uint32_t tokens, uint32_t heads, uint32_t head_size,
    uint32_t num_blocks, uint32_t block_size, uint32_t x) {
  const uint32_t token = blockIdx.x;
  if (token >= tokens) return;
  const int64_t slot = slots[token];
  // -1 is the VLLM padding sentinel. Reject any other invalid slot.
  if (slot < 0 || static_cast<uint64_t>(slot) >=
          static_cast<uint64_t>(num_blocks) * block_size) return;
  const uint32_t page = static_cast<uint64_t>(slot) / block_size;
  const uint32_t within = static_cast<uint64_t>(slot) % block_size;
  const uint64_t stride = static_cast<uint64_t>(heads) * head_size;
  const uint64_t input_base = static_cast<uint64_t>(token) * stride;
  const uint64_t page_base =
      static_cast<uint64_t>(page) * stride * block_size;

  for (uint64_t i = threadIdx.x; i < stride; i += blockDim.x) {
    const uint32_t head = i / head_size;
    const uint32_t d = i % head_size;
    const uint64_t key_index = page_base
        + static_cast<uint64_t>(head) * head_size * block_size
        + static_cast<uint64_t>(d / x) * block_size * x
        + static_cast<uint64_t>(within) * x + d % x;
    const uint64_t value_index = page_base
        + static_cast<uint64_t>(head) * head_size * block_size
        + static_cast<uint64_t>(d) * block_size + within;
    key_cache[key_index] = __nv_cvt_float_to_fp8(
        static_cast<float>(key[input_base + i]) / *k_scale, __NV_SATFINITE, __NV_E4M3);
    value_cache[value_index] = __nv_cvt_float_to_fp8(
        static_cast<float>(value[input_base + i]) / *v_scale, __NV_SATFINITE, __NV_E4M3);
  }
}


template<typename T>
__device__ __forceinline__ void kv_fp8_gather(
    const uint8_t* __restrict__ keys, const uint8_t* __restrict__ values,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    T* __restrict__ key_output, T* __restrict__ value_output,
    const float* k_scale, const float* v_scale,
    uint32_t num_tokens, uint32_t num_seqs,
    uint32_t block_size, uint32_t block_table_stride,
    uint32_t num_heads, uint32_t head_size, uint32_t x, uint32_t num_blocks) {
  const uint32_t token = blockIdx.x;
  if (token >= num_tokens) return;
  uint32_t lo = 0, hi = num_seqs;
  while (lo < hi) {
    const uint32_t mid = (lo + hi + 1) / 2;
    if (cu_seq_lens[mid] <= static_cast<int32_t>(token)) lo = mid;
    else hi = mid - 1;
  }
  if (lo >= num_seqs) return;
  const uint32_t position = token - static_cast<uint32_t>(cu_seq_lens[lo]);
  const uint32_t page_index = position / block_size;
  const uint32_t slot = position % block_size;
  if (page_index >= block_table_stride) return;
  const int32_t page = block_table[
      static_cast<uint64_t>(lo) * block_table_stride + page_index];
  if (page < 0 || static_cast<uint32_t>(page) >= num_blocks) return;

  const uint64_t k_page_stride =
      static_cast<uint64_t>(num_heads) * head_size * block_size;
  const uint64_t v_page_stride = k_page_stride;
  const uint64_t k_head_stride =
      static_cast<uint64_t>(head_size) * block_size;
  const uint64_t v_head_stride = k_head_stride;
  const uint64_t out_base =
      static_cast<uint64_t>(token) * num_heads * head_size;

  for (uint64_t i = threadIdx.x; i < static_cast<uint64_t>(num_heads) * head_size; i += blockDim.x) {
    const uint32_t head = i / head_size;
    const uint32_t d = i % head_size;
    const uint64_t ki = static_cast<uint64_t>(page) * k_page_stride
        + head * k_head_stride + (d / x) * block_size * x + slot * x + d % x;
    const uint64_t vi = static_cast<uint64_t>(page) * v_page_stride
        + head * v_head_stride + d * block_size + slot;
    key_output[out_base + i] = static_cast<T>(
        __half2float(__nv_cvt_fp8_to_halfraw(keys[ki], __NV_E4M3)) * *k_scale);
    value_output[out_base + i] = static_cast<T>(
        __half2float(__nv_cvt_fp8_to_halfraw(values[vi], __NV_E4M3)) * *v_scale);
  }
}


#define DKR_KV_FP8(T, NAME) \
extern "C" __global__ void dkr_kv_fp8_store_##NAME( \
    const T* key, const T* value, uint8_t* kc, uint8_t* vc, \
    const int64_t* slots, const float* ks, const float* vs, \
    uint32_t tokens, uint32_t heads, uint32_t hd, uint32_t blocks, \
    uint32_t page, uint32_t x) { \
  kv_fp8_store<T>(key, value, kc, vc, slots, ks, vs, tokens, heads, hd, blocks, page, x); \
} \
extern "C" __global__ void dkr_kv_fp8_gather_##NAME( \
    const uint8_t* kc, const uint8_t* vc, const int32_t* table, const int32_t* offsets, \
    T* ko, T* vo, const float* ks, const float* vs, \
    uint32_t tokens, uint32_t seqs, uint32_t page, uint32_t stride, \
    uint32_t heads, uint32_t hd, uint32_t x, uint32_t blocks) { \
  kv_fp8_gather<T>(kc, vc, table, offsets, ko, vo, ks, vs, tokens, seqs, page, stride, heads, hd, x, blocks); \
}
DKR_KV_FP8(__half, f16)
DKR_KV_FP8(__nv_bfloat16, bf16)
DKR_KV_FP8(float, f32)

// Rank-1 physical input spans (including row padding), with element strides.
// The Rust adapter bounds each span against its backing CUDA allocation.
template<typename T, typename Cache, bool Scaled>
__device__ __forceinline__ void kv_store_strided(
    const T* key, const T* value, Cache* kc, Cache* vc,
    const int64_t* slots, const float* ks, const float* vs,
    uint32_t tokens, uint32_t heads, uint32_t hd, uint32_t blocks,
    uint32_t page_size, uint32_t x, uint64_t k_stride, uint64_t v_stride) {
  const uint32_t token = blockIdx.x;
  if (token >= tokens) return;
  const int64_t slot = slots[token];
  if (slot < 0 || static_cast<uint64_t>(slot) >=
      static_cast<uint64_t>(blocks)*page_size) return;
  const uint64_t page = static_cast<uint64_t>(slot)/page_size;
  const uint64_t pos = static_cast<uint64_t>(slot)%page_size;
  const uint64_t width = static_cast<uint64_t>(heads)*hd;
  for (uint64_t i=threadIdx.x; i<width; i+=blockDim.x) {
    const uint64_t h=i/hd, d=i%hd;
    const uint64_t base=(page*heads+h)*hd*page_size;
    const uint64_t ki=base+(d/x)*page_size*x+pos*x+d%x;
    const uint64_t vi=base+d*page_size+pos;
    const T k=key[static_cast<uint64_t>(token)*k_stride+i];
    const T v=value[static_cast<uint64_t>(token)*v_stride+i];
    if constexpr (Scaled) {
      kc[ki]=__nv_cvt_float_to_fp8(static_cast<float>(k)/ *ks, __NV_SATFINITE, __NV_E4M3);
      vc[vi]=__nv_cvt_float_to_fp8(static_cast<float>(v)/ *vs, __NV_SATFINITE, __NV_E4M3);
    } else {
      kc[ki]=k;
      vc[vi]=v;
    }
  }
}
#define DKR_KV_STRIDED(T, NAME) \
extern "C" __global__ void dkr_kv_store_strided_##NAME( \
    const T* k, const T* v, T* kc, T* vc, const int64_t* slots, \
    uint32_t tokens, uint32_t heads, uint32_t hd, uint32_t blocks, \
    uint32_t page, uint32_t x, uint64_t k_stride, uint64_t v_stride) { \
  kv_store_strided<T,T,false>(k,v,kc,vc,slots,nullptr,nullptr,tokens,heads,hd,blocks,page,x,k_stride,v_stride); \
} \
extern "C" __global__ void dkr_kv_fp8_store_strided_##NAME( \
    const T* k, const T* v, uint8_t* kc, uint8_t* vc, const int64_t* slots, \
    const float* ks, const float* vs, uint32_t tokens, uint32_t heads, \
    uint32_t hd, uint32_t blocks, uint32_t page, uint32_t x, \
    uint64_t k_stride, uint64_t v_stride) { \
  kv_store_strided<T,uint8_t,true>(k,v,kc,vc,slots,ks,vs,tokens,heads,hd,blocks,page,x,k_stride,v_stride); \
}
DKR_KV_STRIDED(__half, f16)
DKR_KV_STRIDED(__nv_bfloat16, bf16)
DKR_KV_STRIDED(float, f32)
