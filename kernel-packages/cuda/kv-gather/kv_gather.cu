#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// VLLM paged layouts:
// K: [blocks, heads, head_size/x, block_size, x]
// V: [blocks, heads, head_size, block_size]
// Output: [tokens, heads, head_size].
// A CUDA block processes one output token. No scales or quantized KV here.
template<typename T>
__device__ __forceinline__ void gather_kv(
    const T* __restrict__ keys, const T* __restrict__ values,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ cu_seq_lens,
    T* __restrict__ key_output, T* __restrict__ value_output,
    uint32_t num_tokens, uint32_t num_seqs,
    uint32_t block_size, uint32_t block_table_stride,
    uint32_t num_heads, uint32_t head_size, uint32_t x) {
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
  const int32_t page = block_table[
      static_cast<uint64_t>(lo) * block_table_stride + page_index];

  const uint64_t k_page_stride =
      static_cast<uint64_t>(num_heads) * head_size * block_size;
  const uint64_t v_page_stride = k_page_stride;
  const uint64_t k_head_stride =
      static_cast<uint64_t>(head_size) * block_size;
  const uint64_t v_head_stride = k_head_stride;
  const uint64_t out_base =
      static_cast<uint64_t>(token) * num_heads * head_size;

  for (uint32_t i = threadIdx.x; i < num_heads * head_size; i += blockDim.x) {
    const uint32_t head = i / head_size;
    const uint32_t d = i % head_size;
    const uint64_t ki = static_cast<uint64_t>(page) * k_page_stride
        + head * k_head_stride + (d / x) * block_size * x + slot * x + d % x;
    const uint64_t vi = static_cast<uint64_t>(page) * v_page_stride
        + head * v_head_stride + d * block_size + slot;
    key_output[out_base + i] = keys[ki];
    value_output[out_base + i] = values[vi];
  }
}

#define DKR_KV_GATHER(NAME, TYPE) \
extern "C" __global__ void dkr_kv_gather_##NAME( \
    const TYPE* keys, const TYPE* values, const int32_t* block_table, \
    const int32_t* cu_seq_lens, TYPE* key_output, TYPE* value_output, \
    uint32_t tokens, uint32_t seqs, uint32_t block_size, \
    uint32_t block_table_stride, uint32_t heads, uint32_t head_size, \
    uint32_t x) { \
  gather_kv<TYPE>(keys, values, block_table, cu_seq_lens, \
      key_output, value_output, tokens, seqs, block_size, \
      block_table_stride, heads, head_size, x); \
}

DKR_KV_GATHER(f16, __half)
DKR_KV_GATHER(bf16, __nv_bfloat16)
DKR_KV_GATHER(f32, float)
