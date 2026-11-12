#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// DKR-R1.21/S7 — byte-exact KV page copy + canonical VLLM slot write.
// All sources and destinations are validated by the Rust adapter; mappings
// must be independent for copy and slots must be valid for reshape.
// No GPU-side host transfers, extra device allocations or SM80 intrinsics.

extern "C" __global__ void dkr_kv_copy_blocks_bytes(
    const int64_t* __restrict__ key_ptrs,
    const int64_t* __restrict__ value_ptrs,
    const int64_t* __restrict__ block_pairs, // [pair_count, 2]
    uint32_t layers, uint32_t pairs,
    uint64_t key_block_bytes, uint64_t value_block_bytes) {
  const uint32_t layer = blockIdx.x;
  const uint32_t pair = blockIdx.y;
  if (layer >= layers || pair >= pairs) return;

  const int64_t src = block_pairs[static_cast<uint64_t>(pair) * 2u];
  const int64_t dst = block_pairs[static_cast<uint64_t>(pair) * 2u + 1u];
  if (src < 0 || dst < 0 || src == dst) return;

  const uint64_t key_src = static_cast<uint64_t>(src) * key_block_bytes;
  const uint64_t key_dst = static_cast<uint64_t>(dst) * key_block_bytes;
  const uint64_t val_src = static_cast<uint64_t>(src) * value_block_bytes;
  const uint64_t val_dst = static_cast<uint64_t>(dst) * value_block_bytes;

  const uint8_t* key_in = reinterpret_cast<const uint8_t*>(key_ptrs[layer]);
  uint8_t* key_out = reinterpret_cast<uint8_t*>(key_ptrs[layer]);
  const uint8_t* value_in = reinterpret_cast<const uint8_t*>(value_ptrs[layer]);
  uint8_t* value_out = reinterpret_cast<uint8_t*>(value_ptrs[layer]);

  for (uint64_t i = threadIdx.x; i < key_block_bytes; i += blockDim.x) {
    key_out[key_dst + i] = key_in[key_src + i];
  }
  for (uint64_t i = threadIdx.x; i < value_block_bytes; i += blockDim.x) {
    value_out[val_dst + i] = value_in[val_src + i];
  }
}

template<typename T>
__device__ __forceinline__ void kv_reshape_store(
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ key_cache,
    T* __restrict__ value_cache,
    const int64_t* __restrict__ slots,
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
    key_cache[key_index] = key[input_base + i];
    value_cache[value_index] = value[input_base + i];
  }
}

#define DKR_KV_RESHAPE(DT, NAME) \
extern "C" __global__ void dkr_kv_reshape_store_##NAME( \
    const DT* key, const DT* value, DT* key_cache, DT* value_cache, \
    const int64_t* slots, uint32_t tokens, uint32_t heads, \
    uint32_t head_size, uint32_t num_blocks, uint32_t block_size, \
    uint32_t x) { \
  kv_reshape_store<DT>(key, value, key_cache, value_cache, slots, \
                       tokens, heads, head_size, num_blocks, block_size, x); \
}

DKR_KV_RESHAPE(__half, f16)
DKR_KV_RESHAPE(__nv_bfloat16, bf16)
DKR_KV_RESHAPE(float, f32)
