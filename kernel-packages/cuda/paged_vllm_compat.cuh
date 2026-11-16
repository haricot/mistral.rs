#pragma once

#include <cuda_fp16.h>
#include <limits.h>
#include <math_constants.h>
#include <stdint.h>
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
#include "../../mistralrs-paged-attn/src/cuda/pagedattention.cuh"
#endif

namespace dkr_paged {

struct VllmInput {
  const __half* query;
  const __half* key;
  const __half* value;
  const uint32_t* table;
  const uint32_t* lengths;
  uint32_t seqs;
  uint32_t heads;
  uint32_t kv_heads;
  uint32_t head_size;
  uint32_t page_size;
  uint32_t x;
  uint32_t table_stride;
  uint32_t max_context;
  float scale;
};

template <typename Out>
__device__ bool try_vllm_v1(const VllmInput& input, Out* output,
                           float* logsumexp) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
  constexpr uint32_t HEAD_SIZE = 128;
  constexpr uint32_t PAGE_SIZE = 32;
  constexpr uint32_t THREADS = 128;
  constexpr uint32_t MAX_CONTEXT = 512;
  constexpr uint32_t ALIGNMENT = 16;
  const uint64_t query_stride = uint64_t(input.heads) * HEAD_SIZE;
  const uint64_t page_stride = uint64_t(input.kv_heads) * HEAD_SIZE * PAGE_SIZE;
  const uint64_t row = uint64_t(blockIdx.y) * input.heads + blockIdx.x;
  const uintptr_t addresses = reinterpret_cast<uintptr_t>(input.query) |
      reinterpret_cast<uintptr_t>(input.key) | reinterpret_cast<uintptr_t>(input.value);
  if (input.head_size != HEAD_SIZE || input.page_size != PAGE_SIZE ||
      input.x != ALIGNMENT / sizeof(__half) || input.max_context > MAX_CONTEXT ||
      input.lengths[blockIdx.y] > input.max_context || addresses % ALIGNMENT != 0 ||
      query_stride * input.seqs * gridDim.z > INT_MAX || page_stride > INT_MAX ||
      uint64_t(input.table_stride) * input.seqs > INT_MAX) {
    return false;
  }

  if constexpr (std::is_same_v<Out, float>) {
    // This context fits in one VLLM V1 launch. The merge ABI still has a slot per 256 tokens.
    if (blockIdx.z != 0 || input.lengths[blockIdx.y] == 0) {
      const uint64_t slot = row * gridDim.z + blockIdx.z;
      if (threadIdx.x == 0) logsumexp[slot] = -CUDART_INF_F;
      output[slot * HEAD_SIZE + threadIdx.x] = 0.0f;
      return true;
    }
  }

  using Storage = std::conditional_t<std::is_same_v<Out, float>, float, uint16_t>;
  // The archived typed launch ABI reserves no dynamic shared memory.
  vllm::paged_attention_kernel<uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto,
                              HEAD_SIZE, PAGE_SIZE, THREADS, 0, MAX_CONTEXT, Storage>(
      nullptr, nullptr, reinterpret_cast<Storage*>(output),
      reinterpret_cast<const uint16_t*>(input.query),
      reinterpret_cast<const uint16_t*>(input.key),
      reinterpret_cast<const uint16_t*>(input.value), input.kv_heads, input.scale,
      1.0f, input.table, input.lengths, input.table_stride, nullptr,
      static_cast<int>(query_stride), static_cast<int>(page_stride), HEAD_SIZE * PAGE_SIZE,
      nullptr, nullptr, nullptr, 0, logsumexp);
  return true;
#else
  return false;
#endif
}

}  // namespace dkr_paged
