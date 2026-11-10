#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float decode_f16(uint16_t bits) {
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(bits));
  return result;
}

__device__ __forceinline__ uint16_t encode_f16(float value) {
  uint16_t result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float decode_bf16(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t encode_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  uint32_t round = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + round) >> 16);
}

__device__ __forceinline__ float decode(uint16_t value) {
#ifdef DKR_BF16_MODE
  return decode_bf16(value);
#else
  return decode_f16(value);
#endif
}

__device__ __forceinline__ uint16_t encode(float value) {
#ifdef DKR_BF16_MODE
  return encode_bf16(value);
#else
  return encode_f16(value);
#endif
}

extern "C" __global__ void dkr_moe_grouped_gemm(
    const uint16_t *__restrict__ input,
    const uint16_t *__restrict__ weights,
    const uint32_t *__restrict__ sorted_token_ids,
    const uint32_t *__restrict__ expert_ids,
    const float *__restrict__ gating_weights,
    uint16_t *__restrict__ output,
    uint32_t num_experts,
    uint32_t topk,
    uint32_t size_m,
    uint32_t size_n,
    uint32_t size_k,
    uint32_t has_gating) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(size_m) * size_n) return;

  const uint32_t routed_index = static_cast<uint32_t>(index / size_n);
  const uint32_t n = static_cast<uint32_t>(index % size_n);
  const uint32_t expert = expert_ids[routed_index];
  const uint32_t token = sorted_token_ids[routed_index];
  if (expert >= num_experts || token >= size_m || topk == 0) return;

  const uint32_t input_index = has_gating ? token : token / topk;
  const uint32_t input_rows = has_gating ? size_m : size_m / topk;
  if (input_index >= input_rows) return;

  const uint16_t *input_row = input + static_cast<size_t>(input_index) * size_k;
  const uint16_t *weight_row =
      weights + (static_cast<size_t>(expert) * size_n + n) * size_k;

  float sum = 0.0f;
  for (uint32_t k = 0; k < size_k; ++k) {
    sum += decode(input_row[k]) * decode(weight_row[k]);
  }
  if (has_gating) sum *= gating_weights[token];
  output[static_cast<size_t>(token) * size_n + n] = encode(sum);
}
