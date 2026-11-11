#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

template <typename Input, typename Output>
__device__ __forceinline__ void weighted_reduce_body(
    const Input *input, const float *weights, Output *output,
    uint64_t tokens, uint64_t hidden, uint64_t topk) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden)
    return;
  const uint64_t token = index / hidden;
  const uint64_t column = index % hidden;
  float sum = 0.0f;
  for (uint64_t route = 0; route < topk; ++route) {
    const uint64_t row = token * topk + route;
    const float value = static_cast<float>(input[row * hidden + column]);
    sum = __fadd_rn(sum, __fmul_rn(value, weights[row]));
  }
  output[index] = static_cast<Output>(sum);
}

#define DKR_WEIGHTED_ENTRY(IN_TAG, OUT_TAG, IN_TYPE, OUT_TYPE)                 \
  extern "C" __global__ void dkr_weighted_##IN_TAG##_##OUT_TAG(             \
      const IN_TYPE *input, const float *weights, OUT_TYPE *output,           \
      uint64_t tokens, uint64_t hidden, uint64_t topk) {                     \
    weighted_reduce_body(input, weights, output, tokens, hidden, topk);       \
  }

DKR_WEIGHTED_ENTRY(f16, f16, __half, __half)
DKR_WEIGHTED_ENTRY(f16, bf16, __half, __nv_bfloat16)
DKR_WEIGHTED_ENTRY(f16, f32, __half, float)
DKR_WEIGHTED_ENTRY(bf16, f16, __nv_bfloat16, __half)
DKR_WEIGHTED_ENTRY(bf16, bf16, __nv_bfloat16, __nv_bfloat16)
DKR_WEIGHTED_ENTRY(bf16, f32, __nv_bfloat16, float)
DKR_WEIGHTED_ENTRY(f32, f16, float, __half)
DKR_WEIGHTED_ENTRY(f32, bf16, float, __nv_bfloat16)
DKR_WEIGHTED_ENTRY(f32, f32, float, float)

__device__ __forceinline__ float dkr_warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ float dkr_warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

template <typename T>
__device__ __forceinline__ void softmax_sinks_body(
    const T *logits, const T *sinks, T *output,
    uint32_t num_heads, uint32_t q_len, uint32_t k_len) {
  const uint32_t row = blockIdx.x;
  const uint32_t head = (row / q_len) % num_heads;
  const uint64_t offset = static_cast<uint64_t>(row) * k_len;
  const T *row_logits = logits + offset;
  T *row_output = output + offset;
  const float sink_value = static_cast<float>(sinks[head]);

  __shared__ float block_max;
  __shared__ float block_sum;
  __shared__ float warp_maxes[32];
  __shared__ float warp_sums[32];

  const int tid = threadIdx.x;
  const int block_size = blockDim.x;
  float local_max = -INFINITY;
  for (uint32_t column = tid; column < k_len; column += block_size)
    local_max = fmaxf(local_max, static_cast<float>(row_logits[column]));
  if (tid == 0)
    local_max = fmaxf(local_max, sink_value);
  local_max = dkr_warp_max(local_max);
  if (block_size > 32) {
    if (tid % 32 == 0)
      warp_maxes[tid / 32] = local_max;
    __syncthreads();
    if (tid < 32) {
      local_max = tid < (block_size + 31) / 32 ? warp_maxes[tid] : -INFINITY;
      local_max = dkr_warp_max(local_max);
    }
  }
  if (tid == 0)
    block_max = local_max;
  __syncthreads();
  const float row_max = block_max;

  float local_sum = 0.0f;
  for (uint32_t column = tid; column < k_len; column += block_size)
    local_sum += expf(static_cast<float>(row_logits[column]) - row_max);
  if (tid == 0)
    local_sum += expf(sink_value - row_max);
  local_sum = dkr_warp_sum(local_sum);
  if (block_size > 32) {
    if (tid % 32 == 0)
      warp_sums[tid / 32] = local_sum;
    __syncthreads();
    if (tid < 32) {
      local_sum = tid < (block_size + 31) / 32 ? warp_sums[tid] : 0.0f;
      local_sum = dkr_warp_sum(local_sum);
    }
  }
  if (tid == 0)
    block_sum = local_sum;
  __syncthreads();

  const float inverse_sum = 1.0f / block_sum;
  for (uint32_t column = tid; column < k_len; column += block_size) {
    const float value = static_cast<float>(row_logits[column]);
    row_output[column] = static_cast<T>(expf(value - row_max) * inverse_sum);
  }
}

#define DKR_SOFTMAX_ENTRY(TAG, TYPE)                                          \
  extern "C" __global__ void dkr_softmax_sinks_##TAG(                        \
      const TYPE *logits, const TYPE *sinks, TYPE *output,                    \
      uint32_t num_heads, uint32_t q_len, uint32_t k_len) {                  \
    softmax_sinks_body(logits, sinks, output, num_heads, q_len, k_len);       \
  }

DKR_SOFTMAX_ENTRY(f16, __half)
DKR_SOFTMAX_ENTRY(bf16, __nv_bfloat16)
DKR_SOFTMAX_ENTRY(f32, float)
