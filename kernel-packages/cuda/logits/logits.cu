#include <stdint.h>

extern "C" __global__ void dkr_copy_f32(
    const float* __restrict__ input, float* __restrict__ output, uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) output[i] = input[i];
}

extern "C" __global__ void dkr_logits_penalties_sparse_f32(
    float* __restrict__ logits,
    const uint32_t* __restrict__ token_ids,
    const float* __restrict__ counts,
    uint32_t vocab_size, uint32_t count,
    float frequency_penalty, float presence_penalty, float repetition_penalty) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t id = token_ids[i];
  if (id >= vocab_size || counts[i] <= 0.0f) return;
  float value = logits[id];
  value -= counts[i] * frequency_penalty + presence_penalty;
  if (repetition_penalty != 1.0f) {
    value = value > 0.0f ? value / repetition_penalty : value * repetition_penalty;
  }
  logits[id] = value;
}

extern "C" __global__ void dkr_logits_bias_sparse_f32(
    float* __restrict__ logits,
    const uint32_t* __restrict__ token_ids,
    const float* __restrict__ biases,
    uint32_t vocab_size, uint32_t count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t id = token_ids[i];
  if (id < vocab_size) logits[id] += biases[i];
}
