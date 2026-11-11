#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

// Direct external equivalent of the in-place F32 mask in core/sort.cu.
extern "C" __global__ void dkr_causal_mask_f32(
    float *__restrict__ scores, const int32_t batch_heads,
    const int32_t q_len, const int32_t kv_len, const int32_t q_offset,
    const int32_t prefix_len) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_heads * q_len * kv_len;
  if (idx >= total) {
    return;
  }
  const int kv_idx = idx % kv_len;
  const int q_idx = (idx / kv_len) % q_len;
  const int q_pos = prefix_len + q_offset + q_idx;
  if (kv_idx > q_pos) {
    scores[idx] = -std::numeric_limits<float>::infinity();
  }
}
