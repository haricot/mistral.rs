#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

// DKR-R1.21/S5: short-context FP8 E4M3 paged decode on Pascal.
// Contiguous VLLM K/V layout. Q in F16/BF16/F32, K and V in E4M3.
// Explicit F32 scalar K/V scales, no ALiBi, sinks, windows or softcap.
// One cooperative 128-thread block per (sequence, Q-head).
// max_context <= 256, head_size <= 512. No scratch allocation.

template <typename T>
__device__ __forceinline__ float dkr_as_f32(T x) {
  return static_cast<float>(x);
}
template <>
__device__ __forceinline__ float dkr_as_f32<__half>(__half x) {
  return __half2float(x);
}
template <>
__device__ __forceinline__ float dkr_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
template <typename T>
__device__ __forceinline__ T dkr_from_f32(float x) {
  return static_cast<T>(x);
}
template <>
__device__ __forceinline__ __half dkr_from_f32<__half>(float x) {
  return __float2half_rn(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 dkr_from_f32<__nv_bfloat16>(float x) {
  return __float2bfloat16_rn(x);
}

template <typename T>
__device__ __forceinline__ void short_decode(
    const T* __restrict__ q,
    const __nv_fp8_e4m3* __restrict__ key,
    const __nv_fp8_e4m3* __restrict__ value,
    const uint32_t* __restrict__ block_table,
    const uint32_t* __restrict__ context_lens,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    T* __restrict__ output,
    uint32_t num_seqs, uint32_t num_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t block_size, uint32_t x,
    uint32_t table_stride, uint32_t max_context, float scale) {
  const uint32_t head = blockIdx.x;
  const uint32_t seq = blockIdx.y;
  if (seq >= num_seqs || head >= num_heads) return;

  const uint32_t tid = threadIdx.x;
  const uint32_t kvhead = head / (num_heads / kv_heads);
  const uint32_t context = min(context_lens[seq], max_context);
  const uint64_t qoffset = (static_cast<uint64_t>(seq) * num_heads + head) * head_size;
  const uint64_t block_elements =
      static_cast<uint64_t>(kv_heads) * head_size * block_size;
  const uint64_t khead_offset = static_cast<uint64_t>(kvhead) * head_size * block_size;
  const uint64_t vhead_offset = khead_offset;
  __shared__ float scratch[128];

  // Each thread owns at most four output coordinates (head_size <= 512).
  float qval[4] = {};
  float accumulator[4] = {};
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t d = tid + j * blockDim.x;
    if (d < head_size) qval[j] = dkr_as_f32(q[qoffset + d]);
  }

  float running_max = -CUDART_INF_F;
  float denominator = 0.0f;
  for (uint32_t token = 0; token < context; ++token) {
    const uint32_t page = block_table[
        static_cast<uint64_t>(seq) * table_stride + token / block_size];
    const uint32_t slot = token % block_size;
    const uint64_t kbase = static_cast<uint64_t>(page) * block_elements
        + khead_offset;
    const uint64_t vbase = static_cast<uint64_t>(page) * block_elements
        + vhead_offset;
    float dot = 0.0f;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t d = tid + j * blockDim.x;
      if (d < head_size) {
        const uint64_t ki = kbase + static_cast<uint64_t>(d / x) * block_size * x
            + slot * x + d % x;
        dot = fmaf(qval[j], (__half2float(__nv_cvt_fp8_to_halfraw(key[ki].__x, __NV_E4M3)) * k_scale[0]), dot);
      }
    }
    scratch[tid] = dot;
    __syncthreads();
    for (int shift = 64; shift > 0; shift >>= 1) {
      if (tid < static_cast<uint32_t>(shift))
        scratch[tid] += scratch[tid + shift];
      __syncthreads();
    }
    const float logit = scratch[0] * scale;
    const float maximum = fmaxf(running_max, logit);
    const float past = expf(running_max - maximum);
    const float present = expf(logit - maximum);
    denominator = denominator * past + present;
    running_max = maximum;
    for (uint32_t j = 0; j < 4; ++j) {
      const uint32_t d = tid + j * blockDim.x;
      if (d < head_size) {
        const uint64_t vi = vbase + static_cast<uint64_t>(d) * block_size + slot;
        accumulator[j] =
            accumulator[j] * past + present * (__half2float(__nv_cvt_fp8_to_halfraw(value[vi].__x, __NV_E4M3)) * v_scale[0]);
      }
    }
    __syncthreads();
  }
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t d = tid + j * blockDim.x;
    if (d < head_size)
      output[qoffset + d] =
          dkr_from_f32<T>(denominator > 0.0f ? accumulator[j] / denominator : 0.0f);
  }
}

#define DKR_DECODE(NAME, TYPE)                                         \
extern "C" __global__ void dkr_paged_fp8_decode_##NAME(                   \
    const TYPE* q, const __nv_fp8_e4m3* k, const __nv_fp8_e4m3* v,                        \
    const uint32_t* table, const uint32_t* lens,                       \
    const float* k_scale, const float* v_scale, TYPE* output,          \
    uint32_t seqs, uint32_t q_heads, uint32_t kv_heads,                \
    uint32_t head_size, uint32_t block_size, uint32_t x,               \
    uint32_t table_stride, uint32_t max_context, float scale) {        \
  short_decode<TYPE>(q, k, v, table, lens, k_scale, v_scale, output, seqs, q_heads,       \
                     kv_heads, head_size, block_size, x, table_stride, \
                     max_context, scale);                              \
}
DKR_DECODE(f16, __half)
DKR_DECODE(bf16, __nv_bfloat16)
DKR_DECODE(f32, float)
