#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// DKR KV-cache FP8 scale update. Match the static reference:
// scale <- max(scale, max(abs(K or V)) / 240.0f).
// Requires initialized nonnegative finite scalar scales.
// 512 threads, grid=min(ceil(elements/512), 65535), 4096 dynamic shared bytes.
template<typename T>
__device__ __forceinline__ float dkr_scale_abs(T value) {
  return fabsf(static_cast<float>(value));
}
template<>
__device__ __forceinline__ float dkr_scale_abs<__half>(__half value) {
  return fabsf(__half2float(value));
}
template<>
__device__ __forceinline__ float dkr_scale_abs<__nv_bfloat16>(__nv_bfloat16 value) {
  return fabsf(__bfloat162float(value));
}

__device__ __forceinline__ void dkr_atomic_max_nonnegative(float* ptr, float candidate) {
  int* address = reinterpret_cast<int*>(ptr);
  int old = *address;
  while (__int_as_float(old) < candidate) {
    const int assumed = old;
    old = atomicCAS(address, assumed, __float_as_int(candidate));
    if (old == assumed) break;
  }
}

template<typename T>
__device__ __forceinline__ void dkr_kv_scale_body(
    const T* __restrict__ keys, const T* __restrict__ values,
    float* __restrict__ k_scale, float* __restrict__ v_scale,
    uint64_t elements) {
  extern __shared__ float scratch[];
  float* local_k = scratch;
  float* local_v = scratch + blockDim.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + lane;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  float k_max = 0.0f;
  float v_max = 0.0f;
  for (uint64_t i = index; i < elements; i += stride) {
    k_max = fmaxf(k_max, dkr_scale_abs(keys[i]));
    v_max = fmaxf(v_max, dkr_scale_abs(values[i]));
  }
  local_k[lane] = k_max;
  local_v[lane] = v_max;
  __syncthreads();
  for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
    if (lane < static_cast<uint32_t>(step)) {
      local_k[lane] = fmaxf(local_k[lane], local_k[lane + step]);
      local_v[lane] = fmaxf(local_v[lane], local_v[lane + step]);
    }
    __syncthreads();
  }
  if (lane == 0) {
    const float candidate_k = local_k[0] / 240.0f;
    const float candidate_v = local_v[0] / 240.0f;
    if (candidate_k > 0.0f) dkr_atomic_max_nonnegative(k_scale, candidate_k);
    if (candidate_v > 0.0f) dkr_atomic_max_nonnegative(v_scale, candidate_v);
  }
}

#define DKR_DEFINE_KV_SCALE(TAG, TYPE) \
extern "C" __global__ void dkr_kv_scale_update_##TAG( \
    const TYPE* key, const TYPE* value, float* k_scale, float* v_scale, \
    uint64_t elements) { \
  dkr_kv_scale_body<TYPE>(key, value, k_scale, v_scale, elements); \
}

DKR_DEFINE_KV_SCALE(f16, __half)
DKR_DEFINE_KV_SCALE(bf16, __nv_bfloat16)
DKR_DEFINE_KV_SCALE(f32, float)
