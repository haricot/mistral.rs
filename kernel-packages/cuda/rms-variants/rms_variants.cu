#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ float as_float(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float as_float<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T> __device__ __forceinline__ T as_dtype(float v) {
  return static_cast<T>(v);
}
template <> __device__ __forceinline__ __half as_dtype<__half>(float v) {
  return __float2half(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 as_dtype<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

constexpr uint32_t WARP_THREADS = 32;
constexpr uint32_t LEGACY_MAX_THREADS = 1024;
constexpr uint32_t VECTOR_ITEMS = 8;

template <typename T, uint32_t ITEMS> struct Fragment {
  T values[ITEMS];
};

template <typename T> struct alignas(16) Fragment<T, VECTOR_ITEMS> {
  T values[VECTOR_ITEMS];
};

__device__ __forceinline__ bool vector_aligned(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % alignof(Fragment<__half, VECTOR_ITEMS>) == 0;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (uint32_t offset = WARP_THREADS / 2; offset; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void store_warp_sum(float value, uint32_t lane,
                                               float* warp_sums) {
  const float sum = warp_sum(value);
  if (lane % WARP_THREADS == 0) warp_sums[lane / WARP_THREADS] = sum;
}

__device__ __forceinline__ float row_sum(uint32_t lanes, float* warp_sums) {
  __syncthreads();
  if (threadIdx.x < WARP_THREADS) {
    const float part = threadIdx.x < lanes / WARP_THREADS ? warp_sums[threadIdx.x] : 0.0f;
    const float sum = warp_sum(part);
    if (threadIdx.x == 0) warp_sums[0] = sum;
  }
  __syncthreads();
  const float sum = warp_sums[0];
  __syncthreads();  // All warps must read the sum before the next reduction reuses it.
  return sum;
}

template <typename T, uint32_t ITEMS, bool SCALED, bool SECOND_NORM>
__device__ void residual_fragments(
    const T* x, const T* residual, const T* weight, const T* scale,
    const T* norm_weight, T* residual_output, T* normalized_output,
    uint32_t width, float eps1, float eps2) {
  using Pack = Fragment<T, ITEMS>;
  const uint32_t columns = width / ITEMS;
  // Match the scalar and vec8 launch geometry in the static RMSNorm kernels.
  const uint32_t lanes = ITEMS == 1
      ? (width < LEGACY_MAX_THREADS ? WARP_THREADS : LEGACY_MAX_THREADS)
      : min((columns + WARP_THREADS - 1) / WARP_THREADS * WARP_THREADS, LEGACY_MAX_THREADS);
  const uint64_t base = uint64_t(blockIdx.x) * columns;
  const auto* x_row = reinterpret_cast<const Pack*>(x) + base;
  const auto* residual_row = reinterpret_cast<const Pack*>(residual) + base;
  const auto* weights = reinterpret_cast<const Pack*>(weight);
  auto* out_row = reinterpret_cast<Pack*>(residual_output) + base;
  __shared__ float warp_sums[LEGACY_MAX_THREADS / WARP_THREADS];

  // The typed ABI fixes the physical block size, so a warp can cover multiple legacy warps.
  for (uint32_t lane = threadIdx.x; lane < lanes; lane += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t col = lane; col < columns; col += lanes) {
      const Pack input = x_row[col];
      if constexpr (ITEMS == 1) {
        const float value = as_float(input.values[0]);
        sum += value * value;
      } else {
        float fragment_sum = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
          const float value = as_float(input.values[i]);
          fragment_sum += value * value;
        }
        sum += fragment_sum;
      }
    }
    store_warp_sum(sum, lane, warp_sums);
  }
  const float inv = rsqrtf(row_sum(lanes, warp_sums) / float(width) + eps1);
  const float multiplier = SCALED ? as_float(scale[0]) : 1.0f;

  for (uint32_t lane = threadIdx.x; lane < lanes; lane += blockDim.x) {
    float residual_sum = 0.0f;
    for (uint32_t col = lane; col < columns; col += lanes) {
      const Pack input = x_row[col];
      const Pack skip = residual_row[col];
      const Pack weight_values = weights[col];
      Pack out;
#pragma unroll
      for (uint32_t i = 0; i < ITEMS; ++i) {
        const float norm = as_float(input.values[i]) * inv * as_float(weight_values.values[i]);
        const float value = (as_float(skip.values[i]) + norm) * multiplier;
        out.values[i] = as_dtype<T>(value);
        if constexpr (SECOND_NORM) residual_sum += value * value;
      }
      out_row[col] = out;
    }
    if constexpr (SECOND_NORM) store_warp_sum(residual_sum, lane, warp_sums);
  }

  if constexpr (SECOND_NORM) {
    const float norm_inv = rsqrtf(row_sum(lanes, warp_sums) / float(width) + eps2);
    const auto* norm_weights = reinterpret_cast<const Pack*>(norm_weight);
    auto* norm_row = reinterpret_cast<Pack*>(normalized_output) + base;
    for (uint32_t col = threadIdx.x; col < columns; col += blockDim.x) {
      const Pack residual_values = out_row[col];
      const Pack weight_values = norm_weights[col];
      Pack out;
#pragma unroll
      for (uint32_t i = 0; i < ITEMS; ++i) {
        out.values[i] = as_dtype<T>(
            as_float(residual_values.values[i]) * norm_inv * as_float(weight_values.values[i]));
      }
      norm_row[col] = out;
    }
  }
}

template <typename T, bool SCALED, bool SECOND_NORM>
__device__ void residual_body(
    const T* x, const T* residual, const T* weight, const T* scale,
    const T* norm_weight, T* residual_output, T* normalized_output,
    uint32_t rows, uint32_t width, float eps1, float eps2) {
  if (blockIdx.x >= rows) return;
  if constexpr (sizeof(T) == 2) {
    if (width % VECTOR_ITEMS == 0 && vector_aligned(x) && vector_aligned(residual)
        && vector_aligned(weight) && vector_aligned(residual_output)
        && (!SECOND_NORM || (vector_aligned(norm_weight) && vector_aligned(normalized_output)))) {
      residual_fragments<T, VECTOR_ITEMS, SCALED, SECOND_NORM>(
          x, residual, weight, scale, norm_weight, residual_output, normalized_output,
          width, eps1, eps2);
      return;
    }
  }
  residual_fragments<T, 1, SCALED, SECOND_NORM>(
      x, residual, weight, scale, norm_weight, residual_output, normalized_output,
      width, eps1, eps2);
}

#define DKR_RESIDUAL(NAME, TYPE) \
extern "C" __global__ void dkr_rms_residual_##NAME##_plain( \
    const TYPE* x, const TYPE* residual, const TYPE* weight, TYPE* output, \
    uint32_t rows, uint32_t width, float eps) { \
  residual_body<TYPE, false, false>(x, residual, weight, nullptr, nullptr, output, \
                                    nullptr, rows, width, eps, 0.0f); \
} \
extern "C" __global__ void dkr_rms_residual_##NAME##_scaled( \
    const TYPE* x, const TYPE* residual, const TYPE* weight, const TYPE* scale, \
    TYPE* output, uint32_t rows, uint32_t width, float eps) { \
  residual_body<TYPE, true, false>(x, residual, weight, scale, nullptr, output, \
                                   nullptr, rows, width, eps, 0.0f); \
} \
extern "C" __global__ void dkr_rms_then_##NAME##_plain( \
    const TYPE* x, const TYPE* residual, const TYPE* rw, const TYPE* nw, \
    TYPE* residual_output, TYPE* normalized_output, \
    uint32_t rows, uint32_t width, float eps1, float eps2) { \
  residual_body<TYPE, false, true>(x, residual, rw, nullptr, nw, residual_output, \
                                   normalized_output, rows, width, eps1, eps2); \
} \
extern "C" __global__ void dkr_rms_then_##NAME##_scaled( \
    const TYPE* x, const TYPE* residual, const TYPE* rw, const TYPE* scale, \
    const TYPE* nw, TYPE* residual_output, TYPE* normalized_output, \
    uint32_t rows, uint32_t width, float eps1, float eps2) { \
  residual_body<TYPE, true, true>(x, residual, rw, scale, nw, residual_output, \
                                  normalized_output, rows, width, eps1, eps2); \
}

DKR_RESIDUAL(f32, float)
DKR_RESIDUAL(f16, __half)
DKR_RESIDUAL(bf16, __nv_bfloat16)
