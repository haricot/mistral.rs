#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// DKR-R1.21/S8 — per-route quantized MoE projections on SM61.
// GEMV: 128 threads per (assignment, output column); T<=16.
// GEMM: 16x16 output tile; T>=17. No route sorting or FP dequant buffer.
// This does NOT replace fused GLU, topk reduction or grouped sorted GEMM.
// Entrypoints have no unused optional weight/scale/global parameters.

namespace dkr_routed {
constexpr uint32_t kBlock = 128;

template <typename T> __device__ __forceinline__ float to_f32(T v) {
  return static_cast<float>(v);
}
template <> __device__ __forceinline__ float to_f32<__half>(__half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T> __device__ __forceinline__ T from_f32(float v) {
  return static_cast<T>(v);
}
template <> __device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half_rn(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
__device__ __forceinline__ float fp8(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}
__device__ __forceinline__ float e2m1(uint8_t code) {
  const uint32_t exponent = (code >> 1u) & 3u;
  const uint32_t mantissa = code & 1u;
  const uint32_t bits = exponent
      ? ((exponent + 126u) << 23) | (mantissa << 22)
      : mantissa * 0x3f000000u;
  return __uint_as_float(bits | (static_cast<uint32_t>(code & 8u) << 28));
}
__device__ __forceinline__ float e8m0(uint8_t exponent) {
  // OCP semantics, including E8M0 subnormal exponent 0 and NaN 255.
  if (exponent == 255) return __uint_as_float(0x7fc00000u);
  return exponent == 0
      ? __uint_as_float(0x00400000u)
      : __uint_as_float(static_cast<uint32_t>(exponent) << 23);
}
enum Format : int { FP8 = 0, MXFP8 = 1, MXFP4 = 2, NVFP4 = 3 };

template <Format F>
__device__ __forceinline__ float weight_value(
    const uint8_t* weights, const void* scales, const float* global,
    uint32_t expert, uint32_t col, uint32_t idx,
    uint32_t n, uint32_t k,
    uint32_t block_y, uint32_t block_x, uint32_t scale_cols) {
  const uint64_t row = static_cast<uint64_t>(expert) * n + col;
  if constexpr (F == FP8) {
    const uint64_t scale_rows = (n + block_y - 1u) / block_y;
    const uint64_t scale_idx =
        (static_cast<uint64_t>(expert) * scale_rows + col / block_y)
        * scale_cols + idx / block_x;
    return fp8(weights[row * k + idx])
        * static_cast<const float*>(scales)[scale_idx];
  } else if constexpr (F == MXFP8) {
    const uint64_t offset = row * k + idx;
    return fp8(weights[offset])
        * e8m0(static_cast<const uint8_t*>(scales)[row * (k / 32u) + idx / 32u]);
  } else {
    const uint8_t pair = weights[row * (k / 2u) + idx / 2u];
    const uint8_t code = (idx & 1u) ? pair >> 4u : pair & 15u;
    if constexpr (F == MXFP4) {
      return e2m1(code)
          * e8m0(static_cast<const uint8_t*>(scales)[row * (k / 32u) + idx / 32u]);
    } else {
      return e2m1(code)
          * fp8(static_cast<const uint8_t*>(scales)[row * (k / 16u) + idx / 16u])
          * global[row];
    }
  }
}

template <Format F, typename T>
__device__ __forceinline__ void gemv(
    const T* input, const uint8_t* weights, const void* scales,
    const float* global, const uint32_t* ids, T* output,
    uint32_t tokens, uint32_t topk, uint32_t n, uint32_t k,
    uint32_t experts, uint32_t input_routes,
    uint32_t block_y, uint32_t block_x, uint32_t scale_cols) {
  const uint32_t col = blockIdx.x;
  const uint32_t route = blockIdx.y;
  const uint64_t routes = static_cast<uint64_t>(tokens) * topk;
  if (col >= n || route >= routes) return;
  const uint32_t expert = ids[route];
  const uint32_t token = route / topk;
  const uint32_t inner = input_routes == 1 ? 0 : route % topk;
  const uint64_t in_base =
      (static_cast<uint64_t>(token) * input_routes + inner) * k;
  float sum = 0.0f;
  if (expert < experts) {
    for (uint32_t idx = threadIdx.x; idx < k; idx += blockDim.x) {
      sum = fmaf(to_f32(input[in_base + idx]),
          weight_value<F>(weights, scales, global, expert, col, idx, n, k,
                          block_y, block_x, scale_cols), sum);
    }
  }
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  __shared__ float warps[4];
  if (lane == 0) warps[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    output[static_cast<uint64_t>(route) * n + col] =
        from_f32<T>(warps[0] + warps[1] + warps[2] + warps[3]);
  }
}

template <Format F, typename T>
__device__ __forceinline__ void gemm(
    const T* input, const uint8_t* weights, const void* scales,
    const float* global, const uint32_t* ids, T* output,
    uint32_t tokens, uint32_t topk, uint32_t n, uint32_t k,
    uint32_t experts, uint32_t input_routes,
    uint32_t block_y, uint32_t block_x, uint32_t scale_cols) {
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t route = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= n || route >= static_cast<uint64_t>(tokens) * topk) return;
  const uint32_t expert = ids[route];
  const uint32_t token = route / topk;
  const uint32_t inner = input_routes == 1 ? 0 : route % topk;
  const uint64_t in_base =
      (static_cast<uint64_t>(token) * input_routes + inner) * k;
  float sum = 0.0f;
  if (expert < experts) {
    for (uint32_t idx = 0; idx < k; ++idx) {
      sum = fmaf(to_f32(input[in_base + idx]),
          weight_value<F>(weights, scales, global, expert, col, idx, n, k,
                          block_y, block_x, scale_cols), sum);
    }
  }
  output[static_cast<uint64_t>(route) * n + col] = from_f32<T>(sum);
}

#define DKR_COMMON_PARAMS(TYPE) \
    const TYPE* input, const uint8_t* weights, const void* scales, \
    const uint32_t* indices, TYPE* output, \
    uint32_t tokens, uint32_t topk, uint32_t n, uint32_t k, \
    uint32_t experts, uint32_t input_routes
#define DKR_NV_PARAMS(TYPE) \
    const TYPE* input, const uint8_t* weights, const uint8_t* scales, \
    const float* globals, const uint32_t* indices, TYPE* output, \
    uint32_t tokens, uint32_t topk, uint32_t n, uint32_t k, \
    uint32_t experts, uint32_t input_routes
#define DKR_FP8_PARAMS(TYPE) \
    DKR_COMMON_PARAMS(TYPE), \
    uint32_t block_y, uint32_t block_x, uint32_t scale_cols

#define DKR_PAIR_FP8(TYPE, DT) \
extern "C" __global__ void dkr_routed_gemv_fp8_##DT(DKR_FP8_PARAMS(TYPE)) { \
  gemv<FP8, TYPE>(input,weights,scales,nullptr,indices,output, \
      tokens,topk,n,k,experts,input_routes,block_y,block_x,scale_cols); \
} \
extern "C" __global__ void dkr_routed_gemm_fp8_##DT(DKR_FP8_PARAMS(TYPE)) { \
  gemm<FP8, TYPE>(input,weights,scales,nullptr,indices,output, \
      tokens,topk,n,k,experts,input_routes,block_y,block_x,scale_cols); \
}
#define DKR_PAIR_MX(TYPE, DT, FORMAT, LABEL) \
extern "C" __global__ void dkr_routed_gemv_##LABEL##_##DT(DKR_COMMON_PARAMS(TYPE)) { \
  gemv<FORMAT, TYPE>(input,weights,scales,nullptr,indices,output, \
      tokens,topk,n,k,experts,input_routes,1,32,1); \
} \
extern "C" __global__ void dkr_routed_gemm_##LABEL##_##DT(DKR_COMMON_PARAMS(TYPE)) { \
  gemm<FORMAT, TYPE>(input,weights,scales,nullptr,indices,output, \
      tokens,topk,n,k,experts,input_routes,1,32,1); \
}
#define DKR_PAIR_NV(TYPE, DT) \
extern "C" __global__ void dkr_routed_gemv_nvfp4_##DT(DKR_NV_PARAMS(TYPE)) { \
  gemv<NVFP4, TYPE>(input,weights,scales,globals,indices,output, \
      tokens,topk,n,k,experts,input_routes,1,16,1); \
} \
extern "C" __global__ void dkr_routed_gemm_nvfp4_##DT(DKR_NV_PARAMS(TYPE)) { \
  gemm<NVFP4, TYPE>(input,weights,scales,globals,indices,output, \
      tokens,topk,n,k,experts,input_routes,1,16,1); \
}

DKR_PAIR_FP8(__half, f16)
DKR_PAIR_FP8(__nv_bfloat16, bf16)
DKR_PAIR_FP8(float, f32)
DKR_PAIR_MX(__half, f16, MXFP8, mxfp8)
DKR_PAIR_MX(__nv_bfloat16, bf16, MXFP8, mxfp8)
DKR_PAIR_MX(float, f32, MXFP8, mxfp8)
DKR_PAIR_MX(__half, f16, MXFP4, mxfp4)
DKR_PAIR_MX(__nv_bfloat16, bf16, MXFP4, mxfp4)
DKR_PAIR_MX(float, f32, MXFP4, mxfp4)
DKR_PAIR_NV(__half, f16)
DKR_PAIR_NV(__nv_bfloat16, bf16)
DKR_PAIR_NV(float, f32)
} // namespace dkr_routed
