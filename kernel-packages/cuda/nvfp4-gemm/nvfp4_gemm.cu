#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// R1.26/S2 reference dense W4A16 prefill. Separate module from qualified
// GEMV and FP8/MXFP4 images. SIMT tiled reuse, F32 accumulation, no
// global dequantized weight buffer. Not a performance-qualified kernel.
namespace {
constexpr unsigned TILE = 16;
template<typename T> __device__ __forceinline__ float f32(T x);
template<> __device__ __forceinline__ float f32(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float f32(__nv_bfloat16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T cast(float x);
template<> __device__ __forceinline__ __half cast(float x) { return __float2half_rn(x); }
template<> __device__ __forceinline__ __nv_bfloat16 cast(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float e2m1(uint8_t code) {
  const unsigned e = (code >> 1) & 3u;
  const unsigned m = code & 1u;
  const float v = e == 0 ? (m ? 0.5f : 0.0f) : (1u << (e-1)) * (1.0f + 0.5f*m);
  return code & 8u ? -v : v;
}

template<typename T, bool Bias>
__device__ __forceinline__ void gemm(
    const T* x, const uint8_t* w, const __nv_fp8_e4m3* s,
    const float* g, const T* bias, T* y, uint32_t m, uint32_t n, uint32_t k) {
  __shared__ float a[TILE][TILE];
  __shared__ float b[TILE][TILE+1];
  const unsigned tx = threadIdx.x, ty = threadIdx.y;
  const uint64_t row = uint64_t(blockIdx.y)*TILE + ty;
  const uint64_t col = uint64_t(blockIdx.x)*TILE + tx;
  const uint64_t load_col = uint64_t(blockIdx.x)*TILE + ty;
  float acc = 0.f;
  // K is a multiple of 16 by the typed adapter's group16 contract.
  for (uint64_t base = 0; base < k; base += TILE) {
    const uint64_t j = base + tx;
    a[ty][tx] = row < m && j < k ? f32(x[row*k+j]) : 0.f;
    float v = 0.f;
    if (load_col < n && j < k) {
      const uint8_t pair = w[load_col*(k/2)+j/2];
      const uint8_t code = j & 1 ? pair >> 4 : pair & 15;
      const float scale = __half2float(__nv_cvt_fp8_to_halfraw(
          s[load_col*(k/16)+j/16].__x, __NV_E4M3));
      v = e2m1(code)*scale;
    }
    b[tx][ty] = v;
    __syncthreads();
    #pragma unroll
    for (unsigned q=0; q<TILE; ++q) acc = fmaf(a[ty][q], b[q][tx], acc);
    __syncthreads();
  }
  if (row < m && col < n) {
    float out = acc*g[col];
    if constexpr (Bias) out += f32(bias[col]);
    y[row*n+col] = cast<T>(out);
  }
}
}

#define DKR_PLAIN(NAME,T) \
extern "C" __global__ void dkr_nvfp4_gemm_##NAME##_plain( \
    const T* x, const uint8_t* w, const __nv_fp8_e4m3* s, \
    const float* g, T* y, uint32_t m, uint32_t n, uint32_t k) { \
  gemm<T,false>(x,w,s,g,nullptr,y,m,n,k); \
}
#define DKR_BIAS(NAME,T) \
extern "C" __global__ void dkr_nvfp4_gemm_##NAME##_bias( \
    const T* x, const uint8_t* w, const __nv_fp8_e4m3* s, \
    const float* g, const T* bias, T* y, uint32_t m, uint32_t n, uint32_t k) { \
  gemm<T,true>(x,w,s,g,bias,y,m,n,k); \
}
DKR_PLAIN(f16,__half)
DKR_BIAS(f16,__half)
DKR_PLAIN(bf16,__nv_bfloat16)
DKR_BIAS(bf16,__nv_bfloat16)
