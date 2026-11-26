#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Exact comparisons in the native domain. Widening a 16-bit float to F32
// is exact, unlike narrowing F64 or I64 to F32.
template<class T> __device__ __forceinline__ auto argsort_value(T x) { return x; }
__device__ __forceinline__ float argsort_value(__half x) { return __half2float(x); }
__device__ __forceinline__ float argsort_value(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ void argsort_init(uint32_t *work, uint32_t padded) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < padded) work[i] = i;
}

// The original input is immutable. Indirect bitonic swaps preserve the
// legacy valid/valid comparisons and stage sequence. Padding is ordered
// after every valid index in the final direction, independently of value.
template<class T, bool ASC>
__device__ __forceinline__ void argsort_compare(const T *input, uint32_t *work,
    uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= padded) return;
  const uint32_t other = i ^ j;
  if (other <= i || other >= padded) return;
  const uint32_t a = work[i], b = work[other];
  const bool a_pad = a >= cols, b_pad = b >= cols;
  const bool up = ASC ? ((i & k) == 0) : ((i & k) != 0);
  bool exchange = false;
  if (a_pad != b_pad) {
    exchange = (up == ASC) ? a_pad : b_pad;
  } else if (!a_pad) {
    const auto av = argsort_value(input[a]), bv = argsort_value(input[b]);
    exchange = up ? av > bv : av < bv;
  }
  if (exchange) { work[i] = b; work[other] = a; }
}

__device__ __forceinline__ void argsort_finish(const uint32_t *work, uint32_t *indices, uint32_t cols) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cols) indices[i] = work[i];
}

// Only called with the permutation produced by this adapter. Guard the
// load even if a corrupted index reaches the kernel.
template<class T>
__device__ __forceinline__ void argsort_gather(const T *input, const uint32_t *indices,
    T *output, uint32_t cols, uint32_t elements) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  const uint32_t id = indices[i];
  const size_t base = static_cast<size_t>(i / cols) * cols;
  output[i] = input[base + (id < cols ? id : 0)];
}
