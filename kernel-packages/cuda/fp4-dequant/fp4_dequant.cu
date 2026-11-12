#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdint.h>

// Exact low-nibble-first E2M1 bit decoding (same as fp_codecs.cuh).
__device__ __forceinline__ float dkr_e2m1_to_f32(uint8_t code) {
  const uint32_t e = (code >> 1) & 3u;
  const uint32_t m = code & 1u;
  const uint32_t magnitude =
      e ? ((e + 126u) << 23) | (m << 22) : m * 0x3f000000u;
  return __uint_as_float(magnitude | (static_cast<uint32_t>(code & 8u) << 28));
}

__device__ __forceinline__ float dkr_e4m3_scale(uint8_t bits) {
  return __half2float(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3));
}

__device__ __forceinline__ float dkr_e8m0_scale(uint8_t bits) {
  // Match the existing mxfp4_gemm.cu e8m0_to_float convention.
  return __uint_as_float(static_cast<uint32_t>(bits) << 23);
}

// NVFP4 E2M1 packed bytes, E4M3 group16 scales, F32 per-row global.
extern "C" __global__ void dkr_nvfp4_dequant_f32(
    const uint8_t* packed, const __nv_fp8_e4m3* scales,
    const float* global, float* output, uint32_t rows, uint32_t k) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = static_cast<uint64_t>(rows) * k;
  if (index >= elements) return;
  const uint32_t row = static_cast<uint32_t>(index / k);
  const uint32_t column = static_cast<uint32_t>(index % k);
  const uint8_t byte = packed[static_cast<uint64_t>(row) * (k / 2) + column / 2];
  const uint8_t code = column & 1u ? byte >> 4 : byte & 15u;
  const uint8_t scale = scales[
      static_cast<uint64_t>(row) * (k / 16) + column / 16].__x;
  const float value = dkr_e2m1_to_f32(code) * dkr_e4m3_scale(scale);
  output[index] = value * global[row];
}

// MXFP4 E2M1 packed bytes with OCP E8M0 group32 exponent scales.
// No global-scale argument: this is a different stable typed ABI.
extern "C" __global__ void dkr_mxfp4_dequant_f32(
    const uint8_t* packed, const uint8_t* scales,
    float* output, uint32_t rows, uint32_t k) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t elements = static_cast<uint64_t>(rows) * k;
  if (index >= elements) return;
  const uint32_t row = static_cast<uint32_t>(index / k);
  const uint32_t column = static_cast<uint32_t>(index % k);
  const uint8_t byte = packed[static_cast<uint64_t>(row) * (k / 2) + column / 2];
  const uint8_t code = column & 1u ? byte >> 4 : byte & 15u;
  const uint8_t scale = scales[
      static_cast<uint64_t>(row) * (k / 32) + column / 32];
  output[index] = dkr_e2m1_to_f32(code) * dkr_e8m0_scale(scale);
}
