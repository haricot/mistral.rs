#!/usr/bin/env python3
"""Preserve FlashInfer SM75+ PTX while providing an SM61-compatible math path.

PTX tanh.approx (all widths) requires SM75, and half/half2 ex2.approx forms
also require newer instruction support. CUDA's float math intrinsics provide
equivalent mathematical operations on Pascal. Keep native PTX unmodified for
newer devices, and fail closed if the upstream header changes.
"""
from pathlib import Path

path = Path("mistralrs-paged-attn/src/cuda/flashinfer/math.cuh")
src = path.read_text()

# A previously promoted Pascal candidate already contains all five guards.
# Do not try to match the obsolete unguarded PTX bodies a second time.
# Require the complete known SM61 fallback contract before accepting a no-op.
pascal_guard = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)"
guard_count = src.count(pascal_guard)
if guard_count:
    required = {
        "half2 exp2": "return __halves2half2(__float2half_rn(::exp2f(__low2float(x))),",
        "half exp2": "return __float2half_rn(::exp2f(__half2float(x)));",
        "float tanh": "return ::tanhf(x);",
        "half2 tanh": "return __halves2half2(__float2half_rn(::tanhf(__low2float(x))),",
        "half tanh": "return __float2half_rn(::tanhf(__half2float(x)));",
    }
    occurrences = {name: src.count(needle) for name, needle in required.items()}
    # Ensure PTX native paths were preserved, not replaced by Pascal math.
    high_arch_ptx = [
        'asm volatile("ex2.approx.f16x2',
        'asm volatile("ex2.approx.f16 ',
        'asm volatile("tanh.approx.f32 ',
        'asm volatile("tanh.approx.f16x2 ',
        'asm volatile("tanh.approx.f16 ',
    ]
    if guard_count != 5 or any(n != 1 for n in occurrences.values()) or any(
        src.count(needle) != 1 for needle in high_arch_ptx
    ):
        raise RuntimeError(
            "FlashInfer SM61 pre-existing math port is partial or duplicated; "
            f"guards={guard_count}, variants={occurrences}; refusing replay"
        )
    print("PORT_FLASHINFER_SM61_MATH_ALREADY_VALIDATED: 5 Pascal branches; native PTX retained")
    raise SystemExit(0)

def replace_once(before: str, after: str, what: str) -> None:
    global src
    n = src.count(before)
    if n != 1:
        raise RuntimeError(f"FlashInfer SM61 {what}: expected one anchor, got {n}")
    src = src.replace(before, after, 1)


old_exp2_half2 = '''__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}'''
new_exp2_half2 = '''__forceinline__ __device__ half2 ptx_exp2(half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  // Pascal: calculate in f32 and round each lane once to half.
  return __halves2half2(__float2half_rn(::exp2f(__low2float(x))),
                        __float2half_rn(::exp2f(__high2float(x))));
#else
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
#endif
}'''
replace_once(old_exp2_half2, new_exp2_half2, "half2 exp2")

old_exp2_half = '''__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}'''
new_exp2_half = '''__forceinline__ __device__ half ptx_exp2(half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  return __float2half_rn(::exp2f(__half2float(x)));
#else
  ushort y_u16;
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
#endif
}'''
replace_once(old_exp2_half, new_exp2_half, "half exp2")

old_tanh_float = '''__forceinline__ __device__ float tanh(float x) {
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}'''
new_tanh_float = '''__forceinline__ __device__ float tanh(float x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  // libdevice emits Pascal-compatible math, including proper saturation.
  return ::tanhf(x);
#else
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#endif
}'''
replace_once(old_tanh_float, new_tanh_float, "float tanh")

old_tanh_half2 = '''__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
}'''
new_tanh_half2 = '''__forceinline__ __device__ half2 tanh(half2 x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  return __halves2half2(__float2half_rn(::tanhf(__low2float(x))),
                        __float2half_rn(::tanhf(__high2float(x))));
#else
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
#endif
}'''
replace_once(old_tanh_half2, new_tanh_half2, "half2 tanh")

old_tanh_half = '''__forceinline__ __device__ half tanh(half x) {
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
}'''
new_tanh_half = '''__forceinline__ __device__ half tanh(half x) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
  return __float2half_rn(::tanhf(__half2float(x)));
#else
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
#endif
}'''
replace_once(old_tanh_half, new_tanh_half, "half tanh")

path.write_text(src)
print("PORT_FLASHINFER_SM61_MATH_R1_READY: guarded 3 tanh and 2 FP16 exp2 PTX variants")
