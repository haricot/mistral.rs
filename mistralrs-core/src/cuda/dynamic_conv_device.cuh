#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int kMaxDynamicConvKernelSize = 8;

template <typename T> __device__ __forceinline__ float dynamic_conv_to_float(T value) {
  return static_cast<float>(value);
}

template <typename T> __device__ __forceinline__ T dynamic_conv_from_float(float value) {
  return static_cast<T>(value);
}

template <typename T, int KernelSize>
__device__ __forceinline__ void dynamic_conv_body(const T *hidden, const T *dynamic,
                                    const T *base, T *output, int batch,
                                    int sequence_length, int hidden_size,
                                    int group_size, int kernel_size,
                                    int64_t hidden_stride_b,
                                    int64_t hidden_stride_s,
                                    int64_t hidden_stride_h,
                                    int64_t dynamic_stride_b,
                                    int64_t dynamic_stride_s,
                                    int64_t dynamic_stride_k,
                                    int64_t dynamic_stride_g,
                                    int64_t base_stride_k,
                                    int64_t base_stride_h) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(batch) * sequence_length *
                           hidden_size;
  if (index >= elements) {
    return;
  }

  const int channel = static_cast<int>(index % hidden_size);
  const int64_t row = index / hidden_size;
  const int position = static_cast<int>(row % sequence_length);
  const int batch_index = static_cast<int>(row / sequence_length);
  const int group = channel / group_size;
  const int taps = KernelSize == 0 ? kernel_size : KernelSize;

  float result = 0.0f;
#pragma unroll
  for (int offset = 0;
       offset < (KernelSize == 0 ? kMaxDynamicConvKernelSize : KernelSize);
       ++offset) {
    if (offset >= taps || offset > position) {
      continue;
    }
    const int64_t hidden_index = static_cast<int64_t>(batch_index) *
                                     hidden_stride_b +
                                 static_cast<int64_t>(position - offset) *
                                     hidden_stride_s +
                                 static_cast<int64_t>(channel) * hidden_stride_h;
    const int64_t dynamic_index =
        static_cast<int64_t>(batch_index) * dynamic_stride_b +
        static_cast<int64_t>(position) * dynamic_stride_s +
        static_cast<int64_t>(offset) * dynamic_stride_k +
        static_cast<int64_t>(group) * dynamic_stride_g;
    const int64_t base_index =
        static_cast<int64_t>(offset) * base_stride_k +
        static_cast<int64_t>(channel) * base_stride_h;
    const float value = dynamic_conv_to_float(hidden[hidden_index]);
    const float weight = dynamic_conv_to_float(base[base_index]) +
                         dynamic_conv_to_float(dynamic[dynamic_index]);
    result += value * weight;
  }
  output[index] = dynamic_conv_from_float<T>(result);
}

