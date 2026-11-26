#include "lfm3_spatial_conv_device.cuh"

constexpr uint32_t kSpatialConvThreads = 256;

template <typename T>
int launch_lfm3_spatial_conv(const void *p, const void *w, void *output,
                             size_t elements, uint32_t height, uint32_t width,
                             uint32_t merge, uint32_t channels, uint32_t kernel,
                             void *stream) {
  lfm3_spatial_conv_kernel<T>
      <<<(elements + kSpatialConvThreads - 1) / kSpatialConvThreads,
         kSpatialConvThreads, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<const T *>(p), static_cast<const T *>(w),
          static_cast<T *>(output), elements, height, width, merge, channels,
          kernel);
  return int(cudaGetLastError());
}

#define WRAPPER(T, NAME)                                                       \
  extern "C" int lfm3_spatial_conv_##NAME(                                     \
      const void *p, const void *w, void *output, size_t elements,             \
      uint32_t height, uint32_t width, uint32_t merge, uint32_t channels,      \
      uint32_t kernel, void *stream) {                                         \
    return launch_lfm3_spatial_conv<T>(p, w, output, elements, height, width,  \
                                       merge, channels, kernel, stream);       \
  }
WRAPPER(float, f32)
WRAPPER(__half, f16)
WRAPPER(__nv_bfloat16, bf16)
