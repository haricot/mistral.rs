#include "dynamic_conv_device.cuh"

template <typename T, int KernelSize>
__global__ void dynamic_conv_kernel(const T *hidden, const T *dynamic, const T *base, T *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<T, KernelSize>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

template <typename T>
int launch_dynamic_conv(const void *hidden, const void *dynamic,
                        const void *base, void *output, int batch,
                        int sequence_length, int hidden_size, int group_size,
                        int kernel_size, int64_t hidden_stride_b,
                        int64_t hidden_stride_s, int64_t hidden_stride_h,
                        int64_t dynamic_stride_b, int64_t dynamic_stride_s,
                        int64_t dynamic_stride_k, int64_t dynamic_stride_g,
                        int64_t base_stride_k, int64_t base_stride_h,
                        int64_t stream) {
  if (batch <= 0 || sequence_length <= 0 || hidden_size <= 0 ||
      group_size <= 0 || hidden_size % group_size != 0 || kernel_size <= 0 ||
      kernel_size > kMaxDynamicConvKernelSize) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr int threads = 256;
  const int64_t elements = static_cast<int64_t>(batch) * sequence_length *
                           hidden_size;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (kernel_size == 2) {
    dynamic_conv_kernel<T, 2><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const T *>(hidden), static_cast<const T *>(dynamic),
        static_cast<const T *>(base), static_cast<T *>(output), batch,
        sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b,
        hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s,
        dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
  } else {
    dynamic_conv_kernel<T, 0><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<const T *>(hidden), static_cast<const T *>(dynamic),
        static_cast<const T *>(base), static_cast<T *>(output), batch,
        sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b,
        hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s,
        dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
  }
  return static_cast<int>(cudaGetLastError());
}

extern "C" int dynamic_conv_bf16(const void *hidden, const void *dynamic,
                                  const void *base, void *output, int batch,
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
                                  int64_t base_stride_h,
                                  int64_t stream) {
  return launch_dynamic_conv<__nv_bfloat16>(
      hidden, dynamic, base, output, batch, sequence_length, hidden_size,
      group_size, kernel_size, hidden_stride_b, hidden_stride_s,
      hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k,
      dynamic_stride_g, base_stride_k, base_stride_h, stream);
}

extern "C" int dynamic_conv_f16(const void *hidden, const void *dynamic,
                                 const void *base, void *output, int batch,
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
                                 int64_t base_stride_h,
                                 int64_t stream) {
  return launch_dynamic_conv<__half>(hidden, dynamic, base, output, batch,
                                     sequence_length, hidden_size, group_size,
                                     kernel_size, hidden_stride_b,
                                     hidden_stride_s, hidden_stride_h,
                                     dynamic_stride_b, dynamic_stride_s,
                                     dynamic_stride_k, dynamic_stride_g,
                                     base_stride_k, base_stride_h, stream);
}

extern "C" int dynamic_conv_f32(const void *hidden, const void *dynamic,
                                 const void *base, void *output, int batch,
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
                                 int64_t base_stride_h,
                                 int64_t stream) {
  return launch_dynamic_conv<float>(hidden, dynamic, base, output, batch,
                                    sequence_length, hidden_size, group_size,
                                    kernel_size, hidden_stride_b,
                                    hidden_stride_s, hidden_stride_h,
                                    dynamic_stride_b, dynamic_stride_s,
                                    dynamic_stride_k, dynamic_stride_g,
                                    base_stride_k, base_stride_h, stream);
}
