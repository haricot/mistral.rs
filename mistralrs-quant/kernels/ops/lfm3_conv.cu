#include "lfm3_conv_device.cuh"

namespace {
template <typename T>
int launch(const void *x, const void *weight, const void *bias,
           const void *state, void *output, void *next_state,
           const uint64_t *params, cudaStream_t stream) {
  ConvParams p;
  for (int i = 0; i < PARAM_COUNT; i++) {
    p.values[i] = params[i];
  }
  const uint64_t count = params[0] * params[1] * params[2];
  const uint32_t blocks = (count + THREADS - 1) / THREADS;
  if (params[3] == OPTIMIZED_WIDTH) {
    gated_conv<T, OPTIMIZED_WIDTH><<<blocks, THREADS, 0, stream>>>(
        static_cast<const T *>(x), static_cast<const T *>(weight),
        static_cast<const T *>(bias), static_cast<const T *>(state),
        static_cast<T *>(output), static_cast<T *>(next_state), p);
  } else {
    gated_conv<T, 0><<<blocks, THREADS, 0, stream>>>(
        static_cast<const T *>(x), static_cast<const T *>(weight),
        static_cast<const T *>(bias), static_cast<const T *>(state),
        static_cast<T *>(output), static_cast<T *>(next_state), p);
  }
  return int(cudaGetLastError());
}

} // namespace

#define LFM3_CONV_EXPORT(name, type)                                           \
  extern "C" int name(const void *x, const void *weight, const void *bias,     \
                      const void *state, void *output, void *next_state,       \
                      const uint64_t *params, cudaStream_t stream) {           \
    return launch<type>(x, weight, bias, state, output, next_state, params,    \
                        stream);                                               \
  }

LFM3_CONV_EXPORT(lfm3_conv_f32, float)
LFM3_CONV_EXPORT(lfm3_conv_f16, __half)
LFM3_CONV_EXPORT(lfm3_conv_bf16, __nv_bfloat16)
