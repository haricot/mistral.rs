#pragma once
#ifdef DKR_LFM3_IMAGE
#define DKR_LFM3_ENTRY __device__ __forceinline__
#else
#define DKR_LFM3_ENTRY __global__
#endif
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int PARAM_COUNT = 15;
constexpr int THREADS = 256;
constexpr int OPTIMIZED_WIDTH = 7;

struct ConvParams {
  uint64_t values[PARAM_COUNT];
};

template <typename T>
__device__ __forceinline__ T gated_input(const T *__restrict__ x,
                                         const ConvParams &params,
                                         uint64_t batch, uint64_t token,
                                         uint64_t channel) {
  const auto *p = params.values;
  const uint64_t offset = batch * p[4] + token * p[5] + channel * p[6];
  return T(float(x[offset]) * float(x[offset + 2 * p[2] * p[6]]));
}

template <typename T, int WIDTH>
DKR_LFM3_ENTRY void gated_conv(const T *__restrict__ x,
                           const T *__restrict__ weight,
                           const T *__restrict__ bias,
                           const T *__restrict__ state, T *__restrict__ output,
                           T *__restrict__ next_state, ConvParams params) {
  const auto *p = params.values;
  const uint64_t seq = p[1], hidden = p[2], width = WIDTH ? WIDTH : p[3];
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= p[0] * seq * hidden) {
    return;
  }
  const uint64_t channel = index % hidden;
  const uint64_t token = index / hidden % seq;
  const uint64_t batch = index / (hidden * seq);
  const uint64_t state_base = batch * p[7] + channel * p[8];
  float sum = 0.0f;
#pragma unroll
  for (int tap = 0; tap < width; tap++) {
    T value = T(0.0f);
    if (token + tap + 1 >= width) {
      value = gated_input(x, params, batch, token + tap + 1 - width, channel);
    } else if (p[13]) {
      value = state[state_base + (token + tap + 1) * p[9]];
    }
    sum += float(value) * float(weight[channel * p[10] + tap * p[11]]);
  }
  if (p[14]) {
    sum += float(bias[channel * p[12]]);
  }
  const T convolved = T(sum);
  const uint64_t gate = batch * p[4] + token * p[5] + (hidden + channel) * p[6];
  output[index] = T(float(convolved) * float(x[gate]));
  if (token + 1 == seq) {
#pragma unroll
    for (int tap = 0; tap < width; tap++) {
      T value = T(0.0f);
      if (seq + tap >= width) {
        value = gated_input(x, params, batch, seq + tap - width, channel);
      } else if (p[13]) {
        value = state[state_base + (seq + tap) * p[9]];
      }
      next_state[(batch * hidden + channel) * width + tap] = value;
    }
  }
}

} // namespace

#undef DKR_LFM3_ENTRY
