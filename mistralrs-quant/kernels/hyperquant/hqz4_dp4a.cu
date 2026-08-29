#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint64_t SPLITMIX_INCREMENT = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t SPLITMIX_MULTIPLIER_1 = 0xbf58476d1ce4e5b9ULL;
constexpr uint64_t SPLITMIX_MULTIPLIER_2 = 0x94d049bb133111ebULL;
constexpr uint64_t GROUP_MIX = 0xd1b54a32d192ed03ULL;
constexpr uint64_t ELEMENT_MIX = 0x8cb92baa7f3dd15bULL;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int PREFILL_M_TILE = 4;

enum Hqz4CudaDtype : uint32_t {
  HQZ4_CUDA_F16 = 0,
  HQZ4_CUDA_F32 = 1,
};

struct Hqz4Dp4aLaunch {
  const void *input;
  const uint8_t *weight;
  const float *weight_scales;
  int8_t *quantized_activation;
  float *activation_scales;
  void *output;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t group_size;
  uint64_t group_offset;
  uint64_t seed;
  uint32_t dtype;
  cudaStream_t stream;
};

struct Hqz4QuantizeLaunch {
  const void *input;
  int8_t *quantized_activation;
  float *activation_scales;
  uint32_t m;
  uint32_t k;
  uint32_t group_size;
  uint64_t group_offset;
  uint64_t seed;
  uint32_t dtype;
  cudaStream_t stream;
};

struct Hqz4QuantizedMatmulLaunch {
  const int8_t *quantized_activation;
  const float *activation_scales;
  const uint8_t *weight;
  const float *weight_scales;
  void *output;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t group_size;
  uint32_t dtype;
  cudaStream_t stream;
};

struct Hqz4QkvLaunch {
  const int8_t *quantized_activation;
  const float *activation_scales;
  const uint8_t *q_weight;
  const float *q_weight_scales;
  void *q_output;
  uint32_t q_rows;
  const uint8_t *k_weight;
  const float *k_weight_scales;
  void *k_output;
  uint32_t k_rows;
  const uint8_t *v_weight;
  const float *v_weight_scales;
  void *v_output;
  uint32_t v_rows;
  uint32_t m;
  uint32_t k;
  uint32_t group_size;
  uint32_t dtype;
  cudaStream_t stream;
};

struct Hqz4SiluGateUpLaunch {
  const int8_t *quantized_activation;
  const float *activation_scales;
  const uint8_t *gate_weight;
  const float *gate_weight_scales;
  const uint8_t *up_weight;
  const float *up_weight_scales;
  void *output;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t group_size;
  uint32_t dtype;
  cudaStream_t stream;
};

struct Hqz4EmbeddingLaunch {
  const uint32_t *ids;
  const uint8_t *weight;
  const float *weight_scales;
  void *output;
  uint32_t id_count;
  uint32_t rows;
  uint32_t cols;
  uint32_t group_size;
  uint64_t group_offset;
  uint64_t seed;
  uint32_t dtype;
  cudaStream_t stream;
};

__device__ __forceinline__ uint64_t splitmix64(uint64_t state) {
  state += SPLITMIX_INCREMENT;
  state = (state ^ (state >> 30)) * SPLITMIX_MULTIPLIER_1;
  state = (state ^ (state >> 27)) * SPLITMIX_MULTIPLIER_2;
  return state ^ (state >> 31);
}

__device__ __forceinline__ bool sign_is_negative(uint64_t seed,
                                                  uint64_t group,
                                                  uint32_t index) {
  const uint64_t state = seed ^ group * GROUP_MIX ^
                         static_cast<uint64_t>(index) * ELEMENT_MIX;
  return (splitmix64(state) & 1ULL) != 0;
}

template <typename T> __device__ __forceinline__ float load_float(const T *ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_float<__half>(const __half *ptr) {
  return __half2float(*ptr);
}

template <typename T>
__device__ __forceinline__ T store_float(float value) {
  return static_cast<T>(value);
}

template <> __device__ __forceinline__ __half store_float(float value) {
  return __float2half_rn(value);
}

template <typename T>
__global__ void transform_quantize_activation(
    const T *__restrict__ input, int8_t *__restrict__ quantized,
    float *__restrict__ scales, uint32_t m, uint32_t k,
    uint32_t group_size, uint64_t group_offset, uint64_t seed) {
  extern __shared__ float shared[];
  const uint32_t groups_per_row = k / group_size;
  const uint64_t flat_group = blockIdx.x;
  const uint32_t row = flat_group / groups_per_row;
  const uint32_t group = flat_group % groups_per_row;
  const uint32_t index = threadIdx.x;
  if (row >= m || index >= group_size)
    return;

  const uint64_t element = static_cast<uint64_t>(row) * k +
                           static_cast<uint64_t>(group) * group_size + index;
  float value = load_float(&input[element]);
  if (sign_is_negative(seed, group_offset + group, index))
    value = -value;
  shared[index] = value;
  __syncthreads();

  for (uint32_t half = 1; half < group_size; half <<= 1) {
    if (index < group_size / 2) {
      const uint32_t lane = index % half;
      const uint32_t start = (index / half) * (half << 1) + lane;
      const float left = shared[start];
      const float right = shared[start + half];
      shared[start] = left + right;
      shared[start + half] = left - right;
    }
    __syncthreads();
  }

  const float rotated = shared[index] * rsqrtf(static_cast<float>(group_size));
  shared[index] = fabsf(rotated);
  __syncthreads();
  for (uint32_t width = group_size / 2; width > 0; width >>= 1) {
    if (index < width)
      shared[index] = fmaxf(shared[index], shared[index + width]);
    __syncthreads();
  }

  const float scale = shared[0] == 0.0f ? 0.0f : shared[0] / 127.0f;
  if (index == 0)
    scales[flat_group] = scale;
  int quantized_value =
      scale == 0.0f ? 0 : __float2int_rn(rotated / scale);
  if (quantized_value > 127)
    quantized_value = 127;
  if (quantized_value < -127)
    quantized_value = -127;
  quantized[element] = static_cast<int8_t>(quantized_value);
}

__device__ __forceinline__ int8_t signed_nibble(uint16_t packed,
                                                uint32_t shift) {
  const uint8_t nibble = static_cast<uint8_t>((packed >> shift) & 0x0f);
  return static_cast<int8_t>(nibble << 4) >> 4;
}

template <typename T>
__global__ void hqz4_embedding(
    const uint32_t *__restrict__ ids, const uint8_t *__restrict__ weight,
    const float *__restrict__ weight_scales, T *__restrict__ output,
    uint32_t id_count, uint32_t rows, uint32_t cols, uint32_t group_size,
    uint64_t group_offset, uint64_t seed) {
  extern __shared__ float shared[];
  const uint32_t groups_per_row = cols / group_size;
  const uint32_t flat_group = blockIdx.x;
  const uint32_t token = flat_group / groups_per_row;
  const uint32_t group = flat_group % groups_per_row;
  const uint32_t index = threadIdx.x;
  if (token >= id_count || index >= group_size)
    return;

  const uint32_t row = ids[token];
  if (row >= rows) {
    output[static_cast<uint64_t>(token) * cols +
           static_cast<uint64_t>(group) * group_size + index] =
        store_float<T>(0.0f);
    return;
  }

  const uint64_t element = static_cast<uint64_t>(row) * cols +
                           static_cast<uint64_t>(group) * group_size + index;
  const uint8_t packed = weight[element / 2];
  const int8_t code = signed_nibble(packed, (element & 1ULL) * 4);
  const float scale =
      weight_scales[static_cast<uint64_t>(row) * groups_per_row + group];
  shared[index] = static_cast<float>(code) * scale;
  __syncthreads();

  // The normalized Hadamard transform is self-inverse. Applying it to the
  // stored rotated row, followed by the original random sign, reconstructs
  // only the embedding groups requested by `ids`.
  for (uint32_t half = 1; half < group_size; half <<= 1) {
    if (index < group_size / 2) {
      const uint32_t lane = index % half;
      const uint32_t start = (index / half) * (half << 1) + lane;
      const float left = shared[start];
      const float right = shared[start + half];
      shared[start] = left + right;
      shared[start + half] = left - right;
    }
    __syncthreads();
  }

  float value = shared[index] * rsqrtf(static_cast<float>(group_size));
  if (sign_is_negative(seed, group_offset + group, index))
    value = -value;
  output[static_cast<uint64_t>(token) * cols +
         static_cast<uint64_t>(group) * group_size + index] =
      store_float<T>(value);
}

__device__ __forceinline__ int pack_four_weights(uint16_t packed) {
  const uint32_t w0 = static_cast<uint8_t>(signed_nibble(packed, 0));
  const uint32_t w1 = static_cast<uint8_t>(signed_nibble(packed, 4));
  const uint32_t w2 = static_cast<uint8_t>(signed_nibble(packed, 8));
  const uint32_t w3 = static_cast<uint8_t>(signed_nibble(packed, 12));
  return static_cast<int>(w0 | (w1 << 8) | (w2 << 16) | (w3 << 24));
}

__device__ __forceinline__ int dp4a_signed(int left, int right, int sum) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(left, right, sum);
#else
#pragma unroll
  for (int byte = 0; byte < 4; ++byte) {
    const int8_t lhs = static_cast<int8_t>(left >> (byte * 8));
    const int8_t rhs = static_cast<int8_t>(right >> (byte * 8));
    sum += static_cast<int>(lhs) * static_cast<int>(rhs);
  }
  return sum;
#endif
}

template <typename T>
__global__ void hqz4_dp4a_matmul(
    const int8_t *__restrict__ activation,
    const float *__restrict__ activation_scales,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ weight_scales, T *__restrict__ output,
    uint32_t m, uint32_t n, uint32_t k, uint32_t group_size) {
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  const uint64_t output_count = static_cast<uint64_t>(m) * n;
  if (output_index >= output_count)
    return;

  const uint32_t activation_row = output_index / n;
  const uint32_t weight_row = output_index % n;
  const uint32_t groups_per_row = k / group_size;
  float result = 0.0f;

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int dot = 0;
    const uint64_t group_element = static_cast<uint64_t>(group) * group_size;
    for (uint32_t offset = lane * 4; offset < group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * k + group_element + offset;
      const uint16_t packed = *reinterpret_cast<const uint16_t *>(
          &weight[weight_element / 2]);
      const int packed_weights = pack_four_weights(packed);
      const uint64_t activation_element =
          static_cast<uint64_t>(activation_row) * k + group_element + offset;
      const int packed_activation = *reinterpret_cast<const int *>(
          &activation[activation_element]);
      dot = dp4a_signed(packed_weights, packed_activation, dot);
    }
    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1)
      dot += __shfl_down_sync(0xffffffff, dot, delta);
    if (lane == 0) {
      const float weight_scale =
          weight_scales[static_cast<uint64_t>(weight_row) * groups_per_row +
                        group];
      const float activation_scale =
          activation_scales[static_cast<uint64_t>(activation_row) *
                                groups_per_row +
                            group];
      result += static_cast<float>(dot) * weight_scale * activation_scale;
    }
  }

  if (lane == 0)
    output[output_index] = store_float<T>(result);
}

// Prefill kernel: each block computes a 4x4 output tile. A warp owns one
// weight row and reuses every packed W4 load across four activation rows.
// This keeps the decode kernel lean while reducing weight traffic and launch
// count once the flattened token count is at least PREFILL_M_TILE.
template <typename T>
__global__ void hqz4_dp4a_prefill_tiled(
    const int8_t *__restrict__ activation,
    const float *__restrict__ activation_scales,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ weight_scales, T *__restrict__ output,
    uint32_t m, uint32_t n, uint32_t k, uint32_t group_size) {
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t weight_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (weight_row >= n)
    return;

  const uint32_t activation_row_base = blockIdx.y * PREFILL_M_TILE;
  const uint32_t groups_per_row = k / group_size;
  float results[PREFILL_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int dots[PREFILL_M_TILE] = {0, 0, 0, 0};
    const uint64_t group_element = static_cast<uint64_t>(group) * group_size;
    for (uint32_t offset = lane * 4; offset < group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * k + group_element + offset;
      const uint16_t packed = *reinterpret_cast<const uint16_t *>(
          &weight[weight_element / 2]);
      const int packed_weights = pack_four_weights(packed);

#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < m) {
          const uint64_t activation_element =
              static_cast<uint64_t>(activation_row) * k + group_element +
              offset;
          const int packed_activation = *reinterpret_cast<const int *>(
              &activation[activation_element]);
          dots[row_in_tile] =
              dp4a_signed(packed_weights, packed_activation, dots[row_in_tile]);
        }
      }
    }

    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile)
        dots[row_in_tile] +=
            __shfl_down_sync(0xffffffff, dots[row_in_tile], delta);
    }

    if (lane == 0) {
      const float weight_scale =
          weight_scales[static_cast<uint64_t>(weight_row) * groups_per_row +
                        group];
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < m) {
          const float activation_scale =
              activation_scales[static_cast<uint64_t>(activation_row) *
                                    groups_per_row +
                                group];
          results[row_in_tile] += static_cast<float>(dots[row_in_tile]) *
                                  weight_scale * activation_scale;
        }
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
         ++row_in_tile) {
      const uint32_t activation_row = activation_row_base + row_in_tile;
      if (activation_row < m)
        output[static_cast<uint64_t>(activation_row) * n + weight_row] =
            store_float<T>(results[row_in_tile]);
    }
  }
}

template <typename T>
__global__ void hqz4_dp4a_qkv(
    Hqz4QkvLaunch params) {
  const uint32_t projection = blockIdx.z;
  const uint8_t *weight = projection == 0   ? params.q_weight
                          : projection == 1 ? params.k_weight
                                            : params.v_weight;
  const float *weight_scales =
      projection == 0   ? params.q_weight_scales
      : projection == 1 ? params.k_weight_scales
                        : params.v_weight_scales;
  T *output = static_cast<T *>(projection == 0   ? params.q_output
                               : projection == 1 ? params.k_output
                                                 : params.v_output);
  const uint32_t n = projection == 0   ? params.q_rows
                     : projection == 1 ? params.k_rows
                                       : params.v_rows;
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  const uint64_t output_count = static_cast<uint64_t>(params.m) * n;
  if (output_index >= output_count)
    return;

  const uint32_t activation_row = output_index / n;
  const uint32_t weight_row = output_index % n;
  const uint32_t groups_per_row = params.k / params.group_size;
  float result = 0.0f;

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int dot = 0;
    const uint64_t group_element =
        static_cast<uint64_t>(group) * params.group_size;
    for (uint32_t offset = lane * 4; offset < params.group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * params.k + group_element +
          offset;
      const uint16_t packed = *reinterpret_cast<const uint16_t *>(
          &weight[weight_element / 2]);
      const int packed_weights = pack_four_weights(packed);
      const uint64_t activation_element =
          static_cast<uint64_t>(activation_row) * params.k + group_element +
          offset;
      const int packed_activation = *reinterpret_cast<const int *>(
          &params.quantized_activation[activation_element]);
      dot = dp4a_signed(packed_weights, packed_activation, dot);
    }
    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1)
      dot += __shfl_down_sync(0xffffffff, dot, delta);
    if (lane == 0) {
      const float weight_scale =
          weight_scales[static_cast<uint64_t>(weight_row) * groups_per_row +
                        group];
      const float activation_scale =
          params.activation_scales[static_cast<uint64_t>(activation_row) *
                                       groups_per_row +
                                   group];
      result += static_cast<float>(dot) * weight_scale * activation_scale;
    }
  }

  if (lane == 0)
    output[output_index] = store_float<T>(result);
}

template <typename T>
__global__ void hqz4_dp4a_qkv_prefill_tiled(Hqz4QkvLaunch params) {
  const uint32_t projection = blockIdx.z;
  const uint8_t *weight = projection == 0   ? params.q_weight
                          : projection == 1 ? params.k_weight
                                            : params.v_weight;
  const float *weight_scales =
      projection == 0   ? params.q_weight_scales
      : projection == 1 ? params.k_weight_scales
                        : params.v_weight_scales;
  T *output = static_cast<T *>(projection == 0   ? params.q_output
                               : projection == 1 ? params.k_output
                                                 : params.v_output);
  const uint32_t n = projection == 0   ? params.q_rows
                     : projection == 1 ? params.k_rows
                                       : params.v_rows;
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t weight_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (weight_row >= n)
    return;

  const uint32_t activation_row_base = blockIdx.y * PREFILL_M_TILE;
  const uint32_t groups_per_row = params.k / params.group_size;
  float results[PREFILL_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int dots[PREFILL_M_TILE] = {0, 0, 0, 0};
    const uint64_t group_element =
        static_cast<uint64_t>(group) * params.group_size;
    for (uint32_t offset = lane * 4; offset < params.group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * params.k + group_element +
          offset;
      const uint16_t packed = *reinterpret_cast<const uint16_t *>(
          &weight[weight_element / 2]);
      const int packed_weights = pack_four_weights(packed);

#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < params.m) {
          const uint64_t activation_element =
              static_cast<uint64_t>(activation_row) * params.k +
              group_element + offset;
          const int packed_activation = *reinterpret_cast<const int *>(
              &params.quantized_activation[activation_element]);
          dots[row_in_tile] =
              dp4a_signed(packed_weights, packed_activation,
                          dots[row_in_tile]);
        }
      }
    }

    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile)
        dots[row_in_tile] +=
            __shfl_down_sync(0xffffffff, dots[row_in_tile], delta);
    }

    if (lane == 0) {
      const float weight_scale =
          weight_scales[static_cast<uint64_t>(weight_row) * groups_per_row +
                        group];
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < params.m) {
          const float activation_scale =
              params.activation_scales[static_cast<uint64_t>(activation_row) *
                                           groups_per_row +
                                       group];
          results[row_in_tile] += static_cast<float>(dots[row_in_tile]) *
                                  weight_scale * activation_scale;
        }
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
         ++row_in_tile) {
      const uint32_t activation_row = activation_row_base + row_in_tile;
      if (activation_row < params.m)
        output[static_cast<uint64_t>(activation_row) * n + weight_row] =
            store_float<T>(results[row_in_tile]);
    }
  }
}

__device__ __forceinline__ float hqz4_silu(float value) {
  return value / (1.0f + expf(-value));
}

template <typename T>
__device__ __forceinline__ T hqz4_rounded_silu_product(float gate,
                                                       float up) {
  const T rounded_gate = store_float<T>(gate);
  const T rounded_up = store_float<T>(up);
  const T activated = store_float<T>(hqz4_silu(load_float(&rounded_gate)));
  return store_float<T>(load_float(&activated) * load_float(&rounded_up));
}

template <typename T>
__global__ void hqz4_dp4a_silu_gate_up(Hqz4SiluGateUpLaunch params) {
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint64_t output_index =
      static_cast<uint64_t>(blockIdx.x) * WARPS_PER_BLOCK + warp;
  const uint64_t output_count = static_cast<uint64_t>(params.m) * params.n;
  if (output_index >= output_count)
    return;

  const uint32_t activation_row = output_index / params.n;
  const uint32_t weight_row = output_index % params.n;
  const uint32_t groups_per_row = params.k / params.group_size;
  float gate_result = 0.0f;
  float up_result = 0.0f;

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int gate_dot = 0;
    int up_dot = 0;
    const uint64_t group_element =
        static_cast<uint64_t>(group) * params.group_size;
    for (uint32_t offset = lane * 4; offset < params.group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * params.k + group_element +
          offset;
      const uint16_t gate_packed = *reinterpret_cast<const uint16_t *>(
          &params.gate_weight[weight_element / 2]);
      const uint16_t up_packed = *reinterpret_cast<const uint16_t *>(
          &params.up_weight[weight_element / 2]);
      const uint64_t activation_element =
          static_cast<uint64_t>(activation_row) * params.k + group_element +
          offset;
      const int packed_activation = *reinterpret_cast<const int *>(
          &params.quantized_activation[activation_element]);
      gate_dot = dp4a_signed(pack_four_weights(gate_packed),
                             packed_activation, gate_dot);
      up_dot = dp4a_signed(pack_four_weights(up_packed), packed_activation,
                           up_dot);
    }
    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
      gate_dot += __shfl_down_sync(0xffffffff, gate_dot, delta);
      up_dot += __shfl_down_sync(0xffffffff, up_dot, delta);
    }
    if (lane == 0) {
      const uint64_t scale_index =
          static_cast<uint64_t>(weight_row) * groups_per_row + group;
      const float activation_scale =
          params.activation_scales[static_cast<uint64_t>(activation_row) *
                                       groups_per_row +
                                   group];
      gate_result += static_cast<float>(gate_dot) *
                     params.gate_weight_scales[scale_index] *
                     activation_scale;
      up_result += static_cast<float>(up_dot) *
                   params.up_weight_scales[scale_index] * activation_scale;
    }
  }

  if (lane == 0)
    static_cast<T *>(params.output)[output_index] =
        hqz4_rounded_silu_product<T>(gate_result, up_result);
}

template <typename T>
__global__ void
hqz4_dp4a_silu_gate_up_prefill_tiled(Hqz4SiluGateUpLaunch params) {
  const uint32_t warp = threadIdx.x / WARP_SIZE;
  const uint32_t lane = threadIdx.x % WARP_SIZE;
  const uint32_t weight_row = blockIdx.x * WARPS_PER_BLOCK + warp;
  if (weight_row >= params.n)
    return;

  const uint32_t activation_row_base = blockIdx.y * PREFILL_M_TILE;
  const uint32_t groups_per_row = params.k / params.group_size;
  float gate_results[PREFILL_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up_results[PREFILL_M_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t group = 0; group < groups_per_row; ++group) {
    int gate_dots[PREFILL_M_TILE] = {0, 0, 0, 0};
    int up_dots[PREFILL_M_TILE] = {0, 0, 0, 0};
    const uint64_t group_element =
        static_cast<uint64_t>(group) * params.group_size;
    for (uint32_t offset = lane * 4; offset < params.group_size;
         offset += WARP_SIZE * 4) {
      const uint64_t weight_element =
          static_cast<uint64_t>(weight_row) * params.k + group_element +
          offset;
      const int gate_weights = pack_four_weights(
          *reinterpret_cast<const uint16_t *>(
              &params.gate_weight[weight_element / 2]));
      const int up_weights = pack_four_weights(
          *reinterpret_cast<const uint16_t *>(
              &params.up_weight[weight_element / 2]));

#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < params.m) {
          const uint64_t activation_element =
              static_cast<uint64_t>(activation_row) * params.k +
              group_element + offset;
          const int packed_activation = *reinterpret_cast<const int *>(
              &params.quantized_activation[activation_element]);
          gate_dots[row_in_tile] =
              dp4a_signed(gate_weights, packed_activation,
                          gate_dots[row_in_tile]);
          up_dots[row_in_tile] =
              dp4a_signed(up_weights, packed_activation,
                          up_dots[row_in_tile]);
        }
      }
    }

    for (uint32_t delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        gate_dots[row_in_tile] +=
            __shfl_down_sync(0xffffffff, gate_dots[row_in_tile], delta);
        up_dots[row_in_tile] +=
            __shfl_down_sync(0xffffffff, up_dots[row_in_tile], delta);
      }
    }

    if (lane == 0) {
      const uint64_t scale_index =
          static_cast<uint64_t>(weight_row) * groups_per_row + group;
      const float gate_scale = params.gate_weight_scales[scale_index];
      const float up_scale = params.up_weight_scales[scale_index];
#pragma unroll
      for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
           ++row_in_tile) {
        const uint32_t activation_row = activation_row_base + row_in_tile;
        if (activation_row < params.m) {
          const float activation_scale =
              params.activation_scales[static_cast<uint64_t>(activation_row) *
                                           groups_per_row +
                                       group];
          gate_results[row_in_tile] +=
              static_cast<float>(gate_dots[row_in_tile]) * gate_scale *
              activation_scale;
          up_results[row_in_tile] +=
              static_cast<float>(up_dots[row_in_tile]) * up_scale *
              activation_scale;
        }
      }
    }
  }

  if (lane == 0) {
#pragma unroll
    for (uint32_t row_in_tile = 0; row_in_tile < PREFILL_M_TILE;
         ++row_in_tile) {
      const uint32_t activation_row = activation_row_base + row_in_tile;
      if (activation_row < params.m) {
        const uint64_t output_index =
            static_cast<uint64_t>(activation_row) * params.n + weight_row;
        static_cast<T *>(params.output)[output_index] =
            hqz4_rounded_silu_product<T>(gate_results[row_in_tile],
                                         up_results[row_in_tile]);
      }
    }
  }
}

template <typename T>
cudaError_t launch_quantize_typed(const Hqz4QuantizeLaunch &params) {
  const uint32_t groups_per_row = params.k / params.group_size;
  const uint64_t activation_groups =
      static_cast<uint64_t>(params.m) * groups_per_row;
  transform_quantize_activation<T>
      <<<static_cast<uint32_t>(activation_groups), params.group_size,
         params.group_size * sizeof(float), params.stream>>>(
          static_cast<const T *>(params.input), params.quantized_activation,
          params.activation_scales, params.m, params.k, params.group_size,
          params.group_offset, params.seed);
  return cudaGetLastError();
}

template <typename T>
cudaError_t
launch_quantized_matmul_typed(const Hqz4QuantizedMatmulLaunch &params) {
  if (params.m >= PREFILL_M_TILE) {
    const dim3 blocks((params.n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                      (params.m + PREFILL_M_TILE - 1) / PREFILL_M_TILE);
    hqz4_dp4a_prefill_tiled<T>
        <<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, params.stream>>>(
            params.quantized_activation, params.activation_scales,
            params.weight, params.weight_scales,
            static_cast<T *>(params.output), params.m, params.n, params.k,
            params.group_size);
  } else {
    const uint64_t outputs = static_cast<uint64_t>(params.m) * params.n;
    const uint32_t blocks =
        static_cast<uint32_t>((outputs + WARPS_PER_BLOCK - 1) /
                              WARPS_PER_BLOCK);
    hqz4_dp4a_matmul<T><<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0,
                          params.stream>>>(
        params.quantized_activation, params.activation_scales, params.weight,
        params.weight_scales, static_cast<T *>(params.output), params.m,
        params.n, params.k, params.group_size);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_qkv_typed(const Hqz4QkvLaunch &params) {
  const uint32_t qk_max =
      params.q_rows > params.k_rows ? params.q_rows : params.k_rows;
  const uint32_t max_rows =
      qk_max > params.v_rows ? qk_max : params.v_rows;
  if (params.m >= PREFILL_M_TILE) {
    const dim3 blocks((max_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                      (params.m + PREFILL_M_TILE - 1) / PREFILL_M_TILE, 3);
    hqz4_dp4a_qkv_prefill_tiled<T>
        <<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, params.stream>>>(params);
  } else {
    const uint64_t outputs = static_cast<uint64_t>(params.m) * max_rows;
    const dim3 blocks(static_cast<uint32_t>(
                          (outputs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK),
                      1, 3);
    hqz4_dp4a_qkv<T>
        <<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, params.stream>>>(params);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_silu_gate_up_typed(const Hqz4SiluGateUpLaunch &params) {
  if (params.m >= PREFILL_M_TILE) {
    const dim3 blocks((params.n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
                      (params.m + PREFILL_M_TILE - 1) / PREFILL_M_TILE);
    hqz4_dp4a_silu_gate_up_prefill_tiled<T>
        <<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, params.stream>>>(params);
  } else {
    const uint64_t outputs = static_cast<uint64_t>(params.m) * params.n;
    const uint32_t blocks =
        static_cast<uint32_t>((outputs + WARPS_PER_BLOCK - 1) /
                              WARPS_PER_BLOCK);
    hqz4_dp4a_silu_gate_up<T>
        <<<blocks, WARPS_PER_BLOCK * WARP_SIZE, 0, params.stream>>>(params);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t launch_typed(const Hqz4Dp4aLaunch &params) {
  const Hqz4QuantizeLaunch quantize = {
      params.input,
      params.quantized_activation,
      params.activation_scales,
      params.m,
      params.k,
      params.group_size,
      params.group_offset,
      params.seed,
      params.dtype,
      params.stream,
  };
  cudaError_t status = launch_quantize_typed<T>(quantize);
  if (status != cudaSuccess)
    return status;

  const Hqz4QuantizedMatmulLaunch matmul = {
      params.quantized_activation,
      params.activation_scales,
      params.weight,
      params.weight_scales,
      params.output,
      params.m,
      params.n,
      params.k,
      params.group_size,
      params.dtype,
      params.stream,
  };
  return launch_quantized_matmul_typed<T>(matmul);
}

template <typename T>
cudaError_t launch_embedding_typed(const Hqz4EmbeddingLaunch &params) {
  const uint32_t groups_per_row = params.cols / params.group_size;
  const uint64_t groups =
      static_cast<uint64_t>(params.id_count) * groups_per_row;
  hqz4_embedding<T>
      <<<static_cast<uint32_t>(groups), params.group_size,
         params.group_size * sizeof(float), params.stream>>>(
          params.ids, params.weight, params.weight_scales,
          static_cast<T *>(params.output), params.id_count, params.rows,
          params.cols, params.group_size, params.group_offset, params.seed);
  return cudaGetLastError();
}

} // namespace

extern "C" int launch_hqz4_quantize(const Hqz4QuantizeLaunch *params) {
  if (params == nullptr || params->input == nullptr ||
      params->quantized_activation == nullptr ||
      params->activation_scales == nullptr || params->m == 0 ||
      params->k == 0 || params->group_size < 4 ||
      params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->k % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_quantize_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_quantize_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}

extern "C" int launch_hqz4_dp4a_quantized(
    const Hqz4QuantizedMatmulLaunch *params) {
  if (params == nullptr || params->quantized_activation == nullptr ||
      params->activation_scales == nullptr || params->weight == nullptr ||
      params->weight_scales == nullptr || params->output == nullptr ||
      params->m == 0 || params->n == 0 || params->k == 0 ||
      params->group_size < 4 || params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->k % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_quantized_matmul_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_quantized_matmul_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}

extern "C" int launch_hqz4_qkv_quantized(const Hqz4QkvLaunch *params) {
  if (params == nullptr || params->quantized_activation == nullptr ||
      params->activation_scales == nullptr || params->q_weight == nullptr ||
      params->q_weight_scales == nullptr || params->q_output == nullptr ||
      params->q_rows == 0 || params->k_weight == nullptr ||
      params->k_weight_scales == nullptr || params->k_output == nullptr ||
      params->k_rows == 0 || params->v_weight == nullptr ||
      params->v_weight_scales == nullptr || params->v_output == nullptr ||
      params->v_rows == 0 || params->m == 0 || params->k == 0 ||
      params->group_size < 4 || params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->k % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_qkv_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_qkv_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}

extern "C" int
launch_hqz4_silu_gate_up_quantized(const Hqz4SiluGateUpLaunch *params) {
  if (params == nullptr || params->quantized_activation == nullptr ||
      params->activation_scales == nullptr || params->gate_weight == nullptr ||
      params->gate_weight_scales == nullptr || params->up_weight == nullptr ||
      params->up_weight_scales == nullptr || params->output == nullptr ||
      params->m == 0 || params->n == 0 || params->k == 0 ||
      params->group_size < 4 || params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->k % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_silu_gate_up_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_silu_gate_up_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}

extern "C" int launch_hqz4_dp4a(const Hqz4Dp4aLaunch *params) {
  if (params == nullptr || params->input == nullptr || params->weight == nullptr ||
      params->weight_scales == nullptr ||
      params->quantized_activation == nullptr ||
      params->activation_scales == nullptr || params->output == nullptr ||
      params->m == 0 || params->n == 0 || params->k == 0 ||
      params->group_size < 4 || params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->k % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}

extern "C" int launch_hqz4_embedding(const Hqz4EmbeddingLaunch *params) {
  if (params == nullptr || params->ids == nullptr || params->weight == nullptr ||
      params->weight_scales == nullptr || params->output == nullptr ||
      params->id_count == 0 || params->rows == 0 || params->cols == 0 ||
      params->group_size < 4 || params->group_size > 1024 ||
      (params->group_size & (params->group_size - 1)) != 0 ||
      params->cols % params->group_size != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaError_t status;
  switch (params->dtype) {
  case HQZ4_CUDA_F16:
    status = launch_embedding_typed<__half>(*params);
    break;
  case HQZ4_CUDA_F32:
    status = launch_embedding_typed<float>(*params);
    break;
  default:
    status = cudaErrorInvalidValue;
    break;
  }
  return static_cast<int>(status);
}
