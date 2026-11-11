#include <math.h>
#include <stdint.h>

template <int ITEMS>
__device__ __forceinline__ void scan(
    const float* x, const float* dt, const float* a,
    const float* b, const float* c, const float* d, const float* dt_bias,
    float* state, float* output,
    int n_heads, int head_dim, int d_state, int seq_len,
    float dt_min, float dt_max) {
  const int head = blockIdx.x / head_dim;
  const int channel = blockIdx.x % head_dim;
  const int batch = blockIdx.y;
  const int lane = threadIdx.x;
  if (head >= n_heads) return;

  const int64_t state_base =
      (int64_t(batch) * n_heads * head_dim + int64_t(head) * head_dim + channel)
      * d_state;
  float local[ITEMS];
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int col = lane * ITEMS + i;
    local[i] = col < d_state ? state[state_base + col] : 0.0f;
  }

  const float av = a[head];
  const float dv = d[head];
  const float bias = dt_bias[head];
  const int64_t token_base = int64_t(batch) * seq_len;
  const int inner = n_heads * head_dim;
  for (int t = 0; t < seq_len; ++t) {
    const int64_t token = token_base + t;
    float delta = dt[token * n_heads + head] + bias;
    delta = delta > 20.0f ? delta : logf(1.0f + expf(delta));
    delta = fminf(fmaxf(delta, dt_min), dt_max);
    const float decay = expf(delta * av);
    const float xv = x[token * inner + head * head_dim + channel];
    const float xdt = xv * delta;
    const int64_t coefficient_base = (token * n_heads + head) * d_state;
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
      const int col = lane * ITEMS + i;
      if (col < d_state) {
        local[i] = local[i] * decay + b[coefficient_base + col] * xdt;
        dot += local[i] * c[coefficient_base + col];
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      dot += __shfl_xor_sync(0xffffffffu, dot, off);
    }
    if (lane == 0) {
      output[token * inner + head * head_dim + channel] = dot + dv * xv;
    }
  }
#pragma unroll
  for (int i = 0; i < ITEMS; ++i) {
    const int col = lane * ITEMS + i;
    if (col < d_state) state[state_base + col] = local[i];
  }
}

#define DKR_SCAN(ITEMS) \
extern "C" __global__ void dkr_ssm_scan_cf##ITEMS( \
    const float* x, const float* dt, const float* a, \
    const float* b, const float* c, const float* d, \
    const float* dt_bias, float* state, float* output, \
    int n_heads, int head_dim, int d_state, int seq_len, \
    float dt_min, float dt_max) { \
  scan<ITEMS>(x, dt, a, b, c, d, dt_bias, state, output, \
              n_heads, head_dim, d_state, seq_len, dt_min, dt_max); \
}

DKR_SCAN(1)
DKR_SCAN(2)
DKR_SCAN(4)
DKR_SCAN(8)
