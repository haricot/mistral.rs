#include <math.h>
#include <stdint.h>

// DKR-R2: independent SM61 candidates. No host adapter or GPU qualification is implied.

#define DKR_INDEX (blockIdx.x * blockDim.x + threadIdx.x)

extern "C" __global__ void dkr_copy_f32(const float* x, float* y, uint32_t n) {
    uint32_t i = DKR_INDEX;
    if (i < n) y[i] = x[i];
}

extern "C" __global__ void dkr_affine_f32(const float* x, float* y, float mul, float add, uint32_t n) {
    uint32_t i = DKR_INDEX;
    if (i < n) y[i] = x[i] * mul + add;
}

#define DKR_BINARY(NAME, EXPR) \
extern "C" __global__ void dkr_##NAME##_f32(const float* a, const float* b, float* out, uint32_t n) { \
    uint32_t i = DKR_INDEX; \
    if (i < n) { float x = a[i], y = b[i]; out[i] = (EXPR); } \
}

DKR_BINARY(add, x + y)
DKR_BINARY(sub, x - y)
DKR_BINARY(mul, x * y)
DKR_BINARY(div, x / y)

#define DKR_UNARY(NAME, EXPR) \
extern "C" __global__ void dkr_##NAME##_f32(const float* x, float* out, uint32_t n) { \
    uint32_t i = DKR_INDEX; \
    if (i < n) { float v = x[i]; out[i] = (EXPR); } \
}

DKR_UNARY(exp, expf(v))
DKR_UNARY(tanh, tanhf(v))
DKR_UNARY(silu, v / (1.0f + expf(-v)))
DKR_UNARY(gelu, 0.5f * v * (1.0f + erff(v * 0.7071067811865475f)))

// Reductions use one 256-thread CTA per row. The adapter validates grid.x == rows.
extern "C" __global__ void dkr_reduce_sum_f32(const float* x, float* out, uint32_t rows, uint32_t width) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float values[256];
    float v = 0.0f;
    for (uint32_t col = threadIdx.x; col < width; col += 256) v += x[row * width + col];
    values[threadIdx.x] = v;
    __syncthreads();
    for (uint32_t step = 128; step; step >>= 1) {
        if (threadIdx.x < step) values[threadIdx.x] += values[threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x) out[row] = values[0];
}

extern "C" __global__ void dkr_reduce_max_f32(const float* x, float* out, uint32_t rows, uint32_t width) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float values[256];
    float v = -INFINITY;
    for (uint32_t col = threadIdx.x; col < width; col += 256) v = fmaxf(v, x[row * width + col]);
    values[threadIdx.x] = v;
    __syncthreads();
    for (uint32_t step = 128; step; step >>= 1) {
        if (threadIdx.x < step) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + step]);
        __syncthreads();
    }
    if (!threadIdx.x) out[row] = values[0];
}

extern "C" __global__ void dkr_softmax_f32(const float* x, float* out, uint32_t rows, uint32_t width) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float values[256];
    __shared__ float maximum;
    __shared__ float denominator;
    float peak = -INFINITY;
    for (uint32_t col = threadIdx.x; col < width; col += 256) peak = fmaxf(peak, x[row * width + col]);
    values[threadIdx.x] = peak;
    __syncthreads();
    for (uint32_t step = 128; step; step >>= 1) {
        if (threadIdx.x < step) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + step]);
        __syncthreads();
    }
    if (!threadIdx.x) maximum = values[0];
    __syncthreads();
    float sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < width; col += 256) sum += expf(x[row * width + col] - maximum);
    values[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t step = 128; step; step >>= 1) {
        if (threadIdx.x < step) values[threadIdx.x] += values[threadIdx.x + step];
        __syncthreads();
    }
    if (!threadIdx.x) denominator = values[0];
    __syncthreads();
    for (uint32_t col = threadIdx.x; col < width; col += 256)
        out[row * width + col] = expf(x[row * width + col] - maximum) / denominator;
}

// Four entry signatures eliminate nullable scale pointers and device-side option checks.
#define DKR_RMSNORM(NAME, RESIDUAL, SCALED) \
extern "C" __global__ void dkr_##NAME##_f32( \
    const float* x, DKR_RES_ARG_##RESIDUAL DKR_SCALE_ARG_##SCALED float* out, \
    uint32_t rows, uint32_t width, float eps) { \
    uint32_t row = blockIdx.x; \
    if (row >= rows) return; \
    __shared__ float values[256]; \
    __shared__ float inverse; \
    float sum = 0.0f; \
    for (uint32_t col = threadIdx.x; col < width; col += 256) { \
        float value = x[row * width + col] DKR_RES_OP_##RESIDUAL; \
        sum += value * value; \
    } \
    values[threadIdx.x] = sum; \
    __syncthreads(); \
    for (uint32_t step = 128; step; step >>= 1) { \
        if (threadIdx.x < step) values[threadIdx.x] += values[threadIdx.x + step]; \
        __syncthreads(); \
    } \
    if (!threadIdx.x) inverse = rsqrtf(values[0] / width + eps); \
    __syncthreads(); \
    for (uint32_t col = threadIdx.x; col < width; col += 256) { \
        float value = x[row * width + col] DKR_RES_OP_##RESIDUAL; \
        out[row * width + col] = value * inverse DKR_SCALE_OP_##SCALED; \
    } \
}

#define DKR_RES_ARG_0
#define DKR_RES_ARG_1 const float* residual,
#define DKR_SCALE_ARG_0
#define DKR_SCALE_ARG_1 const float* scale,
#define DKR_RES_OP_0
#define DKR_RES_OP_1 + residual[row * width + col]
#define DKR_SCALE_OP_0
#define DKR_SCALE_OP_1 * scale[col]

DKR_RMSNORM(rms_norm, 0, 0)
DKR_RMSNORM(rms_norm_scaled, 0, 1)
DKR_RMSNORM(residual_rms_norm, 1, 0)
DKR_RMSNORM(residual_rms_norm_scaled, 1, 1)

extern "C" __global__ void dkr_causal_mask_f32(
    const float* x, float* out, uint32_t rows, uint32_t cols, uint32_t past_tokens) {
    uint32_t i = DKR_INDEX;
    uint32_t n = rows * cols;
    if (i >= n) return;
    uint32_t row = i / cols;
    uint32_t col = i % cols;
    out[i] = col > row + past_tokens ? -INFINITY : x[i];
}

// Sparse ID arrays must be coalesced, unique and in range before either launch.
// A copy launch must precede the penalty launch when preserving the source logits.
extern "C" __global__ void dkr_logits_penalty_sparse_f32(
    float* logits, const uint32_t* token_ids, const uint32_t* counts,
    uint32_t entries, uint32_t vocab, float frequency, float presence, float repetition) {
    uint32_t i = DKR_INDEX;
    if (i >= entries) return;
    uint32_t token = token_ids[i];
    if (token >= vocab) return;
    float value = logits[token];
    if (counts[i] > 0) {
        if (repetition != 1.0f) value = value < 0.0f ? value * repetition : value / repetition;
        value -= frequency * static_cast<float>(counts[i]) + presence;
    }
    logits[token] = value;
}

extern "C" __global__ void dkr_logits_bias_sparse_f32(
    float* logits, const uint32_t* token_ids, const float* biases,
    uint32_t entries, uint32_t vocab) {
    uint32_t i = DKR_INDEX;
    if (i >= entries) return;
    uint32_t token = token_ids[i];
    if (token < vocab) logits[token] += biases[i];
}

extern "C" __global__ void dkr_gather_rows_f32(
    const float* source, const uint32_t* indices, float* out,
    uint32_t rows, uint32_t width, uint32_t source_rows) {
    uint32_t i = DKR_INDEX;
    if (i >= rows * width) return;
    uint32_t row = i / width;
    uint32_t col = i % width;
    uint32_t selected = indices[row];
    if (selected < source_rows) out[i] = source[selected * width + col];
}

extern "C" __global__ void dkr_rope_pairs_f32(
    const float* source, const float* cosine, const float* sine, float* out,
    uint32_t rows, uint32_t width, uint32_t rotary_width) {
    uint32_t i = DKR_INDEX;
    uint32_t pairs = rows * (rotary_width / 2);
    if (i >= pairs) return;
    uint32_t row = i / (rotary_width / 2);
    uint32_t pair = i % (rotary_width / 2);
    uint32_t start = row * width + 2 * pair;
    float x0 = source[start];
    float x1 = source[start + 1];
    float c = cosine[row * (rotary_width / 2) + pair];
    float s = sine[row * (rotary_width / 2) + pair];
    out[start] = x0 * c - x1 * s;
    out[start + 1] = x0 * s + x1 * c;
}
