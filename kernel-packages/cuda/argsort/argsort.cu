#include "../../../mistralrs-core/src/cuda/argsort_device.cuh"

extern "C" __global__ void dkr_argsort_init(uint32_t *work, uint32_t padded) {
  argsort_init(work, padded);
}

extern "C" __global__ void dkr_argsort_asc_f32(const float *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<float, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_f32(const float *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<float, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_f16(const __half *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<__half, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_f16(const __half *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<__half, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_bf16(const __nv_bfloat16 *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<__nv_bfloat16, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_bf16(const __nv_bfloat16 *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<__nv_bfloat16, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_f64(const double *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<double, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_f64(const double *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<double, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_u8(const uint8_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<uint8_t, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_u8(const uint8_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<uint8_t, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_u32(const uint32_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<uint32_t, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_u32(const uint32_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<uint32_t, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_asc_i64(const int64_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<int64_t, true>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_desc_i64(const int64_t *input, uint32_t *work, uint32_t cols, uint32_t padded, uint32_t j, uint32_t k) {
  argsort_compare<int64_t, false>(input, work, cols, padded, j, k);
}

extern "C" __global__ void dkr_argsort_finish(const uint32_t *work, uint32_t *indices, uint32_t cols) {
  argsort_finish(work, indices, cols);
}

extern "C" __global__ void dkr_argsort_gather_f32(const float *input, const uint32_t *indices, float *output, uint32_t cols, uint32_t elements) {
  argsort_gather<float>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_f16(const __half *input, const uint32_t *indices, __half *output, uint32_t cols, uint32_t elements) {
  argsort_gather<__half>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_bf16(const __nv_bfloat16 *input, const uint32_t *indices, __nv_bfloat16 *output, uint32_t cols, uint32_t elements) {
  argsort_gather<__nv_bfloat16>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_f64(const double *input, const uint32_t *indices, double *output, uint32_t cols, uint32_t elements) {
  argsort_gather<double>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_u8(const uint8_t *input, const uint32_t *indices, uint8_t *output, uint32_t cols, uint32_t elements) {
  argsort_gather<uint8_t>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_u32(const uint32_t *input, const uint32_t *indices, uint32_t *output, uint32_t cols, uint32_t elements) {
  argsort_gather<uint32_t>(input, indices, output, cols, elements);
}

extern "C" __global__ void dkr_argsort_gather_i64(const int64_t *input, const uint32_t *indices, int64_t *output, uint32_t cols, uint32_t elements) {
  argsort_gather<int64_t>(input, indices, output, cols, elements);
}
