#ifndef DKR_LARGE_TOPK_IMAGE
#error "select exactly one large top-k image"
#endif
#if DKR_LARGE_TOPK_IMAGE == 1
#define DKR_ROUTER_TOPK_EXTERNAL
#define DKR_LARGE_TOPK_EXTERNAL
#include "../../../mistralrs-core/src/cuda/large_topk_device.cuh"
extern "C" __global__ void dkr_large_topk_stage1_single_softmax_f32(const float *input, float *block_values, uint32_t *block_indices, float *block_maxes, float *block_sums, int32_t ncols, int32_t k, int32_t chunk_size, float inv_temperature) {
  topk_large_stage1<float, false, true>(input, block_values, block_indices, block_maxes, block_sums, ncols, k, chunk_size, nullptr, inv_temperature);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_softmax_f32(const float *input, float *block_values, uint32_t *block_indices, float *block_maxes, float *block_sums, int32_t ncols, int32_t k, int32_t chunk_size, const float *inv_temperatures) {
  topk_large_stage1<float, true, true>(input, block_values, block_indices, block_maxes, block_sums, ncols, k, chunk_size, inv_temperatures, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_ranked_f32(const float *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k, int32_t chunk_size) {
  topk_large_stage1<float, true, false>(input, block_values, block_indices, nullptr, nullptr, ncols, k, chunk_size, nullptr, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_softmax_f16(const __half *input, float *block_values, uint32_t *block_indices, float *block_maxes, float *block_sums, int32_t ncols, int32_t k, int32_t chunk_size, const float *inv_temperatures) {
  topk_large_stage1<__half, true, true>(input, block_values, block_indices, block_maxes, block_sums, ncols, k, chunk_size, inv_temperatures, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_ranked_f16(const __half *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k, int32_t chunk_size) {
  topk_large_stage1<__half, true, false>(input, block_values, block_indices, nullptr, nullptr, ncols, k, chunk_size, nullptr, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_softmax_bf16(const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices, float *block_maxes, float *block_sums, int32_t ncols, int32_t k, int32_t chunk_size, const float *inv_temperatures) {
  topk_large_stage1<__nv_bfloat16, true, true>(input, block_values, block_indices, block_maxes, block_sums, ncols, k, chunk_size, inv_temperatures, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage1_batch_ranked_bf16(const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k, int32_t chunk_size) {
  topk_large_stage1<__nv_bfloat16, true, false>(input, block_values, block_indices, nullptr, nullptr, ncols, k, chunk_size, nullptr, 0.0f);
}
extern "C" __global__ void dkr_large_topk_stage2_single_split(const float *block_values, const uint32_t *block_indices, const float *block_maxes, const float *block_sums, float *values_out, uint32_t *indices_out, float *softmax_info_out, int32_t nblocks, int32_t k) {
  topk_large_stage2_f32(block_values, block_indices, block_maxes, block_sums, values_out, indices_out, softmax_info_out, nblocks, k);
}
extern "C" __global__ void dkr_large_topk_stage2_single_packed(const float *block_values, const uint32_t *block_indices, const float *block_maxes, const float *block_sums, float *packed_out, int32_t nblocks, int32_t k) {
  topk_large_stage2_f32_packed<false, true>(block_values, block_indices, block_maxes, block_sums, packed_out, nblocks, k);
}
extern "C" __global__ void dkr_large_topk_stage2_batch_packed(const float *block_values, const uint32_t *block_indices, const float *block_maxes, const float *block_sums, float *packed_out, int32_t nblocks, int32_t k) {
  topk_large_stage2_f32_packed<true, true>(block_values, block_indices, block_maxes, block_sums, packed_out, nblocks, k);
}
extern "C" __global__ void dkr_large_topk_stage2_batch_ranked(const float *block_values, const uint32_t *block_indices, float *packed_out, int32_t nblocks, int32_t k) {
  topk_large_stage2_f32_packed<true, false>(block_values, block_indices, nullptr, nullptr, packed_out, nblocks, k);
}
extern "C" __global__ void dkr_ranked_radix_stage1_f32(const float *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k) {
  topk_large_ranked_stage1_radix<float>(input, block_values, block_indices, ncols, k);
}
extern "C" __global__ void dkr_ranked_radix_stage1_f16(const __half *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k) {
  topk_large_ranked_stage1_radix<__half>(input, block_values, block_indices, ncols, k);
}
extern "C" __global__ void dkr_ranked_radix_stage1_bf16(const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t k) {
  topk_large_ranked_stage1_radix<__nv_bfloat16>(input, block_values, block_indices, ncols, k);
}
extern "C" __global__ void dkr_ranked_radix_stage2(const float *block_values, const uint32_t *block_indices, float *packed_out, int32_t nblocks, int32_t k) {
  topk_large_ranked_stage2_radix(block_values, block_indices, packed_out, nblocks, k);
}
#elif DKR_LARGE_TOPK_IMAGE == 2
#define DKR_COOPERATIVE_RADIX_EXTERNAL
#include "../../../mistralrs-core/src/cuda/radix_topk.cuh"
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f32_v1(const float *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<float, 1>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f32_v2(const float *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<float, 2>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f32_v4(const float *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<float, 4>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f16_v1(const __half *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__half, 1>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f16_v2(const __half *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__half, 2>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f16_v4(const __half *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__half, 4>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_f16_v8(const __half *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__half, 8>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_bf16_v1(const __nv_bfloat16 *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__nv_bfloat16, 1>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_bf16_v2(const __nv_bfloat16 *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__nv_bfloat16, 2>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_bf16_v4(const __nv_bfloat16 *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__nv_bfloat16, 4>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ __launch_bounds__(1024) void dkr_ranked_cooperative_bf16_v8(const __nv_bfloat16 *input, float *packed_out, int32_t nrows, int32_t ncols, int32_t k, uint32_t *row_states, uint32_t chunk_size, uint32_t ctas_per_group) {
  mistralrs_radix_topk::ranked_radix_select<__nv_bfloat16, 8>(input, packed_out, nrows, ncols, k, reinterpret_cast<mistralrs_radix_topk::RowState *>(row_states), chunk_size, ctas_per_group);
}
extern "C" __global__ void dkr_ranked_cooperative_sort(float *packed_out, int32_t k) {
  mistralrs_radix_topk::ranked_radix_sort_pack(packed_out, k);
}
#else
#error "invalid large top-k image"
#endif
