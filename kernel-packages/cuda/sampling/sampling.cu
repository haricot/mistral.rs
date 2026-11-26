#ifndef DKR_SAMPLING_IMAGE
#error "select exactly one sampling image"
#endif
#define DKR_SAMPLING_EXTERNAL
#if DKR_SAMPLING_IMAGE == 1
#define DKR_ROUTER_TOPK_EXTERNAL
#define DKR_LARGE_TOPK_EXTERNAL
#include "../../../mistralrs-core/src/cuda/sampling_device.cuh"
extern "C" __global__ void dkr_top1_stage1_single_f32(const float *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<float, false, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage1_batch_f32(const float *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<float, true, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage1_single_f16(const __half *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<__half, false, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage1_batch_f16(const __half *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<__half, true, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage1_single_bf16(const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<__nv_bfloat16, false, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage1_batch_bf16(const __nv_bfloat16 *input, float *block_values, uint32_t *block_indices, int32_t ncols, int32_t chunk_size) {
  top1_large_stage1<__nv_bfloat16, true, false>(input, block_values, block_indices, nullptr, ncols, chunk_size, nullptr);
}
extern "C" __global__ void dkr_top1_stage2_single_tokens(const float *block_values, const uint32_t *block_indices, uint32_t *token_ids_out, int32_t nblocks) {
  top1_large_stage2_f32_packed<false>(block_values, block_indices, nullptr, token_ids_out, nblocks);
}
extern "C" __global__ void dkr_top1_stage2_single_both(const float *block_values, const uint32_t *block_indices, float *packed_out, uint32_t *token_ids_out, int32_t nblocks) {
  top1_large_stage2_f32_packed<false>(block_values, block_indices, packed_out, token_ids_out, nblocks);
}
extern "C" __global__ void dkr_top1_stage2_batch_tokens(const float *block_values, const uint32_t *block_indices, uint32_t *token_ids_out, int32_t nblocks) {
  top1_large_stage2_f32_packed<true>(block_values, block_indices, nullptr, token_ids_out, nblocks);
}
extern "C" __global__ void dkr_top1_stage2_batch_both(const float *block_values, const uint32_t *block_indices, float *packed_out, uint32_t *token_ids_out, int32_t nblocks) {
  top1_large_stage2_f32_packed<true>(block_values, block_indices, packed_out, token_ids_out, nblocks);
}
extern "C" __global__ void dkr_top1_stage2_batch_packed(const float *block_values, const uint32_t *block_indices, float *packed_out, int32_t nblocks) {
  top1_large_stage2_f32_packed<true>(block_values, block_indices, packed_out, nullptr, nblocks);
}
extern "C" __global__ void dkr_categorical_stage1(const float *input, float *block_values, float *block_sums, int32_t ncols, int32_t chunk_size, const float *inv_temperatures) {
  top1_large_stage1<float, true, true>(input, block_values, nullptr, block_sums, ncols, chunk_size, inv_temperatures);
}
extern "C" __global__ void dkr_categorical_stage2(const float *input, const float *inv_temperatures, const float *uniforms, const float *block_values, const float *block_sums, float *packed_out, int32_t ncols, int32_t chunk_size, int32_t nblocks) {
  categorical_large_stage2_f32_packed(input, inv_temperatures, uniforms, block_values, block_sums, packed_out, ncols, chunk_size, nblocks);
}
extern "C" __global__ void dkr_sample_ranked_topk(const float *packed, const float *params, uint32_t *tokens, int32_t nrows, int32_t packed_k) {
  sample_ranked_topk_kernel(packed, params, tokens, nrows, packed_k);
}
#elif DKR_SAMPLING_IMAGE == 2
#include "../../../mistralrs-core/src/cuda/dflash_selector_device.cuh"
extern "C" __global__ void dkr_dflash_greedy_f32_f32_f32(const float *packed_topk, const float *projected_hidden, const float *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_greedy_f32_f32_bf16(const float *packed_topk, const float *projected_hidden, const float *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_greedy_f32_bf16_f32(const float *packed_topk, const float *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_greedy_f32_bf16_bf16(const float *packed_topk, const float *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_greedy_bf16_f32_f32(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const float *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_greedy_bf16_f32_bf16(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const float *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_greedy_bf16_bf16_f32(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_greedy_bf16_bf16_bf16(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, uint32_t *selected, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_greedy_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, selected, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_sample_f32_f32_f32(const float *packed_topk, const float *projected_hidden, const float *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_sample_f32_f32_bf16(const float *packed_topk, const float *projected_hidden, const float *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_sample_f32_bf16_f32(const float *packed_topk, const float *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_sample_f32_bf16_bf16(const float *packed_topk, const float *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_sample_bf16_f32_f32(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const float *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_sample_bf16_f32_bf16(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const float *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32, DFLASH_SELECTOR_BF16);
}
extern "C" __global__ void dkr_dflash_sample_bf16_bf16_f32(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const float *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_F32);
}
extern "C" __global__ void dkr_dflash_sample_bf16_bf16_bf16(const float *packed_topk, const __nv_bfloat16 *projected_hidden, const __nv_bfloat16 *predecessor_codebook, const __nv_bfloat16 *successor_codebook, const uint32_t *anchors, const float *inverse_temperatures, const float *uniforms, uint32_t *selected, uint32_t *candidate_ids, float *candidate_probs, int32_t batch, int32_t positions, int32_t rank, int32_t vocab, int32_t k, int32_t packed_width) {
  dflash_sample_select_kernel(packed_topk, projected_hidden, predecessor_codebook, successor_codebook, anchors, inverse_temperatures, uniforms, selected, candidate_ids, candidate_probs, batch, positions, rank, vocab, k, packed_width, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16, DFLASH_SELECTOR_BF16);
}
#elif DKR_SAMPLING_IMAGE == 3
#include "../../../mistralrs-core/src/cuda/speculative_rejection_device.cuh"
extern "C" __global__ void dkr_sparse_rejection_categorical_f32(const float *target_logits, const uint32_t *draft_tokens, const uint32_t *q_token_ids, const float *q_probs, const float *inverse_temperatures, const uint32_t *target_top_k, const float *top_p, const float *min_p, const float *accept_uniforms, const float *sample_uniforms, uint32_t *outcomes, int32_t batch, int32_t drafts, int32_t vocab, int32_t q_width) {
  sparse_rejection_categorical_f32_kernel(target_logits, draft_tokens, q_token_ids, q_probs, inverse_temperatures, target_top_k, top_p, min_p, accept_uniforms, sample_uniforms, outcomes, batch, drafts, vocab, q_width);
}
extern "C" __global__ void dkr_sparse_rejection_topk_f32(const float *packed_target, const uint32_t *draft_tokens, const uint32_t *q_token_ids, const float *q_probs, const float *inverse_temperatures, const uint32_t *target_top_k, const float *top_p, const float *min_p, const float *accept_uniforms, const float *sample_uniforms, uint32_t *outcomes, int32_t batch, int32_t drafts, int32_t vocab, int32_t q_width, int32_t packed_k) {
  sparse_rejection_topk_f32_kernel(packed_target, draft_tokens, q_token_ids, q_probs, inverse_temperatures, target_top_k, top_p, min_p, accept_uniforms, sample_uniforms, outcomes, batch, drafts, vocab, q_width, packed_k);
}
#else
#error "invalid sampling image"
#endif
