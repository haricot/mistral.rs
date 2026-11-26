#include "speculative_rejection_device.cuh"

extern "C" void sparse_rejection_categorical_f32(
    const float *target_logits, const uint32_t *draft_tokens,
    const uint32_t *q_token_ids, const float *q_probs,
    const float *inverse_temperatures, const uint32_t *target_top_k,
    const float *top_p, const float *min_p, const float *accept_uniforms,
    const float *sample_uniforms, uint32_t *outcomes, int batch, int drafts,
    int vocab, int q_width, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sparse_rejection_categorical_f32_kernel
      <<<batch, SPEC_REJECTION_BLOCK_SIZE, 0, cuda_stream>>>(
          target_logits, draft_tokens, q_token_ids, q_probs,
          inverse_temperatures, target_top_k, top_p, min_p, accept_uniforms,
          sample_uniforms, outcomes, batch, drafts, vocab, q_width);
}

extern "C" void sparse_rejection_topk_f32(
    const float *packed_target, const uint32_t *draft_tokens,
    const uint32_t *q_token_ids, const float *q_probs,
    const float *inverse_temperatures, const uint32_t *target_top_k,
    const float *top_p, const float *min_p, const float *accept_uniforms,
    const float *sample_uniforms, uint32_t *outcomes, int batch, int drafts,
    int vocab, int q_width, int packed_k, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  sparse_rejection_topk_f32_kernel<<<batch, 1, 0, cuda_stream>>>(
      packed_target, draft_tokens, q_token_ids, q_probs,
      inverse_temperatures, target_top_k, top_p, min_p, accept_uniforms,
      sample_uniforms, outcomes, batch, drafts, vocab, q_width, packed_k);
}
