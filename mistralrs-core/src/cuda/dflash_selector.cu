#include "dflash_selector_device.cuh"

extern "C" void dflash_greedy_select(
    const float *packed_topk, const void *projected_hidden,
    const void *predecessor_codebook, const void *successor_codebook,
    const uint32_t *anchors, uint32_t *selected, int batch, int positions,
    int rank, int vocab, int k, int packed_width, int hidden_dtype,
    int predecessor_dtype, int successor_dtype, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dflash_greedy_select_kernel<<<batch, DFLASH_SELECTOR_BLOCK_SIZE, 0,
                                cuda_stream>>>(
      packed_topk, projected_hidden, predecessor_codebook, successor_codebook,
      anchors, selected, batch, positions, rank, vocab, k, packed_width,
      hidden_dtype, predecessor_dtype, successor_dtype);
}

extern "C" void dflash_sample_select(
    const float *packed_topk, const void *projected_hidden,
    const void *predecessor_codebook, const void *successor_codebook,
    const uint32_t *anchors, const float *inverse_temperatures,
    const float *uniforms, uint32_t *selected, uint32_t *candidate_ids,
    float *candidate_probs, int batch, int positions, int rank, int vocab,
    int k, int packed_width, int hidden_dtype, int predecessor_dtype,
    int successor_dtype, int64_t stream) {
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dflash_sample_select_kernel<<<batch, DFLASH_SELECTOR_BLOCK_SIZE, 0,
                                cuda_stream>>>(
      packed_topk, projected_hidden, predecessor_codebook, successor_codebook,
      anchors, inverse_temperatures, uniforms, selected, candidate_ids,
      candidate_probs, batch, positions, rank, vocab, k, packed_width,
      hidden_dtype, predecessor_dtype, successor_dtype);
}
