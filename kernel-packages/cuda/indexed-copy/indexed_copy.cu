#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
__device__ __forceinline__ void copy_row(
    const T *src, T *dst, const uint32_t *indices,
    uint32_t source_rows, uint32_t destination_rows, uint64_t row_elements) {
  const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t total = uint64_t(source_rows) * row_elements;
  if (index >= total) return;
  const uint64_t source_row = index / row_elements;
  const uint32_t destination_row = indices[source_row];
  if (destination_row >= destination_rows) return;
  dst[uint64_t(destination_row) * row_elements + index % row_elements] = src[index];
}

extern "C" __global__ void dkr_indexed_copy_u16(
    const uint16_t *src, uint16_t *dst, const uint32_t *indices,
    uint32_t source_rows, uint32_t destination_rows, uint64_t row_elements) {
  copy_row(src, dst, indices, source_rows, destination_rows, row_elements);
}

extern "C" __global__ void dkr_indexed_copy_u32(
    const uint32_t *src, uint32_t *dst, const uint32_t *indices,
    uint32_t source_rows, uint32_t destination_rows, uint64_t row_elements) {
  copy_row(src, dst, indices, source_rows, destination_rows, row_elements);
}
