#include <stdint.h>

extern "C" __global__ void dkr_pad_decode_input_u32(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    int input_rows, int output_rows, int width) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = int64_t(output_rows) * width;
  if (index >= elements) return;
  const int row = int(index / width);
  const int col = int(index - int64_t(row) * width);
  output[index] = input[int64_t(row < input_rows ? row : 0) * width + col];
}
