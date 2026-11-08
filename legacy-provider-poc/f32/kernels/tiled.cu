constexpr int TILE = 16;
extern "C" __global__ void linear_tiled(
    const float* x, const float* w, float* y, int m, int n, int k) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    float sum = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        xs[ty][tx] = row < m && base + tx < k ? x[row * k + base + tx] : 0.0f;
        const int weight_row = blockIdx.x * TILE + ty;
        ws[ty][tx] = weight_row < n && base + tx < k ? w[weight_row * k + base + tx] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) sum += xs[ty][i] * ws[tx][i];
        __syncthreads();
    }
    if (row < m && col < n) y[row * n + col] = sum;
}
