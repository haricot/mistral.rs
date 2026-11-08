extern "C" __global__ void linear_simple(
    const float* x, const float* w, float* y, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) return;
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += x[row * k + i] * w[col * k + i];
    y[row * n + col] = sum;
}
