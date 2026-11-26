#include "../../../mistralrs-core/src/cuda/dynamic_conv_device.cuh"

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_f16_k2(const __half *hidden, const __half *dynamic, const __half *base, __half *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<__half, 2>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_f16_generic(const __half *hidden, const __half *dynamic, const __half *base, __half *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<__half, 0>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_bf16_k2(const __nv_bfloat16 *hidden, const __nv_bfloat16 *dynamic, const __nv_bfloat16 *base, __nv_bfloat16 *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<__nv_bfloat16, 2>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_bf16_generic(const __nv_bfloat16 *hidden, const __nv_bfloat16 *dynamic, const __nv_bfloat16 *base, __nv_bfloat16 *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<__nv_bfloat16, 0>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_f32_k2(const float *hidden, const float *dynamic, const float *base, float *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<float, 2>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

extern "C" __global__ __launch_bounds__(256) void dkr_dynamic_conv_f32_generic(const float *hidden, const float *dynamic, const float *base, float *output, int batch, int sequence_length, int hidden_size, int group_size, int kernel_size, int64_t hidden_stride_b, int64_t hidden_stride_s, int64_t hidden_stride_h, int64_t dynamic_stride_b, int64_t dynamic_stride_s, int64_t dynamic_stride_k, int64_t dynamic_stride_g, int64_t base_stride_k, int64_t base_stride_h) {
    dynamic_conv_body<float, 0>(hidden, dynamic, base, output, batch, sequence_length, hidden_size, group_size, kernel_size, hidden_stride_b, hidden_stride_s, hidden_stride_h, dynamic_stride_b, dynamic_stride_s, dynamic_stride_k, dynamic_stride_g, base_stride_k, base_stride_h);
}

