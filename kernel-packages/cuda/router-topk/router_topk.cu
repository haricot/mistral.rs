#define DKR_ROUTER_TOPK_EXTERNAL
#include "../../../mistralrs-core/src/cuda/router_topk_device.cuh"

#ifndef DKR_ROUTER_TOPK_IMAGE
#error "select exactly one router/top-k image"
#endif
#if DKR_ROUTER_TOPK_IMAGE == 1
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 1, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 1, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 1, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 1, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 1, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 1, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 1, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 1, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 1, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 1, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 1, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e1_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 1, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 2
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 2, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 2, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 2, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 2, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 2, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 2, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 2, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 2, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 2, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 2, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 2, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e2_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 2, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 4
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 4, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 4, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 4, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 4, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 4, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 4, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 4, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 4, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 4, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 4, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 4, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e4_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 4, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 8
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 8, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 8, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 8, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 8, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 8, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 8, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 8, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 8, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 8, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 8, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 8, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e8_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 8, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 16
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 16, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 16, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 16, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 16, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 16, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 16, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 16, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 16, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 16, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 16, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 16, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e16_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 16, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 32
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 32, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 32, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 32, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 32, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 32, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 32, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 32, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 32, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 32, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 32, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 32, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e32_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 32, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 64
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 64, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 64, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 64, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 64, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 64, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 64, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 64, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 64, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 64, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 64, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 64, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e64_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 64, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 128
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 128, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 128, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 128, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 128, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 128, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 128, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 128, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 128, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 128, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 128, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 128, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e128_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 128, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 256
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 256, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 256, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 256, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 256, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 256, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 256, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 256, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 256, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 256, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 256, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 256, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e256_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 256, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 512
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 512, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 512, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 512, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 512, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 512, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 512, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 512, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 512, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 512, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 512, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 512, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e512_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 512, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 576
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f32_b0_s0(const float *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 576, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f32_b0_s1(const float *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 576, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f32_b1_s0(const float *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 576, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f32_b1_s1(const float *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<float, 576, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f16_b0_s0(const __half *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 576, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f16_b0_s1(const __half *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 576, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f16_b1_s0(const __half *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 576, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_f16_b1_s1(const __half *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__half, 576, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_bf16_b0_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 576, false, false>(logits, weights, ids, nullptr, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_bf16_b0_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 576, false, true>(logits, weights, ids, nullptr, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_bf16_b1_s0(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 576, true, false>(logits, weights, ids, bias, nullptr, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
extern "C" __global__ __launch_bounds__(128, 1) void dkr_moe_router_e576_bf16_b1_s1(const __nv_bfloat16 *logits, float *weights, uint32_t *ids, const float *bias, const float *scale, int32_t rows, int32_t k, int32_t score_mode, int32_t weight_mode, uint32_t renormalize, uint32_t clamp_logits, float clip_min, float clip_max, float norm_min, float output_scale) {
  moe_router_topk_kernel<__nv_bfloat16, 576, true, true>(logits, weights, ids, bias, scale, rows, k, score_mode, weight_mode, renormalize != 0, clamp_logits != 0, clip_min, clip_max, norm_min, output_scale);
}
#endif

#if DKR_ROUTER_TOPK_IMAGE == 0
extern "C" __global__ void dkr_small_topk_f32_b64(const float *input, float *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<float>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f32_b128(const float *input, float *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<float>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f32_b256(const float *input, float *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<float>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f32_b512(const float *input, float *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<float>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f16_b64(const __half *input, __half *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__half>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f16_b128(const __half *input, __half *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__half>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f16_b256(const __half *input, __half *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__half>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_f16_b512(const __half *input, __half *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__half>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_bf16_b64(const __nv_bfloat16 *input, __nv_bfloat16 *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__nv_bfloat16>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_bf16_b128(const __nv_bfloat16 *input, __nv_bfloat16 *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__nv_bfloat16>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_bf16_b256(const __nv_bfloat16 *input, __nv_bfloat16 *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__nv_bfloat16>(input, values, indices, rows, cols, k);
}
extern "C" __global__ void dkr_small_topk_bf16_b512(const __nv_bfloat16 *input, __nv_bfloat16 *values, uint32_t *indices, int32_t rows, int32_t cols, int32_t k) {
  topk_kernel<__nv_bfloat16>(input, values, indices, rows, cols, k);
}
#endif
