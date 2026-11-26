#define DKR_LFM3_IMAGE
#include "../../../mistralrs-quant/kernels/ops/lfm3_conv_device.cuh"
#include "../../../mistralrs-quant/kernels/ops/lfm3_spatial_conv_device.cuh"

extern "C" __global__ void dkr_lfm3_causal_f32_generic_nobias(
    const float* x, const float* weight, const float* state,
    float* output, float* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<float,0>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f32_generic_bias(
    const float* x, const float* weight, const float* state,
    const float* bias, float* output, float* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<float,0>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f32_width7_nobias(
    const float* x, const float* weight, const float* state,
    float* output, float* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<float,7>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f32_width7_bias(
    const float* x, const float* weight, const float* state,
    const float* bias, float* output, float* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<float,7>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_spatial_f32(
    const float* x, const float* weight, float* output, uint64_t elements,
    uint32_t height, uint32_t width, uint32_t merge, uint32_t channels, uint32_t kernel) {
    lfm3_spatial_conv_kernel<float>(x,weight,output,elements,height,width,merge,channels,kernel);
}

extern "C" __global__ void dkr_lfm3_causal_f16_generic_nobias(
    const __half* x, const __half* weight, const __half* state,
    __half* output, __half* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<__half,0>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f16_generic_bias(
    const __half* x, const __half* weight, const __half* state,
    const __half* bias, __half* output, __half* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<__half,0>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f16_width7_nobias(
    const __half* x, const __half* weight, const __half* state,
    __half* output, __half* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<__half,7>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_f16_width7_bias(
    const __half* x, const __half* weight, const __half* state,
    const __half* bias, __half* output, __half* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<__half,7>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_spatial_f16(
    const __half* x, const __half* weight, __half* output, uint64_t elements,
    uint32_t height, uint32_t width, uint32_t merge, uint32_t channels, uint32_t kernel) {
    lfm3_spatial_conv_kernel<__half>(x,weight,output,elements,height,width,merge,channels,kernel);
}

extern "C" __global__ void dkr_lfm3_causal_bf16_generic_nobias(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* state,
    __nv_bfloat16* output, __nv_bfloat16* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<__nv_bfloat16,0>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_bf16_generic_bias(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* state,
    const __nv_bfloat16* bias, __nv_bfloat16* output, __nv_bfloat16* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<__nv_bfloat16,0>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_bf16_width7_nobias(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* state,
    __nv_bfloat16* output, __nv_bfloat16* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,0,use_state,0}};
    gated_conv<__nv_bfloat16,7>(x,weight,nullptr,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_causal_bf16_width7_bias(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* state,
    const __nv_bfloat16* bias, __nv_bfloat16* output, __nv_bfloat16* next_state,
    uint64_t batch, uint64_t seq, uint64_t hidden, uint64_t width,
    uint64_t xs0, uint64_t xs1, uint64_t xs2,
    uint64_t ss0, uint64_t ss1, uint64_t ss2,
    uint64_t ws0, uint64_t ws2, uint64_t bs, uint64_t use_state) {
    const ConvParams params={{batch,seq,hidden,width,xs0,xs1,xs2,ss0,ss1,ss2,ws0,ws2,bs,use_state,1}};
    gated_conv<__nv_bfloat16,7>(x,weight,bias,state,output,next_state,params);
}

extern "C" __global__ void dkr_lfm3_spatial_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* output, uint64_t elements,
    uint32_t height, uint32_t width, uint32_t merge, uint32_t channels, uint32_t kernel) {
    lfm3_spatial_conv_kernel<__nv_bfloat16>(x,weight,output,elements,height,width,merge,channels,kernel);
}

