// Typed entries; CUDA math is shared with the historical producer.
#include <cuda_runtime.h>
#include <cuda_fp8.h>
using gdn_fp8_e4m3 = __nv_fp8_e4m3;
#define DKR_GDN_NORM_IMAGE
#define DKR_GDN_PREPARATION_IMAGE
#include "../../../mistralrs-core/src/cuda/gdn_preparation_device.cuh"

#define RAGGED(T,DT) \
extern "C" __global__ void dkr_gdn_pack_##DT(const T* src,T* out,const uint32_t* slots,uint64_t total,int padded,int width,int64_t stride,float pad){ \
    gdn_packed_to_padded_kernel<T>(src,out,slots,total,padded,width,stride,pad); \
} \
extern "C" __global__ void dkr_gdn_unpack_##DT(const T* src,T* out,const uint32_t* slots,int width,int64_t sb,int64_t st,int64_t sf,int inner){ \
    gdn_padded_to_packed_kernel<T>(src,out,slots,width,sb,st,sf,inner); \
} \
extern "C" __global__ void dkr_gdn_extract_##DT(const T* src,const T* initial,T* out,const uint32_t* slots,uint64_t total,int padded,int channels,int width,int64_t sb,int64_t st){ \
    gdn_extract_ragged_conv_state_kernel<T>(src,initial,out,slots,total,padded,channels,width,sb,st); \
}
RAGGED(__nv_bfloat16,bf16)
RAGGED(float,f32)

#define PREP(T,O,DT,ODT) \
extern "C" __global__ void dkr_gdn_prepare_##DT##_##ODT(const T* qkv,const T* b,const T* a,const float* al,const float* db,O* q,O* k,O* v,float* g,float* beta,int batch,int seq,int stride,int kh,int vh,int kd,int vd,int tiled){ \
    gdn_prepare_recurrence_kernel<T,O>(qkv,b,a,al,db,q,k,v,g,beta,batch,seq,stride,kh,vh,kd,vd,tiled); \
}
PREP(__half,float,f16,f32)
PREP(__half,__nv_bfloat16,f16,bf16)
PREP(__nv_bfloat16,float,bf16,f32)
PREP(__nv_bfloat16,__nv_bfloat16,bf16,bf16)
#define GATE(T,DT) \
extern "C" __global__ void dkr_gdn_gating_##DT(const T* b,const T* a,const float* al,const float* db,T* beta,T* g,int total,int heads){ \
    fused_gdn_gating_kernel<T>(b,a,al,db,beta,g,total,heads); \
}
GATE(__half,f16)
GATE(__nv_bfloat16,bf16)
#define FP8PTR const __nv_bfloat16* x,const __nv_bfloat16* gate,const __nv_bfloat16* weight,gdn_fp8_e4m3* out,float* scales
extern "C" __global__ void dkr_gdn_fp8_warp(FP8PTR,int rows,int groups,int sm,int layout,int d1,int d2,int64_t x0,int64_t x1,int64_t x2,int64_t x3,int64_t g0,int64_t g1,int64_t g2,int64_t g3,float eps,int sigmoid){
    gdn_rmsnorm_gated_quantized_warp_kernel<8>(x,gate,weight,out,scales,rows,groups,sm,layout,d1,d2,x0,x1,x2,x3,g0,g1,g2,g3,eps,sigmoid);
}
extern "C" __global__ void dkr_gdn_fp8_rows4(FP8PTR,int rows,int groups,int sm,int layout,float eps,int sigmoid){
    gdn_rmsnorm_gated_quantized_rows4_kernel(x,gate,weight,out,scales,rows,groups,sm,layout,eps,sigmoid);
}
