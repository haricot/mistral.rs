// Typed wrapper for the original GGUF projected sigmoid-weighted stream mix.
#define DKR_GGUF_GATED_MIX_IMAGE 1
#include "../../../mistralrs-quant/kernels/gguf/gated_mix.cu"
#define SILU(dt,T) \
extern "C" __global__ void dkr_gguf_mix_silu_##dt(const T* x,block_q8_1* y,int k,int padded,float divisor){ \
    quantize_silu_q8_1<T>(x,y,k,padded,divisor); \
}
SILU(f16,half) SILU(bf16,__nv_bfloat16) SILU(f32,float)
#define MIX(tag,B,QK,QI,VDR,Dot,dt,T) \
extern "C" __global__ void dkr_gguf_mix_##tag##_##dt(const B* w,const block_q8_1* q,const T* r,T* o,int rank,int hidden,int streams){ \
    gated_mix_kernel<T,B,QK,QI,VDR,Dot>(w,q,r,o,rank,hidden,streams); \
}
#define FORMAT(tag,rust,code,qk,qi,B,vdr,dot) \
MIX(tag,B,qk,qi,vdr,dot,f16,half) \
MIX(tag,B,qk,qi,vdr,dot,bf16,__nv_bfloat16) \
MIX(tag,B,qk,qi,vdr,dot,f32,float)
GGUF_VECTOR_FORMATS(FORMAT)
