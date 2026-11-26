// Existing device bodies, flat typed ABI. No host launcher is included.
#if !defined(DKR_GGUF_RESIDUAL_IMAGE) || DKR_GGUF_RESIDUAL_IMAGE<1 || DKR_GGUF_RESIDUAL_IMAGE>3
#error "choose residual image 1..3"
#endif
#if DKR_GGUF_RESIDUAL_IMAGE <= 2
#define DKR_GGUF_INDEXED_DEVICE_ONLY 1
#define DKR_GGUF_MOE_DECODE_IMAGE 1
#include "../../../mistralrs-quant/kernels/indexed_moe/indexed_moe.cu"
#define LORA(tag,B,QK,QI,VDR,DOT,dt,TYPE) \
extern "C" __global__ void dkr_lora_gate_##tag##_##dt(const void* w,const void* up,const void* q,const unsigned* ids,void* out,int n,int k,int batch,int topk,int experts,int padded){ \
    moe_gemv_gate_up_pair_impl<QK,QI,B,VDR,DOT>(w,up,q,ids,out,n,k,batch,topk,experts,padded,TYPE); \
} \
extern "C" __global__ void dkr_lora_down_##tag##_##dt(const void* w,const void* q,const unsigned* ids,void* out,int n,int k,int batch,int topk,int experts,int padded){ \
    moe_gemv_lora_down_impl<QK,QI,B,VDR,DOT>(w,q,ids,out,n,k,batch,topk,experts,padded,TYPE); \
}
#define FORMAT(tag,B,QK,QI,VDR,DOT) \
LORA(tag,B,QK,QI,VDR,DOT,f32,MOE_OUTPUT_F32) \
LORA(tag,B,QK,QI,VDR,DOT,f16,MOE_OUTPUT_F16) \
LORA(tag,B,QK,QI,VDR,DOT,bf16,MOE_OUTPUT_BF16)
#if DKR_GGUF_RESIDUAL_IMAGE == 1
FORMAT(q4_0,block_q4_0,QK4_0,QI4_0,VDR_Q4_0_Q8_1_MMVQ,vec_dot_q4_0_q8_1)
FORMAT(q4_1,block_q4_1,QK4_1,QI4_1,VDR_Q4_1_Q8_1_MMVQ,vec_dot_q4_1_q8_1)
FORMAT(q5_0,block_q5_0,QK5_0,QI5_0,VDR_Q5_0_Q8_1_MMVQ,vec_dot_q5_0_q8_1)
FORMAT(q5_1,block_q5_1,QK5_1,QI5_1,VDR_Q5_1_Q8_1_MMVQ,vec_dot_q5_1_q8_1)
FORMAT(q8_0,block_q8_0,QK8_0,QI8_0,VDR_Q8_0_Q8_1_MMVQ,vec_dot_q8_0_q8_1)
FORMAT(q8_1,block_q8_1,QK8_1,QI8_1,VDR_Q8_1_Q8_1_MMVQ,vec_dot_q8_1_q8_1)
#else
FORMAT(q2_k,block_q2_K,QK_K,QI2_K,VDR_Q2_K_Q8_1_MMVQ,vec_dot_q2_K_q8_1)
FORMAT(q3_k,block_q3_K,QK_K,QI3_K,VDR_Q3_K_Q8_1_MMVQ,vec_dot_q3_K_q8_1)
FORMAT(q4_k,block_q4_K,QK_K,QI4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1)
FORMAT(q5_k,block_q5_K,QK_K,QI5_K,VDR_Q5_K_Q8_1_MMVQ,vec_dot_q5_K_q8_1)
FORMAT(q6_k,block_q6_K,QK_K,QI6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1)
#endif
#else
#define DKR_GGUF_MMVQ_IMAGE 1
#define DKR_GGUF_MIXED_PAIR_IMAGE 1
#include "../../../mistralrs-quant/kernels/mmvq_gguf/mmvq_gguf.cu"
#define MIX(tag,B,QK,QI,VDR,DOT,dt,T,V,mode,P) \
extern "C" __global__ void __launch_bounds__(256,1) dkr_mixed_##tag##_##dt##_##mode( \
 const void* w,const void* q,const T* dense,const T* x,T* qo,T* dout,int columns,int qr,int dr){ \
    mmvq_dense_pair<T,V,QK,QI,B,VDR,DOT,P>(MixedMmvqArgs{w,q,dense,x,qo,dout,columns,qr,dr,int(P)}); \
}
#define DT(tag,B,QK,QI,VDR,DOT,dt,T,V) \
MIX(tag,B,QK,QI,VDR,DOT,dt,T,V,direct,false) \
MIX(tag,B,QK,QI,VDR,DOT,dt,T,V,prefetch,true)
#define FORMAT(tag,B,QK,QI,VDR,DOT) \
DT(tag,B,QK,QI,VDR,DOT,f16,half,half2) \
DT(tag,B,QK,QI,VDR,DOT,bf16,__nv_bfloat16,__nv_bfloat162) \
DT(tag,B,QK,QI,VDR,DOT,f32,float,float2)
FORMAT(q4_0,block_q4_0,QK4_0,QI4_0,VDR_Q4_0_Q8_1_MMVQ,vec_dot_q4_0_q8_1)
FORMAT(q4_1,block_q4_1,QK4_1,QI4_1,VDR_Q4_1_Q8_1_MMVQ,vec_dot_q4_1_q8_1)
FORMAT(q5_0,block_q5_0,QK5_0,QI5_0,VDR_Q5_0_Q8_1_MMVQ,vec_dot_q5_0_q8_1)
FORMAT(q5_1,block_q5_1,QK5_1,QI5_1,VDR_Q5_1_Q8_1_MMVQ,vec_dot_q5_1_q8_1)
FORMAT(q8_0,block_q8_0,QK8_0,QI8_0,VDR_Q8_0_Q8_1_MMVQ,vec_dot_q8_0_q8_1)
FORMAT(q2_k,block_q2_K,QK_K,QI2_K,VDR_Q2_K_Q8_1_MMVQ,vec_dot_q2_K_q8_1)
FORMAT(q3_k,block_q3_K,QK_K,QI3_K,VDR_Q3_K_Q8_1_MMVQ,vec_dot_q3_K_q8_1)
FORMAT(q4_k,block_q4_K,QK_K,QI4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1)
FORMAT(q5_k,block_q5_K,QK_K,QI5_K,VDR_Q5_K_Q8_1_MMVQ,vec_dot_q5_K_q8_1)
FORMAT(q6_k,block_q6_K,QK_K,QI6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1)
#endif
