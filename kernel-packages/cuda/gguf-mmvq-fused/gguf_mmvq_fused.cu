// Recover exact historical fused MMVQ device bodies and indexed DP4A gather.
#if !defined(DKR_GGUF_MMVQ_FUSED_IMAGE) || DKR_GGUF_MMVQ_FUSED_IMAGE < 1 || DKR_GGUF_MMVQ_FUSED_IMAGE > 10
#error "select one fused MMVQ image"
#endif
#define DKR_GGUF_MMVQ_IMAGE DKR_GGUF_MMVQ_FUSED_IMAGE
#include "../../../mistralrs-quant/kernels/mmvq_gguf/mmvq_gguf.cu"
#define DKR_GGUF_INDEXED_DEVICE_ONLY 1
#include "../../../mistralrs-quant/kernels/indexed_moe/indexed_moe.cu"

#define INDEXED(tag,B,QK,QI,VDR,DOT) \
extern "C" __global__ void dkr_moe_indexed_##tag(const void* w,const void* y,const unsigned* ids,float* o, \
    int n,int k,int batch,int topk,int padded,int input_routes,int experts){ \
    indexed_moe_forward<QK,QI,B,VDR,DOT>(w,y,ids,o,n,k,batch,topk,padded,input_routes,experts); \
}
#define PREFETCH(tag,B,QK,QI,VDR,DOT,dt,T) \
extern "C" __global__ void __launch_bounds__(256,1) dkr_mmvq_fused_##tag##_##dt##_prefetch( \
 const void* q,const void* k,const void* v,const block_q8_1* y,T* qo,T* ko,T* vo, \
 int columns,int nq,int nk,int nv,int stride){ \
    int row=blockIdx.x;const void* w=q;T* out=qo; \
    if(row>=nq){row-=nq;w=k;out=ko;if(row>=nk){row-=nk;w=v;out=vo;if(row>=nv)return;}} \
    prefetch_row<T,QK,QI,B,VDR,DOT>(static_cast<const B*>(w)+(size_t)row*(columns/QK),y,out+row,columns); \
}
#define PREFETCH_SET(tag,B,QK,QI,VDR,DOT) \
PREFETCH(tag,B,QK,QI,VDR,DOT,f16,half) \
PREFETCH(tag,B,QK,QI,VDR,DOT,bf16,__nv_bfloat16) \
PREFETCH(tag,B,QK,QI,VDR,DOT,f32,float)
#define BOTH(tag,B,QK,QI,VDR,DOT,dt,T) \
MMVQ_FUSED_GLU_BATCH_SET(tag,B,QK,QI,VDR,DOT,dt,T) \
MMVQ_FUSED_QKV_BATCH_SET(tag,B,QK,QI,VDR,DOT,dt,T)
#define FORMAT(tag,B,QK,QI,VDR,DOT) \
BOTH(tag,B,QK,QI,VDR,DOT,f16,half) \
BOTH(tag,B,QK,QI,VDR,DOT,bf16,__nv_bfloat16) \
BOTH(tag,B,QK,QI,VDR,DOT,f32,float) \
INDEXED(tag,B,QK,QI,VDR,DOT)
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 1
FORMAT(q4_0,block_q4_0,QK4_0,QI4_0,VDR_Q4_0_Q8_1_MMVQ,vec_dot_q4_0_q8_1)
PREFETCH_SET(q4_0,block_q4_0,QK4_0,QI4_0,VDR_Q4_0_Q8_1_MMVQ,vec_dot_q4_0_q8_1)
INDEXED(q8_1,block_q8_1,QK8_1,QI8_1,VDR_Q8_1_Q8_1_MMVQ,vec_dot_q8_1_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 2
FORMAT(q4_1,block_q4_1,QK4_1,QI4_1,VDR_Q4_1_Q8_1_MMVQ,vec_dot_q4_1_q8_1)
PREFETCH_SET(q4_1,block_q4_1,QK4_1,QI4_1,VDR_Q4_1_Q8_1_MMVQ,vec_dot_q4_1_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 3
FORMAT(q5_0,block_q5_0,QK5_0,QI5_0,VDR_Q5_0_Q8_1_MMVQ,vec_dot_q5_0_q8_1)
PREFETCH_SET(q5_0,block_q5_0,QK5_0,QI5_0,VDR_Q5_0_Q8_1_MMVQ,vec_dot_q5_0_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 4
FORMAT(q5_1,block_q5_1,QK5_1,QI5_1,VDR_Q5_1_Q8_1_MMVQ,vec_dot_q5_1_q8_1)
PREFETCH_SET(q5_1,block_q5_1,QK5_1,QI5_1,VDR_Q5_1_Q8_1_MMVQ,vec_dot_q5_1_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 5
FORMAT(q8_0,block_q8_0,QK8_0,QI8_0,VDR_Q8_0_Q8_1_MMVQ,vec_dot_q8_0_q8_1)
PREFETCH_SET(q8_0,block_q8_0,QK8_0,QI8_0,VDR_Q8_0_Q8_1_MMVQ,vec_dot_q8_0_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 6
FORMAT(q2_k,block_q2_K,QK_K,QI2_K,VDR_Q2_K_Q8_1_MMVQ,vec_dot_q2_K_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 7
FORMAT(q3_k,block_q3_K,QK_K,QI3_K,VDR_Q3_K_Q8_1_MMVQ,vec_dot_q3_K_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 8
FORMAT(q4_k,block_q4_K,QK_K,QI4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1)
PREFETCH_SET(q4_k,block_q4_K,QK_K,QI4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 9
FORMAT(q5_k,block_q5_K,QK_K,QI5_K,VDR_Q5_K_Q8_1_MMVQ,vec_dot_q5_K_q8_1)
PREFETCH_SET(q5_k,block_q5_K,QK_K,QI5_K,VDR_Q5_K_Q8_1_MMVQ,vec_dot_q5_K_q8_1)
#endif
#if DKR_GGUF_MMVQ_FUSED_IMAGE == 10
FORMAT(q6_k,block_q6_K,QK_K,QI6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1)
PREFETCH_SET(q6_k,block_q6_K,QK_K,QI6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1)
#endif
