// Exact historic direct, L2-prefetch and aligned shared-stage MoE decode.
#if !defined(DKR_GGUF_MOE_DECODE_IMAGE) || DKR_GGUF_MOE_DECODE_IMAGE<1 || DKR_GGUF_MOE_DECODE_IMAGE>2
#error "choose a MoE decode image"
#endif
#define DKR_GGUF_INDEXED_DEVICE_ONLY 1
#include "../../../mistralrs-quant/kernels/indexed_moe/indexed_moe.cu"
#include "../../../mistralrs-quant/kernels/indexed_moe/moe_decode_prefetch.cu"
#define GATE_ARGS const void* w,const void* up,const block_q8_1* q,const unsigned* ids,float* out,int n,int k,int batch,int topk,int padded,int act,int experts
#define DOWN_ARGS const void* w,const block_q8_1* q,const unsigned* ids,const float* routing,float* out,int n,int k,int batch,int topk,int padded,int experts
#define GATE_STRUCT MoeDecodeArgs{w,up,q,ids,nullptr,out,n,k,topk,padded,act,experts}
#define DOWN_STRUCT MoeDecodeArgs{w,w,q,ids,routing,out,n,k,topk,padded,0,experts}
#define STAGED(tag,B,QK,QI,VDR,DOT,RPW,Bytes) \
extern "C" __global__ void dkr_moe_decode_gate_##tag##_stage##Bytes(GATE_ARGS){ \
    moe_decode_staged<QK,QI,B,VDR,DOT,true,RPW,Bytes>(GATE_STRUCT); \
} \
extern "C" __global__ void dkr_moe_decode_down_##tag##_stage##Bytes(DOWN_ARGS){ \
    moe_decode_staged<QK,QI,B,VDR,DOT,false,RPW,Bytes>(DOWN_STRUCT); \
}
#define FORMAT(tag,B,QK,QI,VDR,DOT) \
extern "C" __global__ void dkr_moe_decode_gate_##tag##_direct(GATE_ARGS){ \
    moe_gemv_fused_gate_up_impl<QK,QI,B,VDR,DOT>(w,up,q,ids,out,n,k,batch,topk,padded,act,experts); \
} \
extern "C" __global__ void dkr_moe_decode_down_##tag##_direct(DOWN_ARGS){ \
    moe_gemv_down_aggregate_impl<QK,QI,B,VDR,DOT>(w,q,ids,routing,out,n,k,batch,topk,padded,experts); \
} \
extern "C" __global__ void dkr_moe_decode_gate_##tag##_prefetch(GATE_ARGS){ \
    moe_decode_prefetch<QK,QI,B,VDR,DOT,true>(GATE_STRUCT); \
} \
extern "C" __global__ void dkr_moe_decode_down_##tag##_prefetch(DOWN_ARGS){ \
    moe_decode_prefetch<QK,QI,B,VDR,DOT,false>(DOWN_STRUCT); \
} \
STAGED(tag,B,QK,QI,VDR,DOT,1,16) \
STAGED(tag,B,QK,QI,VDR,DOT,1,8) \
STAGED(tag,B,QK,QI,VDR,DOT,2,4)
#if DKR_GGUF_MOE_DECODE_IMAGE==1
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
