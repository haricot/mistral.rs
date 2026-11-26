// Historical row-Q8_1 grouped GEMM, sorted vs weighted output kept distinct.
// Input metadata is produced by typed dispatch; validate bounded reads on device.
#define DKR_GGUF_GROUPED_IMAGE 1
#include "../../../mistralrs-quant/kernels/moe_grouped/moe_grouped.cu"
__device__ __forceinline__ bool dkr_group_valid(const int32_t* bounds,const int32_t* ids,int experts,int routes,int topk,int input_dim1) {
    if(blockIdx.y>=experts)return false;
    const int lo=bounds[blockIdx.y], hi=bounds[blockIdx.y+1];
    if(lo<0||hi<lo||hi>routes)return false;
    __shared__ int bad;
    const int tid=threadIdx.y*32+threadIdx.x;
    if(tid==0)bad=0;
    __syncthreads();
    for(int i=lo+tid;i<hi;i+=256)if(ids[i]<0||ids[i]>=routes)atomicExch(&bad,1);
    __syncthreads();
    return !bad;
}
#define ARGS const void* w,const void* q,const int32_t* bounds,const int32_t* ids,const float* routing,float* out,int n,int k,int padded,int experts,int topk,int input_dim1,int routes
#define CHECK if(!dkr_group_valid(bounds,ids,experts,routes,topk,input_dim1))return;
#define ENTRY(tag,B,QK,QI,VDR,DOT,KTB,mode,WEIGHTED) \
extern "C" __global__ void dkr_grouped_##tag##_##mode(ARGS){ \
    CHECK \
    moe_tiled_gemm_impl<QK,QI,B,VDR,DOT,KTB>(w,q,bounds,ids,WEIGHTED?routing:nullptr,out,n,k,padded,experts,topk,input_dim1); \
}
#define FORMAT(tag,B,QK,QI,VDR,DOT,KTB) \
ENTRY(tag,B,QK,QI,VDR,DOT,KTB,sorted,false) \
ENTRY(tag,B,QK,QI,VDR,DOT,KTB,weighted,true)
FORMAT(q4_0,block_q4_0,QK4_0,QI4_0,VDR_Q4_0_Q8_1_MMVQ,vd_q4_0_q8_1,8)
FORMAT(q4_1,block_q4_1,QK4_1,QI4_1,VDR_Q4_1_Q8_1_MMVQ,vd_q4_1_q8_1,8)
FORMAT(q5_0,block_q5_0,QK5_0,QI5_0,VDR_Q5_0_Q8_1_MMVQ,vd_q5_0_q8_1,8)
FORMAT(q5_1,block_q5_1,QK5_1,QI5_1,VDR_Q5_1_Q8_1_MMVQ,vd_q5_1_q8_1,8)
FORMAT(q8_1,block_q8_1,QK8_1,QI8_1,VDR_Q8_1_Q8_1_MMVQ,vd_q8_1_q8_1,8)
FORMAT(q2_k,block_q2_K,QK_K,QI2_K,VDR_Q2_K_Q8_1_MMVQ,vd_q2_K_q8_1,1)
FORMAT(q3_k,block_q3_K,QK_K,QI3_K,VDR_Q3_K_Q8_1_MMVQ,vd_q3_K_q8_1,1)
FORMAT(q4_k,block_q4_K,QK_K,QI4_K,VDR_Q4_K_Q8_1_MMVQ,vd_q4_K_q8_1,1)
FORMAT(q5_k,block_q5_K,QK_K,QI5_K,VDR_Q5_K_Q8_1_MMVQ,vd_q5_K_q8_1,1)
FORMAT(q6_k,block_q6_K,QK_K,QI6_K,VDR_Q6_K_Q8_1_MMVQ,vd_q6_K_q8_1,1)
extern "C" __global__ void dkr_grouped_q8_0_sorted(ARGS){
    CHECK
    moe_tiled_gemm_q8_0_impl(w,q,bounds,ids,nullptr,out,n,k,padded,experts,topk,input_dim1);
}
extern "C" __global__ void dkr_grouped_q8_0_weighted(ARGS){
    CHECK
    moe_tiled_gemm_q8_0_impl(w,q,bounds,ids,routing,out,n,k,padded,experts,topk,input_dim1);
}
