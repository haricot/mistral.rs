// Typed operation names select the pointee dtypes. The table itself is
// I64 storage owned and assembled by the internal guarded Rust adapter.
#define DKR_GDN_TRANSACTION_IMAGE
#include "../../../mistralrs-core/src/cuda/gdn_transition_device.cuh"

#define COMMIT(T,S,DT,SD) \
extern "C" __global__ void dkr_gdn_tx_commit_##DT##_##SD(const uint64_t* table,const uint32_t* keep,const uint32_t* slots,int layers,int batch,int seq,int kh,int vh,int kd,int vd,int cd,int cw,int tiled,int vm,int capacity){ \
    gdn_speculative_transition_commit_batched_kernel<T,S>(table,keep,slots,layers,batch,seq,kh,vh,kd,vd,cd,cw,tiled,vm,capacity); \
}
#define APPLY(T,S,DT,SD) \
extern "C" __global__ void dkr_gdn_tx_apply_##DT##_##SD(const uint64_t* table,const uint32_t* slots,int layers,int batch,int maxrows,int capacity,int kh,int vh,int kd,int vd,int cd,int cw,int tiled,int blocks){ \
    gdn_pending_transition_apply_batched_kernel<T,S>(table,slots,layers,batch,maxrows,capacity,kh,vh,kd,vd,cd,cw,tiled,blocks); \
}
#define STATES(M,T,DT) M(T,__half,DT,f16) M(T,__nv_bfloat16,DT,bf16) M(T,float,DT,f32)
STATES(COMMIT,__half,f16)
STATES(COMMIT,__nv_bfloat16,bf16)
STATES(APPLY,__half,f16)
STATES(APPLY,__nv_bfloat16,bf16)
#define STAGE(T,DT) \
extern "C" __global__ void dkr_gdn_tx_stage_##DT(const uint64_t* table,const uint32_t* keep,const uint32_t* slots,int layers,int batch,int seq,int maxrows,int capacity,int kh,int vh,int kd,int vd,int cd){ \
    gdn_speculative_transition_stage_batched_kernel<T>(table,keep,slots,layers,batch,seq,maxrows,capacity,kh,vh,kd,vd,cd); \
}
STAGE(__half,f16)
STAGE(__nv_bfloat16,bf16)
extern "C" __global__ void dkr_gdn_tx_publish(const uint64_t* table,const uint32_t* keep,const uint32_t* slots,int layers,int batch,int maxrows,int capacity){
    gdn_pending_transition_publish_batched_kernel(table,keep,slots,layers,batch,maxrows,capacity);
}
