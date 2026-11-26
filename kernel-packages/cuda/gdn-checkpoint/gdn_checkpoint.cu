#define DKR_GDN_ENTRY __device__ __forceinline__
#define DKR_GDN_WARP_ENTRY(N) __device__ __forceinline__
#include "../../../mistralrs-core/src/cuda/gdn_recurrence_conv_device.cuh"
#define DKR_GDN_CHECKPOINT_IMAGE
#include "../../../mistralrs-core/src/cuda/gdn_checkpoint_device.cuh"

#define COMMIT_CONV(T,DT) \
extern "C" __global__ void dkr_gdn_commit_conv_##DT(const T* x,const T* initial,T* pool,const uint32_t* keep,const uint32_t* slots,int batch,int seq,int cd,int cw,int cap){ \
    gdn_speculative_conv_state_commit_kernel<T>(x,initial,pool,keep,slots,batch,seq,cd,cw,cap); \
}
COMMIT_CONV(__half,f16)
COMMIT_CONV(__nv_bfloat16,bf16)
#define REC(T,S,DT,SD,VM,TAG) \
extern "C" __global__ void dkr_gdn_commit_rec_##DT##_##SD##_##TAG(const T* qkv,const T* b,const T* a,const S* initial,const float* al,const float* db,S* pool,const uint32_t* keep,const uint32_t* slots,int batch,int seq,int kh,int vh,int kd,int vd,int tiled,int cap){ \
    gdn_speculative_recurrent_state_commit_kernel<T,S,VM>(qkv,b,a,initial,al,db,pool,keep,slots,batch,seq,kh,vh,kd,vd,tiled,cap); \
}
#define LAYOUTS(T,S,DT,SD) REC(T,S,DT,SD,false,km) REC(T,S,DT,SD,true,vm)
#define STATES(T,DT) LAYOUTS(T,__half,DT,f16) LAYOUTS(T,__nv_bfloat16,DT,bf16) LAYOUTS(T,float,DT,f32)
STATES(__half,f16)
STATES(__nv_bfloat16,bf16)

#define CPTR(T) const T* x,const T* w,T* pool,T* out,const uint32_t* slots
#define PENDING(T) T* pc,const uint32_t* keep,const uint32_t* epoch,uint32_t* applied
#define CDIMS int batch,int seq,int cd,int cw,int lanes,int write,int cap
#define STRIDES int64_t sb,int64_t ss,int64_t sc
#define META int maxrows,int pcap,int blocks
#define CONV(T,DT) \
extern "C" __global__ void dkr_gdn_cp_conv_##DT(CPTR(T),CDIMS,STRIDES){ \
    gdn_speculative_conv_checkpoints_kernel<T>(x,w,pool,out,slots,batch,seq,cd,cw,lanes,write!=0,nullptr,nullptr,nullptr,nullptr,0,0,0,sb,ss,sc,cap); \
} \
extern "C" __global__ void dkr_gdn_cp_conv4_##DT(CPTR(T),CDIMS,STRIDES){ \
    gdn_speculative_conv_checkpoints_width4_kernel<T>(x,w,pool,out,slots,batch,seq,cd,lanes,write!=0,nullptr,nullptr,nullptr,nullptr,0,0,0,sb,ss,sc,cap); \
} \
extern "C" __global__ void dkr_gdn_cp_pending_##DT(CPTR(T),PENDING(T),CDIMS,META,STRIDES){ \
    gdn_speculative_conv_checkpoints_kernel<T>(x,w,pool,out,slots,batch,seq,cd,cw,lanes,write!=0,pc,keep,epoch,applied,maxrows,pcap,blocks,sb,ss,sc,cap); \
} \
extern "C" __global__ void dkr_gdn_cp_pending4_##DT(CPTR(T),PENDING(T),CDIMS,META,STRIDES){ \
    gdn_speculative_conv_checkpoints_width4_kernel<T>(x,w,pool,out,slots,batch,seq,cd,lanes,write!=0,pc,keep,epoch,applied,maxrows,pcap,blocks,sb,ss,sc,cap); \
}
CONV(__half,f16)
CONV(__nv_bfloat16,bf16)
