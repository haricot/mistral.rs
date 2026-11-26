// Keep the historical selection policy in Rust, and the device math here.
#define DKR_GDN_ENTRY __device__ __forceinline__
#define DKR_GDN_WARP_ENTRY(N) __device__ __forceinline__
#include "../../../mistralrs-core/src/cuda/gdn_recurrence_conv_device.cuh"
#define DKR_GDN_DECODE_IMAGE
#include "../../../mistralrs-core/src/cuda/gdn_decode_device.cuh"

#define ARGS(T,S) const T* qkv,const T* b,const T* a,const float* al,const float* db,S* state,T* out,const int32_t* slots,int batch,int kh,int vh,int kd,int vd,int tiled,int64_t bs,int64_t bh,int64_t as,int64_t ah,int has
#define COMMON qkv,b,a,al,db,state,out,batch,kh,vh
#define TAIL tiled,bs,bh,as,ah,(has?slots:nullptr)
#define ENTRY(DT,SD,NAME) dkr_gdn_decode_##DT##_##SD##_##NAME
#define BASE(T,S,DT,SD) \
extern "C" __global__ void ENTRY(DT,SD,k64)(ARGS(T,S)){gdn_decode_recurrence_kernel<T,S,64,64>(COMMON,vd,TAIL);} \
extern "C" __global__ void ENTRY(DT,SD,k128)(ARGS(T,S)){gdn_decode_recurrence_kernel<T,S,128,64>(COMMON,vd,TAIL);} \
extern "C" __global__ void ENTRY(DT,SD,fallback)(ARGS(T,S)){gdn_decode_recurrence_kernel_fallback<T,S,64,256>(COMMON,kd,vd,TAIL);}
#define STATES(M,T,DT) M(T,__half,DT,f16) M(T,__nv_bfloat16,DT,bf16) M(T,float,DT,f32)
STATES(BASE,__half,f16)
STATES(BASE,__nv_bfloat16,bf16)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define OPT(T,S,DT,SD) \
extern "C" __global__ void ENTRY(DT,SD,cooperative)(ARGS(T,S)){gdn_decode_recurrence_kernel_cooperative<T,S>(COMMON,vd,TAIL);} \
extern "C" __global__ void ENTRY(DT,SD,pipelined)(ARGS(T,S)){gdn_decode_recurrence_kernel_pipelined<T,S>(COMMON,vd,TAIL);}
STATES(OPT,__half,f16)
STATES(OPT,__nv_bfloat16,bf16)
#define VM(T,S,DT,SD) \
extern "C" __global__ __launch_bounds__(32) void ENTRY(DT,SD,vm4)(ARGS(T,S)){gdn_decode_recurrence_kernel_value_major<T,S,4>(COMMON,TAIL);} \
extern "C" __global__ __launch_bounds__(32) void ENTRY(DT,SD,vm32)(ARGS(T,S)){gdn_decode_recurrence_kernel_value_major<T,S,32>(COMMON,TAIL);}
STATES(VM,__nv_bfloat16,bf16)
#endif
