// Reuse the exact historical device bodies, never the host launchers.
#define DKR_GDN_ENTRY __device__ __forceinline__
#define DKR_GDN_WARP_ENTRY(N) __device__ __forceinline__
#include "../../../mistralrs-core/src/cuda/gdn_recurrence_conv_device.cuh"

#if DKR_GDN_DTYPE == 1
using State = __half;
#define DT f16
#elif DKR_GDN_DTYPE == 2
using State = __nv_bfloat16;
#define DT bf16
#elif DKR_GDN_DTYPE == 3
using State = float;
#define DT f32
#else
#error invalid DKR_GDN_DTYPE
#endif
#define NAME_I(kind,dt,variant) dkr_gdn_##kind##_##dt##_##variant
#define NAME(kind,dt,variant) NAME_I(kind,dt,variant)
#define RARGS const float* q,const float* k,const float* v,const float* g,const float* beta,State* state,float* out,const int32_t* slots,int seq,int kd,int vd,int heads,int has_slots
#define RP q,k,v,g,beta,state,out
#define SLOTS (has_slots ? slots : nullptr)
#define TILED(tag,K) extern "C" __global__ void NAME(recurrence,DT,tag)(RARGS){gated_delta_rule_recurrence_kernel_tiled<State,K,64>(RP,seq,vd,SLOTS,heads);}
#define WARP(tag,K,VM) extern "C" __global__ __launch_bounds__(128,2) void NAME(recurrence,DT,tag)(RARGS){gated_delta_rule_recurrence_kernel_warp<State,K,4,VM>(RP,seq,vd,SLOTS,heads);}
#define GROUP(tag,N) extern "C" __global__ __launch_bounds__(128,2) void NAME(recurrence,DT,tag)(RARGS){gated_delta_rule_recurrence_kernel_vmajor_grouped<State,N,4>(RP,seq,SLOTS,heads);}
#define CHUNK(tag,BT,BK,VM) extern "C" __global__ void NAME(recurrence,DT,tag)(RARGS){chunked_gated_delta_rule_kernel<State,BT,BK,64,VM>(RP,seq,vd,SLOTS,heads);}
TILED(scalar64,64)
TILED(scalar128,128)
extern "C" __global__ void NAME(recurrence,DT,fallback)(RARGS){gated_delta_rule_recurrence_kernel_fallback<State,64,256>(RP,seq,kd,vd,SLOTS,heads);}
WARP(warp64,64,false)
WARP(warp128,128,false)
WARP(vm1,128,true)
GROUP(vm2,2)
GROUP(vm4,4)
GROUP(vm8,8)
CHUNK(chunk32k64,32,64,false)
CHUNK(chunk64k64,64,64,false)
CHUNK(chunk32k128,32,128,false)
CHUNK(vmchunk32,32,128,true)
#if DKR_GDN_LARGE_CHUNK
CHUNK(chunk64k128,64,128,false)
CHUNK(vmchunk64,64,128,true)
#endif

#if DKR_GDN_DTYPE != 3
#define STRIDES int64_t sb,int64_t ss,int64_t sc,int has_slots
#define UARGS const State* x,const State* w,State* state,State* out,const int32_t* slots,int batch,int channels,int width,STRIDES
#define FARGS const State* x,const State* w,const State* state,State* out,const int32_t* slots,int batch,int channels,int seq,int width,STRIDES
extern "C" __global__ void NAME(conv,DT,update)(UARGS){causal_conv1d_update_kernel<State>(x,w,state,out,batch,channels,width,sb,ss,sc,SLOTS);}
extern "C" __global__ void NAME(conv,DT,update4)(UARGS){causal_conv1d_update_width4_kernel<State>(x,w,state,out,batch,channels,sb,ss,sc,SLOTS);}
extern "C" __global__ void NAME(conv,DT,full)(FARGS){causal_conv1d_full_kernel<State>(x,w,state,out,batch,channels,seq,width,sb,ss,sc,SLOTS);}
extern "C" __global__ void NAME(conv,DT,full4)(FARGS){causal_conv1d_full_width4_tiled_kernel<State,64>(x,w,state,out,batch,channels,seq,sb,ss,sc,SLOTS);}
extern "C" __global__ void NAME(conv,DT,save)(const State* x,const State* state_in,State* state_out,const int32_t* slots,int batch,int channels,int seq,int width,STRIDES){save_conv_state_kernel<State>(x,state_in,state_out,batch,channels,seq,width,sb,ss,sc,SLOTS);}
#endif
