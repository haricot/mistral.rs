#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

// HND cache [block,head,page,dim]. Uniform dense input rows with independent
// physical strides, canonical cache storage. No host dispatch or FFI.
template<typename T,typename C>
__device__ __forceinline__ void store_hnd(
    const T* k,const T* v,C* kc,C* vc,const int64_t* slots,
    uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,
    uint64_t ks,uint64_t vs,float sk,float sv) {
    const uint64_t t=blockIdx.x;
    if(t>=tokens) return;
    const int64_t s=slots[t];
    if(s<0 || static_cast<uint64_t>(s)>=static_cast<uint64_t>(blocks)*page) return;
    const uint64_t b=static_cast<uint64_t>(s)/page, pos=static_cast<uint64_t>(s)%page;
    const uint64_t width=static_cast<uint64_t>(heads)*hd;
    for(uint64_t i=threadIdx.x;i<width;i+=blockDim.x) {
        const uint64_t idx=((b*heads+i/hd)*page+pos)*hd+i%hd;
        if constexpr(std::is_same_v<T,C>) {
            kc[idx]=k[t*ks+i]; vc[idx]=v[t*vs+i];
        } else {
            kc[idx]=static_cast<C>(static_cast<float>(k[t*ks+i])/sk);
            vc[idx]=static_cast<C>(static_cast<float>(v[t*vs+i])/sv);
        }
    }
}
template<typename C,typename T,typename B,typename O>
__device__ __forceinline__ void gather_hnd(
    const C* kc,const C* vc,const B* table,const O* offsets,T* k,T* v,
    uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,
    uint32_t page,uint32_t stride,float sk,float sv) {
    const uint32_t t=blockIdx.x;
    if(t>=tokens || seqs==0) return;
    // Metadata is monotone under the caller contract. Upper bound handles
    // empty sequences without a linear scan across the batch.
    uint32_t lo=0,hi=seqs;
    while(lo<hi) {
        const uint32_t mid=lo+(hi-lo)/2;
        if(static_cast<int64_t>(offsets[mid+1])<=t) lo=mid+1; else hi=mid;
    }
    if(lo>=seqs) return;
    const int64_t begin=offsets[lo],end=offsets[lo+1];
    if(begin<0 || end<begin || end>tokens || t<begin || t>=end) return;
    const uint64_t local=t-begin, logical=local/page;
    if(logical>=stride) return;
    const int64_t b=table[static_cast<uint64_t>(lo)*stride+logical];
    if(b<0 || b>=blocks) return;
    const uint64_t width=static_cast<uint64_t>(heads)*hd;
    for(uint64_t i=threadIdx.x;i<width;i+=blockDim.x) {
        const uint64_t idx=((static_cast<uint64_t>(b)*heads+i/hd)*page+local%page)*hd+i%hd;
        const uint64_t out=static_cast<uint64_t>(t)*width+i;
        if constexpr(std::is_same_v<T,C>) { k[out]=kc[idx]; v[out]=vc[idx]; }
        else { k[out]=static_cast<T>(static_cast<float>(kc[idx])*sk);
               v[out]=static_cast<T>(static_cast<float>(vc[idx])*sv); }
    }
}

extern "C" __global__ void dkr_hnd_store_plain_f16(const __half* k,const __half* v,__half* kc,__half* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_i32_f16(const __half* kc,const __half* vc,const int32_t* table,const int32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_u32_f16(const __half* kc,const __half* vc,const int32_t* table,const uint32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_i32_f16(const __half* kc,const __half* vc,const uint32_t* table,const int32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_u32_f16(const __half* kc,const __half* vc,const uint32_t* table,const uint32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_store_fp8_f16(const __half* k,const __half* v,__nv_fp8_e4m3* kc,__nv_fp8_e4m3* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs,float sk,float sv) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_i32_f16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const int32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_u32_f16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const uint32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_i32_f16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const int32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_u32_f16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const uint32_t* offsets,__half* k,__half* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_store_plain_bf16(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_i32_bf16(const __nv_bfloat16* kc,const __nv_bfloat16* vc,const int32_t* table,const int32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_u32_bf16(const __nv_bfloat16* kc,const __nv_bfloat16* vc,const int32_t* table,const uint32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_i32_bf16(const __nv_bfloat16* kc,const __nv_bfloat16* vc,const uint32_t* table,const int32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_u32_bf16(const __nv_bfloat16* kc,const __nv_bfloat16* vc,const uint32_t* table,const uint32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_store_fp8_bf16(const __nv_bfloat16* k,const __nv_bfloat16* v,__nv_fp8_e4m3* kc,__nv_fp8_e4m3* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs,float sk,float sv) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_i32_bf16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const int32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_u32_bf16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const uint32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_i32_bf16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const int32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_u32_bf16(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const uint32_t* offsets,__nv_bfloat16* k,__nv_bfloat16* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_store_plain_f32(const float* k,const float* v,float* kc,float* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_i32_f32(const float* kc,const float* vc,const int32_t* table,const int32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_i32_u32_f32(const float* kc,const float* vc,const int32_t* table,const uint32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_i32_f32(const float* kc,const float* vc,const uint32_t* table,const int32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_gather_plain_u32_u32_f32(const float* kc,const float* vc,const uint32_t* table,const uint32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,1.0f,1.0f); }

extern "C" __global__ void dkr_hnd_store_fp8_f32(const float* k,const float* v,__nv_fp8_e4m3* kc,__nv_fp8_e4m3* vc,const int64_t* slots,uint32_t tokens,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint64_t ks,uint64_t vs,float sk,float sv) { store_hnd(k,v,kc,vc,slots,tokens,blocks,heads,hd,page,ks,vs,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_i32_f32(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const int32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_i32_u32_f32(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const int32_t* table,const uint32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_i32_f32(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const int32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }

extern "C" __global__ void dkr_hnd_gather_fp8_u32_u32_f32(const __nv_fp8_e4m3* kc,const __nv_fp8_e4m3* vc,const uint32_t* table,const uint32_t* offsets,float* k,float* v,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t heads,uint32_t hd,uint32_t page,uint32_t stride,float sk,float sv) { gather_hnd(kc,vc,table,offsets,k,v,tokens,seqs,blocks,heads,hd,page,stride,sk,sv); }
