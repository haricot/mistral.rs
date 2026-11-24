// Reuse the vendored FlashInfer device body, not its host launcher.
// Separate typed pointer/scalar arguments avoid a C++ struct ABI at admission.
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/decode.cuh"
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/default_decode_params.cuh"
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/variants.cuh"

using namespace flashinfer;

template <typename T>
__device__ __forceinline__ void decode_mla(
    T* qn,T* qp,T* ck,T* pk,int32_t* indptr,int32_t* indices,int32_t* last,
    int32_t* requests,int32_t* tiles,int32_t* out_indptr,int32_t* chunk,T* out,
    uint32_t batch,uint32_t heads,uint32_t page,uint32_t magic,uint32_t shift,
    uint32_t add,float scale) {
    using Params=BatchDecodeParamsMLA<T,T,T,int32_t>;
    using Variant=DefaultAttention<false,false,false,false>;
    Params p;
    p.q_nope=qn; p.q_pe=qp; p.o=out; p.sm_scale=scale;
    p.paged_kv.page_size.d=page;
    p.paged_kv.page_size.m=magic;
    p.paged_kv.page_size.s=shift;
    p.paged_kv.page_size.a=add;
    p.paged_kv.head_dim_ckv=512; p.paged_kv.head_dim_kpe=64;
    p.paged_kv.batch_size=batch;
    p.paged_kv.stride_page_ckv=page*512;
    p.paged_kv.stride_page_kpe=page*64;
    p.paged_kv.stride_n_ckv=512; p.paged_kv.stride_n_kpe=64;
    p.paged_kv.ckv_data=ck; p.paged_kv.kpe_data=pk;
    p.paged_kv.indices=indices; p.paged_kv.indptr=indptr;
    p.paged_kv.last_page_len=last;
    p.padded_batch_size=batch; p.num_qo_heads=heads;
    p.window_left=-1; p.logits_soft_cap=0.f;
    p.rope_rcp_scale=1.f; p.rope_rcp_theta=1.f;
    p.request_indices=requests; p.kv_tile_indices=tiles;
    p.o_indptr=out_indptr; p.kv_chunk_size_ptr=chunk;
    // Historical Mistral MLA wrapper passes null tmp_v/tmp_s.
    p.partition_kv=false;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t stages=2;
#else
    constexpr uint32_t stages=1;
#endif
    BatchDecodeWithPagedKVCacheDeviceMLA<stages,16,2,32,8,1,2,Variant,Params>(p);
}

#define DKR_MLA_DECODE(T,N) \
extern "C" __global__ void dkr_mla_decode_##N( \
    T* qn,T* qp,T* ck,T* pk,int32_t* indptr,int32_t* indices,int32_t* last, \
    int32_t* requests,int32_t* tiles,int32_t* out_indptr,int32_t* chunk,T* out, \
    uint32_t batch,uint32_t heads,uint32_t page,uint32_t magic,uint32_t shift,uint32_t add,float scale) { \
    decode_mla(qn,qp,ck,pk,indptr,indices,last,requests,tiles,out_indptr,chunk,out, \
               batch,heads,page,magic,shift,add,scale); \
}
DKR_MLA_DECODE(__half,f16)
DKR_MLA_DECODE(__nv_bfloat16,bf16)
DKR_MLA_DECODE(float,f32)
