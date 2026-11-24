// Typed entry points over the existing vendored FlashInfer device body.
// One Cudaforge translation unit per bounded image; no host launcher.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/decode.cuh"
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/default_decode_params.cuh"
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/variants.cuh"

using namespace flashinfer;
#ifndef DKR_FLASHINFER_IMAGE
#error "Cudaforge must supply the image identity from the packaging plan"
#endif
#if (DKR_FLASHINFER_IMAGE & 3) == 0
using DkrQ = __half;
#elif (DKR_FLASHINFER_IMAGE & 3) == 1
using DkrQ = __nv_bfloat16;
#elif (DKR_FLASHINFER_IMAGE & 3) == 2
using DkrQ = float;
#else
#error "invalid activation dtype"
#endif
#if (DKR_FLASHINFER_IMAGE & 4)
using DkrKV = __nv_fp8_e4m3;
#else
using DkrKV = DkrQ;
#endif
constexpr uint32_t DKR_HEAD = 64U << ((DKR_FLASHINFER_IMAGE >> 3) & 3);

template <uint32_t G, bool W, bool S, bool P>
__device__ __forceinline__ void dkr_decode(
    DkrQ* q, DkrKV* k, DkrKV* v, int32_t* indptr, int32_t* indices,
    int32_t* last, int32_t* requests, int32_t* tiles, int32_t* out_indptr,
    int32_t* chunk, uint8_t* mask, DkrQ* out, float* lse,
    uint32_t batch, uint32_t padded, uint32_t heads, uint32_t kv_heads,
    uint32_t page, uint32_t magic, uint32_t shift, uint32_t add,
    int32_t stride_n, int32_t stride_h, uint32_t use_mask,
    int32_t window_left, float softcap, float scale, float k_scale, float v_scale,
    uint8_t* smem) {
    // The historical body reads request/tile metadata before checking the mask.
    // Skip invalid padded blocks before either access; the planner must still
    // provide full physical spans and valid metadata for all active blocks.
    if (use_mask && !mask[blockIdx.x]) return;
    using Params = BatchDecodeParams<DkrQ,DkrKV,DkrQ,int32_t>;
    using Variant = DefaultAttention<false,W,S,false>;
    Params p;
    p.q=q; p.o=out; p.lse=P ? lse : nullptr;
    p.paged_kv.page_size.d=page; p.paged_kv.page_size.m=magic;
    p.paged_kv.page_size.s=shift; p.paged_kv.page_size.a=add;
    p.paged_kv.num_heads=kv_heads; p.paged_kv.head_dim=DKR_HEAD;
    p.paged_kv.batch_size=batch;
    p.paged_kv.stride_page=kv_heads*page*DKR_HEAD;
    p.paged_kv.stride_n=DKR_HEAD; p.paged_kv.stride_h=page*DKR_HEAD;
    p.paged_kv.k_data=k; p.paged_kv.v_data=v;
    p.paged_kv.indices=indices; p.paged_kv.indptr=indptr; p.paged_kv.last_page_len=last;
    p.padded_batch_size=padded; p.num_qo_heads=heads;
    p.q_stride_n=stride_n; p.q_stride_h=stride_h;
    p.window_left=window_left; p.logits_soft_cap=softcap;
    p.sm_scale=scale*k_scale; p.v_scale=v_scale;
    p.rope_rcp_scale=1.f; p.rope_rcp_theta=1.f;
    p.request_indices=requests; p.kv_tile_indices=tiles;
    p.o_indptr=out_indptr; p.kv_chunk_size_ptr=chunk;
    p.block_valid_mask=nullptr; p.partition_kv=P;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr uint32_t stages=2;
#else
    constexpr uint32_t stages=1;
#endif
    constexpr uint32_t width=sizeof(DkrKV);
    constexpr uint32_t vec=(16/width > DKR_HEAD/32) ? 16/width : DKR_HEAD/32;
    constexpr uint32_t x=DKR_HEAD/vec;
    constexpr uint32_t intended=(128 > x*G) ? 128 : x*G;
    constexpr uint32_t z=intended/(x*G);
    constexpr uint32_t tile=G==1 ? (width==1 ? 2 : 4) : 1;
    static_assert(x*G*z%32==0, "partial warp is quarantined");
    static_assert(z==1 || 2*stages*tile*width>=sizeof(float),
                  "state scratch overlaps shared metadata");
    constexpr uint32_t offsets=tile*intended*sizeof(DkrKV*);
    constexpr uint32_t md=2*G*z*sizeof(float);
    constexpr uint32_t shared=2*stages*tile*G*z*DKR_HEAD*width+(offsets>md?offsets:md);
    static_assert(shared<=48*1024, "staged shared-memory limit");
    BatchDecodeWithPagedKVCacheDevice<PosEncodingMode::kNone,stages,tile,vec,x,G,z,Variant>(p,smem);
}

#define DKR_INPUTS DkrQ* q,DkrKV* k,DkrKV* v,int32_t* indptr,int32_t* indices, \
    int32_t* last,int32_t* requests,int32_t* tiles,int32_t* out_indptr,int32_t* chunk,uint8_t* mask
#define DKR_SCALARS uint32_t batch,uint32_t padded,uint32_t heads,uint32_t kv_heads, \
    uint32_t page,uint32_t magic,uint32_t shift,uint32_t add,int32_t stride_n,int32_t stride_h, \
    uint32_t use_mask,int32_t window_left,float softcap,float scale,float k_scale,float v_scale
#define DKR_FORWARD q,k,v,indptr,indices,last,requests,tiles,out_indptr,chunk,mask
#define DKR_VALUES batch,padded,heads,kv_heads,page,magic,shift,add,stride_n,stride_h, \
    use_mask,window_left,softcap,scale,k_scale,v_scale,smem
#define DKR_PAIR(G,W,S) \
extern "C" __global__ void dkr_flashinfer_decode_g##G##_w##W##_s##S##_direct( \
    DKR_INPUTS,DkrQ* out,DKR_SCALARS) { \
    extern __shared__ __align__(16) uint8_t smem[]; \
    dkr_decode<G,W,S,false>(DKR_FORWARD,out,nullptr,DKR_VALUES); \
} \
extern "C" __global__ void dkr_flashinfer_decode_g##G##_w##W##_s##S##_partition( \
    DKR_INPUTS,DkrQ* out,float* lse,DKR_SCALARS) { \
    extern __shared__ __align__(16) uint8_t smem[]; \
    dkr_decode<G,W,S,true>(DKR_FORWARD,out,lse,DKR_VALUES); \
}
#define DKR_GROUP(G) DKR_PAIR(G,0,0) DKR_PAIR(G,0,1) DKR_PAIR(G,1,0) DKR_PAIR(G,1,1)
#if (DKR_FLASHINFER_IMAGE & (1 << 5))
DKR_GROUP(1)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 6))
DKR_GROUP(2)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 7))
DKR_GROUP(3)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 8))
DKR_GROUP(4)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 9))
DKR_GROUP(5)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 10))
DKR_GROUP(6)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 11))
DKR_GROUP(7)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 12))
DKR_GROUP(8)
#endif
#if (DKR_FLASHINFER_IMAGE & (1 << 13))
DKR_GROUP(16)
#endif
