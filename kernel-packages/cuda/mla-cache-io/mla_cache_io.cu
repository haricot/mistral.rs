#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Separate latent and positional caches; no expansion into per-head KV.
// Each CTA copies one token. Metadata is device resident.
template <typename T>
__device__ __forceinline__ void mla_store(
    const T* c, const T* p, T* cc, T* pc, const int64_t* slots,
    uint32_t tokens, uint32_t blocks, uint32_t page,
    uint32_t rank, uint32_t rope, uint64_t cs, uint64_t ps) {
    const uint64_t t=blockIdx.x;
    if(t>=tokens) return;
    const int64_t slot=slots[t];
    if(slot<0 || static_cast<uint64_t>(slot)>=static_cast<uint64_t>(blocks)*page) return;
    for(uint32_t d=threadIdx.x;d<rank;d+=blockDim.x)
        cc[static_cast<uint64_t>(slot)*rank+d]=c[t*cs+d];
    for(uint32_t d=threadIdx.x;d<rope;d+=blockDim.x)
        pc[static_cast<uint64_t>(slot)*rope+d]=p[t*ps+d];
}

template <typename T>
__device__ __forceinline__ void mla_gather(
    const T* cc,const T* pc,const int32_t* table,const int32_t* offsets,
    const int32_t* token_to_seq,T* c,T* p,
    uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t page,
    uint32_t stride,uint32_t rank,uint32_t rope) {
    const uint32_t t=blockIdx.x;
    if(t>=tokens) return;
    const int32_t seq=token_to_seq[t];
    if(seq<0 || static_cast<uint32_t>(seq)>=seqs) return;
    const int32_t begin=offsets[seq],end=offsets[static_cast<uint32_t>(seq)+1];
    if(begin<0 || end<begin || static_cast<uint32_t>(end)>tokens ||
       t<static_cast<uint32_t>(begin) || t>=static_cast<uint32_t>(end)) return;
    const uint32_t local=t-static_cast<uint32_t>(begin), logical=local/page;
    if(logical>=stride) return;
    const int32_t block=table[static_cast<uint64_t>(seq)*stride+logical];
    if(block<0 || static_cast<uint32_t>(block)>=blocks) return;
    const uint64_t slot=static_cast<uint64_t>(block)*page+local%page;
    for(uint32_t d=threadIdx.x;d<rank;d+=blockDim.x)
        c[static_cast<uint64_t>(t)*rank+d]=cc[slot*rank+d];
    for(uint32_t d=threadIdx.x;d<rope;d+=blockDim.x)
        p[static_cast<uint64_t>(t)*rope+d]=pc[slot*rope+d];
}

#define DKR_MLA_IO(T, N) \
extern "C" __global__ void dkr_mla_store_##N(const T* c,const T* p,T* cc,T* pc,const int64_t* slots, \
    uint32_t tokens,uint32_t blocks,uint32_t page,uint32_t rank,uint32_t rope,uint64_t cs,uint64_t ps) { \
    mla_store(c,p,cc,pc,slots,tokens,blocks,page,rank,rope,cs,ps); \
} \
extern "C" __global__ void dkr_mla_gather_##N(const T* cc,const T* pc,const int32_t* table,const int32_t* offsets, \
    const int32_t* map,T* c,T* p,uint32_t tokens,uint32_t seqs,uint32_t blocks,uint32_t page,uint32_t stride,uint32_t rank,uint32_t rope) { \
    mla_gather(cc,pc,table,offsets,map,c,p,tokens,seqs,blocks,page,stride,rank,rope); \
}
DKR_MLA_IO(__half,f16)
DKR_MLA_IO(__nv_bfloat16,bf16)
DKR_MLA_IO(float,f32)
