// Preserve FlashInfer's persistent merge, including base-2 logsumexp.
// This is not the natural-log F32-partial DKR paged-v2 merge.
#include "../../../mistralrs-paged-attn/src/cuda/flashinfer/attention/decode.cuh"
using namespace flashinfer;

template<typename T,uint32_t D>
__device__ __forceinline__ void merge_hnd(
    T* partial,float* lse,int32_t* offsets,T* output,
    uint32_t batch,uint32_t heads,uint8_t* smem) {
    constexpr uint32_t vec = (16/sizeof(T)>D/32 ? 16/sizeof(T) : D/32);
    constexpr uint32_t x = D/vec;
    constexpr uint32_t y = 128/x;
    // Same four stages as VariableLengthMergeStates host dispatcher.
    PersistentVariableLengthMergeStatesDevice<vec,x,y,4,T,T,int32_t>(
        partial,lse,offsets,output,nullptr,batch,nullptr,heads,smem);
}
#define DKR_MERGE(T,N,D) \
extern "C" __global__ void dkr_flashinfer_merge_##N##_d##D( \
    T* partial,float* lse,int32_t* offsets,T* output,uint32_t batch,uint32_t heads) { \
    extern __shared__ uint8_t smem[]; \
    merge_hnd<T,D>(partial,lse,offsets,output,batch,heads,smem); \
}
DKR_MERGE(__half,f16,64)
DKR_MERGE(__half,f16,128)
DKR_MERGE(__half,f16,256)
DKR_MERGE(__half,f16,512)
DKR_MERGE(__nv_bfloat16,bf16,64)
DKR_MERGE(__nv_bfloat16,bf16,128)
DKR_MERGE(__nv_bfloat16,bf16,256)
DKR_MERGE(__nv_bfloat16,bf16,512)
DKR_MERGE(float,f32,64)
DKR_MERGE(float,f32,128)
DKR_MERGE(float,f32,256)
DKR_MERGE(float,f32,512)
