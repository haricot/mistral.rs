// Pascal conventional MMQ. Reuse the original packed tile load, DP4A,
// reduction and write-back. No host launcher, fastdiv structs or Stream-K.
#if !defined(DKR_GGUF_MMQ_IMAGE) || DKR_GGUF_MMQ_IMAGE < 1 || DKR_GGUF_MMQ_IMAGE > 11
#error "select one bounded GGUF MMQ image (1..11)"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 610
#error "GGUF MMQ conventional qualification is SM61 only"
#endif
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh"
static_assert(sizeof(block_q8_1_mmq)==144, "MMQ activation ABI");
#if DKR_GGUF_MMQ_IMAGE == 11
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_quantize.cu"
#define ENCODE(layout, Layout, dtype, T) \
extern "C" __global__ void __launch_bounds__(128) \
dkr_mmq_encode_##layout##_##dtype(const T* x, void* y, int k, int padded, int batch) { \
    quantize_mmq_q8_1<T, Layout>(x, nullptr, y, k, k, 0, 0, padded, batch, 1); \
}
#define ENCODE_SET(layout, Layout) \
ENCODE(layout, Layout, f16, half) \
ENCODE(layout, Layout, bf16, __nv_bfloat16) \
ENCODE(layout, Layout, f32, float)
ENCODE_SET(d4, MMQ_Q8_1_DS_LAYOUT_D4)
ENCODE_SET(ds4, MMQ_Q8_1_DS_LAYOUT_DS4)
ENCODE_SET(d2s6, MMQ_Q8_1_DS_LAYOUT_D2S6)
#else
template<ggml_type Weight, int X, bool Check, ggml_type Output>
static __device__ __forceinline__ void dkr_mmq_dense(
    const char* w, const int* y, void* dst, int n, int batch, int blocks_k) {
    constexpr int Y=64;
    extern __shared__ int ids[];
    const int tid=threadIdx.y*32+threadIdx.x;
    if (tid<X) ids[tid]=tid;
    __syncthreads();
    const int row=blockIdx.x*Y;
    const int col=blockIdx.y*X;
    const int64_t dst_offset=(int64_t)col*n+row;
    // Always mask the activation tail, including rounded-up cooperative
    // loads of the final tile. K must be a multiple of ITER_K=256.
    mul_mat_q_process_tile<Weight,X,Check,false>(
        w,row*blocks_k,y+col*(sizeof(block_q8_1_mmq)/sizeof(int)),
        ids,(char*)dst+dst_offset*mmq_type_size(Output),Output,nullptr,
        blocks_k,batch,n,n-row-1,batch-col-1,0,blocks_k,true);
}
#define ENTRY(tag, Weight, X, bounds, Check, dtype, Output, T) \
extern "C" __global__ void __launch_bounds__(256,2) \
dkr_mmq_##tag##_##dtype##_x##X##_##bounds(const char* w,const int* y,T* dst,int n,int batch,int blocks_k) { \
    dkr_mmq_dense<Weight,X,Check,Output>(w,y,dst,n,batch,blocks_k); \
}
#define DTYPES(tag, Weight, X, bounds, Check) \
ENTRY(tag,Weight,X,bounds,Check,f16,GGML_TYPE_F16,half) \
ENTRY(tag,Weight,X,bounds,Check,bf16,GGML_TYPE_BF16,__nv_bfloat16) \
ENTRY(tag,Weight,X,bounds,Check,f32,GGML_TYPE_F32,float)
#define TILE(tag,Weight,X) \
DTYPES(tag,Weight,X,full,false) \
DTYPES(tag,Weight,X,tail,true)
#define FORMAT(tag,Weight) \
TILE(tag,Weight,8) TILE(tag,Weight,16) TILE(tag,Weight,24) TILE(tag,Weight,32) \
TILE(tag,Weight,40) TILE(tag,Weight,48) TILE(tag,Weight,56) TILE(tag,Weight,64)
#if DKR_GGUF_MMQ_IMAGE == 1
FORMAT(q4_0,GGML_TYPE_Q4_0)
#endif
#if DKR_GGUF_MMQ_IMAGE == 2
FORMAT(q4_1,GGML_TYPE_Q4_1)
#endif
#if DKR_GGUF_MMQ_IMAGE == 3
FORMAT(q5_0,GGML_TYPE_Q5_0)
#endif
#if DKR_GGUF_MMQ_IMAGE == 4
FORMAT(q5_1,GGML_TYPE_Q5_1)
#endif
#if DKR_GGUF_MMQ_IMAGE == 5
FORMAT(q8_0,GGML_TYPE_Q8_0)
#endif
#if DKR_GGUF_MMQ_IMAGE == 6
FORMAT(q2_k,GGML_TYPE_Q2_K)
#endif
#if DKR_GGUF_MMQ_IMAGE == 7
FORMAT(q3_k,GGML_TYPE_Q3_K)
#endif
#if DKR_GGUF_MMQ_IMAGE == 8
FORMAT(q4_k,GGML_TYPE_Q4_K)
#endif
#if DKR_GGUF_MMQ_IMAGE == 9
FORMAT(q5_k,GGML_TYPE_Q5_K)
#endif
#if DKR_GGUF_MMQ_IMAGE == 10
FORMAT(q6_k,GGML_TYPE_Q6_K)
#endif
#endif
