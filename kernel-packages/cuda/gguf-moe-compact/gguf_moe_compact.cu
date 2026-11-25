// SM61 compact grouped MMQ: packed weights, GPU-only route metadata.
// Reuses the historical DP4A tile body and MMQ activation encoder.
#if !defined(DKR_GGUF_MOE_IMAGE) || DKR_GGUF_MOE_IMAGE < 1 || DKR_GGUF_MOE_IMAGE > 11
#error "select GGUF MoE image 1..11"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 610
#error "SM61 compact provider only"
#endif
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh"
static_assert(sizeof(block_q8_1_mmq)==144, "MMQ Q8_1 ABI");

static __device__ __forceinline__ int dkr_columns(int count) {
    int x=8;
    while (x<64 && x<count) x*=2;
    return x;
}
#if DKR_GGUF_MOE_IMAGE == 11
#define DKR_GGUF_MMQ_IMAGE 11
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_quantize.cu"

// Four ordered dispatch phases. Out-of-range expert IDs are excluded,
// unused source slots remain zero, and their result rows stay NaN.
extern "C" __global__ void dkr_moe_init(uint32_t* counts,uint32_t* src,
                                       uint32_t* dst,int experts,int routes) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<experts) counts[i]=0;
    if(i<routes) {src[i]=0;dst[i]=0;}
}
extern "C" __global__ void dkr_moe_count(const uint32_t* ids,uint32_t* counts,
                                        int experts,int routes) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<routes && ids[i]<(unsigned)experts) atomicAdd(counts+ids[i],1u);
}
extern "C" __global__ void dkr_moe_prefix(const uint32_t* counts,uint32_t* bounds,
                                         uint32_t* cursors,int experts) {
    unsigned offset=0;
    for(int e=0;e<experts;e++) {
        bounds[e]=offset;cursors[e]=offset;offset+=counts[e];
    }
    bounds[experts]=offset;
}
extern "C" __global__ void dkr_moe_scatter(const uint32_t* ids,uint32_t* cursors,
    uint32_t* src,uint32_t* dst,int experts,int routes,int topk,int input_routes) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<routes && ids[i]<(unsigned)experts) {
        const unsigned p=atomicAdd(cursors+ids[i],1u);
        dst[p]=i;src[p]=input_routes==1 ? i/topk : i;
    }
}
// Parallel block prefix scan, following historical compact_mmq_worklist.
// Output is a bounded [capacity,2] table and a GPU-resident count.
extern "C" __global__ void dkr_moe_worklist(const uint32_t* bounds,uint32_t* tiles,
    uint32_t* total,int experts,int routes,int capacity) {
    __shared__ int scan[256], preceding, bad;
    const int tid=threadIdx.x;
    if(tid==0){preceding=0;bad=(bounds[0]!=0 || bounds[experts]>(unsigned)routes);*total=0;}
    __syncthreads();
    for(int base=0;base<experts;base+=256) {
        const int e=base+tid;
        int count=0;
        if(e<experts) {
            const unsigned lo=bounds[e],hi=bounds[e+1];
            if(lo>hi || hi>(unsigned)routes) atomicExch(&bad,1);
            else count=(hi-lo+dkr_columns(hi-lo)-1)/dkr_columns(hi-lo);
        }
        scan[tid]=count;
        __syncthreads();
        for(int shift=1;shift<256;shift*=2) {
            const int v=tid>=shift ? scan[tid-shift]:0;
            __syncthreads();
            scan[tid]+=v;
            __syncthreads();
        }
        const int offset=preceding+scan[tid]-count;
        if(offset+count>capacity) atomicExch(&bad,1);
        __syncthreads();
        if(!bad) for(int j=0;j<count;j++){
            tiles[2*(offset+j)]=e;tiles[2*(offset+j)+1]=j;
        }
        __syncthreads();
        if(tid==0) preceding+=scan[255];
        __syncthreads();
    }
    if(tid==0 && !bad)*total=preceding;
}
extern "C" __global__ void dkr_moe_fill(float* out,int count) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=__int_as_float(0x7fc00000);
}
#define ENCODE(layout,Layout,dt,T) \
extern "C" __global__ void __launch_bounds__(128) \
dkr_moe_encode_##layout##_##dt(const T* x,const uint32_t* src,void* y,int k,int padded,int routes,int source_rows){ \
    quantize_mmq_q8_1<T,Layout>(x,reinterpret_cast<const int32_t*>(src),y,k,k,0,0,padded,routes,1,source_rows); \
}
#define ENCODE_SET(layout,Layout) \
ENCODE(layout,Layout,f16,half) ENCODE(layout,Layout,bf16,__nv_bfloat16) ENCODE(layout,Layout,f32,float)
ENCODE_SET(d4,MMQ_Q8_1_DS_LAYOUT_D4)
ENCODE_SET(ds4,MMQ_Q8_1_DS_LAYOUT_DS4)
ENCODE_SET(d2s6,MMQ_Q8_1_DS_LAYOUT_D2S6)
#else
template<ggml_type W,bool Check>
static __device__ __forceinline__ void compact(const char* w,const int* y,
 const uint32_t* dst,const uint32_t* bounds,const uint32_t* tiles,const uint32_t* total,
 float* out,int n,int routes,int blocks_k,int experts,int capacity) {
    if(blockIdx.y>=*total || blockIdx.y>=capacity)return;
    const unsigned e=tiles[2*blockIdx.y],j=tiles[2*blockIdx.y+1];
    if(e>=experts)return;
    const unsigned lo=bounds[e],hi=bounds[e+1];
    if(lo>=hi || hi>(unsigned)routes)return;
    const int count=hi-lo,x=dkr_columns(count);
    if(j>=(unsigned)((count+x-1)/x))return;
    const int col=j*x, row=blockIdx.x*64, tid=threadIdx.y*32+threadIdx.x;
    extern __shared__ int ids[];
    __shared__ int invalid;
    if(tid==0)invalid=0;
    __syncthreads();
    if(tid<x) {
        const unsigned route=col+tid<count ? dst[lo+col+tid]:0;
        if(route>=(unsigned)routes)atomicExch(&invalid,1);
        ids[tid]=route;
    }
    __syncthreads();
    if(invalid)return;
    const int wo=(e*n+row)*blocks_k;
    const int* act=y+(lo+col)*36;
#define TILE(X) mul_mat_q_process_tile<W,X,Check,false>(w,wo,act,ids,out+row,GGML_TYPE_F32,nullptr,blocks_k,routes,n,n-row-1,count-col-1,0,blocks_k,true)
    if(x==8){TILE(8);}else if(x==16){TILE(16);}else if(x==32){TILE(32);}else{TILE(64);}
#undef TILE
}
#define ENTRY(tag,W,bounds,Check) \
extern "C" __global__ void __launch_bounds__(256,2) \
dkr_moe_compact_##tag##_##bounds(const char* w,const int* y,const uint32_t* dst,const uint32_t* bounds_ptr,const uint32_t* tiles,const uint32_t* count,float* out,int n,int routes,int blocks_k,int experts,int capacity){ \
    compact<W,Check>(w,y,dst,bounds_ptr,tiles,count,out,n,routes,blocks_k,experts,capacity); \
}
#define FORMAT(tag,W) ENTRY(tag,W,full,false) ENTRY(tag,W,tail,true)
#if DKR_GGUF_MOE_IMAGE == 1
FORMAT(q4_0,GGML_TYPE_Q4_0)
#endif
#if DKR_GGUF_MOE_IMAGE == 2
FORMAT(q4_1,GGML_TYPE_Q4_1)
#endif
#if DKR_GGUF_MOE_IMAGE == 3
FORMAT(q5_0,GGML_TYPE_Q5_0)
#endif
#if DKR_GGUF_MOE_IMAGE == 4
FORMAT(q5_1,GGML_TYPE_Q5_1)
#endif
#if DKR_GGUF_MOE_IMAGE == 5
FORMAT(q8_0,GGML_TYPE_Q8_0)
#endif
#if DKR_GGUF_MOE_IMAGE == 6
FORMAT(q2_k,GGML_TYPE_Q2_K)
#endif
#if DKR_GGUF_MOE_IMAGE == 7
FORMAT(q3_k,GGML_TYPE_Q3_K)
#endif
#if DKR_GGUF_MOE_IMAGE == 8
FORMAT(q4_k,GGML_TYPE_Q4_K)
#endif
#if DKR_GGUF_MOE_IMAGE == 9
FORMAT(q5_k,GGML_TYPE_Q5_K)
#endif
#if DKR_GGUF_MOE_IMAGE == 10
FORMAT(q6_k,GGML_TYPE_Q6_K)
#endif
#endif
