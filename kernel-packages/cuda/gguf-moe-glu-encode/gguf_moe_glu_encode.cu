// Typed indexed GLU -> transposed MMQ Q8_1, preserving historical rounding.
#define DKR_GGUF_MMQ_IMAGE 11
#define DKR_GGUF_GLU_IMAGE 1
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_quantize.cu"
#define ENTRY(layout,L,dt,T) \
extern "C" __global__ void __launch_bounds__(128) dkr_moe_glu_##layout##_##dt( \
 const T* gate,const T* up,const unsigned* src,void* scratch,int k,int padded,int routes,int source_rows,int activation,int use_src){ \
 quantize_mmq_q8_1_glu<T,L>(gate,up,use_src?reinterpret_cast<const int32_t*>(src):nullptr,scratch,k,k,padded,routes,activation,source_rows); \
}
#define SET(layout,L) ENTRY(layout,L,f16,half) ENTRY(layout,L,bf16,__nv_bfloat16) ENTRY(layout,L,f32,float)
SET(d4,MMQ_Q8_1_DS_LAYOUT_D4)
SET(ds4,MMQ_Q8_1_DS_LAYOUT_DS4)
SET(d2s6,MMQ_Q8_1_DS_LAYOUT_D2S6)
