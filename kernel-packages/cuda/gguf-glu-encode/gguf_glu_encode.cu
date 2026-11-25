// Reuse historical GLU + MMQ Q8_1 fused encoder. No static host entry.
#define DKR_GGUF_MMQ_IMAGE 11
#define DKR_GGUF_GLU_IMAGE 1
#include "../../../mistralrs-quant/kernels/mmq_gguf/mmq_quantize.cu"
#define ENTRY(layout,L,dt,T) \
extern "C" __global__ void __launch_bounds__(128) dkr_gguf_glu_##layout##_##dt( \
    const T* gate,const T* up,void* scratch,int k,int padded,int batch,int activation){ \
    quantize_mmq_q8_1_glu<T,L>(gate,up,nullptr,scratch,k,k,padded,batch,activation); \
}
#define SET(layout,L) ENTRY(layout,L,f16,half) ENTRY(layout,L,bf16,__nv_bfloat16) ENTRY(layout,L,f32,float)
SET(d4,MMQ_Q8_1_DS_LAYOUT_D4)
SET(ds4,MMQ_Q8_1_DS_LAYOUT_DS4)
SET(d2s6,MMQ_Q8_1_DS_LAYOUT_D2S6)
