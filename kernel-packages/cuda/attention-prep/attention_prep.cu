// Historical rotary arithmetic and attention preparation device bodies.
// Block shape for norm fusion is selected by typed profile, not CC.
#define DKR_ATTENTION_PREP_IMAGE 1
#include "../../../mistralrs-core/src/cuda/attention_prep.cu"
#define DKR_ROTARY_IMAGE 1
#include "../../../mistralrs-quant/kernels/rotary/rotary.cu"
#define STRIDES(p) int64_t p##_b,int64_t p##_h,int64_t p##_s,int64_t p##_d
#define S(p) p##_b,p##_h,p##_s,p##_d
#define DIMS int batch,int q_heads,int k_heads,int seq_len,int head_dim,int rot_dim
#define D batch,q_heads,k_heads,seq_len,head_dim,rot_dim
#define QK_ARGS(T) const T* q,const T* k,const T* qw,const T* kw,const T* cos,const T* sin
#define QK_OUT(T) T* qo,T* ko,STRIDES(q),STRIDES(k),DIMS
#define QK_CALL q,k,qw,kw,cos,sin,qo,ko,S(q),S(k),D
#define POS_CALL q,k,qw,kw,cos,sin,pos,qo,ko,S(q),S(k),D
#define QKV_ARGS(T) const T* q,const T* k,const T* v,const T* qw,const T* kw,const T* vw,const T* cos,const T* sin,const uint32_t* pos,T* qo,T* ko,T* vo,STRIDES(q),STRIDES(k),STRIDES(v),DIMS,float qe,float ke,float ve,int neox,int cache_rows
#define QKV_CALL q,k,v,qw,kw,vw,cos,sin,pos,qo,ko,vo,S(q),S(k),S(v),D,qe,ke,ve,cache_rows
#define AP_BLOCK(tag,T,B) \
extern "C" __global__ __launch_bounds__(B) void dkr_ap_qk_##tag##_b##B(QK_ARGS(T),QK_OUT(T),int cos_stride,float qe,float ke,int neox,int token_major){ \
    if(neox)qk_rms_norm_rope_kernel<T,true>(QK_CALL,cos_stride,qe,ke,token_major!=0); \
    else qk_rms_norm_rope_kernel<T,false>(QK_CALL,cos_stride,qe,ke,token_major!=0); \
} \
extern "C" __global__ __launch_bounds__(B) void dkr_ap_qk_pos_##tag##_b##B(QK_ARGS(T),const uint32_t* pos,QK_OUT(T),float qe,float ke,int neox,int token_major,int cache_rows){ \
    if(neox)qk_rms_norm_rope_positions_kernel<T,true>(POS_CALL,qe,ke,token_major!=0,cache_rows); \
    else qk_rms_norm_rope_positions_kernel<T,false>(POS_CALL,qe,ke,token_major!=0,cache_rows); \
} \
extern "C" __global__ __launch_bounds__(B) void dkr_ap_qkv_pos_##tag##_b##B(QKV_ARGS(T)){ \
    if(neox)qkv_rms_norm_rope_positions_kernel<T,true>(QKV_CALL); \
    else qkv_rms_norm_rope_positions_kernel<T,false>(QKV_CALL); \
}
#define DTYPE(tag,T) \
AP_BLOCK(tag,T,32) AP_BLOCK(tag,T,64) AP_BLOCK(tag,T,128) \
AP_BLOCK(tag,T,256) AP_BLOCK(tag,T,512) AP_BLOCK(tag,T,1024) \
extern "C" __global__ void dkr_ap_sincos_##tag(const uint32_t* pos,const float* freq,T* cos,T* sin,int rows,int width){ \
    rope_sincos_positions_kernel<T>(pos,freq,cos,sin,width,int64_t(rows)*width); \
} \
extern "C" __global__ void dkr_rotary_##tag(T* q,const T* cos,const T* sin,int head,int rot,int heads,int64_t stride,int neox){ \
    if(neox)vllm::rotary_embedding_kernel<T,true>(q,nullptr,cos,sin,rot,stride,0,heads,0,head); \
    else vllm::rotary_embedding_kernel<T,false>(q,nullptr,cos,sin,rot,stride,0,heads,0,head); \
} \
extern "C" __global__ void dkr_rotary_pos_##tag(T* q,const T* cos,const T* sin,const uint32_t* pos,int head,int rot,int heads,int64_t stride,int neox,int cache_rows){ \
    if(neox)vllm::rotary_embedding_positions_kernel<T,true>(q,nullptr,cos,sin,pos,rot,cache_rows,stride,0,heads,0,head); \
    else vllm::rotary_embedding_positions_kernel<T,false>(q,nullptr,cos,sin,pos,rot,cache_rows,stride,0,heads,0,head); \
}
DTYPE(f16,half)
DTYPE(bf16,__nv_bfloat16)
DTYPE(f32,float)
