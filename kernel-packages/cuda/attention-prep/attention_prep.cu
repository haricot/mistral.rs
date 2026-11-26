// Historical rotary arithmetic and attention preparation device bodies.
// Block shape for norm fusion is selected by typed profile, not CC.
#define DKR_ATTENTION_PREP_IMAGE 1
#include "../../../mistralrs-core/src/cuda/attention_prep.cu"
#define DKR_ROTARY_IMAGE 1
#include "../../../mistralrs-quant/kernels/rotary/rotary.cu"
#define STRIDES(p) uint64_t p##_bh,uint64_t p##_sd
#define S(p) int64_t(uint32_t(p##_bh)),int64_t(p##_bh>>32),int64_t(uint32_t(p##_sd)),int64_t(p##_sd>>32)
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
#define NORM_BLOCK(tag,T,B) \
extern "C" __global__ __launch_bounds__(B) void dkr_ap_norm_##tag##_b##B(const T* x,const T* w,T* y,STRIDES(x),int batch,int heads,int seq,int head,float eps){ \
    __shared__ float reduce[1024]; \
    int row=blockIdx.x;int s=row%seq;int h=(row/seq)%heads;int b=row/seq/heads; \
    int64_t base=int64_t(b)*uint32_t(x_bh)+int64_t(h)*(x_bh>>32)+int64_t(s)*uint32_t(x_sd); \
    write_norm_row<T>(x,w,y+int64_t(row)*head,base,int64_t(x_sd>>32),head,eps,reduce); \
}
#define DTYPE(tag,T) \
NORM_BLOCK(tag,T,32) NORM_BLOCK(tag,T,64) NORM_BLOCK(tag,T,128) \
NORM_BLOCK(tag,T,256) NORM_BLOCK(tag,T,512) NORM_BLOCK(tag,T,1024) \
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
#if DKR_AP_DTYPE==1
DTYPE(f16,half)
#elif DKR_AP_DTYPE==2
DTYPE(bf16,__nv_bfloat16)
#elif DKR_AP_DTYPE==3
DTYPE(f32,float)
#else
#error "select DKR_AP_DTYPE"
#endif

#if DKR_AP_DTYPE==1
__device__ __forceinline__ uint32_t dkr_staged_token(int row,int column,const uint32_t* r0,const uint32_t* r1,const uint32_t* r2,const uint32_t* r3,const uint32_t* r4,const uint32_t* r5,const uint32_t* r6,const uint32_t* r7,const uint32_t* r8,const uint32_t* r9,const uint32_t* r10,const uint32_t* r11,const uint32_t* r12,const uint32_t* r13,const uint32_t* r14,const uint32_t* r15,const uint32_t* r16,const uint32_t* r17,const uint32_t* r18,const uint32_t* r19,const uint32_t* r20,const uint32_t* r21,const uint32_t* r22,const uint32_t* r23){
    switch(row){
        case 0: return r0[column];
        case 1: return r1[column];
        case 2: return r2[column];
        case 3: return r3[column];
        case 4: return r4[column];
        case 5: return r5[column];
        case 6: return r6[column];
        case 7: return r7[column];
        case 8: return r8[column];
        case 9: return r9[column];
        case 10: return r10[column];
        case 11: return r11[column];
        case 12: return r12[column];
        case 13: return r13[column];
        case 14: return r14[column];
        case 15: return r15[column];
        case 16: return r16[column];
        case 17: return r17[column];
        case 18: return r18[column];
        case 19: return r19[column];
        case 20: return r20[column];
        case 21: return r21[column];
        case 22: return r22[column];
        case 23: return r23[column];
        default:return 0;
    }
}
extern "C" __global__ void dkr_pack_completion_u32(const uint32_t* host,const uint32_t* r0,const uint32_t* r1,const uint32_t* r2,const uint32_t* r3,const uint32_t* r4,const uint32_t* r5,const uint32_t* r6,const uint32_t* r7,const uint32_t* r8,const uint32_t* r9,const uint32_t* r10,const uint32_t* r11,const uint32_t* r12,const uint32_t* r13,const uint32_t* r14,const uint32_t* r15,const uint32_t* r16,const uint32_t* r17,const uint32_t* r18,const uint32_t* r19,const uint32_t* r20,const uint32_t* r21,const uint32_t* r22,const uint32_t* r23,uint32_t* out,int rows,int host_width,int staged_width){
    int64_t width=int64_t(host_width)+staged_width;
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=int64_t(rows)*width)return;
    int row=i/width,col=i%width;
    out[i]=col<host_width?host[int64_t(row)*host_width+col]:
        dkr_staged_token(row,col-host_width,r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23);
}
#endif
