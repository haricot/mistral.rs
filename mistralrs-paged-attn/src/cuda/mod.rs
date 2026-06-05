pub const USE_FP8: bool = cfg!(has_fp8);
pub const USE_FLASHINFER: bool = cfg!(has_flashinfer);

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, copy_blocks, flash_attn_sinks,
    flash_attn_sinks_varlen, flashinfer_decode, flashinfer_mla_decode, flashinfer_prefill,
    gather_kv_cache, gather_kv_cache_flashinfer, gather_mla_cache, is_flashinfer_cache,
    kv_scale_update, kvarn_flash_attn_decode, mtp_paged_attention, paged_attention,
    reshape_and_cache, reshape_and_cache_flashinfer, swap_blocks, turboquant_gather_kv_cache,
    turboquant_reshape_and_cache,
};
