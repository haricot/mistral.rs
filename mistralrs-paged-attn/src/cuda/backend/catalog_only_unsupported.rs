//! Explicit capability errors for operations not yet externalized.
//! No FFI shims or fake successful outputs.
#![allow(unused_variables)]
use candle_core::{DType, Result, Tensor};
use crate::KvCacheScales;

#[derive(Clone, Copy)]
pub struct FlashInferDecodeScratch<'a> {
    pub tmp_v: &'a Tensor,
    pub tmp_s: &'a Tensor,
}

pub fn is_flashinfer_cache(key_cache: &Tensor, value_cache: &Tensor) -> bool {
    key_cache.dims().len() == 4
        && value_cache.dims().len() == 4
        && key_cache.dims() == value_cache.dims()
}

#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_flashinfer(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    scales: KvCacheScales,
) -> Result<()> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: reshape_and_cache_flashinfer is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn flashinfer_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    scales: KvCacheScales,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    paged_kv_last_page_len: &Tensor,
    request_indices: &Tensor,
    kv_tile_indices: &Tensor,
    o_indptr: &Tensor,
    kv_chunk_size: &Tensor,
    block_valid_mask: &Tensor,
    sm_scale: f32,
    window_left: Option<usize>,
    logits_soft_cap: Option<f32>,
    scratch: Option<FlashInferDecodeScratch<'_>>,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: flashinfer_decode is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn gather_kv_cache_flashinfer(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    num_tokens: usize, // Must equal cu_seq_lens[-1].
    out_dtype: DType,
    scales: KvCacheScales,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: gather_kv_cache_flashinfer is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn concat_and_cache_mla(
    ckv: &Tensor,
    k_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: concat_and_cache_mla is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn flashinfer_mla_decode(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    paged_kv_indptr: &Tensor,
    paged_kv_indices: &Tensor,
    paged_kv_last_page_len: &Tensor,
    request_indices: &Tensor,
    kv_tile_indices: &Tensor,
    o_indptr: &Tensor,
    kv_chunk_size: &Tensor,
    sm_scale: f32,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: flashinfer_mla_decode is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn gather_mla_cache(
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    token_to_seq: &Tensor,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: gather_mla_cache is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn context_attention_fwd_mla(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_new: &Tensor,
    kpe_new: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    query_lens: &Tensor,
    w_uk: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: context_attention_fwd_mla is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: flash_attn_sinks is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen_packed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    max_q_len: usize,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: flash_attn_sinks_varlen_packed is not yet externalized");
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    candle_core::bail!("DKR_PAGED_CATALOG_ONLY: flash_attn_sinks_varlen is not yet externalized");
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn dkr_paged_catalog_only_rejects_unmigrated_routes() -> Result<()> {
        let t = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu)?;
        assert!(is_flashinfer_cache(&t, &t));
        let error = flash_attn_sinks(&t, &t, &t, None, 1., 0)
            .unwrap_err().to_string();
        assert!(error.contains("DKR_PAGED_CATALOG_ONLY: flash_attn_sinks"));
        let error = concat_and_cache_mla(&t, &t, &t, &t, &t)
            .unwrap_err().to_string();
        assert!(error.contains("DKR_PAGED_CATALOG_ONLY: concat_and_cache_mla"));
        Ok(())
    }
}
