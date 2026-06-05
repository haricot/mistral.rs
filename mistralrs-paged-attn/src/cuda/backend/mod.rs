mod cache;
mod context_attention_mla;
mod flash_attn_sinks;
#[cfg(has_flashinfer)]
mod flashinfer;
#[cfg(not(has_flashinfer))]
mod flashinfer {
    use candle_core::{DType, Result, Tensor};

    pub fn is_flashinfer_cache(_key_cache: &Tensor, _value_cache: &Tensor) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reshape_and_cache_flashinfer(
        _key: &Tensor,
        _value: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _slot_mapping: &Tensor,
    ) -> Result<()> {
        candle_core::bail!(
            "FlashInfer paged-attention kernels are not available on this CUDA compute capability"
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flashinfer_decode(
        _query: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _paged_kv_indptr: &Tensor,
        _paged_kv_indices: &Tensor,
        _paged_kv_last_page_len: &Tensor,
        _request_indices: &Tensor,
        _kv_tile_indices: &Tensor,
        _o_indptr: &Tensor,
        _kv_chunk_size: &Tensor,
        _block_valid_mask: &Tensor,
        _sm_scale: f32,
        _window_left: Option<usize>,
        _logits_soft_cap: Option<f32>,
        _use_tensor_cores: bool,
    ) -> Result<Tensor> {
        candle_core::bail!("FlashInfer decode is not available on this CUDA compute capability")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flashinfer_prefill(
        _query: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _paged_kv_indptr: &Tensor,
        _paged_kv_indices: &Tensor,
        _paged_kv_last_page_len: &Tensor,
        _q_indptr: &Tensor,
        _request_indices: &Tensor,
        _qo_tile_indices: &Tensor,
        _kv_tile_indices: &Tensor,
        _o_indptr: &Tensor,
        _kv_chunk_size: &Tensor,
        _block_valid_mask: &Tensor,
        _batch_size: usize,
        _sm_scale: f32,
        _window_left: Option<usize>,
        _logits_soft_cap: Option<f32>,
    ) -> Result<Tensor> {
        candle_core::bail!("FlashInfer prefill is not available on this CUDA compute capability")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gather_kv_cache_flashinfer(
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _block_table: &Tensor,
        _cu_seq_lens: &Tensor,
        _out_dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        candle_core::bail!("FlashInfer KV gather is not available on this CUDA compute capability")
    }
}
mod gather_kv;
mod kvarn;
mod legacy_flash_attn;
mod legacy_flash_attn_turboquant;
mod mla;
mod mtp_paged_attention;
mod paged_attention;
mod scale_update;
mod turboquant;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};
pub use context_attention_mla::context_attention_fwd_mla;
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
pub use flashinfer::{
    flashinfer_decode, flashinfer_prefill, gather_kv_cache_flashinfer, is_flashinfer_cache,
    reshape_and_cache_flashinfer,
};
pub use gather_kv::gather_kv_cache;
pub use kvarn::kvarn_flash_attn_decode;
pub use legacy_flash_attn::{legacy_flash_attn_decode_dense, legacy_flash_attn_decode_paged};
pub use legacy_flash_attn_turboquant::{
    legacy_flash_attn_decode_turboquant, legacy_flash_attn_decode_turboquant_head512_twopass,
    legacy_flash_attn_turboquant_allowed,
};
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use mtp_paged_attention::mtp_paged_attention;
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;
pub use turboquant::{turboquant_gather_kv_cache, turboquant_reshape_and_cache};

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}
