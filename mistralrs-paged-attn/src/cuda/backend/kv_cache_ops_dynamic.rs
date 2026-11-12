//! DKR-R1.21/S7: typed prevalidated SM61 KV page copy and slot write.
//!
//! Copy is bit-exact across FP8/F16/BF16/F32 by transferring raw bytes
//! from the already validated per-layer pointer tables, with no duplicate
//! destination or cross-pair read-after-write aliasing.
//! Slot writes use canonical contiguous VLLM K-blocked-x/V-token-strided
//! layouts. Scaled FP8 writes, CPU/GPU swap and noncontiguous tensors keep
//! their existing static/host backend.
//! No static NVCC flags or CUDA_COMPUTE_CAP modifications.

use super::slice_ptr_on_stream;
use candle_core::{
    cuda::cudarc::driver::{CudaStream, PushKernelArg},
    DType, Device, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, path::PathBuf, sync::{Arc, OnceLock}};

const ENV: &str = "DKR_KERNEL_PACKAGES";
const BLOCK: u32 = 256;
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

fn enabled(name: &str) -> bool {
    match name {
        "copy" => {
            static COPY: OnceLock<bool> = OnceLock::new();
            *COPY.get_or_init(|| {
                std::env::var("MISTRALRS_DKR_KV_COPY")
                    .map_or(true, |value| value != "0")
            })
        }
        "reshape" => {
            static RESHAPE: OnceLock<bool> = OnceLock::new();
            *RESHAPE.get_or_init(|| {
                std::env::var("MISTRALRS_DKR_KV_RESHAPE")
                    .map_or(true, |value| value != "0")
            })
        }
        _ => false,
    }
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| {
        std::env::var_os(ENV).map(PathBuf::from)
            .ok_or_else(|| format!("{ENV} missing for DKR KV cache operations"))
            .and_then(|root| {
                KernelExecutionCache::discover(&root)
                    .map_err(|error| error.to_string())
            })
    }).as_ref().map_err(|error| candle_core::Error::Msg(error.clone()))
}

fn tensor(dtype: ElementType, shape: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: repr.map(str::to_owned),
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct CopyKey {
    layers: usize,
    pairs: usize,
    key_bytes: usize,
    value_bytes: usize,
}

thread_local! {
    static COPIES: RefCell<LaunchPlanCache<CopyKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn prepare_copy(key: CopyKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let mut args = vec![
        tensor(ElementType::I64, vec![key.layers],
               Some("kv_cache.layer_ptr_table.i64.v1")),
        tensor(ElementType::I64, vec![key.layers],
               Some("kv_cache.layer_ptr_table.i64.v1")),
        tensor(ElementType::I64, vec![key.pairs, 2],
               Some("kv_cache.block_pairs.i64.v1")),
    ];
    args.extend((0..2).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    args.extend((0..2).map(|_| Argument::Scalar { dtype: ElementType::U64 }));
    let mut req = LaunchRequest {
        operation: "kv_cache.copy_blocks.bytes.b256.v1".to_owned(),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.layers).map_err(candle_core::Error::wrap)?,
            u32::try_from(key.pairs).map_err(candle_core::Error::wrap)?,
            1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?.prepare_for_device(&mut req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1]
        || prepared.manifest.shared_mem_bytes != 0
    {
        candle_core::bail!("DKR KV block copy: launch geometry mismatch");
    }
    prepared.prevalidate_launch(&req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// The caller owns and retains the CUDA metadata allocation, the cache
/// storage read guards and the stream. The original CPU mapping has already
/// been validated. No device buffers are created in this adapter.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_copy_blocks(
    device: &Device,
    stream: &Arc<CudaStream>,
    key_table_ptr: u64,
    value_table_ptr: u64,
    pairs_ptr: u64,
    layers: usize,
    pairs: usize,
    key_block_bytes: usize,
    value_block_bytes: usize,
) -> Result<bool> {
    if !enabled("copy") || pairs == 0 || layers == 0
        || key_block_bytes == 0 || value_block_bytes == 0
        || layers > u32::MAX as usize
        || pairs > u32::MAX as usize
    {
        return Ok(false);
    }
    let key = CopyKey { layers, pairs, key_bytes: key_block_bytes,
                         value_bytes: value_block_bytes };
    let layers_u32 = u32::try_from(layers).map_err(candle_core::Error::wrap)?;
    let pairs_u32 = u32::try_from(pairs).map_err(candle_core::Error::wrap)?;
    let kb = u64::try_from(key_block_bytes).map_err(candle_core::Error::wrap)?;
    let vb = u64::try_from(value_block_bytes).map_err(candle_core::Error::wrap)?;
    COPIES.with(|cache| {
        cache.borrow_mut().with_plan(
            key, device, || prepare_copy(key, device),
            |plan| {
                unsafe {
                    plan.with_prevalidated_launch(|function, config| {
                        let mut launch = stream.launch_builder(function);
                        launch.arg(&key_table_ptr).arg(&value_table_ptr)
                            .arg(&pairs_ptr).arg(&layers_u32).arg(&pairs_u32)
                            .arg(&kb).arg(&vb);
                        launch.launch(config)?;
                        Ok(())
                    })
                }.map_err(|error| candle_core::Error::msg(error.to_string()))?;
                Ok(true)
            },
        )
    })
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct ReshapeKey {
    dtype: DType,
    tokens: usize,
    heads: usize,
    hd: usize,
    blocks: usize,
    block_size: usize,
    x: usize,
}

thread_local! {
    static WRITES: RefCell<LaunchPlanCache<ReshapeKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn prepare_reshape(key: ReshapeKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dtype = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("DKR KV reshape: unsupported dtype"),
    };
    let mut args = vec![
        tensor(dtype, vec![key.tokens, key.heads, key.hd], None),
        tensor(dtype, vec![key.tokens, key.heads, key.hd], None),
        tensor(dtype, vec![
            key.blocks, key.heads, key.hd / key.x, key.block_size, key.x,
        ], Some("vllm_k_blocked_x.v1")),
        tensor(dtype, vec![
            key.blocks, key.heads, key.hd, key.block_size,
        ], Some("vllm_v_token_strided.v1")),
        tensor(ElementType::I64, vec![key.tokens], None),
    ];
    args.extend((0..6).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut req = LaunchRequest {
        operation: "kv_cache.reshape_store.vllm.b256.v1".to_owned(),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.tokens).map_err(candle_core::Error::wrap)?,
            1, 1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?.prepare_for_device(&mut req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1]
        || prepared.manifest.shared_mem_bytes != 0
    {
        candle_core::bail!("DKR KV reshape: launch geometry mismatch");
    }
    prepared.prevalidate_launch(&req, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// A safe subset of reshape_and_cache. The original VLLM writer remains
/// responsible for FP8 scaled quantization and all strided variants.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_reshape(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
) -> Result<bool> {
    if !enabled("reshape")
        || k_scale.is_some() || v_scale.is_some()
        || !key.device().is_cuda()
        || !matches!(key.dtype(), DType::F16 | DType::BF16 | DType::F32)
        || key.dtype() != value.dtype()
        || key.dtype() != key_cache.dtype()
        || key.dtype() != value_cache.dtype()
        || slot_mapping.dtype() != DType::I64
        || !key.is_contiguous() || !value.is_contiguous()
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !slot_mapping.is_contiguous()
        || key.dims() != value.dims()
    {
        return Ok(false);
    }
    let (tokens, heads, hd) = match key.dims() {
        [tokens, heads, hd] => (*tokens, *heads, *hd),
        [batch, seq, heads, hd] => {
            let Some(tokens) = batch.checked_mul(*seq) else {
                return Ok(false);
            };
            (tokens, *heads, *hd)
        }
        _ => return Ok(false),
    };
    if tokens == 0 || heads == 0 || hd == 0
        || slot_mapping.dims() != [tokens]
        || [tokens, heads, hd].iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(false);
    }
    let (blocks, kh, khx, block_size, x) = key_cache.dims5()?;
    let (vblocks, vh, vhd, vbs) = value_cache.dims4()?;
    if blocks == 0 || block_size == 0 || x == 0
        || heads != kh || heads != vh || hd != vhd
        || khx.checked_mul(x) != Some(hd)
        || blocks != vblocks || block_size != vbs
        || [blocks, block_size, x].iter().any(|&v| v > u32::MAX as usize)
        || blocks.checked_mul(block_size).is_none()
        || blocks.checked_mul(block_size).is_some_and(|v| v > i64::MAX as usize)
        || heads.checked_mul(hd).is_none()
    {
        return Ok(false);
    }
    if ![value, key_cache, value_cache, slot_mapping].iter()
        .all(|other| key.device().same_device(other.device()))
    {
        return Ok(false);
    }
    let device = key.device();
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let (ks, kl) = key.storage_and_layout();
    let (vs, vl) = value.storage_and_layout();
    let (kcs, kcl) = key_cache.storage_and_layout();
    let (vcs, vcl) = value_cache.storage_and_layout();
    let (slots, sl) = slot_mapping.storage_and_layout();
    let (
        Storage::Cuda(ks), Storage::Cuda(vs), Storage::Cuda(kcs),
        Storage::Cuda(vcs), Storage::Cuda(slots)
    ) = (&*ks, &*vs, &*kcs, &*vcs, &*slots) else {
        candle_core::bail!("DKR KV reshape requires CUDA storages");
    };
    let launch_key = ReshapeKey {
        dtype: key.dtype(), tokens, heads, hd, blocks, block_size, x,
    };
    let tokens32 = u32::try_from(tokens).map_err(candle_core::Error::wrap)?;
    let heads32 = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let hd32 = u32::try_from(hd).map_err(candle_core::Error::wrap)?;
    let blocks32 = u32::try_from(blocks).map_err(candle_core::Error::wrap)?;
    let block32 = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let x32 = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    WRITES.with(|cache| {
        cache.borrow_mut().with_plan(
            launch_key, device,
            || prepare_reshape(launch_key, device),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (kp, _kg) = slice_ptr_on_stream(
                            ks.as_cuda_slice::<$ty>()?, kl.start_offset(), &stream,
                        );
                        let (vp, _vg) = slice_ptr_on_stream(
                            vs.as_cuda_slice::<$ty>()?, vl.start_offset(), &stream,
                        );
                        let (kcp, _kcg) = slice_ptr_on_stream(
                            kcs.as_cuda_slice::<$ty>()?, kcl.start_offset(), &stream,
                        );
                        let (vcp, _vcg) = slice_ptr_on_stream(
                            vcs.as_cuda_slice::<$ty>()?, vcl.start_offset(), &stream,
                        );
                        let (sp, _sg) = slice_ptr_on_stream(
                            slots.as_cuda_slice::<i64>()?, sl.start_offset(), &stream,
                        );
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut launch = stream.launch_builder(function);
                                launch.arg(&kp).arg(&vp).arg(&kcp).arg(&vcp)
                                    .arg(&sp).arg(&tokens32).arg(&heads32).arg(&hd32)
                                    .arg(&blocks32).arg(&block32).arg(&x32);
                                launch.launch(config)?;
                                Ok(())
                            })
                        }.map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        Ok(true)
                    }};
                }
                match launch_key.dtype {
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
                    DType::F32 => launch!(f32),
                    _ => unreachable!(),
                }
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    #[ignore = "requires prebuilt SM61 DKR KV cache ops PTX"]
    fn dkr_kv_cache_copy_gpu() -> Result<()> {
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F8E4M3] {
            let device = Device::new_cuda(0)?;
            let key = Tensor::from_vec(
                (0..24).map(|i| i as f32).collect::<Vec<_>>(),
                (6,4), &device,
            )?.to_dtype(dtype)?;
            let value = Tensor::from_vec(
                (100..124).map(|i| i as f32).collect::<Vec<_>>(),
                (6,4), &device,
            )?.to_dtype(dtype)?;
            let mut kc = key.narrow(0,1,4)?;
            let mut vc = value.narrow(0,1,4)?;
            let mapping = HashMap::from([(0,vec![2]),(1,vec![3])]);
            let source_k = key.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let source_v = value.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            super::super::cache::copy_blocks(vec![&mut kc],vec![&mut vc],&mapping)?;
            device.synchronize()?;
            let actual_key = key.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let actual_value = value.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            let mut expected_k = source_k.clone();
            let mut expected_v = source_v.clone();
            expected_k[3] = source_k[1].clone();
            expected_k[4] = source_k[2].clone();
            expected_v[3] = source_v[1].clone();
            expected_v[4] = source_v[2].clone();
            assert_eq!(actual_key,expected_k,"KV copy K {dtype:?}");
            assert_eq!(actual_value,expected_v,"KV copy V {dtype:?}");
            assert!(COPIES.with(|cache| cache.borrow().len()) > 0);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires prebuilt SM61 DKR KV cache ops PTX"]
    fn dkr_kv_cache_reshape_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let tokens = 3usize;
            let heads = 1usize;
            let hd = 8usize;
            let blocks = 2usize;
            let page = 2usize;
            let x = if dtype == DType::F32 {4} else {8};
            let keys: Vec<f32> = (0..tokens*hd)
                .map(|i| (i as f32) * 0.25).collect();
            let values: Vec<f32> = (0..tokens*hd)
                .map(|i| (100+i) as f32 * 0.25).collect();
            let k = Tensor::from_vec(keys.clone(), (tokens,heads,hd), &device)?
                .to_dtype(dtype)?;
            let v = Tensor::from_vec(values.clone(), (tokens,heads,hd), &device)?
                .to_dtype(dtype)?;
            let kc = Tensor::zeros((blocks, heads, hd/x, page, x), dtype, &device)?;
            let vc = Tensor::zeros((blocks, heads, hd, page), dtype, &device)?;
            let slot = Tensor::from_vec(vec![2i64,-1,1], tokens, &device)?;
            super::super::paged_attention::reshape_and_cache(
                &k,&v,None,None,&kc,&vc,&slot,
            )?;
            device.synchronize()?;
            let mut expected_key = vec![0.0f32;blocks*heads*hd*page];
            let mut expected_value = expected_key.clone();
            for (token, slot) in [(0usize,2usize),(2,1)] {
                let block = slot / page;
                let pos = slot % page;
                for d in 0..hd {
                    let ki = block*hd*page + (d/x)*page*x + pos*x + d%x;
                    let vi = block*hd*page + d*page + pos;
                    expected_key[ki] = keys[token*hd+d];
                    expected_value[vi] = values[token*hd+d];
                }
            }
            assert_eq!(
                kc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                expected_key, "KV reshape key {dtype:?}",
            );
            assert_eq!(
                vc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                expected_value, "KV reshape value {dtype:?}",
            );
            assert!(WRITES.with(|cache|cache.borrow().len()) > 0);
        }
        Ok(())
    }
}
