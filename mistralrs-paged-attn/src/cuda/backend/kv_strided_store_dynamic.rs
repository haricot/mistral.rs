//! Strided VLLM cache write with bounded physical spans and cached typed plans.
use super::slice_ptr_on_stream;
use candle_core::{
    cuda::cudarc::driver::{DeviceSlice, PushKernelArg},
    DType, Device, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, sync::OnceLock};

const ENV: &str = "DKR_KERNEL_PACKAGES";
const BLOCK: u32 = 256;
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| {
        KernelExecutionCache::package_root()
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
struct ReshapeKey {
    dtype: DType,
    tokens: usize,
    heads: usize,
    hd: usize,
    blocks: usize,
    block_size: usize,
    x: usize,
    k_stride: usize,
    v_stride: usize,
    k_span: usize,
    v_span: usize,
    scaled: bool,
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
    let cache_dtype = if key.scaled { ElementType::Fp8E4m3 } else { dtype };
    // These are contiguous physical memory intervals, not logical tensor views.
    let mut args = vec![
        tensor(dtype, vec![key.k_span], Some("kv_cache.row_span.v1")),
        tensor(dtype, vec![key.v_span], Some("kv_cache.row_span.v1")),
        tensor(cache_dtype, vec![
            key.blocks, key.heads, key.hd / key.x, key.block_size, key.x,
        ], Some("vllm_k_blocked_x.v1")),
        tensor(cache_dtype, vec![
            key.blocks, key.heads, key.hd, key.block_size,
        ], Some("vllm_v_token_strided.v1")),
        tensor(ElementType::I64, vec![key.tokens], None),
    ];
    if key.scaled {
        args.extend([tensor(ElementType::F32, vec![1], None),
                     tensor(ElementType::F32, vec![1], None)]);
    }
    args.extend((0..6).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    args.extend((0..2).map(|_| Argument::Scalar { dtype: ElementType::U64 }));
    let mut req = LaunchRequest {
        operation: if key.scaled {
            "kv_cache.fp8_store_strided.vllm.b256.v1"
        } else {
            "kv_cache.store_strided.vllm.b256.v1"
        }.to_owned(),
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

fn row_span(tokens: usize, width: usize, stride: usize) -> Option<usize> {
    if tokens == 0 || width == 0 || stride < width {
        return None;
    }
    (tokens - 1).checked_mul(stride)?.checked_add(width)
}

/// A uniform token stride and dense heads are required; cache storage remains canonical.
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
    let scaled = key_cache.dtype() == DType::F8E4M3;
    if !key.device().is_cuda()
        || !(match (scaled, k_scale, v_scale) {
            (true, Some(k), Some(v)) => super::kv_fp8_scales_valid(k, v, key.device()),
            (false, None, None) => key_cache.dtype() == key.dtype(),
            _ => false,
        })
        || !matches!(key.dtype(), DType::F16 | DType::BF16 | DType::F32)
        || key.dtype() != value.dtype()
        || value_cache.dtype() != key_cache.dtype()
        || key_cache.rank() != 5 || value_cache.rank() != 4
        || slot_mapping.dtype() != DType::I64
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !slot_mapping.is_contiguous()
        || key.dims() != value.dims()
    {
        return Ok(false);
    }
    // Validate logical layouts before borrowing the underlying buffers for launch.
    let (tokens, heads, hd, k_stride) = {
        let (_, layout) = key.storage_and_layout();
        let Ok(shape) = super::cache_input_layout(layout, "key", "DKR strided store") else {
            return Ok(false);
        };
        shape
    };
    let (vt, vh, vd, v_stride) = {
        let (_, layout) = value.storage_and_layout();
        let Ok(shape) = super::cache_input_layout(layout, "value", "DKR strided store") else {
            return Ok(false);
        };
        shape
    };
    if (tokens, heads, hd) != (vt, vh, vd) {
        return Ok(false);
    }
    let Some(width) = heads.checked_mul(hd) else { return Ok(false); };
    let (Some(k_span), Some(v_span)) = (row_span(tokens, width, k_stride), row_span(tokens, width, v_stride)) else {
        return Ok(false);
    };
    // Leave the old fast path in charge when both tensors are contiguous.
    if key.is_contiguous() && value.is_contiguous() { return Ok(false); }
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
    let scale_storage = match (k_scale, v_scale) {
        (Some(k), Some(v)) => Some((k.storage_and_layout(), v.storage_and_layout())),
        _ => None,
    };
    let scale_ptrs = if let Some(((kss, ksl), (vss, vsl))) = &scale_storage {
        let (Storage::Cuda(kss), Storage::Cuda(vss)) = (&**kss, &**vss) else {
            candle_core::bail!("DKR strided FP8 scales require CUDA storage");
        };
        let (ksp, ksg) = slice_ptr_on_stream(kss.as_cuda_slice::<f32>()?, ksl.start_offset(), &stream);
        let (vsp, vsg) = slice_ptr_on_stream(vss.as_cuda_slice::<f32>()?, vsl.start_offset(), &stream);
        Some((ksp, vsp, ksg, vsg))
    } else { None };
    let (
        Storage::Cuda(ks), Storage::Cuda(vs), Storage::Cuda(kcs),
        Storage::Cuda(vcs), Storage::Cuda(slots)
    ) = (&*ks, &*vs, &*kcs, &*vcs, &*slots) else {
        candle_core::bail!("DKR KV reshape requires CUDA storages");
    };
    let launch_key = ReshapeKey {
        dtype: key.dtype(), tokens, heads, hd, blocks, block_size, x,
        k_stride, v_stride, k_span, v_span, scaled,
    };
    let tokens32 = u32::try_from(tokens).map_err(candle_core::Error::wrap)?;
    let heads32 = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let hd32 = u32::try_from(hd).map_err(candle_core::Error::wrap)?;
    let blocks32 = u32::try_from(blocks).map_err(candle_core::Error::wrap)?;
    let block32 = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let x32 = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    let k_stride64 = u64::try_from(k_stride).map_err(candle_core::Error::wrap)?;
    let v_stride64 = u64::try_from(v_stride).map_err(candle_core::Error::wrap)?;
    WRITES.with(|cache| {
        cache.borrow_mut().with_plan(
            launch_key, device,
            || prepare_reshape(launch_key, device),
            |plan| {
                macro_rules! launch {
                    ($ty:ty, $cache_ty:ty) => {{
                        let k_slice = ks.as_cuda_slice::<$ty>()?;
                        let v_slice = vs.as_cuda_slice::<$ty>()?;
                        if kl.start_offset().checked_add(k_span).is_none_or(|end| end > k_slice.len())
                            || vl.start_offset().checked_add(v_span).is_none_or(|end| end > v_slice.len()) {
                            candle_core::bail!("DKR strided KV span exceeds backing allocation");
                        }
                        let (kp, _kg) = slice_ptr_on_stream(
                            k_slice, kl.start_offset(), &stream,
                        );
                        let (vp, _vg) = slice_ptr_on_stream(
                            v_slice, vl.start_offset(), &stream,
                        );
                        let (kcp, _kcg) = slice_ptr_on_stream(
                            kcs.as_cuda_slice::<$cache_ty>()?, kcl.start_offset(), &stream,
                        );
                        let (vcp, _vcg) = slice_ptr_on_stream(
                            vcs.as_cuda_slice::<$cache_ty>()?, vcl.start_offset(), &stream,
                        );
                        let (sp, _sg) = slice_ptr_on_stream(
                            slots.as_cuda_slice::<i64>()?, sl.start_offset(), &stream,
                        );
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut launch = stream.launch_builder(function);
                                launch.arg(&kp).arg(&vp).arg(&kcp).arg(&vcp)
                                    .arg(&sp);
                                if let Some((ksp, vsp, _, _)) = &scale_ptrs {
                                    launch.arg(ksp).arg(vsp);
                                }
                                launch.arg(&tokens32).arg(&heads32).arg(&hd32)
                                    .arg(&blocks32).arg(&block32).arg(&x32)
                                    .arg(&k_stride64).arg(&v_stride64);
                                launch.launch(config)?;
                                Ok(())
                            })
                        }.map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        Ok(true)
                    }};
                }
                match (launch_key.dtype, scaled) {
                    (DType::F16, false) => launch!(half::f16, half::f16),
                    (DType::BF16, false) => launch!(half::bf16, half::bf16),
                    (DType::F32, false) => launch!(f32, f32),
                    (DType::F16, true) => launch!(half::f16, float8::F8E4M3),
                    (DType::BF16, true) => launch!(half::bf16, float8::F8E4M3),
                    (DType::F32, true) => launch!(f32, float8::F8E4M3),
                    _ => unreachable!(),
                }
            },
        )
    })
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dkr_kv_strided_span_checked() {
        assert_eq!(row_span(4, 32, 48), Some(176));
        assert_eq!(row_span(1, 32, 48), Some(32));
        assert_eq!(row_span(4, 32, 31), None);
        assert_eq!(row_span(0, 32, 48), None);
        assert_eq!(row_span(usize::MAX, 32, 48), None);
    }

    #[test]
    #[ignore = "requires staged kv-fp8-io image and physical CUDA"]
    fn dkr_kv_strided_store_gpu() -> Result<()> {
        let _evidence = mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev = Device::new_cuda(0)?;
        let (tokens, heads, hd, page, blocks) = (4usize, 2usize, 16usize, 2usize, 2usize);
        let slots = Tensor::from_vec(vec![2i64, -1, 0, 3], tokens, &dev)?;
        let ks = Tensor::from_vec(vec![0.5f32], 1, &dev)?;
        let vs = Tensor::from_vec(vec![2.0f32], 1, &dev)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for scaled in [false, true] {
                for rank4 in [false, true] {
                    // K has one padding head and V two; narrow starting
                    // at head 1 makes both base pointers nonzero offsets.
                    let make = |extra: usize, shift: usize, scale: f32| -> Result<Tensor> {
                        let backing = Tensor::from_vec((0..tokens*(heads+extra)*hd)
                            .map(|i| {
                                let t = i/((heads+extra)*hd);
                                let h = i/hd%(heads+extra);
                                let d = i%hd;
                                if h == 0 || h > heads { 100.0 }
                                else { ((t+h+d+shift)%8) as f32*scale }
                            }).collect::<Vec<_>>(), (tokens,heads+extra,hd), &dev)?.to_dtype(dtype)?;
                        if rank4 {
                            backing.reshape((2,2,heads+extra,hd))?.narrow(2,1,heads)
                        } else {
                            backing.narrow(1,1,heads)
                        }
                    };
                    let k = make(1, 0, if scaled {0.5} else {1.0})?;
                    let v = make(2, 3, if scaled {2.0} else {1.0})?;
                    assert!(!k.is_contiguous() && !v.is_contiguous());
                    let x = if scaled {16} else if dtype == DType::F32 {4} else {8};
                    let cache_dtype = if scaled {DType::F8E4M3} else {dtype};
                    let kc = Tensor::zeros((blocks,heads,hd/x,page,x), DType::F32,&dev)?.to_dtype(cache_dtype)?;
                    let vc = Tensor::zeros((blocks,heads,hd,page), DType::F32,&dev)?.to_dtype(cache_dtype)?;
                    let scales = if scaled {(Some(&ks),Some(&vs))} else {(None,None)};
                    super::super::paged_attention::reshape_and_cache(&k,&v,scales.0,scales.1,&kc,&vc,&slots)?;
                    dev.synchronize()?;
                    let mut ek = vec![0f32;blocks*heads*hd*page];
                    let mut ev = ek.clone();
                    for (t, slot) in [(0usize,2usize),(2,0),(3,3)] {
                        for h in 0..heads {
                            for d in 0..hd {
                                let base=(slot/page*heads+h)*hd*page;
                                ek[base+(d/x)*page*x+(slot%page)*x+d%x] = ((t+h+1+d)%8) as f32;
                                ev[base+d*page+slot%page] = ((t+h+1+d+3)%8) as f32;
                            }
                        }
                    }
                    assert_eq!(kc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, ek);
                    assert_eq!(vc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, ev);
                    eprintln!("DKR_R126_KV_STRIDED=PASS dtype={dtype:?} scaled={scaled} rank4={rank4}");
                }
            }
        }
        Ok(())
    }
}
