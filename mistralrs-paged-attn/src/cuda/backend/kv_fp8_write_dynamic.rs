//! Scaled E4M3 VLLM cache write; separate typed operation and cached launch plan.
use super::slice_ptr_on_stream;
use candle_core::{
    cuda::cudarc::driver::PushKernelArg,
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
        tensor(ElementType::Fp8E4m3, vec![
            key.blocks, key.heads, key.hd / key.x, key.block_size, key.x,
        ], Some("vllm_k_blocked_x.v1")),
        tensor(ElementType::Fp8E4m3, vec![
            key.blocks, key.heads, key.hd, key.block_size,
        ], Some("vllm_v_token_strided.v1")),
        tensor(ElementType::I64, vec![key.tokens], None),
        tensor(ElementType::F32, vec![1], None),
        tensor(ElementType::F32, vec![1], None),
    ];
    args.extend((0..6).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut req = LaunchRequest {
        operation: "kv_cache.fp8_store.vllm.b256.v1".to_owned(),
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

/// Scaled FP8 canonical contiguous VLLM write. Other layouts are not admitted.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_reshape(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    k_scale: &Tensor,
    v_scale: &Tensor,
) -> Result<bool> {
    if !key.device().is_cuda()
        || !super::kv_fp8_scales_valid(k_scale, v_scale, key.device())
        || !matches!(key.dtype(), DType::F16 | DType::BF16 | DType::F32)
        || key.dtype() != value.dtype()
        || key_cache.dtype() != DType::F8E4M3
        || value_cache.dtype() != DType::F8E4M3
        || key_cache.rank() != 5 || value_cache.rank() != 4
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
    let (kss, ksl) = k_scale.storage_and_layout();
    let (vss, vsl) = v_scale.storage_and_layout();
    let (Storage::Cuda(kss), Storage::Cuda(vss)) = (&*kss, &*vss) else {
        candle_core::bail!("DKR FP8 scales must be CUDA storage");
    };
    let (ksp, _ksg) = slice_ptr_on_stream(kss.as_cuda_slice::<f32>()?, ksl.start_offset(), &stream);
    let (vsp, _vsg) = slice_ptr_on_stream(vss.as_cuda_slice::<f32>()?, vsl.start_offset(), &stream);
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
                            kcs.as_cuda_slice::<float8::F8E4M3>()?, kcl.start_offset(), &stream,
                        );
                        let (vcp, _vcg) = slice_ptr_on_stream(
                            vcs.as_cuda_slice::<float8::F8E4M3>()?, vcl.start_offset(), &stream,
                        );
                        let (sp, _sg) = slice_ptr_on_stream(
                            slots.as_cuda_slice::<i64>()?, sl.start_offset(), &stream,
                        );
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut launch = stream.launch_builder(function);
                                launch.arg(&kp).arg(&vp).arg(&kcp).arg(&vcp)
                                    .arg(&sp).arg(&ksp).arg(&vsp).arg(&tokens32).arg(&heads32).arg(&hd32)
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

    // Exact E4M3 values, saturation and ties-to-even, independently specified.
    // K/V use different binary-exact scales to catch swapped scale arguments.
    #[test]
    #[ignore = "requires staged kv-fp8-io CUBIN and CUDA GPU"]
    fn dkr_kv_fp8_io_gpu() -> Result<()> {
        let _evidence = mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev = Device::new_cuda(0)?;
        let input = [0.0f32, 1.0, -1.0, 1.0625, 1.1875, 500.0, -500.0, 0.001953125];
        let rounded = [0.0f32, 1.0, -1.0, 1.0, 1.25, 448.0, -448.0, 0.001953125];
        let (tokens, heads, hd, blocks, page, x) = (4usize, 2usize, 16usize, 2usize, 2usize, 16usize);
        let width = heads * hd;
        let ks = Tensor::from_vec(vec![0.5f32], 1, &dev)?;
        let vs = Tensor::from_vec(vec![2.0f32], 1, &dev)?;
        let slots = Tensor::from_vec(vec![2i64, -1, 0, 3], tokens, &dev)?;
        let table = Tensor::from_vec(vec![1i32, 0], (2, 1), &dev)?;
        let offsets = Tensor::from_vec(vec![0i32, 2, 4], 3, &dev)?;

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let k = Tensor::from_vec((0..tokens*width).map(|i| {
                input[(i + i/width + i/hd) % input.len()] * 0.5
            }).collect::<Vec<_>>(), (tokens, heads, hd), &dev)?.to_dtype(dtype)?;
            let v = Tensor::from_vec((0..tokens*width).map(|i| {
                input[(i + 3*i/width + i/hd + 2) % input.len()] * 2.0
            }).collect::<Vec<_>>(), (tokens, heads, hd), &dev)?.to_dtype(dtype)?;
            let kc = Tensor::zeros((blocks, heads, hd/x, page, x), DType::F32, &dev)?
                .to_dtype(DType::F8E4M3)?;
            let vc = Tensor::zeros((blocks, heads, hd, page), DType::F32, &dev)?
                .to_dtype(DType::F8E4M3)?;
            super::super::paged_attention::reshape_and_cache(
                &k, &v, Some(&ks), Some(&vs), &kc, &vc, &slots,
            )?;
            dev.synchronize()?;
            assert!(WRITES.with(|c| c.borrow().len()) > 0);

            let mut raw_k = vec![0f32; blocks * width * page];
            let mut raw_v = raw_k.clone();
            let mut expected_k = vec![0f32; tokens * width];
            let mut expected_v = expected_k.clone();
            for (token, slot, out_token) in [(0usize,2usize,0usize), (2,0,2), (3,3,1)] {
                let block = slot/page;
                let pos = slot%page;
                for h in 0..heads {
                    for d in 0..hd {
                        let i = token*width + h*hd + d;
                        let rk = rounded[(i + i/width + i/hd) % rounded.len()];
                        let rv = rounded[(i + 3*i/width + i/hd + 2) % rounded.len()];
                        let base = block*width*page + h*hd*page;
                        raw_k[base + (d/x)*page*x + pos*x + d%x] = rk;
                        raw_v[base + d*page + pos] = rv;
                        expected_k[out_token*width + h*hd + d] = rk*0.5;
                        expected_v[out_token*width + h*hd + d] = rv*2.0;
                    }
                }
            }
            assert_eq!(kc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, raw_k);
            assert_eq!(vc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, raw_v);

            for _ in 0..2 {
                let (ko, vo) = super::super::gather_kv::gather_kv_cache(
                    &kc, &vc, Some(&ks), Some(&vs), &table, &offsets, tokens, dtype,
                )?;
                assert_eq!(ko.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, expected_k);
                assert_eq!(vo.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?, expected_v);
            }
            println!("DKR_R126_KV_FP8_IO=PASS dtype={dtype:?} scaled=true padding=true saturation=true ties=true");
        }
        Ok(())
    }
}
