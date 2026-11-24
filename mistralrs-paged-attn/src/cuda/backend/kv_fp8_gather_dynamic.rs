//! Scaled E4M3 VLLM gather; device-resident scales and cached typed plan.
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

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    dtype: DType,
    tokens: usize,
    seqs: usize,
    blocks: usize,
    table_stride: usize,
    heads: usize,
    head_size: usize,
    block_size: usize,
    x: usize,
}

thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| {
        KernelExecutionCache::package_root()
            .ok_or_else(|| format!("{ENV} not set for external KV gather"))
            .and_then(|path| KernelExecutionCache::discover(&path)
                .map_err(|err| err.to_string()))
    }).as_ref().map_err(|err| candle_core::Error::Msg(err.clone()))
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

fn prepare(key: Key, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dt = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("external KV gather dtype not supported"),
    };
    let mut args = vec![
        tensor(ElementType::Fp8E4m3, vec![key.blocks, key.heads, key.head_size / key.x,
                        key.block_size, key.x], Some("vllm_k_blocked_x.v1")),
        tensor(ElementType::Fp8E4m3, vec![key.blocks, key.heads, key.head_size, key.block_size],
               Some("vllm_v_token_strided.v1")),
        tensor(ElementType::I32, vec![key.seqs, key.table_stride], None),
        tensor(ElementType::I32, vec![key.seqs + 1], None),
        tensor(dt, vec![key.tokens, key.heads, key.head_size], None),
        tensor(dt, vec![key.tokens, key.heads, key.head_size], None),
        tensor(ElementType::F32, vec![1], None),
        tensor(ElementType::F32, vec![1], None),
    ];
    args.extend((0..8).map(|_| Argument::Scalar { dtype: ElementType::U32 }));
    let mut request = LaunchRequest {
        operation: "kv_cache.fp8_gather.vllm.b256.v1".into(),
        cc: 0,
        args,
        grid: [
            u32::try_from(key.tokens).map_err(candle_core::Error::wrap)?,
            1, 1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] {
        candle_core::bail!("DKR KV gather block must be 256 threads");
    }
    prepared.prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Returns false for all layouts not yet externally supported.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_gather(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    cu_seq_lens: &Tensor,
    key_output: &Tensor,
    value_output: &Tensor,
    tokens: usize,
    k_scale: &Tensor,
    v_scale: &Tensor,
) -> Result<bool> {
    let dtype = key_output.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || key_cache.dtype() != DType::F8E4M3
        || value_cache.dtype() != DType::F8E4M3
        || !key_cache.device().is_cuda()
        || !super::kv_fp8_scales_valid(k_scale, v_scale, key_cache.device())
        || key_cache.rank() != 5 || value_cache.rank() != 4
        || value_output.dtype() != dtype
        || block_table.dtype() != DType::I32
        || cu_seq_lens.dtype() != DType::I32
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !block_table.is_contiguous() || !cu_seq_lens.is_contiguous()
        || !key_output.is_contiguous() || !value_output.is_contiguous()
        || tokens == 0 || tokens > i32::MAX as usize
    {
        return Ok(false);
    }
    let (blocks, heads, hd_over_x, block_size, x) = key_cache.dims5()?;
    let (vblocks, vheads, head_size, vblock_size) = value_cache.dims4()?;
    let (seqs, table_stride) = block_table.dims2()?;
    if blocks != vblocks || heads != vheads || block_size != vblock_size
        || x == 0 || block_size == 0 || heads == 0 || head_size == 0
        || hd_over_x.checked_mul(x) != Some(head_size)
        || cu_seq_lens.dims() != [seqs + 1]
        || key_output.dims() != [tokens, heads, head_size]
        || value_output.dims() != [tokens, heads, head_size]
        || seqs == 0 || table_stride == 0 || blocks == 0
        || blocks > u32::MAX as usize
        || heads.checked_mul(head_size).is_none()
        || [seqs, table_stride, heads, head_size, block_size, x]
            .iter().any(|&v| v > i32::MAX as usize)
    {
        return Ok(false);
    }
    let dev = key_cache.device();
    if ![
        value_cache, block_table, cu_seq_lens, key_output, value_output
    ].iter().all(|tensor| dev.same_device(tensor.device())) {
        return Ok(false);
    }

    let (ks, kl) = key_cache.storage_and_layout();
    let (vs, vl) = value_cache.storage_and_layout();
    let (ts, tl) = block_table.storage_and_layout();
    let (ss, sl) = cu_seq_lens.storage_and_layout();
    let (ko, kol) = key_output.storage_and_layout();
    let (vo, vol) = value_output.storage_and_layout();
    let (Storage::Cuda(ks), Storage::Cuda(vs), Storage::Cuda(ts),
        Storage::Cuda(ss), Storage::Cuda(ko), Storage::Cuda(vo)) =
        (&*ks, &*vs, &*ts, &*ss, &*ko, &*vo)
    else {
        candle_core::bail!("DKR KV gather requires CUDA tensor storage");
    };
    let cuda = dev.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let (kss, ksl) = k_scale.storage_and_layout();
    let (vss, vsl) = v_scale.storage_and_layout();
    let (Storage::Cuda(kss), Storage::Cuda(vss)) = (&*kss, &*vss) else {
        candle_core::bail!("DKR FP8 scales must be CUDA storage");
    };
    let (ksp, _ksg) = slice_ptr_on_stream(kss.as_cuda_slice::<f32>()?, ksl.start_offset(), &stream);
    let (vsp, _vsg) = slice_ptr_on_stream(vss.as_cuda_slice::<f32>()?, vsl.start_offset(), &stream);
    let nblocks = u32::try_from(blocks).map_err(candle_core::Error::wrap)?;
    let key = Key { dtype, tokens, seqs, blocks, table_stride,
                    heads, head_size, block_size, x };
    let toks = u32::try_from(tokens).map_err(candle_core::Error::wrap)?;
    let nseqs = u32::try_from(seqs).map_err(candle_core::Error::wrap)?;
    let page = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let stride = u32::try_from(table_stride).map_err(candle_core::Error::wrap)?;
    let nheads = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let hsize = u32::try_from(head_size).map_err(candle_core::Error::wrap)?;
    let x_size = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    HOT.with(|cache| {
        cache.borrow_mut().with_plan(
            key, dev,
            || prepare(key, dev),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (kp, _kg) = slice_ptr_on_stream(
                            ks.as_cuda_slice::<float8::F8E4M3>()?, kl.start_offset(), &stream);
                        let (vp, _vg) = slice_ptr_on_stream(
                            vs.as_cuda_slice::<float8::F8E4M3>()?, vl.start_offset(), &stream);
                        let (tp, _tg) = slice_ptr_on_stream(
                            ts.as_cuda_slice::<i32>()?, tl.start_offset(), &stream);
                        let (sp, _sg) = slice_ptr_on_stream(
                            ss.as_cuda_slice::<i32>()?, sl.start_offset(), &stream);
                        let (kop, _kog) = slice_ptr_on_stream(
                            ko.as_cuda_slice::<$ty>()?, kol.start_offset(), &stream);
                        let (vop, _vog) = slice_ptr_on_stream(
                            vo.as_cuda_slice::<$ty>()?, vol.start_offset(), &stream);
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut launch = stream.launch_builder(function);
                                launch.arg(&kp).arg(&vp).arg(&tp).arg(&sp)
                                    .arg(&kop).arg(&vop).arg(&ksp).arg(&vsp)
                                    .arg(&toks).arg(&nseqs).arg(&page).arg(&stride)
                                    .arg(&nheads).arg(&hsize).arg(&x_size).arg(&nblocks);
                                launch.launch(config)?;
                                Ok(())
                            })
                        }.map_err(|err| candle_core::Error::msg(err.to_string()))?;
                        Ok(true)
                    }};
                }
                match dtype {
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
                    DType::F32 => launch!(f32),
                    _ => unreachable!(),
                }
            },
        )
    })
}

