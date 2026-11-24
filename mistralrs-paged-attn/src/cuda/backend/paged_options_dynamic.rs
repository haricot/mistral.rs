//! Typed paged options: exact trailing window, ALiBi and one global sink.
//! Reuses the V2 workspace and no-sink merge; no legacy FFI.

use super::{paged_attention::workspace_ensure, slice_ptr_on_stream};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, sync::OnceLock};

const ENV: &str = "DKR_KERNEL_PACKAGES";
const PARTITION: usize = 256;
const MAX_CONTEXT: usize = 32_768;
const BLOCK: u32 = 128;
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();
static ENABLED: OnceLock<bool> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| {
        std::env::var("MISTRALRS_DKR_PAGED_OPTIONS")
            .map_or(true, |value| value != "0")
    })
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG
        .get_or_init(|| {
            KernelExecutionCache::package_root()
                .ok_or_else(|| format!("{ENV} not set for external paged-v2"))
                .and_then(|path| {
                    KernelExecutionCache::discover(&path)
                        .map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Stage {
    Partition,
    Merge,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Partition => "partition",
            Self::Merge => "merge",
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    scaled: bool,
    alibi: bool,
    sink: bool,
    window: u32,
    cap_bits: u32,
    dtype: DType,
    seqs: usize,
    heads: usize,
    kv_heads: usize,
    head_size: usize,
    blocks: usize,
    block_size: usize,
    x: usize,
    table_stride: usize,
    max_context: usize,
    parts: usize,
    scale_bits: u32,
}

thread_local! {
    static PLANS: RefCell<LaunchPlanCache<(Key, Stage), PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
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

fn prepare(
    key: Key,
    stage: Stage,
    device: &Device,
) -> Result<PrevalidatedLaunchPlan> {
    let dt = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("unsupported DKR paged-v2 activation dtype"),
    };
    let partial_shape = vec![key.seqs, key.heads, key.parts, key.head_size];
    let lse_shape = vec![key.seqs, key.heads, key.parts];
    let (args, grid) = match stage {
        Stage::Partition => {
            let cache_dt = if key.scaled { ElementType::Fp8E4m3 } else { dt };
            let mut args = vec![
                tensor(dt, vec![key.seqs, key.heads, key.head_size], None),
                tensor(cache_dt, vec![
                    key.blocks, key.kv_heads, key.head_size / key.x,
                    key.block_size, key.x,
                ], Some("vllm_k_blocked_x.v1")),
                tensor(cache_dt, vec![
                    key.blocks, key.kv_heads, key.head_size, key.block_size,
                ], Some("vllm_v_token_strided.v1")),
                tensor(ElementType::U32, vec![key.seqs, key.table_stride], None),
                tensor(ElementType::U32, vec![key.seqs], None),
            ];
            if key.scaled {
                args.extend([tensor(ElementType::F32, vec![1], None), tensor(ElementType::F32, vec![1], None)]);
            }
            if key.alibi { args.push(tensor(ElementType::F32, vec![key.heads], None)); }
            args.extend([tensor(ElementType::F32, partial_shape, None), tensor(ElementType::F32, lse_shape, None)]);
            args.extend(
                (0..9).map(|_| Argument::Scalar { dtype: ElementType::U32 }),
            );
            args.extend([Argument::Scalar { dtype: ElementType::F32 }, Argument::Scalar { dtype: ElementType::U32 }, Argument::Scalar { dtype: ElementType::F32 }]);
            (
                args,
                [
                    u32::try_from(key.heads).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.seqs).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.parts).map_err(candle_core::Error::wrap)?,
                ],
            )
        }
        Stage::Merge => {
            let mut args = vec![
                tensor(ElementType::F32, partial_shape, None),
                tensor(ElementType::F32, lse_shape, None),
            ];
            if key.sink { args.push(tensor(ElementType::F32, vec![key.heads], None)); }
            args.push(tensor(dt, vec![key.seqs, key.heads, key.head_size], None));
            args.extend(
                (0..4).map(|_| Argument::Scalar { dtype: ElementType::U32 }),
            );
            (
                args,
                [
                    u32::try_from(key.heads).map_err(candle_core::Error::wrap)?,
                    u32::try_from(key.seqs).map_err(candle_core::Error::wrap)?,
                    1,
                ],
            )
        }
    };
    let mut request = LaunchRequest {
        operation: match stage {
            Stage::Partition => format!("attention.paged.options.{}_{}.p256.v1", if key.scaled {"fp8"} else {"plain"}, if key.alibi {"alibi"} else {"noalibi"}),
            Stage::Merge if key.sink => "attention.paged.options.sink_merge.p256.v1".into(),
            Stage::Merge => "attention.paged.v2.merge.p256.v1".into(),
        },
        cc: 0,
        args,
        grid,
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK, 1, 1] ||
        prepared.manifest.shared_mem_bytes != 0
    {
        candle_core::bail!("DKR paged-v2: unexpected block/shared-memory geometry");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

fn workspace_layout(key: Key) -> Result<(usize, usize)> {
    let count = key.seqs.checked_mul(key.heads)
        .and_then(|count| count.checked_mul(key.parts))
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partition count overflow"))?;
    let partial_elems = count.checked_mul(key.head_size)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partial count overflow"))?;
    let partial_bytes = partial_elems.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 partial allocation overflow"))?;
    let lse_offset = partial_bytes.checked_add(15)
        .map(|value| value & !15)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 offset overflow"))?;
    let lse_bytes = count.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 lse allocation overflow"))?;
    let total_bytes = lse_offset.checked_add(lse_bytes)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 workspace overflow"))?;
    Ok((lse_offset, total_bytes))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_decode(
    q: &CudaStorage,
    q_layout: &Layout,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_table: &Tensor,
    context_lens: &Tensor,
    softmax_scale: f32,
    softcapping: f32,
    max_context_len: usize,
    scales: Option<(&Tensor, &Tensor)>,
    alibi: Option<&Tensor>,
    sinks: Option<&Tensor>,
    window: Option<usize>,
) -> Result<Option<(CudaStorage, Shape)>> {
    if !enabled() {
        return Ok(None);
    }
    let dtype = q.dtype();
    let scaled = scales.is_some();
    let cache_dtype = if scaled { DType::F8E4M3 } else { dtype };
    if alibi.is_none() && sinks.is_none() && window.is_none() && softcapping == 1.0 { return Ok(None); }
    let window = match window { None => 0, Some(w) if w > 0 && w <= u32::MAX as usize => w as u32, _ => return Ok(None) };
    if !softcapping.is_finite() || softcapping <= 0.0
        || !softmax_scale.is_finite()
        || max_context_len == 0 || max_context_len > MAX_CONTEXT
        || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || key_cache.dtype() != cache_dtype || value_cache.dtype() != cache_dtype
        || block_table.dtype() != DType::U32
        || context_lens.dtype() != DType::U32
        || !q_layout.is_contiguous()
        || !key_cache.is_contiguous() || !value_cache.is_contiguous()
        || !block_table.is_contiguous() || !context_lens.is_contiguous()
        || q_layout.shape().rank() != 3
        || key_cache.rank() != 5 || value_cache.rank() != 4
        || block_table.rank() != 2 || context_lens.rank() != 1
    {
        return Ok(None);
    }
    let (seqs, heads, head_size) = q_layout.shape().dims3()?;
    let (blocks, kv_heads, head_width, block_size, x) = key_cache.dims5()?;
    let (vblocks, vheads, vhead_size, vblock_size) = value_cache.dims4()?;
    let (table_seqs, table_stride) = block_table.dims2()?;
    let parts = max_context_len.div_ceil(PARTITION);
    if seqs == 0 || seqs > 65_535
        || heads == 0 || kv_heads == 0 || blocks == 0
        || block_size == 0 || x == 0 || table_stride == 0
        || heads % kv_heads != 0
        || head_width.checked_mul(x) != Some(head_size)
        || blocks != vblocks || kv_heads != vheads
        || head_size != vhead_size || block_size != vblock_size
        || table_seqs != seqs || context_lens.dims() != [seqs]
        || max_context_len > table_stride.saturating_mul(block_size)
        || !matches!(head_size, 64 | 80 | 96 | 112 | 128 | 192 | 256 | 512)
        || [blocks, heads, kv_heads, head_size, block_size, x, table_stride, parts]
            .iter().any(|&v| v > u32::MAX as usize)
    {
        return Ok(None);
    }
    let cuda = q.device();
    let device = Device::Cuda(cuda.clone());
    if ![key_cache, value_cache, block_table, context_lens]
        .iter().all(|tensor| device.same_device(tensor.device()))
    {
        return Ok(None);
    }
    if let Some((k, v)) = scales {
        if ![k,v].iter().all(|t| t.dtype() == DType::F32 && t.elem_count() == 1 && t.is_contiguous() && device.same_device(t.device())) {
            return Ok(None);
        }
    }
    for t in [alibi, sinks].into_iter().flatten() {
        if t.dtype() != DType::F32 || t.dims() != [heads] || !t.is_contiguous() || !device.same_device(t.device()) { return Ok(None); }
    }
    let output_elements = seqs.checked_mul(heads)
        .and_then(|v| v.checked_mul(head_size))
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 output overflow"))?;
    let key = Key {
        scaled, alibi: alibi.is_some(), sink: sinks.is_some(), window, cap_bits: softcapping.to_bits(), dtype, seqs, heads, kv_heads, head_size,
        blocks, block_size, x, table_stride,
        max_context: max_context_len, parts,
        scale_bits: softmax_scale.to_bits(),
    };
    let (lse_offset, workspace_bytes) = workspace_layout(key)?;
    let (ks, kl) = key_cache.storage_and_layout();
    let (vs, vl) = value_cache.storage_and_layout();
    let (ts, tl) = block_table.storage_and_layout();
    let (ls, ll) = context_lens.storage_and_layout();
    let (Storage::Cuda(ks), Storage::Cuda(vs),
        Storage::Cuda(ts), Storage::Cuda(ls)) =
        (&*ks, &*vs, &*ts, &*ls)
    else {
        candle_core::bail!("DKR paged-v2 requires CUDA page and metadata tensors");
    };
    let stream = cuda.cuda_stream();
    let scale_storage = scales.map(|(k,v)| (k.storage_and_layout(), v.storage_and_layout()));
    let scale_ptrs = if let Some(((ks,kl),(vs,vl))) = &scale_storage {
        let (Storage::Cuda(ks), Storage::Cuda(vs)) = (&**ks,&**vs) else { candle_core::bail!("FP8 V2 scales require CUDA storage"); };
        let (kp,kg) = slice_ptr_on_stream(ks.as_cuda_slice::<f32>()?, kl.start_offset(), &stream);
        let (vp,vg) = slice_ptr_on_stream(vs.as_cuda_slice::<f32>()?, vl.start_offset(), &stream);
        Some((kp,vp,kg,vg))
    } else { None };
    let alibi_storage = alibi.map(Tensor::storage_and_layout);
    let sink_storage = sinks.map(Tensor::storage_and_layout);
    let alibi_ptr = if let Some((storage, layout)) = &alibi_storage {
        let Storage::Cuda(storage) = &**storage else { candle_core::bail!("ALiBi requires CUDA storage"); };
        Some(slice_ptr_on_stream(storage.as_cuda_slice::<f32>()?, layout.start_offset(), &stream))
    } else { None };
    let sink_ptr = if let Some((storage, layout)) = &sink_storage {
        let Storage::Cuda(storage) = &**storage else { candle_core::bail!("sinks require CUDA storage"); };
        Some(slice_ptr_on_stream(storage.as_cuda_slice::<f32>()?, layout.start_offset(), &stream))
    } else { None };
    // Shared with the historical v2 backend. Its guard serializes host
    // dispatches on a device while both kernels enqueue; capacity grows
    // geometrically, and old buffers survive for graph capture.
    let (workspace_ptr, _workspace_guard) = workspace_ensure(cuda, workspace_bytes)?;
    let partial_ptr = workspace_ptr;
    let lse_ptr = workspace_ptr
        .checked_add(u64::try_from(lse_offset).map_err(candle_core::Error::wrap)?)
        .ok_or_else(|| candle_core::Error::msg("DKR paged-v2 GPU offset overflow"))?;
    let num_blocks = u32::try_from(blocks).map_err(candle_core::Error::wrap)?;
    let num_seqs = u32::try_from(seqs).map_err(candle_core::Error::wrap)?;
    let num_heads = u32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let num_kv_heads = u32::try_from(kv_heads).map_err(candle_core::Error::wrap)?;
    let head_size_u32 = u32::try_from(head_size).map_err(candle_core::Error::wrap)?;
    let block_size_u32 = u32::try_from(block_size).map_err(candle_core::Error::wrap)?;
    let x_u32 = u32::try_from(x).map_err(candle_core::Error::wrap)?;
    let table_stride_u32 = u32::try_from(table_stride).map_err(candle_core::Error::wrap)?;
    let max_context_u32 = u32::try_from(max_context_len).map_err(candle_core::Error::wrap)?;
    let parts_u32 = u32::try_from(parts).map_err(candle_core::Error::wrap)?;
    let mut output = None;
    let out_shape = q_layout.shape().clone();

    macro_rules! launch_stages {
        ($ty:ty, $cache_ty:ty) => {{
            let (qp, _qg) = slice_ptr_on_stream(
                q.as_cuda_slice::<$ty>()?, q_layout.start_offset(), &stream,
            );
            let (kp, _kg) = slice_ptr_on_stream(
                ks.as_cuda_slice::<$cache_ty>()?, kl.start_offset(), &stream,
            );
            let (vp, _vg) = slice_ptr_on_stream(
                vs.as_cuda_slice::<$cache_ty>()?, vl.start_offset(), &stream,
            );
            let (tp, _tg) = slice_ptr_on_stream(
                ts.as_cuda_slice::<u32>()?, tl.start_offset(), &stream,
            );
            let (lp, _lg) = slice_ptr_on_stream(
                ls.as_cuda_slice::<u32>()?, ll.start_offset(), &stream,
            );
            let mut result = unsafe { cuda.alloc::<$ty>(output_elements)? };
            {
                let (op, _og) = result.device_ptr_mut(&stream);
                PLANS.with(|plans| {
                    let mut plans = plans.borrow_mut();
                    plans.with_plan(
                        (key, Stage::Partition), &device,
                        || prepare(key, Stage::Partition, &device),
                        |plan| {
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&qp).arg(&kp).arg(&vp)
                                        .arg(&tp).arg(&lp);
                                    if let Some((kp,vp,_,_)) = &scale_ptrs { launch.arg(kp).arg(vp); }
                                    if let Some((p,_)) = &alibi_ptr { launch.arg(p); }
                                    launch.arg(&partial_ptr).arg(&lse_ptr)
                                        .arg(&num_seqs).arg(&num_heads).arg(&num_kv_heads)
                                        .arg(&head_size_u32).arg(&block_size_u32).arg(&x_u32)
                                        .arg(&table_stride_u32).arg(&max_context_u32);
                                    launch.arg(&num_blocks).arg(&softmax_scale).arg(&window).arg(&softcapping);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|error| candle_core::Error::msg(error.to_string()))
                        },
                    )?;
                    plans.with_plan(
                        (key, Stage::Merge), &device,
                        || prepare(key, Stage::Merge, &device),
                        |plan| {
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut launch = stream.launch_builder(function);
                                    launch.arg(&partial_ptr).arg(&lse_ptr);
                                    if let Some((p,_)) = &sink_ptr { launch.arg(p); }
                                    launch.arg(&op)
                                        .arg(&num_seqs).arg(&num_heads).arg(&head_size_u32)
                                        .arg(&parts_u32);
                                    launch.launch(config)?;
                                    Ok(())
                                })
                            }.map_err(|error| candle_core::Error::msg(error.to_string()))
                        },
                    )
                })?;
            }
            output = Some(CudaStorage::wrap_cuda_slice(result, cuda.clone()));
        }};
    }
    match (dtype, scaled) {
        (DType::F16, false) => launch_stages!(half::f16, half::f16),
        (DType::BF16, false) => launch_stages!(half::bf16, half::bf16),
        (DType::F32, false) => launch_stages!(f32, f32),
        (DType::F16, true) => launch_stages!(half::f16, float8::F8E4M3),
        (DType::BF16, true) => launch_stages!(half::bf16, float8::F8E4M3),
        (DType::F32, true) => launch_stages!(f32, float8::F8E4M3),
        _ => unreachable!(),
    }
    Ok(output.map(|storage| (storage, out_shape)))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires physical CUDA and staged paged-options and paged-v2 CUBINs"]
    fn dkr_paged_options_gpu() -> Result<()> {
        let _evidence = mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev = Device::new_cuda(0)?;
        let (seqs, heads, hd, page, stride) = (3usize, 2usize, 64usize, 16usize, 34usize);
        let blocks = seqs * stride;
        let table_data: Vec<u32> = (0..blocks).map(|i| (blocks-1-i) as u32).collect();
        let table = Tensor::from_vec(table_data.clone(), (seqs,stride), &dev)?;
        let ks = Tensor::from_vec(vec![0.5f32],1,&dev)?;
        let vs = Tensor::from_vec(vec![2f32],1,&dev)?;
        let slopes = Tensor::from_vec(vec![0.03125f32,0.0625],heads,&dev)?;
        let sinks = Tensor::from_vec(vec![2f32,-1.0],heads,&dev)?;
        let kval = |t:usize| (t%5) as f32-2.0;
        let vval = |t:usize,d:usize,seq:usize| (((t+d+seq)%13) as f32-6.0)*0.125;
        for dtype in [DType::F16,DType::BF16,DType::F32] {
            let mut queries = vec![0f32;seqs*heads*hd];
            for row in 0..seqs*heads { queries[row*hd]=if row%2==0 {2.0} else {-2.0}; }
            let q=Tensor::from_vec(queries,(seqs,heads,hd),&dev)?.to_dtype(dtype)?;
            for scaled in [false,true] {
                let x = if scaled {16} else if dtype==DType::F32 {4} else {8};
                let mut k=vec![0f32;blocks*hd*page];
                let mut v=k.clone();
                for seq in 0..seqs {
                    for t in 0..stride*page {
                        let p=table_data[seq*stride+t/page] as usize;
                        let slot=t%page;
                        k[p*hd*page+slot*x]=kval(t);
                        for d in 0..hd { v[p*hd*page+d*page+slot]=vval(t,d,seq); }
                    }
                }
                let cd=if scaled {DType::F8E4M3} else {dtype};
                let kc=Tensor::from_vec(k,(blocks,1,hd/x,page,x),&dev)?.to_dtype(cd)?;
                let vc=Tensor::from_vec(v,(blocks,1,hd,page),&dev)?.to_dtype(cd)?;
                for max in [256usize,257,513] {
                    let contexts=[max,max/2,0];
                    let lens=Tensor::from_vec(contexts.map(|n|n as u32).to_vec(),seqs,&dev)?;
                    // Isolate each optional term, then combine them. Windows 1
                    // and 17 mask whole partitions; 257 crosses a boundary.
                    for (alibi,sink,window,cap) in [
                        (true,false,None,1f32),
                        (false,true,None,1.0),
                        (false,false,Some(1usize),1.0),
                        (false,false,None,2.0),
                        (true,true,Some(17),2.0),
                        (true,true,Some(257),1.0),
                    ] {
                        let run=||super::super::paged_attention::paged_attention_with_window(
                            &q, scaled.then_some(&ks), scaled.then_some(&vs),
                            &kc,&vc,&table,&lens,alibi.then_some(&slopes),max,1.0,cap,
                            sink.then_some(&sinks),window);
                        let actual=run()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let mut error=0f32;
                        for seq in 0..seqs {
                            let ctx=contexts[seq];
                            let start=ctx.saturating_sub(window.unwrap_or(ctx));
                            for h in 0..heads {
                                let sign=if h==0 {2.0f64} else {-2.0};
                                let logit=|t:usize| {
                                    let mut l=sign*kval(t) as f64*if scaled {0.5} else {1.0};
                                    if cap!=1.0 {l=(l/cap as f64).tanh()*cap as f64;}
                                    if alibi {l+=(if h==0 {0.03125} else {0.0625}) * (t as f64-ctx as f64+1.0);}
                                    l
                                };
                                let sink_log=if h==0 {2.0f64} else {-1.0};
                                let m=(start..ctx).map(logit).fold(if sink {sink_log} else {f64::NEG_INFINITY},f64::max);
                                for d in 0..hd {
                                    let mut denom=if sink {(sink_log-m).exp()} else {0.0};
                                    let mut numer=0f64;
                                    for t in start..ctx {
                                        let w=(logit(t)-m).exp();
                                        denom+=w;
                                        numer+=w*vval(t,d,seq) as f64*if scaled {2.0} else {1.0};
                                    }
                                    let expected=if denom>0.0 {(numer/denom) as f32} else {0.0};
                                    let got=actual[(seq*heads+h)*hd+d];
                                    assert!(got.is_finite());
                                    error=error.max((got-expected).abs());
                                }
                            }
                        }
                        let tol=match dtype {DType::F32=>3e-5,DType::F16=>0.001,_=>0.006};
                        assert!(error<=tol,"paged options {dtype:?} fp8={scaled} max={max} alibi={alibi} sink={sink} window={window:?} cap={cap} error={error}");
                        let cached=PLANS.with(|p|p.borrow().len());
                        assert!(cached>0);
                        let again=run()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        assert_eq!(actual,again);
                        assert_eq!(cached,PLANS.with(|p|p.borrow().len()));
                        eprintln!("DKR_R126_PAGED_OPTIONS=PASS dtype={dtype:?} fp8={scaled} context={max} alibi={alibi} sink={sink} window={window:?} cap={cap} max_abs={error:.8}");
                    }
                }
            }
        }
        Ok(())
    }
}
