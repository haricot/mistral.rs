//! Typed routing and small Top-K; preserve the historical specialized kernels.
use crate::{
    cuda::kernel_runtime::{kernel_cache, LaunchPlanCache},
    ops::{MoeRouterTopKConfig, TopKOutput},
};
use candle_core::{
    backend::BackendStorage,
    cuda_backend::{cudarc::driver::{DevicePtr, DevicePtrMut, DeviceSlice, PushKernelArg}, CudaStorage, CudaStorageSlice},
    DType, Device, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    model_scope::OperationRequirement,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const EXPERTS: [usize; 11] = [1,2,4,8,16,32,64,128,256,512,576];
const DEFAULT_SHARED_CEILING: u32 = 48*1024;

#[derive(Clone, Copy, Eq, PartialEq)]
enum Kind { Small, Router { bias: bool, scale: bool } }

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key { kind: Kind, dtype: ElementType, rows: i32, cols: i32, k: i32 }

thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn dtype(dt: DType) -> Result<ElementType> {
    match dt {
        DType::F32 => Ok(ElementType::F32),
        DType::F16 => Ok(ElementType::F16),
        DType::BF16 => Ok(ElementType::Bf16),
        _ => candle_core::bail!("router/top-k requires F32/F16/BF16"),
    }
}
fn dtype_tag(dt: ElementType) -> &'static str {
    match dt {
        ElementType::F32 => "f32",
        ElementType::F16 => "f16",
        ElementType::Bf16 => "bf16",
        _ => unreachable!("checked router dtype"),
    }
}
fn small_block(cols: usize) -> u32 {
    if cols <= 64 { 64 } else if cols <= 128 { 128 } else if cols <= 256 { 256 } else { 512 }
}
fn names(kind: Kind, dt: ElementType, cols: usize) -> (String,String) {
    let tag = dtype_tag(dt);
    match kind {
        Kind::Small => {
            let b = small_block(cols);
            (format!("sampling.small_topk.b{b}.v1"),format!("cuda.dkr_small_topk_{tag}_b{b}.v1"))
        }
        Kind::Router {bias,scale} => {
            let b=u8::from(bias); let s=u8::from(scale);
            (format!("sampling.moe_router.e{cols}.b{b}.s{s}.v1"),
             format!("cuda.dkr_moe_router_e{cols}_{tag}_b{b}_s{s}.v1"))
        }
    }
}

/// Capability producer: provide the actual per-layer expert count, router
/// activation dtype and presence of optional tensors. Union over all layers,
/// target/draft and adapters in the caller; this is not full model coverage.
pub fn dkr_moe_router_requirement(
    experts: usize, dt: DType, bias: bool, scale: bool,
) -> Result<OperationRequirement> {
    if !EXPERTS.contains(&experts) {
        candle_core::bail!("no specialized DKR router for {experts} experts");
    }
    let (operation,implementation)=names(Kind::Router{bias,scale},dtype(dt)?,experts);
    Ok(OperationRequirement{operation,implementations:Some(vec![implementation])})
}

/// Small Top-K preserves the original block-size policy. The producer must
/// enumerate every reachable input width (or conservatively all four tiers).
pub fn dkr_small_topk_requirement(cols: usize, dt: DType) -> Result<OperationRequirement> {
    let dt=dtype(dt)?;
    if cols==0 || cols.checked_mul(dt.size_bytes()+1).is_none_or(|n| n>DEFAULT_SHARED_CEILING as usize) {
        candle_core::bail!("small top-k width outside default shared memory ceiling");
    }
    let (operation,implementation)=names(Kind::Small,dt,cols);
    Ok(OperationRequirement{operation,implementations:Some(vec![implementation])})
}

fn key(kind: Kind, dt: DType, rows: usize, cols: usize, k: usize) -> Result<Key> {
    if rows==0 || cols==0 || k==0 || k>cols ||
        rows.checked_mul(cols).is_none_or(|n| n>i32::MAX as usize) ||
        rows.checked_mul(k).is_none_or(|n| n>i32::MAX as usize) {
        candle_core::bail!("invalid router/top-k dimensions or signed device indexing overflow");
    }
    match kind {
        Kind::Small => { dkr_small_topk_requirement(cols,dt)?; }
        Kind::Router{bias,scale} => { dkr_moe_router_requirement(cols,dt,bias,scale)?; }
    }
    Ok(Key{kind,dtype:dtype(dt)?,rows:i32::try_from(rows).map_err(candle_core::Error::wrap)?,
        cols:i32::try_from(cols).map_err(candle_core::Error::wrap)?,
        k:i32::try_from(k).map_err(candle_core::Error::wrap)?})
}
fn t(dt:ElementType,n:usize)->Argument {
    Argument::Tensor{dtype:dt,shape:vec![n],contiguous:true,start_offset:0,representation:None}
}
fn s(dt:ElementType)->Argument { Argument::Scalar{dtype:dt} }
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let rows=usize::try_from(key.rows).map_err(candle_core::Error::wrap)?;
    let cols=usize::try_from(key.cols).map_err(candle_core::Error::wrap)?;
    let k=usize::try_from(key.k).map_err(candle_core::Error::wrap)?;
    let (operation,implementation)=names(key.kind,key.dtype,cols);
    let mut args=vec![t(key.dtype,rows*cols)];
    let (grid,block,ceiling,shared)=match key.kind {
        Kind::Small => {
            args.extend([t(key.dtype,rows*k),t(ElementType::U32,rows*k)]);
            args.extend(std::iter::repeat_n(s(ElementType::I32),3));
            let block=small_block(cols);
            let bytes_per_col=u32::try_from(key.dtype.size_bytes()+1).map_err(candle_core::Error::wrap)?;
            let shared=u32::try_from(cols).map_err(candle_core::Error::wrap)?*bytes_per_col;
            (u32::try_from(rows).map_err(candle_core::Error::wrap)?,[block,1,1],
                if block<512 {block*bytes_per_col} else {DEFAULT_SHARED_CEILING},shared)
        }
        Kind::Router{bias,scale} => {
            args.extend([t(ElementType::F32,rows*k),t(ElementType::U32,rows*k)]);
            if bias {args.push(t(ElementType::F32,cols));}
            if scale {args.push(t(ElementType::F32,cols));}
            args.extend(std::iter::repeat_n(s(ElementType::I32),4));
            args.extend(std::iter::repeat_n(s(ElementType::U32),2));
            args.extend(std::iter::repeat_n(s(ElementType::F32),4));
            (u32::try_from(rows.div_ceil(4)).map_err(candle_core::Error::wrap)?,[32,4,1],0,0)
        }
    };
    let mut request=LaunchRequest{operation,cc:0,args,grid:[grid,1,1],max_dynamic_shared_mem_bytes:ceiling};
    let p=kernel_cache()?.prepare_for_device(&mut request,device)
        .map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=block || p.manifest.shared_mem_bytes!=ceiling ||
        p.manifest.implementation_id!=implementation {
        candle_core::bail!("router/top-k selected profile differs from exact launch contract");
    }
    match key.kind {
        Kind::Small => unsafe {p.prevalidate_launch_with_default_shared(&request,device,shared)},
        Kind::Router{..} => p.prevalidate_launch(&request,device),
    }.map_err(|e|candle_core::Error::msg(e.to_string()))
}
fn with_plan<T>(key:Key,device:&Device,launch:impl FnOnce(&PrevalidatedLaunchPlan)->Result<T>)->Result<T>{
    PLANS.with(|p|p.borrow_mut().with_plan(key,device,||prepare(key,device),launch))
}
fn span(offset:usize,n:usize,len:usize)->Result<std::ops::Range<usize>>{
    let end=offset.checked_add(n).ok_or_else(||candle_core::Error::msg("router/top-k storage offset overflow"))?;
    if end>len {candle_core::bail!("router/top-k storage range exceeds allocation");}
    Ok(offset..end)
}

pub(crate) fn small_topk(input:&Tensor,k:usize)->Result<TopKOutput>{
    // Caller preserves final_logits_row policy; this adapter handles offsets
    // even if contiguous() returns a contiguous view of the original storage.
    let cols=input.dim(candle_core::D::Minus1)?;
    if cols==0 {candle_core::bail!("empty top-k columns");}
    let rows=input.elem_count()/cols;
    let key=key(Kind::Small,input.dtype(),rows,cols,k)?;
    let mut dims=input.dims().to_vec();
    *dims.last_mut().ok_or_else(||candle_core::Error::msg("empty top-k dims"))?=k;
    let count=rows*k;
    let (storage,layout)=input.storage_and_layout();
    let Storage::Cuda(storage)=&*storage else {candle_core::bail!("top-k requires CUDA");};
    if !layout.is_contiguous(){candle_core::bail!("top-k requires contiguous input");}
    let dev=storage.device();
    let stream=dev.cuda_stream();
    let device=Device::Cuda(dev.clone());
    let mut ids=unsafe{dev.alloc::<u32>(count)?};
    let (ids_ptr,ids_guard)=ids.device_ptr_mut(&stream);
    macro_rules! launch {
        ($variant:ident,$ty:ty)=>{{
            let CudaStorageSlice::$variant(src)=&storage.slice else {candle_core::bail!("top-k storage dtype mismatch");};
            let view=src.slice(span(layout.start_offset(),input.elem_count(),src.len())?);
            let (src_ptr,_src_guard)=view.device_ptr(&stream);
            let mut dst=unsafe{dev.alloc::<$ty>(count)?};
            let (dst_ptr,dst_guard)=dst.device_ptr_mut(&stream);
            with_plan(key,&device,|plan|{
                unsafe{plan.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&src_ptr).arg(&dst_ptr).arg(&ids_ptr).arg(&key.rows).arg(&key.cols).arg(&key.k);
                    b.launch(cfg)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))
            })?;
            drop(dst_guard);
            CudaStorageSlice::$variant(dst)
        }};
    }
    let values=match input.dtype(){
        DType::F32=>launch!(F32,f32),
        DType::F16=>launch!(F16,half::f16),
        DType::BF16=>launch!(BF16,half::bf16),
        _=>candle_core::bail!("top-k dtype"),
    };
    drop(ids_guard);
    Ok(TopKOutput{
        values:Tensor::from((Storage::Cuda(CudaStorage{slice:values,device:dev.clone()}),Shape::from_dims(&dims))),
        indices:Tensor::from((Storage::Cuda(CudaStorage{slice:CudaStorageSlice::U32(ids),device:dev.clone()}),Shape::from_dims(&dims))),
    })
}

pub(crate) fn moe_router_topk(
    logits:&Tensor, config:MoeRouterTopKConfig, selection_bias:Option<&Tensor>, expert_scale:Option<&Tensor>,
)->Result<TopKOutput>{
    let logits=logits.contiguous()?;
    let cols=logits.dim(candle_core::D::Minus1)?;
    if cols==0 {candle_core::bail!("empty expert dimension");}
    let rows=logits.elem_count()/cols;
    let key=key(Kind::Router{bias:selection_bias.is_some(),scale:expert_scale.is_some()},
                logits.dtype(),rows,cols,config.top_k)?;
    for v in [selection_bias,expert_scale].into_iter().flatten(){
        if v.dtype()!=DType::F32 || v.elem_count()!=cols || !v.device().same_device(logits.device()){
            candle_core::bail!("router optional tensor must be F32, same device and expert length");
        }
    }
    let bias=selection_bias.map(Tensor::contiguous).transpose()?;
    let scale=expert_scale.map(Tensor::contiguous).transpose()?;
    let bias_storage=bias.as_ref().map(Tensor::storage_and_layout);
    let scale_storage=scale.as_ref().map(Tensor::storage_and_layout);
    let (storage,layout)=logits.storage_and_layout();
    let Storage::Cuda(storage)=&*storage else{candle_core::bail!("router requires CUDA");};
    let dev=storage.device();
    let stream=dev.cuda_stream();
    let device=Device::Cuda(dev.clone());
    // Bind exactly the admitted contiguous span, not the allocation base.
    macro_rules! opt_view {
        ($binding:expr)=>{{
            if let Some((storage,layout))=$binding.as_ref(){
                let Storage::Cuda(storage)=&**storage else{candle_core::bail!("router optional CUDA storage");};
                let CudaStorageSlice::F32(src)=&storage.slice else{candle_core::bail!("router optional dtype");};
                if !layout.is_contiguous(){candle_core::bail!("router optional layout");}
                Some(src.slice(span(layout.start_offset(),cols,src.len())?))
            }else{None}
        }};
    }
    let bias_view=opt_view!(bias_storage);
    let scale_view=opt_view!(scale_storage);
    let bias_ptr=bias_view.as_ref().map(|v|v.device_ptr(&stream));
    let scale_ptr=scale_view.as_ref().map(|v|v.device_ptr(&stream));
    let count=rows*config.top_k;
    let mut values=unsafe{dev.alloc::<f32>(count)?};
    let mut ids=unsafe{dev.alloc::<u32>(count)?};
    let (values_ptr,values_guard)=values.device_ptr_mut(&stream);
    let (ids_ptr,ids_guard)=ids.device_ptr_mut(&stream);
    let score=config.score_function.as_i32();
    let weight=config.selected_weight.as_i32();
    let renorm=u32::from(config.renormalize);
    let (clip_min,clip_max,clamp)=match config.logit_clip{
        Some((lo,hi))=>(lo,hi,1u32),None=>(0.,0.,0u32),
    };
    if !layout.is_contiguous(){candle_core::bail!("router logits layout");}
    macro_rules! launch{
        ($variant:ident)=>{{
            let CudaStorageSlice::$variant(src)=&storage.slice else{candle_core::bail!("router logits dtype");};
            let view=src.slice(span(layout.start_offset(),logits.elem_count(),src.len())?);
            let (src_ptr,_src_guard)=view.device_ptr(&stream);
            with_plan(key,&device,|plan|{
                unsafe{plan.with_prevalidated_launch(|func,cfg|{
                    let mut b=stream.launch_builder(func);
                    b.arg(&src_ptr).arg(&values_ptr).arg(&ids_ptr);
                    if let Some((ptr,_))=&bias_ptr {b.arg(ptr);}
                    if let Some((ptr,_))=&scale_ptr {b.arg(ptr);}
                    b.arg(&key.rows).arg(&key.k).arg(&score).arg(&weight).arg(&renorm).arg(&clamp)
                        .arg(&clip_min).arg(&clip_max).arg(&config.norm_min).arg(&config.output_scale);
                    b.launch(cfg)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))
            })?;
        }};
    }
    match logits.dtype(){
        DType::F32=>launch!(F32),DType::F16=>launch!(F16),DType::BF16=>launch!(BF16),
        _=>candle_core::bail!("router dtype"),
    }
    drop(values_guard);drop(ids_guard);
    let mut dims=logits.dims().to_vec();
    *dims.last_mut().ok_or_else(||candle_core::Error::msg("router dims"))?=config.top_k;
    Ok(TopKOutput{
        values:Tensor::from((Storage::Cuda(CudaStorage{slice:CudaStorageSlice::F32(values),device:dev.clone()}),Shape::from_dims(&dims))),
        indices:Tensor::from((Storage::Cuda(CudaStorage{slice:CudaStorageSlice::U32(ids),device:dev.clone()}),Shape::from_dims(&dims))),
    })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests{
    use super::*;
    use crate::ops::{MoeRouterScoreFunction,MoeRouterSelectedWeight};
    #[test]
    fn dkr_router_topk_capabilities(){
        let mut identities=std::collections::BTreeSet::new();
        for e in EXPERTS{
            for dt in [DType::F32,DType::F16,DType::BF16]{
                for b in [false,true]{
                    for s in [false,true]{
                        let req=dkr_moe_router_requirement(e,dt,b,s).unwrap();
                        assert!(identities.insert(req.implementations.unwrap()[0].clone()));
                    }
                }
            }
        }
        assert_eq!(identities.len(),132);
        assert!(dkr_moe_router_requirement(96,DType::F32,false,false).is_err());
        assert_eq!([64,65,129,257].map(small_block),[64,128,256,512]);
        assert!(key(Kind::Small,DType::F32,1,0,1).is_err());
        assert!(key(Kind::Small,DType::F32,1,2,3).is_err());
        assert!(span(4,4,7).is_err());
    }
    #[test]
    #[ignore="requires GPU and external router/top-k catalogue"]
    fn dkr_router_topk_gpu()->Result<()>{
        let device=Device::new_cuda(0)?;
        for dt in [DType::F32,DType::F16,DType::BF16]{
            for width in [7,64,65,128,129,256,257,576,1024]{
                let data:Vec<f32>=(0..width).map(|i| (i%37) as f32-18.).collect();
                let input=Tensor::from_vec(data.clone(),width,&device)?.to_dtype(dt)?;
                let got=small_topk(&input,4)?;
                // A contiguous nonzero-offset view must not read the allocation base.
                let padded=Tensor::cat(&[&Tensor::zeros(3,dt,&device)?,&input],0)?;
                let offset=padded.narrow(0,3,width)?;
                let offset_got=small_topk(&offset,4)?;
                assert_eq!(offset_got.indices.to_vec1::<u32>()?,got.indices.to_vec1::<u32>()?);
                let mut expected:Vec<usize>=(0..width).collect();
                expected.sort_by(|&a,&b|data[b].total_cmp(&data[a]).then(a.cmp(&b)));
                assert_eq!(got.indices.to_vec1::<u32>()?,expected[..4].iter().map(|&i|u32::try_from(i).unwrap()).collect::<Vec<_>>());
                let expected_values:Vec<f32>=expected[..4].iter().map(|&i|data[i]).collect();
                assert_eq!(got.values.to_dtype(DType::F32)?.to_vec1::<f32>()?,expected_values);
            }
            for e in EXPERTS{
                let data:Vec<f32>=(0..3*e).map(|i| ((i%e) as f32)/(e as f32)).collect();
                let cpu=Tensor::from_vec(data,(3,e),&Device::Cpu)?.to_dtype(dt)?;
                let gpu=cpu.to_device(&device)?;
                let bias=Tensor::from_vec(vec![-2.0f32;e],e,&device)?;
                let scale=Tensor::from_vec(vec![1.5f32;e],e,&device)?;
                let rounded=cpu.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                for b in [None,Some(&bias)]{
                    for s in [None,Some(&scale)]{
                        for score_function in [MoeRouterScoreFunction::Raw,MoeRouterScoreFunction::Sigmoid,MoeRouterScoreFunction::Softmax]{
                            let cfg=MoeRouterTopKConfig{top_k:e.min(3),score_function,
                                selected_weight:MoeRouterSelectedWeight::Score,renormalize:false,norm_min:0.,output_scale:1.,logit_clip:None};
                            let got=moe_router_topk(&gpu,cfg,b,s)?;
                            let indices=got.indices.to_vec2::<u32>()?;
                            let weights=got.values.to_vec2::<f32>()?;
                            for (row,(ids,values)) in rounded.iter().zip(indices.iter().zip(&weights)){
                                let max=row.iter().copied().fold(f32::NEG_INFINITY,f32::max);
                                let denom: f32=row.iter().map(|x| (x-max).exp()).sum();
                                let scores:Vec<f32>=row.iter().map(|x|match score_function{
                                    MoeRouterScoreFunction::Raw=>*x,
                                    MoeRouterScoreFunction::Sigmoid=>1./(1.+(-x).exp()),
                                    MoeRouterScoreFunction::Softmax=>(x-max).exp()/denom,
                                }).collect();
                                let mut order:Vec<usize>=(0..e).collect();
                                order.sort_by(|&a,&b|row[b].total_cmp(&row[a]).then(a.cmp(&b)));
                                // Constant bias does not change order, but negative bias on
                                // sub-warp routers catches padded-lane selection.
                                for (j,&expected) in order[..cfg.top_k].iter().enumerate(){
                                    let actual=usize::try_from(ids[j]).unwrap();
                                    assert!(actual<e,"padded expert selected");
                                    assert!((row[actual]-row[expected]).abs()<1e-6);
                                    let expected_weight=scores[actual]*if s.is_some(){1.5}else{1.};
                                    assert!((values[j]-expected_weight).abs()<2e-5);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
