//! FlashInfer HND cache IO; independent of the split-KV decode scheduler.
use super::{cache_input_layout,slice_ptr_on_stream};
use crate::{KvCacheScales,DEFAULT_FP8_KV_CACHE_SCALES};
use candle_core::{cuda::cudarc::driver::{DeviceSlice,PushKernelArg},
    DType,Device,Result,Storage,Tensor};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument,ElementType,LaunchRequest},
    universal_cuda::{KernelExecutionCache,PrevalidatedLaunchPlan},
};
use std::{cell::RefCell,sync::OnceLock};

#[derive(Clone,Copy,Eq,PartialEq)]
struct Key {
    dtype:DType,scaled:bool,store:bool,tokens:usize,blocks:usize,heads:usize,hd:usize,page:usize,
    seqs:usize,stride:usize,bt:DType,ot:DType,ks:usize,vs:usize,kn:usize,vn:usize,
}
thread_local! {
    static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());
}
static CATALOG:OnceLock<std::result::Result<KernelExecutionCache,String>>=OnceLock::new();
fn catalog()->Result<&'static KernelExecutionCache>{
    CATALOG.get_or_init(||KernelExecutionCache::package_root()
        .ok_or_else(||"DKR HND IO catalogue unavailable".to_owned())
        .and_then(|p|KernelExecutionCache::discover(&p).map_err(|e|e.to_string())))
        .as_ref().map_err(|e|candle_core::Error::Msg(e.clone()))
}
fn elem(dt:DType)->Result<ElementType>{
    Ok(match dt {DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,
        DType::F32=>ElementType::F32,DType::I32=>ElementType::I32,DType::U32=>ElementType::U32,
        _=>candle_core::bail!("DKR HND unsupported dtype")})
}
fn meta_name(dt:DType)->&'static str {if dt==DType::I32 {"i32"}else{"u32"}}
fn arg(dtype:ElementType,shape:Vec<usize>,repr:Option<&str>)->Argument{
    Argument::Tensor{dtype,shape,contiguous:true,start_offset:0,representation:repr.map(str::to_owned)}
}
fn prepare(k:Key,dev:&Device)->Result<PrevalidatedLaunchPlan>{
    let dt=elem(k.dtype)?;
    let cd=if k.scaled{ElementType::Fp8E4m3}else{dt};
    let cache=||arg(cd,vec![k.blocks,k.heads,k.page,k.hd],Some("flashinfer.hnd.v1"));
    let kind=if k.scaled{"fp8"}else{"plain"};
    let mut args=if k.store{
        vec![arg(dt,vec![k.kn],Some("kv_cache.row_span.v1")),
             arg(dt,vec![k.vn],Some("kv_cache.row_span.v1")),cache(),cache(),
             arg(ElementType::I64,vec![k.tokens],None)]
    }else{
        vec![cache(),cache(),arg(elem(k.bt)?,vec![k.seqs,k.stride],None),
             arg(elem(k.ot)?,vec![k.seqs+1],None),
             arg(dt,vec![k.tokens,k.heads,k.hd],None),arg(dt,vec![k.tokens,k.heads,k.hd],None)]
    };
    args.extend((0..if k.store{5}else{7}).map(|_|Argument::Scalar{dtype:ElementType::U32}));
    if k.store{args.extend((0..2).map(|_|Argument::Scalar{dtype:ElementType::U64}));}
    if k.scaled{args.extend((0..2).map(|_|Argument::Scalar{dtype:ElementType::F32}));}
    let mode=if k.store{format!("store_{kind}")}else{format!("gather_{kind}_{}_{}",meta_name(k.bt),meta_name(k.ot))};
    let mut req=LaunchRequest{operation:format!("kv_cache.flashinfer.hnd.{mode}.b256.v1"),
        cc:0,args,grid:[k.tokens as u32,1,1],max_dynamic_shared_mem_bytes:0};
    let p=catalog()?.prepare_for_device(&mut req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[256,1,1] || p.manifest.shared_mem_bytes!=0{
        candle_core::bail!("DKR HND IO launch geometry mismatch");
    }
    p.prevalidate_launch(&req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))
}
fn base(kc:&Tensor,vc:&Tensor,dtype:DType,scales:KvCacheScales)->Result<Key>{
    let (blocks,heads,page,hd)=kc.dims4()?;
    let scaled=kc.dtype()==DType::F8E4M3;
    if !kc.device().is_cuda() || !kc.device().same_device(vc.device()) ||
        kc.dims()!=vc.dims() || kc.dtype()!=vc.dtype() || !kc.is_contiguous() || !vc.is_contiguous() ||
        !matches!(dtype,DType::F16|DType::BF16|DType::F32) || (!scaled && kc.dtype()!=dtype) ||
        !scales.k.is_finite() || !scales.v.is_finite() || scales.k<=0.0 || scales.v<=0.0 ||
        (!scaled && scales!=DEFAULT_FP8_KV_CACHE_SCALES) ||
        [blocks,heads,page,hd].iter().any(|&x|x==0 || x>i32::MAX as usize) {
        candle_core::bail!("DKR HND IO unsupported cache shape, dtype, device or scales");
    }
    Ok(Key{dtype,scaled,store:false,tokens:0,blocks,heads,hd,page,seqs:0,stride:0,
        bt:DType::I32,ot:DType::I32,ks:0,vs:0,kn:0,vn:0})
}
fn span(t:usize,w:usize,s:usize)->Result<usize>{
    if s<w || w==0 || t==0{candle_core::bail!("DKR HND invalid row span");}
    (t-1).checked_mul(s).and_then(|v|v.checked_add(w))
        .ok_or_else(||candle_core::Error::msg("DKR HND row span overflow"))
}
macro_rules! dispatch {
    ($key:expr,$run:ident)=>{{
        match ($key.dtype,$key.scaled){
            (DType::F16,false)=>$run!(half::f16,half::f16),
            (DType::BF16,false)=>$run!(half::bf16,half::bf16),
            (DType::F32,false)=>$run!(f32,f32),
            (DType::F16,true)=>$run!(half::f16,float8::F8E4M3),
            (DType::BF16,true)=>$run!(half::bf16,float8::F8E4M3),
            (DType::F32,true)=>$run!(f32,float8::F8E4M3),
            _=>unreachable!(),
        }
    }}
}
pub fn reshape_and_cache_flashinfer(k:&Tensor,v:&Tensor,kc:&Tensor,vc:&Tensor,slots:&Tensor,scales:KvCacheScales)->Result<()>{
    let mut key=base(kc,vc,k.dtype(),scales)?;
    let dev=kc.device();
    if ![k,v,slots].iter().all(|t|dev.same_device(t.device())) || v.dtype()!=k.dtype() ||
        slots.dtype()!=DType::I64 || !slots.is_contiguous(){
        candle_core::bail!("DKR HND store incompatible inputs");
    }
    let (ks,kl)=k.storage_and_layout();let (vs,vl)=v.storage_and_layout();
    let (t,h,d,kr)=cache_input_layout(kl,"key","DKR HND store")?;
    let (vt,vh,vd,vr)=cache_input_layout(vl,"value","DKR HND store")?;
    if (t,h,d)!=(vt,vh,vd) || (h,d)!=(key.heads,key.hd) || slots.dims()!=[t] || t>i32::MAX as usize{
        candle_core::bail!("DKR HND store shape mismatch");
    }
    if t==0{return Ok(());}
    let width=h.checked_mul(d).ok_or_else(||candle_core::Error::msg("DKR HND width overflow"))?;
    let kn=span(t,width,kr)?;let vn=span(t,width,vr)?;
    key.store=true;key.tokens=t;key.ks=kr;key.vs=vr;key.kn=kn;key.vn=vn;
    let (kcs,kcl)=kc.storage_and_layout();let (vcs,vcl)=vc.storage_and_layout();
    let (ss,sl)=slots.storage_and_layout();
    let (Storage::Cuda(ks),Storage::Cuda(vs),Storage::Cuda(kcs),Storage::Cuda(vcs),Storage::Cuda(ss))=
        (&*ks,&*vs,&*kcs,&*vcs,&*ss) else {candle_core::bail!("DKR HND store requires CUDA");};
    let stream=dev.as_cuda_device()?.cuda_stream();
    let (tokens,blocks,heads,hd,page)=(t as u32,key.blocks as u32,h as u32,d as u32,key.page as u32);
    let (kr,vr)=(kr as u64,vr as u64);
    macro_rules! run {($ty:ty,$ct:ty)=>{{
        let kb=ks.as_cuda_slice::<$ty>()?;let vb=vs.as_cuda_slice::<$ty>()?;
        if kl.start_offset().checked_add(kn).is_none_or(|n|n>kb.len()) ||
            vl.start_offset().checked_add(vn).is_none_or(|n|n>vb.len()){
            candle_core::bail!("DKR HND store span exceeds allocation");
        }
        let (kp,_kg)=slice_ptr_on_stream(kb,kl.start_offset(),&stream);
        let (vp,_vg)=slice_ptr_on_stream(vb,vl.start_offset(),&stream);
        let (kcp,_kcg)=slice_ptr_on_stream(kcs.as_cuda_slice::<$ct>()?,kcl.start_offset(),&stream);
        let (vcp,_vcg)=slice_ptr_on_stream(vcs.as_cuda_slice::<$ct>()?,vcl.start_offset(),&stream);
        let (sp,_sg)=slice_ptr_on_stream(ss.as_cuda_slice::<i64>()?,sl.start_offset(),&stream);
        PLANS.with(|p|p.borrow_mut().with_plan(key,dev,||prepare(key,dev),|plan|unsafe{
            plan.with_prevalidated_launch(|fun,config|{
                let mut l=stream.launch_builder(fun);
                l.arg(&kp).arg(&vp).arg(&kcp).arg(&vcp).arg(&sp)
                    .arg(&tokens).arg(&blocks).arg(&heads).arg(&hd).arg(&page).arg(&kr).arg(&vr);
                if key.scaled{l.arg(&scales.k).arg(&scales.v);}
                l.launch(config)?;Ok(())
            })
        }))
    }}}
    dispatch!(key,run)
}
pub fn gather_kv_cache_flashinfer(kc:&Tensor,vc:&Tensor,table:&Tensor,offsets:&Tensor,
    num_tokens:usize,out_dtype:DType,scales:KvCacheScales)->Result<(Tensor,Tensor)>{
    let mut key=base(kc,vc,out_dtype,scales)?;
    let dev=kc.device();
    let (seqs,stride)=table.dims2()?;
    if ![table,offsets].iter().all(|t|dev.same_device(t.device()) &&
        matches!(t.dtype(),DType::I32|DType::U32)) ||
        offsets.dims()!=[seqs+1] || [seqs,stride].iter().any(|&x|x==0 || x>i32::MAX as usize) ||
        num_tokens>i32::MAX as usize{
        candle_core::bail!("DKR HND gather incompatible metadata");
    }
    let ko=Tensor::zeros((num_tokens,key.heads,key.hd),out_dtype,dev)?;
    let vo=Tensor::zeros((num_tokens,key.heads,key.hd),out_dtype,dev)?;
    if num_tokens==0{return Ok((ko,vo));}
    let table=table.contiguous()?;let offsets=offsets.contiguous()?;
    key.tokens=num_tokens;key.seqs=seqs;key.stride=stride;key.bt=table.dtype();key.ot=offsets.dtype();
    {
        let (kcs,kcl)=kc.storage_and_layout();let (vcs,vcl)=vc.storage_and_layout();
        let (ts,tl)=table.storage_and_layout();let (os,ol)=offsets.storage_and_layout();
        let (ks,kl)=ko.storage_and_layout();let (vs,vl)=vo.storage_and_layout();
        let (Storage::Cuda(kcs),Storage::Cuda(vcs),Storage::Cuda(ts),Storage::Cuda(os),Storage::Cuda(ks),Storage::Cuda(vs))=
            (&*kcs,&*vcs,&*ts,&*os,&*ks,&*vs) else {candle_core::bail!("DKR HND gather requires CUDA");};
        let stream=dev.as_cuda_device()?.cuda_stream();
        let (tp,_tg)=if key.bt==DType::I32{slice_ptr_on_stream(ts.as_cuda_slice::<i32>()?,tl.start_offset(),&stream)}
            else{slice_ptr_on_stream(ts.as_cuda_slice::<u32>()?,tl.start_offset(),&stream)};
        let (op,_og)=if key.ot==DType::I32{slice_ptr_on_stream(os.as_cuda_slice::<i32>()?,ol.start_offset(),&stream)}
            else{slice_ptr_on_stream(os.as_cuda_slice::<u32>()?,ol.start_offset(),&stream)};
        let (tokens,seqs,blocks,heads,hd,page,stride)=(num_tokens as u32,seqs as u32,key.blocks as u32,
            key.heads as u32,key.hd as u32,key.page as u32,stride as u32);
        macro_rules! run {($ty:ty,$ct:ty)=>{{
            let (kcp,_kcg)=slice_ptr_on_stream(kcs.as_cuda_slice::<$ct>()?,kcl.start_offset(),&stream);
            let (vcp,_vcg)=slice_ptr_on_stream(vcs.as_cuda_slice::<$ct>()?,vcl.start_offset(),&stream);
            let (kp,_kg)=slice_ptr_on_stream(ks.as_cuda_slice::<$ty>()?,kl.start_offset(),&stream);
            let (vp,_vg)=slice_ptr_on_stream(vs.as_cuda_slice::<$ty>()?,vl.start_offset(),&stream);
            PLANS.with(|p|p.borrow_mut().with_plan(key,dev,||prepare(key,dev),|plan|unsafe{
                plan.with_prevalidated_launch(|fun,config|{
                    let mut l=stream.launch_builder(fun);
                    l.arg(&kcp).arg(&vcp).arg(&tp).arg(&op).arg(&kp).arg(&vp)
                        .arg(&tokens).arg(&seqs).arg(&blocks).arg(&heads).arg(&hd).arg(&page).arg(&stride);
                    if key.scaled{l.arg(&scales.k).arg(&scales.v);}
                    l.launch(config)?;Ok(())
                })
            }))
        }}}
        dispatch!(key,run)?;
    }
    Ok((ko,vo))
}
