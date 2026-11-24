//! Typed MLA latent and positional cache IO, with explicit physical row spans.
use super::slice_ptr_on_stream;
use candle_core::{cuda::cudarc::driver::{DeviceSlice, PushKernelArg},
    DType, Device, Layout, Result, Storage, Tensor};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, sync::OnceLock};

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    dtype:DType, store:bool, tokens:usize, blocks:usize, page:usize,
    rank:usize, rope:usize, seqs:usize, stride:usize,
    cs:usize, ps:usize, cn:usize, pn:usize,
}
thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}
static CATALOG:OnceLock<std::result::Result<KernelExecutionCache,String>>=OnceLock::new();
fn catalog()->Result<&'static KernelExecutionCache>{
    CATALOG.get_or_init(||KernelExecutionCache::package_root()
        .ok_or_else(||"DKR MLA IO catalogue unavailable".to_owned())
        .and_then(|p|KernelExecutionCache::discover(&p).map_err(|e|e.to_string())))
        .as_ref().map_err(|e|candle_core::Error::Msg(e.clone()))
}
fn arg(dtype:ElementType,shape:Vec<usize>,repr:Option<&str>)->Argument {
    Argument::Tensor{dtype,shape,contiguous:true,start_offset:0,representation:repr.map(str::to_owned)}
}
fn prepare(k:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let dt=match k.dtype {DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,
        DType::F32=>ElementType::F32,_=>candle_core::bail!("unsupported MLA dtype")};
    let cc=arg(dt,vec![k.blocks,k.page,k.rank],Some("mla.ckv.paged.v1"));
    let pc=arg(dt,vec![k.blocks,k.page,k.rope],Some("mla.kpe.paged.v1"));
    let mut args=if k.store {
        vec![arg(dt,vec![k.cn],Some("mla.row_span.v1")),arg(dt,vec![k.pn],Some("mla.row_span.v1")),
            cc,pc,arg(ElementType::I64,vec![k.tokens],None)]
    }else{
        vec![cc,pc,arg(ElementType::I32,vec![k.seqs,k.stride],None),
            arg(ElementType::I32,vec![k.seqs+1],None),arg(ElementType::I32,vec![k.tokens],None),
            arg(dt,vec![k.tokens,k.rank],None),arg(dt,vec![k.tokens,k.rope],None)]
    };
    args.extend((0..if k.store{5}else{7}).map(|_|Argument::Scalar{dtype:ElementType::U32}));
    if k.store{args.extend((0..2).map(|_|Argument::Scalar{dtype:ElementType::U64}));}
    let mut request=LaunchRequest{
        operation:format!("kv_cache.mla.{}.b256.v1",if k.store{"store"}else{"gather"}),
        cc:0,args,grid:[k.tokens as u32,1,1],max_dynamic_shared_mem_bytes:0,
    };
    let p=catalog()?.prepare_for_device(&mut request,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[256,1,1] || p.manifest.shared_mem_bytes!=0{
        candle_core::bail!("DKR MLA IO launch geometry mismatch");
    }
    p.prevalidate_launch(&request,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}
fn base_key(c:&Tensor,p:&Tensor)->Result<Key>{
    let (blocks,page,rank)=c.dims3()?;
    let (pb,pp,rope)=p.dims3()?;
    if !c.device().is_cuda() || !c.device().same_device(p.device()) ||
        c.dtype()!=p.dtype() || !matches!(c.dtype(),DType::F16|DType::BF16|DType::F32) ||
        !c.is_contiguous() || !p.is_contiguous() || (blocks,page)!=(pb,pp) ||
        [blocks,page,rank,rope].iter().any(|&n|n==0 || n>i32::MAX as usize) {
        candle_core::bail!("DKR MLA IO unsupported cache geometry");
    }
    Ok(Key{dtype:c.dtype(),store:false,tokens:0,blocks,page,rank,rope,
        seqs:0,stride:0,cs:0,ps:0,cn:0,pn:0})
}
fn span(l:&Layout,t:usize,w:usize)->Result<(usize,usize)>{
    if t==0 || w==0 || l.dims()!=[t,w] || l.stride()[1]!=1 || l.stride()[0]<w{
        candle_core::bail!("DKR MLA expects dense non-overlapping rows");
    }
    let s=l.stride()[0];
    let n=(t-1).checked_mul(s).and_then(|n|n.checked_add(w))
        .ok_or_else(||candle_core::Error::msg("DKR MLA row span overflow"))?;
    Ok((s,n))
}
fn compatible(device:&Device,t:&Tensor,dt:DType)->bool{
    device.same_device(t.device()) && t.dtype()==dt
}
pub fn concat_and_cache_mla(c:&Tensor,p:&Tensor,cc:&Tensor,pc:&Tensor,slots:&Tensor)->Result<()>{
    let mut k=base_key(cc,pc)?;
    let dev=cc.device();
    let (t,w)=c.dims2()?;
    if w!=k.rank || p.dims2()? != (t,k.rope) ||
        !compatible(dev,c,k.dtype) || !compatible(dev,p,k.dtype) ||
        !compatible(dev,slots,DType::I64) || !slots.is_contiguous() ||
        slots.dims()!=[t] || t>i32::MAX as usize{
        candle_core::bail!("DKR MLA store incompatible inputs");
    }
    if t==0{return Ok(());}
    let (c_s,c_l)=c.storage_and_layout();
    let (p_s,p_l)=p.storage_and_layout();
    let (cc_s,cc_l)=cc.storage_and_layout();
    let (pc_s,pc_l)=pc.storage_and_layout();
    let (s_s,s_l)=slots.storage_and_layout();
    let (cs,cn)=span(c_l,t,k.rank)?;
    let (ps,pn)=span(p_l,t,k.rope)?;
    k.store=true;k.tokens=t;k.cs=cs;k.ps=ps;k.cn=cn;k.pn=pn;
    let (Storage::Cuda(c_s),Storage::Cuda(p_s),Storage::Cuda(cc_s),Storage::Cuda(pc_s),Storage::Cuda(s_s))=
        (&*c_s,&*p_s,&*cc_s,&*pc_s,&*s_s) else {candle_core::bail!("DKR MLA requires CUDA storage");};
    let stream=dev.as_cuda_device()?.cuda_stream();
    let (tokens,blocks,page,rank,rope)=(t as u32,k.blocks as u32,k.page as u32,k.rank as u32,k.rope as u32);
    let (cs,ps)=(cs as u64,ps as u64);
    macro_rules! run {($ty:ty)=>{{
        let cbuf=c_s.as_cuda_slice::<$ty>()?;
        let pbuf=p_s.as_cuda_slice::<$ty>()?;
        if c_l.start_offset().checked_add(cn).is_none_or(|n|n>cbuf.len()) ||
            p_l.start_offset().checked_add(pn).is_none_or(|n|n>pbuf.len()){
            candle_core::bail!("DKR MLA input span exceeds allocation");
        }
        let (cp,_cg)=slice_ptr_on_stream(cbuf,c_l.start_offset(),&stream);
        let (pp,_pg)=slice_ptr_on_stream(pbuf,p_l.start_offset(),&stream);
        let (ccp,_ccg)=slice_ptr_on_stream(cc_s.as_cuda_slice::<$ty>()?,cc_l.start_offset(),&stream);
        let (pcp,_pcg)=slice_ptr_on_stream(pc_s.as_cuda_slice::<$ty>()?,pc_l.start_offset(),&stream);
        let (sp,_sg)=slice_ptr_on_stream(s_s.as_cuda_slice::<i64>()?,s_l.start_offset(),&stream);
        PLANS.with(|cache|cache.borrow_mut().with_plan(k,dev,||prepare(k,dev),|plan|unsafe{
            plan.with_prevalidated_launch(|fun,config|{
                let mut l=stream.launch_builder(fun);
                l.arg(&cp).arg(&pp).arg(&ccp).arg(&pcp).arg(&sp).arg(&tokens).arg(&blocks)
                    .arg(&page).arg(&rank).arg(&rope).arg(&cs).arg(&ps);
                l.launch(config)?;Ok(())
            })
        }))
    }}}
    match k.dtype{DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}
}
pub fn gather_mla_cache(cc:&Tensor,pc:&Tensor,table:&Tensor,offsets:&Tensor,map:&Tensor)->Result<(Tensor,Tensor)>{
    let mut k=base_key(cc,pc)?;
    let dev=cc.device();
    let (seqs,stride)=table.dims2()?;
    let t=map.dims1()?;
    if [seqs,stride].iter().any(|&n|n==0 || n>i32::MAX as usize) || t>i32::MAX as usize ||
        offsets.dims()!=[seqs+1] ||
        ![table,offsets,map].iter().all(|x|compatible(dev,x,DType::I32)){
        candle_core::bail!("DKR MLA gather incompatible metadata");
    }
    let c=Tensor::zeros((t,k.rank),k.dtype,dev)?;
    let p=Tensor::zeros((t,k.rope),k.dtype,dev)?;
    if t==0{return Ok((c,p));}
    // Match legacy behavior for noncontiguous metadata, never repack the cache.
    let table=table.contiguous()?;let offsets=offsets.contiguous()?;let map=map.contiguous()?;
    k.tokens=t;k.seqs=seqs;k.stride=stride;
    {
        let (cc_s,cc_l)=cc.storage_and_layout();let (pc_s,pc_l)=pc.storage_and_layout();
        let (t_s,t_l)=table.storage_and_layout();let (o_s,o_l)=offsets.storage_and_layout();
        let (m_s,m_l)=map.storage_and_layout();let (c_s,c_l)=c.storage_and_layout();
        let (p_s,p_l)=p.storage_and_layout();
        let (Storage::Cuda(cc_s),Storage::Cuda(pc_s),Storage::Cuda(t_s),Storage::Cuda(o_s),
            Storage::Cuda(m_s),Storage::Cuda(c_s),Storage::Cuda(p_s))=
            (&*cc_s,&*pc_s,&*t_s,&*o_s,&*m_s,&*c_s,&*p_s) else {candle_core::bail!("DKR MLA gather CUDA storage required");};
        let stream=dev.as_cuda_device()?.cuda_stream();
        let (tokens,seqs,blocks,page,stride,rank,rope)=(t as u32,seqs as u32,k.blocks as u32,k.page as u32,stride as u32,k.rank as u32,k.rope as u32);
        macro_rules! run {($ty:ty)=>{{
            let (ccp,_ccg)=slice_ptr_on_stream(cc_s.as_cuda_slice::<$ty>()?,cc_l.start_offset(),&stream);
            let (pcp,_pcg)=slice_ptr_on_stream(pc_s.as_cuda_slice::<$ty>()?,pc_l.start_offset(),&stream);
            let (tp,_tg)=slice_ptr_on_stream(t_s.as_cuda_slice::<i32>()?,t_l.start_offset(),&stream);
            let (op,_og)=slice_ptr_on_stream(o_s.as_cuda_slice::<i32>()?,o_l.start_offset(),&stream);
            let (mp,_mg)=slice_ptr_on_stream(m_s.as_cuda_slice::<i32>()?,m_l.start_offset(),&stream);
            let (cp,_cg)=slice_ptr_on_stream(c_s.as_cuda_slice::<$ty>()?,c_l.start_offset(),&stream);
            let (pp,_pg)=slice_ptr_on_stream(p_s.as_cuda_slice::<$ty>()?,p_l.start_offset(),&stream);
            PLANS.with(|cache|cache.borrow_mut().with_plan(k,dev,||prepare(k,dev),|plan|unsafe{
                plan.with_prevalidated_launch(|fun,config|{
                    let mut l=stream.launch_builder(fun);
                    l.arg(&ccp).arg(&pcp).arg(&tp).arg(&op).arg(&mp).arg(&cp).arg(&pp)
                        .arg(&tokens).arg(&seqs).arg(&blocks).arg(&page).arg(&stride).arg(&rank).arg(&rope);
                    l.launch(config)?;Ok(())
                })
            }))
        }}}
        match k.dtype{DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}?;
    }
    Ok((c,p))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_mla_row_span_checked()->Result<()> {
        assert_eq!(span(&Layout::new((3,5).into(),vec![8,1],2),3,5)?,(8,21));
        assert!(span(&Layout::new((3,5).into(),vec![4,1],0),3,5).is_err());
        Ok(())
    }
    #[test]
    #[ignore="requires staged MLA IO CUBIN and physical CUDA"]
    fn dkr_mla_cache_io_gpu()->Result<()> {
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        for dt in [DType::F16,DType::BF16,DType::F32] {
            let (t,rank,rope)=(4usize,7usize,3usize);
            let c=Tensor::from_vec((0..t*9).map(|i|(i%31) as f32).collect::<Vec<_>>(),(t,9),&dev)?.to_dtype(dt)?.narrow(1,1,rank)?;
            let p=Tensor::from_vec((0..t*5).map(|i|-(i as f32)).collect::<Vec<_>>(),(t,5),&dev)?.to_dtype(dt)?.narrow(1,2,rope)?;
            let cc=Tensor::zeros((2,2,rank),dt,&dev)?;
            let pc=Tensor::zeros((2,2,rope),dt,&dev)?;
            let slots=Tensor::from_vec(vec![2i64,-1,0,3],t,&dev)?;
            super::super::concat_and_cache_mla(&c,&p,&cc,&pc,&slots)?;
            let mut ec=vec![0f32;4*rank];let mut ep=vec![0f32;4*rope];
            for (row,slot) in [(0usize,2usize),(2,0),(3,3)] {
                for d in 0..rank {ec[slot*rank+d]=((row*9+1+d)%31) as f32;}
                for d in 0..rope {ep[slot*rope+d]=-((row*5+2+d) as f32);}
            }
            assert_eq!(cc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,ec);
            assert_eq!(pc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,ep);
            let table=Tensor::from_vec(vec![1i32,0],(2,1),&dev)?;
            let offs=Tensor::from_vec(vec![0i32,2,4],3,&dev)?;
            let map=Tensor::from_vec(vec![0i32,0,1,1],4,&dev)?;
            let (co,po)=super::super::gather_mla_cache(&cc,&pc,&table,&offs,&map)?;
            let want_c=[2usize,3,0,1].iter().flat_map(|&i|ec[i*rank..(i+1)*rank].iter().copied()).collect::<Vec<_>>();
            let want_p=[2usize,3,0,1].iter().flat_map(|&i|ep[i*rope..(i+1)*rope].iter().copied()).collect::<Vec<_>>();
            assert_eq!(co.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,want_c);
            assert_eq!(po.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,want_p);
            eprintln!("DKR_R126_MLA_CACHE_IO=PASS dtype={dt:?} strided=true padding=true permuted=true");
        }
        Ok(())
    }
}
