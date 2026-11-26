//! Typed attention preparation launch plans. All addresses are offset-adjusted
//! by the existing guarded tensor consumers; this layer validates ABI/geometry.
use candle_core::{DType,Device,Result,Layout};
use crate::cuda::kernel_runtime::{kernel_cache,LaunchPlanCache};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType,LaunchRequest},universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;
#[derive(Clone,Copy,Eq,PartialEq)]
pub(crate) enum Kind{Qk,QkPos,QkvPos,Sincos,Norm}
#[derive(Clone,Copy,Eq,PartialEq)]
struct Key{kind:Kind,dtype:ElementType,spans:[usize;12],grid:u32,block:u32}
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}

pub(crate) fn span(l:&Layout)->Result<usize>{
    if l.dims().contains(&0){candle_core::bail!("empty attention span");}
    l.dims().iter().zip(l.stride()).try_fold(1usize,|a,(&n,&s)|{
        a.checked_add((n-1).checked_mul(s)?)
    }).filter(|n|*n<=i64::MAX as usize).ok_or_else(||candle_core::Error::msg("attention span overflow"))
}
pub(crate) fn strides(s:&[usize])->Result<[u64;2]>{
    if s.len()!=4 || s.iter().any(|&x|x>u32::MAX as usize){candle_core::bail!("attention stride exceeds packed ABI");}
    Ok([s[0] as u64 | ((s[1] as u64)<<32),s[2] as u64 | ((s[3] as u64)<<32)])
}
pub(crate) fn geometry(batch:usize,q:usize,k:usize,seq:usize,head:usize,triple:bool)->Result<(u32,u32)>{
    if batch==0||q==0||seq==0||(triple&&k==0){candle_core::bail!("empty attention geometry");}
    let heads=k.checked_mul(if triple{2}else{1}).and_then(|v|v.checked_add(q));
    let rows=heads.and_then(|v|v.checked_mul(batch)).and_then(|v|v.checked_mul(seq))
        .filter(|&v|v>0&&v<=i32::MAX as usize).ok_or_else(||candle_core::Error::msg("attention grid overflow"))?;
    if head==0{candle_core::bail!("empty attention head");}
    Ok((rows as u32,head.min(1024).max(32).next_power_of_two() as u32))
}
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let (kind,n,positions)=match key.kind{Kind::Qk=>("qk",2,false),Kind::QkPos=>("qk_pos",2,true),
        Kind::QkvPos=>("qkv_pos",3,true),Kind::Sincos=>("sincos",0,true),Kind::Norm=>("norm",0,false)};
    let t=|i:usize,dt|Argument::Tensor{dtype:dt,shape:vec![key.spans[i]],contiguous:true,start_offset:0,representation:None};
    let s=|dtype|Argument::Scalar{dtype};
    let mut args=Vec::new();
    let op=if key.kind==Kind::Norm{
        args.extend([t(0,key.dtype),t(1,key.dtype),t(2,key.dtype),s(ElementType::U64),s(ElementType::U64),
            s(ElementType::I32),s(ElementType::I32),s(ElementType::I32),s(ElementType::I32),s(ElementType::F32)]);
        format!("attention.prep.norm.b{}.v1",key.block)
    }else if key.kind==Kind::Sincos{
        args.extend([t(0,ElementType::U32),t(1,ElementType::F32),t(2,key.dtype),t(3,key.dtype),
            s(ElementType::I32),s(ElementType::I32)]);
        "attention.prep.sincos.v1".to_string()
    }else{
        let pos_index=n*2+2;
        let pointers=n*3+2+usize::from(positions);
        for i in 0..pointers{args.push(t(i,if positions&&i==pos_index{ElementType::U32}else{key.dtype}));}
        args.extend((0..n*2).map(|_|s(ElementType::U64)));
        args.extend((0..6).map(|_|s(ElementType::I32)));
        if !positions{args.push(s(ElementType::I32));}
        args.extend((0..n).map(|_|s(ElementType::F32)));
        args.push(s(ElementType::I32)); // neox
        if n==2{args.push(s(ElementType::I32));} // token major
        if positions{args.push(s(ElementType::I32));} // cache rows
        format!("attention.prep.{kind}.b{}.v1",key.block)
    };
    let mut req=LaunchRequest{operation:op,cc:0,args,grid:[key.grid,1,1],max_dynamic_shared_mem_bytes:0};
    let p=kernel_cache()?.prepare_for_device(&mut req,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[key.block,1,1]||p.manifest.shared_mem_bytes!=0{candle_core::bail!("attention prep block mismatch");}
    p.prevalidate_launch(&req,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}
pub(crate) fn with_plan<R>(kind:Kind,dtype:DType,spans:&[usize],grid:u32,block:u32,device:&Device,
    launch:impl FnOnce(&PrevalidatedLaunchPlan)->Result<R>)->Result<R>{
    let dtype=match dtype{DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,DType::F32=>ElementType::F32,
        _=>candle_core::bail!("attention prep dtype")};
    if spans.len()>12||spans.contains(&0)||grid==0||grid>i32::MAX as u32{candle_core::bail!("attention prep ABI span");}
    let mut packed=[1;12];packed[..spans.len()].copy_from_slice(spans);
    let key=Key{kind,dtype,spans:packed,grid,block};
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,device,||prepare(key,device),launch))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_attention_prep_geometry_and_stride(){
        assert_eq!(geometry(2,8,2,3,80,false).unwrap(),(60,128));
        assert_eq!(geometry(2,8,2,3,80,true).unwrap(),(72,128));
        assert_eq!(strides(&[9,7,5,1]).unwrap(),[9|(7u64<<32),5|(1u64<<32)]);
        assert!(strides(&[usize::MAX,1,1,1]).is_err());
        assert!(geometry(usize::MAX,8,2,3,80,true).is_err());
    }
}

const PACK_ROWS:usize=24;
#[derive(Clone,Copy,Eq,PartialEq)]
struct PackKey{rows:usize,host:usize,staged:usize}
thread_local!{static PACK:RefCell<LaunchPlanCache<PackKey,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn pack_plan(key:PackKey,dev:&Device)->Result<PrevalidatedLaunchPlan>{
    let t=|n|Argument::Tensor{dtype:ElementType::U32,shape:vec![n],contiguous:true,start_offset:0,representation:None};
    let elements=key.rows*(key.host+key.staged);
    let mut args=vec![t(key.rows*key.host)];
    args.extend((0..PACK_ROWS).map(|_|t(key.staged)));
    args.push(t(elements));
    args.extend((0..3).map(|_|Argument::Scalar{dtype:ElementType::I32}));
    let mut req=LaunchRequest{operation:"layout.pack_completion_u32.v1".into(),cc:0,args,
        grid:[u32::try_from(elements.div_ceil(256)).map_err(candle_core::Error::wrap)?,1,1],max_dynamic_shared_mem_bytes:0};
    let p=kernel_cache()?.prepare_for_device(&mut req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[256,1,1]||p.manifest.shared_mem_bytes!=0{candle_core::bail!("pack block mismatch");}
    p.prevalidate_launch(&req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))
}
// Called only while original host/output/row allocation guards are held.
pub(crate) fn pack_completion(device:&Device,host:u64,rows:&[u64],out:u64,hw:usize,sw:usize)->Result<()>{
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    let width=hw.checked_add(sw).filter(|&v|v<=i32::MAX as usize).ok_or_else(||candle_core::Error::msg("pack width overflow"))?;
    let total=rows.len().checked_mul(width).and_then(|v|v.checked_mul(4)).ok_or_else(||candle_core::Error::msg("pack span overflow"))?;
    if rows.is_empty()||hw==0||sw==0||host.checked_add(total as u64).is_none()||out.checked_add(total as u64).is_none(){
        candle_core::bail!("invalid pack span");
    }
    let dev=device.as_cuda_device()?;let stream=dev.cuda_stream();
    let hi=hw as i32;let si=sw as i32;
    for (chunk,staged) in rows.chunks(PACK_ROWS).enumerate(){
        let start=chunk*PACK_ROWS;
        let hp=host+(start*hw*4) as u64;
        let op=out+(start*width*4) as u64;
        let mut ptrs=[staged[0];PACK_ROWS];
        ptrs[..staged.len()].copy_from_slice(staged);
        let ri=staged.len() as i32;
        let key=PackKey{rows:staged.len(),host:hw,staged:sw};
        PACK.with(|cache|cache.borrow_mut().with_plan(key,device,||pack_plan(key,device),|p|{
            unsafe{p.with_prevalidated_launch(|fun,cfg|{
                let mut b=stream.launch_builder(fun);b.arg(&hp);
                for ptr in &ptrs{b.arg(ptr);}
                b.arg(&op).arg(&ri).arg(&hi).arg(&si);b.launch(cfg)?;Ok(())
            })}.map_err(|e|candle_core::Error::msg(e.to_string()))
        }))?;
    }
    Ok(())
}

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
enum GdnMode{Generic,Warp,Rows4}
#[derive(Clone,Copy,Eq,PartialEq)]
struct GdnKey{dtype:ElementType,mode:GdnMode,spans:[usize;4],rows:u32}
thread_local!{static GDN:RefCell<LaunchPlanCache<GdnKey,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn gdn_geometry(mode:GdnMode,rows:u32)->([u32;3],u32){
    match mode{GdnMode::Generic=>([128,1,1],rows),GdnMode::Warp=>([32,8,1],rows.div_ceil(8)),
        GdnMode::Rows4=>([32,1,1],rows.div_ceil(4))}
}
fn gdn_plan(key:GdnKey,dev:&Device)->Result<PrevalidatedLaunchPlan>{
    let tag=match key.mode{GdnMode::Generic=>"generic",GdnMode::Warp=>"warp",GdnMode::Rows4=>"rows4"};
    let mut args=key.spans.iter().map(|&n|Argument::Tensor{dtype:key.dtype,shape:vec![n],contiguous:true,start_offset:0,representation:None}).collect::<Vec<_>>();
    let s=|dtype|Argument::Scalar{dtype};
    args.push(s(ElementType::I32));
    if key.mode==GdnMode::Generic{args.push(s(ElementType::I32));}
    if key.mode!=GdnMode::Rows4{
        args.extend([s(ElementType::I32),s(ElementType::I32)]);
        args.extend((0..4).map(|_|s(ElementType::U64)));
    }
    args.extend([s(ElementType::F32),s(ElementType::I32)]);
    let (block,grid)=gdn_geometry(key.mode,key.rows);
    let mut req=LaunchRequest{operation:format!("attention.prep.gdn_norm.{tag}.v1"),cc:0,args,grid:[grid,1,1],max_dynamic_shared_mem_bytes:0};
    let p=kernel_cache()?.prepare_for_device(&mut req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=block||p.manifest.shared_mem_bytes!=0{candle_core::bail!("GDN norm block mismatch");}
    p.prevalidate_launch(&req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))
}
pub(crate) fn gdn_norm(device:&Device,dt:DType,spans:[usize;4],pointers:[u64;4],dims:[usize;4],xs:[usize;4],gs:[usize;4],eps:f32,sigmoid:i32)->Result<()>{
    use candle_core::cuda::cudarc::driver::PushKernelArg;
    let [o0,o1,o2,hidden]=dims;
    let rows=o0.checked_mul(o1).and_then(|v|v.checked_mul(o2)).filter(|&v|v>0&&v<=i32::MAX as usize)
        .ok_or_else(||candle_core::Error::msg("GDN norm rows overflow"))?;
    let elements=rows.checked_mul(hidden).ok_or_else(||candle_core::Error::msg("GDN norm size overflow"))?;
    if hidden==0||spans.contains(&0)||spans[3]!=elements||spans[2]!=hidden||pointers.contains(&0){
        candle_core::bail!("GDN norm span mismatch");
    }
    let (ri,hi,oi1,oi2)=(rows as i32,i32::try_from(hidden).map_err(candle_core::Error::wrap)?,
        i32::try_from(o1).map_err(candle_core::Error::wrap)?,i32::try_from(o2).map_err(candle_core::Error::wrap)?);
    let xp=strides(&xs)?;let gp=strides(&gs)?;
    let dense=|s:[usize;4]|s[3]==1&&(o2<=1||s[2]==128)&&(o1<=1||s[1]==o2*128)&&(o0<=1||s[0]==o1*o2*128);
    let mode=if hidden==128&&xs[3]==1&&gs[3]==1{
        if rows>=1024&&dense(xs)&&dense(gs)&&pointers.iter().all(|p|p%16==0){GdnMode::Rows4}else{GdnMode::Warp}
    }else{GdnMode::Generic};
    let dtype=match dt{DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,_=>candle_core::bail!("GDN norm dtype")};
    let key=GdnKey{dtype,mode,spans,rows:rows as u32};
    let stream=device.as_cuda_device()?.cuda_stream();
    GDN.with(|cache|cache.borrow_mut().with_plan(key,device,||gdn_plan(key,device),|p|{
        unsafe{p.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            for ptr in &pointers{b.arg(ptr);}
            b.arg(&ri);
            if mode==GdnMode::Generic{b.arg(&hi);}
            if mode!=GdnMode::Rows4{b.arg(&oi1).arg(&oi2).arg(&xp[0]).arg(&xp[1]).arg(&gp[0]).arg(&gp[1]);}
            b.arg(&eps).arg(&sigmoid);b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}

#[cfg(test)]
mod gpu_tests{
    use super::*;
    use candle_core::{Tensor,D};
    #[test]
    #[ignore="requires attention-prep catalogue and CUDA"]
    fn dkr_attention_prep_strided_norm_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        for dt in [DType::F16,DType::BF16,DType::F32]{
            for width in [7usize,64,80,128,256,513]{
                let x=Tensor::from_vec((0..2*3*4*width).map(|i|((i%19)as f32-9.)/8.).collect::<Vec<_>>(),(2,3,4,width),&dev)?
                    .to_dtype(dt)?.transpose(1,2)?;
                let w=Tensor::ones(width,dt,&dev)?;
                let y=crate::ops::try_cuda_rms_norm_strided_4d(&x,&w,1e-5)?.unwrap().to_dtype(DType::F32)?;
                let xf=x.to_dtype(DType::F32)?;
                let reference=xf.broadcast_div(&(xf.sqr()?.mean_keepdim(D::Minus1)?+1e-5f64)?.sqrt()?)?;
                let delta=(y-reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
                let tol=match dt{DType::F32=>2e-5,DType::F16=>2e-3,_=>2e-2};
                assert!(delta<=tol,"{dt:?} width={width}: {delta}");
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires attention-prep catalogue and CUDA"]
    fn dkr_attention_prep_sincos_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let positions=Tensor::from_vec(vec![0u32,7,37,1024],4,&dev)?;
        let inv=Tensor::from_vec(vec![1f32,0.5,0.125,0.03125],4,&dev)?;
        let (c,s)=crate::ops::try_cuda_rope_sincos_positions(&positions,&inv,DType::F32)?.unwrap();
        let angles=positions.to_dtype(DType::F32)?.unsqueeze(1)?.broadcast_mul(&inv.unsqueeze(0)?)?;
        let cd=(c-angles.cos()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let sd=(s-angles.sin()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(cd<2e-5&&sd<2e-5,"sincos deltas {cd} {sd}");
        Ok(())
    }
    #[test]
    #[ignore="requires attention-prep catalogue and CUDA"]
    fn dkr_attention_prep_qk_indexed_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let positions=Tensor::from_vec(vec![0u32,1,3,7],4,&dev)?;
        let cos=Tensor::ones((8,16),DType::F32,&dev)?;
        let sin=Tensor::zeros((8,16),DType::F32,&dev)?;
        let w=Tensor::ones(64,DType::F32,&dev)?;
        let x=Tensor::from_vec((0..2*2*3*64).map(|i|((i%13)as f32-6.)/8.).collect::<Vec<_>>(),(2,2,3,64),&dev)?.transpose(1,2)?;
        let y=crate::ops::try_cuda_qk_rms_norm_rope_positions(&x,None,&w,None,1e-5,1e-5,
            &cos,&sin,&positions,true,crate::ops::QkRopeOutputLayout::HeadsFirst)?.unwrap().0;
        let expected=x.broadcast_div(&(x.sqr()?.mean_keepdim(D::Minus1)?+1e-5f64)?.sqrt()?)?;
        let delta=(y-expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(delta<2e-5,"indexed Q-only delta {delta}");
        Ok(())
    }
}
