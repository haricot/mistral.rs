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
    if spans.len()>12||spans.contains(&0)||grid==0{candle_core::bail!("attention prep ABI span");}
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
