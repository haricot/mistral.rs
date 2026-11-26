//! Typed GDN preparation/FP8 launches. Pointer guards are owned by the
//! calling tensor adapter; only shape and dtype information enters the cache.
use candle_core::{Device,DType,Result,Layout};
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;
use crate::cuda::kernel_runtime::{kernel_cache,LaunchPlanCache};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType,LaunchRequest},universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;
pub(super) use mistralrs_kernel_runtime::universal::ElementType as E;
pub(super) use crate::cuda::attention_prep_dynamic::span;

#[derive(Clone,Copy)]
pub(super) enum S{I32(i32),I64(i64),U64(u64),F32(f32)}
impl S{
    fn dtype(&self)->E{match self{Self::I32(_)=>E::I32,Self::I64(_)=>E::I64,Self::U64(_)=>E::U64,Self::F32(_)=>E::F32}}
}
pub(super) fn n(v:usize)->Result<S>{Ok(S::I32(i32::try_from(v).map_err(candle_core::Error::wrap)?))}
pub(super) fn stride(v:usize)->Result<S>{Ok(S::I64(i64::try_from(v).map_err(candle_core::Error::wrap)?))}
pub(super) fn dt(dtype:DType)->Result<E>{
    match dtype{DType::F16=>Ok(E::F16),DType::BF16=>Ok(E::Bf16),DType::F32=>Ok(E::F32),_=>candle_core::bail!("unsupported GDN preparation dtype")}
}
pub(super) fn mul(a:usize,b:usize)->Result<usize>{a.checked_mul(b).ok_or_else(||candle_core::Error::msg("GDN preparation size overflow"))}
pub(super) fn grid(x:usize,y:usize)->Result<[u32;3]>{
    if x==0||x>i32::MAX as usize||y==0||y>65535{candle_core::bail!("invalid GDN preparation grid");}
    Ok([x as u32,y as u32,1])
}
pub(super) fn blocks(n:usize)->usize{n.div_ceil(256).min(65535)}
pub(super) fn shifted(base:u64,layout:&Layout,bytes:usize)->Result<u64>{
    layout.start_offset().checked_mul(bytes).and_then(|v|base.checked_add(v as u64))
        .ok_or_else(||candle_core::Error::msg("GDN preparation pointer offset overflow"))
}
#[derive(Clone,Copy,Eq,PartialEq)]
struct Key{
    op:&'static str,
    tensors:[Option<(usize,E)>;10],
    scalars:[Option<E>;16],
    grid:[u32;3],
    block:[u32;3],
    shared:u32,
    ceiling:u32,
}
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let mut args=Vec::new();
    for (size,dtype) in key.tensors.into_iter().flatten(){
        args.push(Argument::Tensor{dtype,shape:vec![size],contiguous:true,start_offset:0,representation:None});
    }
    for dtype in key.scalars.into_iter().flatten(){args.push(Argument::Scalar{dtype});}
    let mut req=LaunchRequest{operation:key.op.into(),cc:0,args,grid:key.grid,max_dynamic_shared_mem_bytes:key.ceiling};
    let p=kernel_cache()?.prepare_for_device(&mut req,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=key.block||p.manifest.shared_mem_bytes!=key.ceiling{candle_core::bail!("GDN preparation catalogue geometry mismatch");}
    if key.shared!=key.ceiling{
        // The caller has validated the exact K-dependent dynamic allocation.
        unsafe{p.prevalidate_launch_with_shared(&req,device,key.shared)}
    }else{p.prevalidate_launch_with_optin(&req,device)}
        .map_err(|e|candle_core::Error::msg(e.to_string()))
}
/// Safety: addresses are valid spans of the declared dtype, guards live across
/// the launch, and operation-specific scalar geometry matches the tensor spans.
pub(super) unsafe fn launch(device:&Device,op:&'static str,pointers:&[(u64,usize,E)],scalars:&[S],grid:[u32;3],block:[u32;3])->Result<()>{
    unsafe{launch_shared(device,op,pointers,scalars,grid,block,0,0)}
}
/// Same safety contract, plus a validated exact dynamic shared-memory size.
pub(super) unsafe fn launch_shared(device:&Device,op:&'static str,pointers:&[(u64,usize,E)],scalars:&[S],grid:[u32;3],block:[u32;3],shared:u32,ceiling:u32)->Result<()>{
    if shared>ceiling{candle_core::bail!("GDN dynamic shared memory exceeds ceiling");}
    if pointers.len()>10||scalars.len()>16||pointers.iter().any(|(p,n,_)|*p==0||*n==0){
        candle_core::bail!("invalid GDN preparation ABI");
    }
    let mut key=Key{op,tensors:[None;10],scalars:[None;16],grid,block,shared,ceiling};
    for (dst,&(_,n,dt)) in key.tensors.iter_mut().zip(pointers){*dst=Some((n,dt));}
    for (dst,s) in key.scalars.iter_mut().zip(scalars){*dst=Some(s.dtype());}
    let stream=device.as_cuda_device()?.cuda_stream();
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,device,||prepare(key,device),|plan|{
        unsafe{plan.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            for (p,_,_) in pointers{b.arg(p);}
            for s in scalars{match s{
                S::I32(v)=>{b.arg(v);},S::I64(v)=>{b.arg(v);},S::U64(v)=>{b.arg(v);},S::F32(v)=>{b.arg(v);},
            }}
            b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}
pub(super) fn dense_rows(rows:usize,dims:[usize;4],strides:[usize;4])->bool{
    dims[1].checked_mul(dims[2]).is_some_and(|plane|plane>0&&rows%plane==0&&strides[3]==1&&
        (dims[2]<=1||strides[2]==128)&&
        (dims[1]<=1||dims[2].checked_mul(128)==Some(strides[1]))&&
        (rows/plane<=1||plane.checked_mul(128)==Some(strides[0])))
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_gdn_prep_geometry(){
        assert_eq!(blocks(1),1);
        assert_eq!(blocks(usize::MAX),65535);
        assert!(n(usize::MAX).is_err());
        assert!(grid(1,65536).is_err());
        assert!(dense_rows(12,[2,2,3,128],[768,384,128,1]));
        assert!(!dense_rows(12,[2,2,3,128],[768,384,129,1]));
    }
}
