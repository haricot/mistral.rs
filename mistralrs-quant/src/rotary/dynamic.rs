//! Typed in-place rotary. Strided token rows are admitted only when
//! heads/columns are packed and token rows do not overlap.
use candle_core::{CudaDevice,DType,Device,Layout,Result};
use candle_core::cuda::cudarc::driver::PushKernelArg;
use crate::dkr_cuda::{catalog,dtype,tensor,LaunchPlanCache};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType,LaunchRequest},universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;

#[derive(Clone,Copy,Eq,PartialEq)]
pub(super) struct Key{
    dtype:ElementType,tokens:i32,heads:i32,head:i32,rot:i32,cache_rows:i32,stride:i64,
    span:usize,cache_span:usize,indexed:bool,neox:bool,
}
impl Key{
    pub(super) fn new(dt:DType,q:&Layout,cos:&Layout,sin:&Layout,pos:Option<&Layout>,neox:bool)->Result<Self>{
        let (tokens,heads,head)=q.shape().dims3()?;
        let (cache_rows,rot)=cos.shape().dims2()?;
        let packed=heads.checked_mul(head).ok_or_else(||candle_core::Error::msg("RoPE row overflow"))?;
        let stride=q.stride()[0];
        if tokens==0||heads==0||head==0||rot==0||rot>head/2||cache_rows==0
            ||q.stride()[1]!=head||q.stride()[2]!=1||(tokens>1&&stride<packed)
            ||!cos.is_contiguous()||!sin.is_contiguous()||sin.shape()!=cos.shape()
            ||pos.is_some_and(|p|!p.is_contiguous()||p.dims()!=[tokens])
            ||(pos.is_none()&&cache_rows!=tokens){
            candle_core::bail!("DKR rotary layout mismatch");
        }
        let span=(tokens-1).checked_mul(stride).and_then(|v|v.checked_add(packed))
            .filter(|&v|v<=i64::MAX as usize).ok_or_else(||candle_core::Error::msg("RoPE query span overflow"))?;
        let cache_span=cache_rows.checked_mul(rot).filter(|&v|v<=i32::MAX as usize)
            .ok_or_else(||candle_core::Error::msg("RoPE cache span overflow"))?;
        if heads.checked_mul(rot).is_none_or(|v|v>i32::MAX as usize){candle_core::bail!("RoPE work overflow");}
        let i=|v|i32::try_from(v).map_err(candle_core::Error::wrap);
        Ok(Self{dtype:dtype(dt)?,tokens:i(tokens)?,heads:i(heads)?,head:i(head)?,rot:i(rot)?,cache_rows:i(cache_rows)?,
            stride:i64::try_from(stride).map_err(candle_core::Error::wrap)?,span,cache_span,indexed:pos.is_some(),neox})
    }
}
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let mut args=vec![tensor(key.dtype,vec![key.span]),tensor(key.dtype,vec![key.cache_span]),tensor(key.dtype,vec![key.cache_span])];
    if key.indexed{args.push(tensor(ElementType::U32,vec![key.tokens as usize]));}
    args.extend([ElementType::I32,ElementType::I32,ElementType::I32,ElementType::I64,ElementType::I32].map(|dtype|Argument::Scalar{dtype}));
    if key.indexed{args.push(Argument::Scalar{dtype:ElementType::I32});}
    let op=if key.indexed{"attention.prep.rotary_pos.v1"}else{"attention.prep.rotary.v1"};
    let mut req=LaunchRequest{operation:op.into(),cc:0,args,grid:[key.tokens as u32,1,1],max_dynamic_shared_mem_bytes:0};
    let p=catalog()?.prepare_for_device(&mut req,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[256,1,1]||p.manifest.shared_mem_bytes!=0{candle_core::bail!("DKR rotary block mismatch");}
    p.prevalidate_launch(&req,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}
pub(super) fn launch(key:Key,dev:&CudaDevice,q:u64,cos:u64,sin:u64,pos:Option<u64>)->Result<()>{
    if pos.is_some()!=key.indexed{candle_core::bail!("DKR rotary position contract");}
    let device=Device::Cuda(dev.clone());let stream=dev.cuda_stream();
    let neox=i32::from(key.neox);
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,&device,||prepare(key,&device),|p|{
        unsafe{p.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            b.arg(&q).arg(&cos).arg(&sin);
            if let Some(ref positions)=pos{b.arg(positions);}
            b.arg(&key.head).arg(&key.rot).arg(&key.heads).arg(&key.stride).arg(&neox);
            if key.indexed{b.arg(&key.cache_rows);}
            b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}
