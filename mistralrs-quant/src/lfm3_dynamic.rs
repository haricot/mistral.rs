//! LFM3 typed launch plan cache. The legacy fused CUDA bodies are retained;
//! the catalogue ABI exposes tensor addresses and individual scalar strides.
use candle_core::{CudaDevice, DType, Device, Layout, Result};
use candle_core::cuda::cudarc::driver::PushKernelArg;
use crate::dkr_cuda::{catalog,dtype,tensor,LaunchPlanCache};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType,LaunchRequest},universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;

#[derive(Clone,Copy,Eq,PartialEq)]
enum Kind{Causal{width7:bool,bias:bool},Spatial}
#[derive(Clone,Copy,Eq,PartialEq)]
pub(super) struct Key{
    kind:Kind,dt:ElementType,spans:[usize;6],count:usize,blocks:u32,
}
pub(super) fn checked_product(values:&[usize])->Result<usize>{
    values.iter().try_fold(1usize,|n,&v|n.checked_mul(v))
        .filter(|&n|n!=0).ok_or_else(||candle_core::Error::msg("LFM3 empty or overflowing geometry"))
}
pub(super) fn span(layout:&Layout,storage_len:usize)->Result<usize>{
    if layout.dims().contains(&0){candle_core::bail!("LFM3 empty tensor span");}
    let len=layout.dims().iter().zip(layout.stride()).try_fold(1usize,|n,(&d,&s)|n.checked_add((d-1).checked_mul(s)?))
        .ok_or_else(||candle_core::Error::msg("LFM3 strided span overflow"))?;
    if layout.start_offset().checked_add(len).is_none_or(|end|end>storage_len){
        candle_core::bail!("LFM3 strided span exceeds CUDA storage");
    }
    Ok(len)
}
impl Key{
    fn new(kind:Kind,dt:DType,spans:&[usize],elements:usize)->Result<Self>{
        let blocks=elements.div_ceil(256);
        let expected=match kind{Kind::Causal{bias,..}=>5+usize::from(bias),Kind::Spatial=>3};
        if blocks==0||blocks>i32::MAX as usize||spans.len()!=expected||spans.contains(&0){
            candle_core::bail!("LFM3 invalid typed launch geometry");
        }
        let mut sizes=[0;6];sizes[..spans.len()].copy_from_slice(spans);
        Ok(Self{kind,dt:dtype(dt)?,spans:sizes,count:spans.len(),blocks:blocks as u32})
    }
    pub(super) fn causal(dt:DType,spans:&[usize],elements:usize,width:usize,bias:bool)->Result<Self>{
        if width==0||width>i32::MAX as usize{candle_core::bail!("LFM3 invalid causal width");}
        Self::new(Kind::Causal{width7:width==7,bias},dt,spans,elements)
    }
    pub(super) fn spatial(dt:DType,spans:&[usize],elements:usize)->Result<Self>{
        Self::new(Kind::Spatial,dt,spans,elements)
    }
    fn operation(&self)->&'static str{
        match self.kind{
            Kind::Causal{width7:false,bias:false}=>"lfm3.causal.generic.nobias.v1",
            Kind::Causal{width7:false,bias:true}=>"lfm3.causal.generic.bias.v1",
            Kind::Causal{width7:true,bias:false}=>"lfm3.causal.width7.nobias.v1",
            Kind::Causal{width7:true,bias:true}=>"lfm3.causal.width7.bias.v1",
            Kind::Spatial=>"lfm3.spatial.v1",
        }
    }
}
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let mut args=key.spans[..key.count].iter().map(|&n|tensor(key.dt,vec![n])).collect::<Vec<_>>();
    let scalar=|dtype|Argument::Scalar{dtype};
    match key.kind{
        Kind::Causal{bias,..}=>args.extend((0..(13+usize::from(bias))).map(|_|scalar(ElementType::U64))),
        Kind::Spatial=>{
            args.push(scalar(ElementType::U64));
            args.extend((0..5).map(|_|scalar(ElementType::U32)));
        }
    }
    let mut req=LaunchRequest{operation:key.operation().into(),cc:0,args,grid:[key.blocks,1,1],max_dynamic_shared_mem_bytes:0};
    let p=catalog()?.prepare_for_device(&mut req,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[256,1,1]||p.manifest.shared_mem_bytes!=0{candle_core::bail!("LFM3 catalogue geometry mismatch");}
    p.prevalidate_launch(&req,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}
/// Safety: the caller retains the storage and CUDA guards for every pointer,
/// on the same device stream, for the complete launch.
pub(super) unsafe fn causal(key:Key,dev:&CudaDevice,pointers:&[u64],params:&[u64;15])->Result<()>{
    let Kind::Causal{width7,bias}=key.kind else{candle_core::bail!("LFM3 causal key mismatch");};
    if key.count!=pointers.len()||pointers.contains(&0)||width7!=(params[3]==7)||params[14]!=u64::from(bias)||params[13]>1{
        candle_core::bail!("LFM3 causal launch contract mismatch");
    }
    let device=Device::Cuda(dev.clone());let stream=dev.cuda_stream();
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,&device,||prepare(key,&device),|p|{
        unsafe{p.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            for ptr in pointers{b.arg(ptr);}
            for v in &params[..12]{b.arg(v);}
            if bias{b.arg(&params[12]);}
            b.arg(&params[13]);
            b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}
/// Same pointer/stream ownership contract as causal.
pub(super) unsafe fn spatial(key:Key,dev:&CudaDevice,pointers:[u64;3],elements:u64,geometry:[u32;5])->Result<()>{
    if key.kind!=Kind::Spatial||pointers.contains(&0){candle_core::bail!("LFM3 spatial launch contract mismatch");}
    let device=Device::Cuda(dev.clone());let stream=dev.cuda_stream();
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,&device,||prepare(key,&device),|p|{
        unsafe{p.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            for ptr in &pointers{b.arg(ptr);}
            b.arg(&elements);
            for v in &geometry{b.arg(v);}
            b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_lfm3_typed_geometry(){
        assert!(checked_product(&[usize::MAX,2]).is_err());
        assert!(checked_product(&[0,2]).is_err());
        let k=Key::causal(DType::BF16,&[42,7,7,14,7],14,7,false).unwrap();
        assert_eq!(k.operation(),"lfm3.causal.width7.nobias.v1");
        assert!(Key::causal(DType::BF16,&[42,7,7,14,7],14,7,true).is_err());
        assert!(Key::spatial(DType::F32,&[3,1,1],0).is_err());
    }
}
