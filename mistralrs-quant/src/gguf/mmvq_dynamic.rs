//! Typed dense MMVQ, reusing historical Q8_1 workspace and DP4A bodies.
use super::{output_shape, prefers_weight_prefetch, use_single_row_prefetch,
            workspace_ensure, MMVQ_MAX_BATCH};
use crate::dkr_cuda::{catalog, dtype, LaunchPlanCache};
use crate::utils::{slice_ptr_on_stream, slice_ptr_mut_on_stream};
use candle_core::{
    cuda::cudarc::driver::PushKernelArg, quantized::{GgmlDType,QTensor},
    CudaStorage, DType, Device, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument,ElementType,LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

#[derive(Clone,Copy,PartialEq,Eq)]
struct Key {
    format: &'static str,
    dtype: DType,
    batch: usize,
    n: usize,
    k: usize,
    padded: usize,
    weight_row: usize,
    scratch_row: usize,
    prefetch: bool,
}
struct Plans {
    encode: PrevalidatedLaunchPlan,
    matvec: PrevalidatedLaunchPlan,
}
thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key,Plans>> = RefCell::new(LaunchPlanCache::new());
}

fn format(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::Q4_0=>"q4_0", GgmlDType::Q4_1=>"q4_1",
        GgmlDType::Q5_0=>"q5_0", GgmlDType::Q5_1=>"q5_1",
        GgmlDType::Q8_0=>"q8_0", GgmlDType::Q2K=>"q2_k",
        GgmlDType::Q3K=>"q3_k", GgmlDType::Q4K=>"q4_k",
        GgmlDType::Q5K=>"q5_k", GgmlDType::Q6K=>"q6_k",
        _=>return None,
    })
}

fn tensor(dtype: ElementType, shape: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor { dtype, shape, contiguous:true, start_offset:0,
                       representation:repr.map(str::to_owned) }
}
fn scalar() -> Argument { Argument::Scalar {dtype:ElementType::I32} }

fn prepare_one(device:&Device, operation:String, args:Vec<Argument>,
               grid:[u32;3], block:[u32;3]) -> Result<PrevalidatedLaunchPlan> {
    let mut req=LaunchRequest {operation,cc:0,args,grid,max_dynamic_shared_mem_bytes:0};
    let prepared=catalog()?.prepare_for_device(&mut req,device)
        .map_err(|e| candle_core::Error::msg(e.to_string()))?;
    if prepared.manifest.block!=block || prepared.manifest.shared_mem_bytes!=0 {
        candle_core::bail!("DKR GGUF MMVQ geometry differs from historical ABI");
    }
    prepared.prevalidate_launch(&req,device)
        .map_err(|e| candle_core::Error::msg(e.to_string()))
}
fn prepare(key:Key, device:&Device) -> Result<Plans> {
    let dt=dtype(key.dtype)?;
    let q8=|| tensor(ElementType::U8,vec![key.batch,key.scratch_row],Some("gguf.q8_1.row.v1"));
    let encode=prepare_one(device,"quant.gguf.q8_1.row.encode.v1".into(),
        vec![tensor(dt,vec![key.batch,key.k],None),q8(),scalar(),scalar()],
        [key.padded.div_ceil(256) as u32,key.batch as u32,1],[256,1,1])?;
    let repr=format!("gguf.{}.packed.v1",key.format);
    let mut args=vec![tensor(ElementType::U8,vec![key.n,key.weight_row],Some(&repr)),
        q8(),tensor(dt,vec![key.batch,key.n],None),scalar()];
    let (variant,grid,block)=if key.prefetch {
        ("prefetch".to_owned(),key.n,[32,8,1])
    } else {
        args.extend([scalar(),scalar(),scalar()]);
        (format!("b{}",key.batch),key.n.div_ceil(2),
         [32,if key.batch==1 {8} else if key.batch<=4 {4} else {2},1])
    };
    let matvec=prepare_one(device,format!("quant.gguf.mmvq.{}.{variant}.v1",key.format),
        args,[grid as u32,1,1],block)?;
    Ok(Plans {encode,matvec})
}

pub(super) fn plain(w:&QTensor,xs:&Tensor)->Result<Tensor>{
    Ok(shared_lhs(&[w],xs)?.pop().expect("one projection"))
}

pub(super) fn shared_lhs(weights:&[&QTensor], xs:&Tensor) -> Result<Vec<Tensor>> {
    let first=weights.first().ok_or_else(|| candle_core::Error::msg("DKR MMVQ empty group"))?;
    let Device::Cuda(dev)=first.device() else {candle_core::bail!("DKR MMVQ requires CUDA");};
    let (_,k)=first.shape().dims2()?;
    if !xs.device().same_device(&first.device()){candle_core::bail!("DKR MMVQ input device");}
    let Some((&input_k,dims))=xs.dims().split_last() else { candle_core::bail!("DKR MMVQ input rank"); };
    let batch=dims.iter().try_fold(1usize,|a,&b|a.checked_mul(b))
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ batch overflow"))?;

    if input_k!=k || k==0
        || !(1..=MMVQ_MAX_BATCH).contains(&batch) {
        candle_core::bail!("DKR MMVQ shape not admitted");
    }
    dtype(xs.dtype())?;
    let padded=k.checked_add(511).ok_or_else(|| candle_core::Error::msg("DKR MMVQ padding overflow"))? / 512 * 512;
    let scratch_row=(padded/32).checked_mul(36)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ scratch row overflow"))?;
    let scratch_bytes=scratch_row.checked_mul(batch)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ scratch overflow"))?;
    let mut keys=Vec::with_capacity(weights.len());
    for w in weights {
        let (n,wk)=w.shape().dims2()?;
        let fmt=format(w.dtype()).ok_or_else(||candle_core::Error::msg("DKR GGUF format not admitted"))?;
        let block_elements=w.dtype().block_size();
        if n==0 || wk!=k || k%block_elements!=0 || !w.device().same_device(&first.device()){
            candle_core::bail!("DKR MMVQ group weight mismatch");
        }
    let weight_row=(k/block_elements).checked_mul(w.dtype().type_size())
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ weight row overflow"))?;
    let weight_bytes=weight_row.checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ weight span overflow"))?;
    let count=n.checked_mul(batch).ok_or_else(|| candle_core::Error::msg("DKR MMVQ output overflow"))?;
    // Historical device indexing is signed i32, including row/block products.
    if [n,k,padded,count,weight_bytes,scratch_bytes].iter().any(|&v|v>i32::MAX as usize)
        || padded.checked_mul(batch).is_none_or(|v|v>i32::MAX as usize)
        || n.checked_mul(k/block_elements).is_none_or(|v|v>i32::MAX as usize) {
        candle_core::bail!("DKR MMVQ index span exceeds i32");
    }
    let prefetch=batch==1 && !matches!(fmt,"q2_k"|"q3_k")
        && use_single_row_prefetch(w.dtype(),k,weight_bytes) && prefers_weight_prefetch(&dev);
    let key=Key{format:fmt,dtype:xs.dtype(),batch,n,k,padded,weight_row,scratch_row,prefetch};
        keys.push((key,count));

    }
    let stream=dev.cuda_stream();
    let xs=xs.contiguous()?;
    let (storage,layout)=xs.storage_and_layout();
    let Storage::Cuda(storage)=&*storage else { candle_core::bail!("DKR MMVQ requires CUDA activations"); };
    let mut workspace=workspace_ensure(&dev,scratch_bytes,&stream)?;
    let (scratch_ptr,_sg)=workspace.ptr_mut();
    let ki=k as i32;
    let pi=padded as i32;
    let stride=(padded/32) as i32;
    macro_rules! launch {
        ($t:ty)=>{{
            let (xp,_xg)=slice_ptr_on_stream(storage.as_cuda_slice::<$t>()?,layout.start_offset(),&stream);
            let mut outputs=Vec::with_capacity(weights.len());
            for (index,(w,(key,count))) in weights.iter().zip(keys.iter().copied()).enumerate(){
                let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
                let ni=key.n as i32;
                let mut out=unsafe{dev.alloc::<$t>(count)?};
                {
                    let (yp,_yg)=slice_ptr_mut_on_stream(&mut out,0,&stream);
                    PLANS.with(|plans|plans.borrow_mut().with_plan(key,xs.device(),||prepare(key,xs.device()),|plans|{
                        if index==0 {
                            unsafe{plans.encode.with_prevalidated_launch(|fun,cfg|{
                                let mut b=stream.launch_builder(fun);
                                b.arg(&xp).arg(&scratch_ptr).arg(&ki).arg(&pi);b.launch(cfg)?;Ok(())
                            })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                        }
                        unsafe{plans.matvec.with_prevalidated_launch(|fun,cfg|{
                            let mut b=stream.launch_builder(fun);
                            b.arg(&wp).arg(&scratch_ptr).arg(&yp).arg(&ki);
                            if !key.prefetch {b.arg(&ni).arg(&stride).arg(&ni);}
                            b.launch(cfg)?;Ok(())
                        })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                        Ok(())
                    }))?;
                }
                outputs.push(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),output_shape(&xs,key.n))));
            }
            Ok(outputs)
        }}
    }
    match xs.dtype(){DType::F16=>launch!(half::f16),DType::BF16=>launch!(half::bf16),
                     DType::F32=>launch!(f32),_=>unreachable!()}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore="requires staged GGUF MMVQ catalogue and CUDA device"]
    fn dkr_s9c_gguf_shared_lhs_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let q=QTensor::quantize(&Tensor::ones((65,256),DType::F32,&dev)?,GgmlDType::Q4K)?;
        let k=QTensor::quantize(&Tensor::ones((17,256),DType::F32,&dev)?,GgmlDType::Q8_0)?;
        let v=QTensor::quantize(&Tensor::ones((33,256),DType::F32,&dev)?,GgmlDType::Q5K)?;
        for batch in [1usize,3,8] {
            let x=Tensor::from_vec((0..batch*256).map(|i|((i%13)as f32-6.)/8.).collect::<Vec<_>>(),(batch,256),&dev)?;
            for dt in [DType::F16,DType::BF16,DType::F32] {
                let x=x.to_dtype(dt)?;
                let weights=[&q,&k,&v];
                let out=shared_lhs(&weights,&x)?;
                for (w,o) in weights.into_iter().zip(out){
                    let expected=plain(w,&x)?;
                    let delta=(o.to_dtype(DType::F32)?-expected.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert_eq!(delta,0.0);
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires staged gated mix catalogue and CUDA device"]
    fn dkr_s9c_gguf_gated_mix_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let w=QTensor::quantize(&Tensor::from_vec((0..6*256).map(|i|((i%17)as f32-8.)/128.).collect::<Vec<_>>(),(6,256),&dev)?,GgmlDType::Q4K)?;
        let x=Tensor::from_vec((0..256).map(|i|((i%9)as f32-4.)/8.).collect::<Vec<_>>(),(1,256),&dev)?;
        let residual=Tensor::from_vec(vec![1f32,-2.,3.,4.,-5.,6.],(1,6),&dev)?;
        for dt in [DType::F16,DType::BF16,DType::F32]{
            let x=x.to_dtype(dt)?;let residual=residual.to_dtype(dt)?;
            for divisor in [None,Some(2f32)]{
                let actual=gated_mix(&w,&x,&residual,2,divisor)?.to_dtype(DType::F32)?;
                let transformed=if let Some(d)=divisor{candle_nn::ops::silu(&x.affine(1.0/d as f64,0.)?)?}else{x.clone()};
                let gate=plain(&w,&transformed)?.to_dtype(DType::F32)?;
                let expected=candle_nn::ops::sigmoid(&gate)?.broadcast_mul(&residual.to_dtype(DType::F32)?)?.reshape((2,3))?.sum(0)?.affine(0.5,0.)?.reshape((1,3))?;
                let delta=(actual-expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(delta<0.05,"{dt:?} {divisor:?} {delta}");
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires staged GGUF MMVQ catalogue and CUDA device"]
    fn dkr_s9c_gguf_mmvq_gpu() -> Result<()> {
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let device=Device::new_cuda(0)?;
        for format in [GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,
                       GgmlDType::Q8_0,GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,
                       GgmlDType::Q5K,GgmlDType::Q6K] {
            let weights=Tensor::from_vec((0..3*256).map(|i| ((i%17) as f32-8.)/8.).collect::<Vec<_>>(),(3,256),&device)?;
            let weights=QTensor::quantize(&weights,format)?;
            let dense=weights.dequantize(&device)?.t()?;
            for batch in 1..=8 {
                let input=Tensor::from_vec((0..batch*256).map(|i| ((i%9) as f32-4.)/8.).collect::<Vec<_>>(),(batch,256),&device)?;
                let expected=input.matmul(&dense)?.flatten_all()?.to_vec1::<f32>()?;
                for dt in [DType::F16,DType::BF16,DType::F32] {
                    let actual=plain(&weights,&input.to_dtype(dt)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                    assert_eq!(actual.len(),expected.len());
                    for (&a,&e) in actual.iter().zip(&expected) {
                        assert!((a-e).abs()<=0.15+0.025*e.abs(),"{format:?} {dt:?} batch={batch}: {a} vs {e}");
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone,Copy,PartialEq,Eq)]
struct MixKey{fmt:&'static str, dtype:DType, rank:usize,hidden:usize,streams:usize,padded:usize,packed:usize,silu:bool}
struct MixPlans{encode:PrevalidatedLaunchPlan,mix:PrevalidatedLaunchPlan}
thread_local!{static MIX:RefCell<LaunchPlanCache<MixKey,MixPlans>>=RefCell::new(LaunchPlanCache::new());}
fn mix_plans(key:MixKey,device:&Device)->Result<MixPlans>{
    let dt=dtype(key.dtype)?;
    let scratch=||tensor(ElementType::U8,vec![1,key.padded/32*36],Some("gguf.q8_1.row.v1"));
    let mut args=vec![tensor(dt,vec![1,key.rank],None),scratch(),scalar(),scalar()];
    if key.silu {args.push(Argument::Scalar{dtype:ElementType::F32});}
    let encode=prepare_one(device,if key.silu{"quant.gguf.mix.silu.encode.v1".into()}else{"quant.gguf.q8_1.row.encode.v1".into()},
        args,[key.padded.div_ceil(256) as u32,1,1],[256,1,1])?;
    let repr=format!("gguf.{}.packed.v1",key.fmt);
    let mix=prepare_one(device,format!("quant.gguf.mix.{}.v1",key.fmt),
        vec![tensor(ElementType::U8,vec![key.hidden*key.streams,key.packed],Some(&repr)),scratch(),
             tensor(dt,vec![key.streams,key.hidden],None),tensor(dt,vec![key.hidden],None),scalar(),scalar(),scalar()],
        [key.hidden.div_ceil(4) as u32,1,1],[128,1,1])?;
    Ok(MixPlans{encode,mix})
}
pub(crate) fn gated_mix(w:&QTensor,x:&Tensor,r:&Tensor,streams:usize,divisor:Option<f32>)->Result<Tensor>{
    let fmt=if w.dtype()==GgmlDType::Q8_1{"q8_1"}else{format(w.dtype()).ok_or_else(||candle_core::Error::msg("DKR gated mix format"))?};
    let (width,rank)=w.shape().dims2()?;
    if streams==0 || rank==0 || width==0 || width%streams!=0 || x.elem_count()!=rank || r.elem_count()!=width
        || x.dim(candle_core::D::Minus1)?!=rank || r.dim(candle_core::D::Minus1)?!=width
        || r.dtype()!=x.dtype() || !w.device().same_device(x.device()) || !r.device().same_device(x.device())
        || rank%w.dtype().block_size()!=0 || divisor.is_some_and(|d|!d.is_finite()||d==0.0){
        candle_core::bail!("DKR gated mix contract");
    }
    dtype(x.dtype())?;
    let Device::Cuda(dev)=w.device() else{candle_core::bail!("DKR gated mix CUDA required");};
    let hidden=width/streams;
    let padded=rank.checked_add(511).ok_or_else(||candle_core::Error::msg("DKR mix padding"))?/512*512;
    let packed=(rank/w.dtype().block_size()).checked_mul(w.dtype().type_size())
        .ok_or_else(||candle_core::Error::msg("DKR mix packed row"))?;
    let bytes=(padded/32).checked_mul(36).ok_or_else(||candle_core::Error::msg("DKR mix scratch"))?;
    if [rank,padded,width,streams,bytes].iter().any(|&n|n>i32::MAX as usize)
        || width.checked_mul(packed).is_none_or(|n|n>i32::MAX as usize){
        candle_core::bail!("DKR mix index span");
    }
    let key=MixKey{fmt,dtype:x.dtype(),rank,hidden,streams,padded,packed,silu:divisor.is_some()};
    let x=x.contiguous()?;let r=r.contiguous()?;
    let (xs,xl)=x.storage_and_layout();let (rs,rl)=r.storage_and_layout();
    let (Storage::Cuda(xs),Storage::Cuda(rs))=(&*xs,&*rs)else{candle_core::bail!("DKR mix CUDA storage");};
    let stream=dev.cuda_stream();
    let mut ws=workspace_ensure(&dev,bytes,&stream)?;
    let (scratch,_sg)=ws.ptr_mut();
    let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
    let (ki,pi,hi,si)=(rank as i32,padded as i32,hidden as i32,streams as i32);
    let mut shape=r.dims().to_vec();*shape.last_mut().unwrap()=hidden;
    macro_rules! run {($t:ty)=>{{
        let (xp,_xg)=slice_ptr_on_stream(xs.as_cuda_slice::<$t>()?,xl.start_offset(),&stream);
        let (rp,_rg)=slice_ptr_on_stream(rs.as_cuda_slice::<$t>()?,rl.start_offset(),&stream);
        let mut out=unsafe{dev.alloc::<$t>(hidden)?};
        {
            let (op,_og)=slice_ptr_mut_on_stream(&mut out,0,&stream);
            MIX.with(|cache|cache.borrow_mut().with_plan(key,x.device(),||mix_plans(key,x.device()),|p|{
                unsafe{p.encode.with_prevalidated_launch(|fun,cfg|{
                    let mut b=stream.launch_builder(fun);
                    b.arg(&xp).arg(&scratch).arg(&ki).arg(&pi);
                    if let Some(d)=divisor{b.arg(&d);b.launch(cfg)?;}else{b.launch(cfg)?;}
                    Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                unsafe{p.mix.with_prevalidated_launch(|fun,cfg|{
                    let mut b=stream.launch_builder(fun);b.arg(&wp).arg(&scratch).arg(&rp).arg(&op).arg(&ki).arg(&hi).arg(&si);
                    b.launch(cfg)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                Ok(())
            }))?;
        }
        Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),candle_core::Shape::from(shape))))
    }}}
    match x.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}
}

// The historical QKV body also admits a pair (nv=0, packed grid).
// A real one-element V allocation and a valid dummy weight pointer keep
// the tensor ABI non-null; no V read or store is reachable in pair mode.
#[derive(Clone,Copy,PartialEq,Eq)]
struct FusedKey {
    fmt:&'static str,dt:DType,batch:usize,k:usize,padded:usize,packed:usize,
    rows:[usize;3],glu:bool,prefetch:bool,
}
struct FusedPlans{encode:PrevalidatedLaunchPlan,project:PrevalidatedLaunchPlan}
thread_local!{static FUSED:RefCell<LaunchPlanCache<FusedKey,FusedPlans>>=RefCell::new(LaunchPlanCache::new());}
fn fused_grid(key:FusedKey)->([u32;3],[u32;3]){
    let [q,k,v]=key.rows;
    let b=key.batch;
    if key.glu {
        let rows=if (2..=4).contains(&b){1}else{2};
        return ([q.div_ceil(rows) as u32,1,1],[32,if b==1{4}else{2},1]);
    }
    if key.prefetch{return ([(q+k+v) as u32,1,1],[32,8,1]);}
    let max=q.max(k).max(v);let min=q.min(k).min(v);
    let packed=v==0 || max>=2*min;
    let grid=if packed{[(q.div_ceil(2)+k.div_ceil(2)+v.div_ceil(2)) as u32,1,1]}else{[max.div_ceil(2) as u32,3,1]};
    (grid,[32,if b==1{8}else if b<=4{4}else{2},1])
}
fn fused_plans(key:FusedKey,device:&Device)->Result<FusedPlans>{
    let dt=dtype(key.dt)?;
    let q8=||tensor(ElementType::U8,vec![key.batch,key.padded/32*36],Some("gguf.q8_1.row.v1"));
    let enc=prepare_one(device,"quant.gguf.q8_1.row.encode.v1".into(),
        vec![tensor(dt,vec![key.batch,key.k],None),q8(),scalar(),scalar()],
        [key.padded.div_ceil(256) as u32,key.batch as u32,1],[256,1,1])?;
    let repr=format!("gguf.{}.packed.v1",key.fmt);
    let w=|n|tensor(ElementType::U8,vec![n,key.packed],Some(&repr));
    let out=|n|tensor(dt,vec![key.batch,n],None);
    let [q,k,v]=key.rows;
    let (kind,args)=if key.glu {
        ("glu",vec![w(q),w(k),q8(),out(q),scalar(),scalar(),scalar(),scalar(),scalar()])
    }else{
        ("qkv",vec![w(q),w(k),w(if v==0{q}else{v}),q8(),out(q),out(k),
            if v==0{tensor(dt,vec![1,1],None)}else{out(v)},scalar(),scalar(),scalar(),scalar(),scalar()])
    };
    let variant=if key.prefetch{"prefetch".into()}else{format!("b{}",key.batch)};
    let (grid,block)=fused_grid(key);
    let project=prepare_one(device,format!("quant.gguf.mmvq.fused.{kind}.{}.{variant}.v1",key.fmt),args,grid,block)?;
    Ok(FusedPlans{encode:enc,project})
}
fn checked_span(v:usize)->Result<usize>{
    if v>i32::MAX as usize{candle_core::bail!("DKR fused MMVQ span exceeds i32");}Ok(v)
}
fn prod(a:usize,b:usize)->Result<usize>{a.checked_mul(b).ok_or_else(||candle_core::Error::msg("DKR MMVQ span overflow"))}

pub(super) fn fused(q:&QTensor,k:&QTensor,v:Option<&QTensor>,xs:&Tensor,act:Option<i32>)->Result<Vec<Tensor>>{
    let fmt=format(q.dtype()).ok_or_else(||candle_core::Error::msg("DKR fused MMVQ format"))?;
    let (nq,columns)=q.shape().dims2()?;let (nk,kc)=k.shape().dims2()?;
    let (nv,vc)=if let Some(v)=v{v.shape().dims2()?}else{(0,columns)};
    let vw=v.unwrap_or(q);
    let Device::Cuda(dev)=q.device()else{candle_core::bail!("DKR fused MMVQ CUDA required");};
    let Some((&input,dims))=xs.dims().split_last()else{candle_core::bail!("DKR fused MMVQ input rank");};
    let batch=dims.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).ok_or_else(||candle_core::Error::msg("batch overflow"))?;
    if columns==0 || nq==0 || nk==0 || v.is_some_and(|_|nv==0) || kc!=columns || vc!=columns || input!=columns
        || columns%q.dtype().block_size()!=0 || !(1..=MMVQ_MAX_BATCH).contains(&batch)
        || act.is_some_and(|a| !(0..=4).contains(&a)||nq!=nk||v.is_some())
        || [q,k,vw].iter().any(|w|w.dtype()!=q.dtype()||!w.device().same_device(xs.device())){
        candle_core::bail!("DKR fused MMVQ shape/device/format mismatch");
    }
    dtype(xs.dtype())?;
    let packed=prod(columns/q.dtype().block_size(),q.dtype().type_size())?;
    let padded=columns.checked_add(511).ok_or_else(||candle_core::Error::msg("padding overflow"))?/512*512;
    let scratch_bytes=prod(prod(padded/32,36)?,batch)?;
    checked_span(prod(padded,batch)?)?;checked_span(scratch_bytes)?;
    let total=nq.checked_add(nk).and_then(|n|n.checked_add(nv)).ok_or_else(||candle_core::Error::msg("row sum overflow"))?;
    checked_span(total.checked_add(2).ok_or_else(||candle_core::Error::msg("row grid overflow"))?)?;
    checked_span(prod(total,packed)?)?;
    for n in [nq,nk,nv]{checked_span(prod(n,batch)?)?;checked_span(prod(n,columns/q.dtype().block_size())?)?;}
    let prefetch=act.is_none() && batch==1 && !matches!(fmt,"q2_k"|"q3_k")
        && use_single_row_prefetch(q.dtype(),columns,total*packed) && prefers_weight_prefetch(&dev);
    let key=FusedKey{fmt,dt:xs.dtype(),batch,k:columns,padded,packed,rows:[nq,nk,nv],glu:act.is_some(),prefetch};
    let xs=xs.contiguous()?;let (st,l)=xs.storage_and_layout();
    let Storage::Cuda(st)=&*st else{candle_core::bail!("DKR fused MMVQ activation CUDA");};
    let stream=dev.cuda_stream();
    let mut ws=workspace_ensure(&dev,scratch_bytes,&stream)?;
    let (scratch,_sg)=ws.ptr_mut();
    let (qp,_qg)=q.device_ptr_with_guard(&stream)?;
    let (kp,_kg)=k.device_ptr_with_guard(&stream)?;
    let (vp,_vg)=vw.device_ptr_with_guard(&stream)?;
    let (ci,pi,qi,ki,vi,si)=(columns as i32,padded as i32,nq as i32,nk as i32,nv as i32,(padded/32) as i32);
    let activation=act.unwrap_or(0);
    macro_rules! run {($t:ty)=>{{
        let (xp,_xg)=slice_ptr_on_stream(st.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
        let mut qo=unsafe{dev.alloc::<$t>(batch*nq)?};
        let mut ko=unsafe{dev.alloc::<$t>(if key.glu{1}else{batch*nk})?};
        let mut vo=unsafe{dev.alloc::<$t>((batch*nv).max(1))?};
        {
            let (qop,_qog)=slice_ptr_mut_on_stream(&mut qo,0,&stream);
            let (kop,_kog)=slice_ptr_mut_on_stream(&mut ko,0,&stream);
            let (vop,_vog)=slice_ptr_mut_on_stream(&mut vo,0,&stream);
            FUSED.with(|cache|cache.borrow_mut().with_plan(key,xs.device(),||fused_plans(key,xs.device()),|p|{
                unsafe{p.encode.with_prevalidated_launch(|fun,cfg|{
                    let mut b=stream.launch_builder(fun);b.arg(&xp).arg(&scratch).arg(&ci).arg(&pi);b.launch(cfg)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                unsafe{p.project.with_prevalidated_launch(|fun,cfg|{
                    let mut b=stream.launch_builder(fun);
                    if key.glu {
                        b.arg(&qp).arg(&kp).arg(&scratch).arg(&qop).arg(&ci).arg(&qi).arg(&si).arg(&qi).arg(&activation);
                        b.launch(cfg)?;
                    }else{
                        b.arg(&qp).arg(&kp).arg(&vp).arg(&scratch).arg(&qop).arg(&kop).arg(&vop)
                         .arg(&ci).arg(&qi).arg(&ki).arg(&vi).arg(&si);
                        b.launch(cfg)?;
                    }
                    Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                Ok(())
            }))?;
        }
        let mut out=vec![Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(qo,dev.clone())),output_shape(&xs,nq)))];
        if !key.glu{out.push(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(ko,dev.clone())),output_shape(&xs,nk))));}
        if nv!=0{out.push(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(vo,dev.clone())),output_shape(&xs,nv))));}
        Ok(out)
    }}}
    match xs.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}
}

#[derive(Clone,Copy,PartialEq,Eq)]
struct IndexedKey{
    fmt:&'static str,dt:DType,e:usize,n:usize,k:usize,batch:usize,topk:usize,input_routes:usize,padded:usize,packed:usize,
}
struct IndexedPlans{encode:PrevalidatedLaunchPlan,gather:PrevalidatedLaunchPlan}
thread_local!{static INDEXED:RefCell<LaunchPlanCache<IndexedKey,IndexedPlans>>=RefCell::new(LaunchPlanCache::new());}
fn indexed_plans(key:IndexedKey,device:&Device)->Result<IndexedPlans>{
    let rows=key.batch*key.input_routes;
    let q8=||tensor(ElementType::U8,vec![rows,key.padded/32*36],Some("gguf.q8_1.row.v1"));
    let encode=prepare_one(device,"quant.gguf.q8_1.row.encode.v1".into(),
        vec![tensor(dtype(key.dt)?,vec![rows,key.k],None),q8(),scalar(),scalar()],
        [key.padded.div_ceil(256) as u32,rows as u32,1],[256,1,1])?;
    let repr=format!("gguf.{}.packed.v1",key.fmt);
    let gather=prepare_one(device,format!("quant.gguf.moe.indexed_mmvq.{}.v1",key.fmt),
        vec![tensor(ElementType::U8,vec![key.e,key.n,key.packed],Some(&repr)),q8(),
            tensor(ElementType::U32,vec![key.batch,key.topk],None),
            tensor(ElementType::F32,vec![key.batch,key.topk,key.n],None),
            scalar(),scalar(),scalar(),scalar(),scalar(),scalar(),scalar()],
        [key.n as u32,key.batch as u32,key.topk as u32],[32,4,1])?;
    Ok(IndexedPlans{encode,gather})
}
pub(crate) fn indexed(w:&QTensor,x:&Tensor,ids:&Tensor)->Result<Tensor>{
    let fmt=if w.dtype()==GgmlDType::Q8_1{"q8_1"}else{format(w.dtype()).ok_or_else(||candle_core::Error::msg("DKR indexed format"))?};
    let (e,n,k)=w.shape().dims3()?;let (batch,input_routes,xk)=x.dims3()?;let (ib,topk)=ids.dims2()?;
    let Device::Cuda(dev)=w.device()else{candle_core::bail!("DKR indexed MMVQ CUDA required");};
    if e==0||n==0||k==0||batch==0||topk==0||ib!=batch||xk!=k||k%w.dtype().block_size()!=0
        || !(input_routes==1||input_routes==topk)||ids.dtype()!=DType::U32
        || !ids.device().same_device(x.device())||!w.device().same_device(x.device()){
        candle_core::bail!("DKR indexed MMVQ contract");
    }
    dtype(x.dtype())?;
    let rows=prod(batch,input_routes)?;let routes=prod(batch,topk)?;
    if rows>65535||batch>65535||topk>65535{candle_core::bail!("DKR indexed MMVQ grid limit");}
    let padded=k.checked_add(511).ok_or_else(||candle_core::Error::msg("padding overflow"))?/512*512;
    let packed=prod(k/w.dtype().block_size(),w.dtype().type_size())?;
    let scratch_bytes=checked_span(prod(prod(padded/32,36)?,rows)?)?;
    checked_span(prod(rows,padded)?)?;checked_span(prod(prod(e,n)?,packed)?)?;
    let count=checked_span(prod(routes,n)?)?;
    for v in [e,n,k,padded,routes]{checked_span(v)?;}
    // historical task_id uses signed i32 and gridDim.y*gridDim.z
    let key=IndexedKey{fmt,dt:x.dtype(),e,n,k,batch,topk,input_routes,padded,packed};
    let x=x.contiguous()?;let ids=ids.contiguous()?;
    let (st,l)=x.storage_and_layout();let (is,il)=ids.storage_and_layout();
    let (Storage::Cuda(st),Storage::Cuda(is))=(&*st,&*is)else{candle_core::bail!("DKR indexed storage");};
    let stream=dev.cuda_stream();
    let mut ws=workspace_ensure(&dev,scratch_bytes,&stream)?;
    let (scratch,_sg)=ws.ptr_mut();let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
    let (ip,_ig)=slice_ptr_on_stream(is.as_cuda_slice::<u32>()?,il.start_offset(),&stream);
    let (ni,ki,bi,ti,pi,ri,ei)=(n as i32,k as i32,batch as i32,topk as i32,padded as i32,input_routes as i32,e as i32);
    let mut out=unsafe{dev.alloc::<f32>(count)?};
    macro_rules! run {($t:ty)=>{{
        let (xp,_xg)=slice_ptr_on_stream(st.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
        let (op,_og)=slice_ptr_mut_on_stream(&mut out,0,&stream);
        INDEXED.with(|cache|cache.borrow_mut().with_plan(key,x.device(),||indexed_plans(key,x.device()),|p|{
            unsafe{p.encode.with_prevalidated_launch(|fun,cfg|{
                let mut b=stream.launch_builder(fun);b.arg(&xp).arg(&scratch).arg(&ki).arg(&pi);b.launch(cfg)?;Ok(())
            })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
            unsafe{p.gather.with_prevalidated_launch(|fun,cfg|{
                let mut b=stream.launch_builder(fun);b.arg(&wp).arg(&scratch).arg(&ip).arg(&op)
                    .arg(&ni).arg(&ki).arg(&bi).arg(&ti).arg(&pi).arg(&ri).arg(&ei);b.launch(cfg)?;Ok(())
            })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
            Ok(())
        }))?;
    }}}
    match x.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}
    Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),(batch,topk,n))))
}

#[cfg(test)]
mod fused_tests{
    use super::*;
    #[test]
    fn dkr_s9c_fused_grid_policy(){
        let mut key=FusedKey{fmt:"q4_k",dt:DType::F32,batch:1,k:256,padded:512,packed:144,
            rows:[65,17,33],glu:false,prefetch:false};
        assert_eq!(fused_grid(key),([59,1,1],[32,8,1]));
        key.rows=[65,65,65];assert_eq!(fused_grid(key),([33,3,1],[32,8,1]));
        key.rows=[65,17,0];assert_eq!(fused_grid(key),([42,1,1],[32,8,1]));
        key.prefetch=true;assert_eq!(fused_grid(key),([82,1,1],[32,8,1]));
        key.prefetch=false;key.glu=true;key.rows=[65,65,0];key.batch=3;
        assert_eq!(fused_grid(key),([65,1,1],[32,2,1]));
    }
    #[test]
    #[ignore="requires fused GGUF catalogue and SM61 GPU"]
    fn dkr_s9c_exact_fused_mmvq_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        for quant in [GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,GgmlDType::Q8_0,
            GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,GgmlDType::Q5K,GgmlDType::Q6K]{
            let make=|n:usize,salt:usize| QTensor::quantize(&Tensor::from_vec(
                (0..n*256).map(|i|(((i+salt)%17)as f32-8.)/64.).collect::<Vec<_>>(),(n,256),&dev)?,quant);
            let q=make(65,0)?;let k=make(17,2)?;let v=make(33,4)?;let up=make(65,5)?;
            for batch in 1..=8 {
                let x=Tensor::from_vec((0..batch*256).map(|i|((i%9)as f32-4.)/8.).collect::<Vec<_>>(),(batch,256),&dev)?;
                for dt in [DType::F16,DType::BF16,DType::F32]{
                    let x=x.to_dtype(dt)?;
                    let exact=fused(&q,&k,Some(&v),&x,None)?;
                    let expected=shared_lhs(&[&q,&k,&v],&x)?;
                    for (a,b) in exact.into_iter().zip(expected){
                        let delta=(a.to_dtype(DType::F32)?-b.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        assert!(delta<=0.01,"QKV {quant:?} {dt:?} {batch}: {delta}");
                    }
                    let pair=fused(&q,&k,None,&x,None)?;
                    assert_eq!(pair.len(),2);
                    for (a,b) in pair.into_iter().zip(shared_lhs(&[&q,&k],&x)?){
                        let delta=(a.to_dtype(DType::F32)?-b.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        assert!(delta<=0.01,"pair {quant:?} {dt:?} {batch}: {delta}");
                    }
                    for act in [crate::GluActivationType::Silu,crate::GluActivationType::Gelu,crate::GluActivationType::Relu,
                        crate::GluActivationType::GeluErf,crate::GluActivationType::Sigmoid]{
                        let actual=fused(&q,&up,None,&x,Some(act as i32))?.pop().unwrap();
                        let parts=shared_lhs(&[&q,&up],&x)?;
                        let expected=super::super::super::group::typed_glu(&parts[0],&parts[1],act)?;
                        let delta=(actual.to_dtype(DType::F32)?-expected.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                        assert!(delta<=0.025,"GLU {quant:?} {dt:?} {batch}: {delta}");
                    }
                }
            }
        }
        Ok(())
    }
}
