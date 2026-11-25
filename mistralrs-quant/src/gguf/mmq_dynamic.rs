//! Conventional SM61 MMQ: exact packed contracts, no FFI launcher or fixup.
use super::{output_shape, q8_1, qk_for, ds_layout_for, DsLayout, workspace_ensure,
            MMQ_WORKSPACE, MMQ_X_MAX};
use crate::{cuda_device::cuda_device_info, dkr_cuda::{catalog,dtype,LaunchPlanCache},
            utils::{slice_ptr_on_stream,slice_ptr_mut_on_stream}};
use candle_core::{
    cuda::cudarc::driver::PushKernelArg, quantized::{GgmlDType,QTensor},
    CudaStorage,DType,Device,Result,Storage,Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument,ElementType,LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const SHARED_LIMIT:usize=48*1024;
const Y:usize=64;
const ITER_K:usize=256;

#[derive(Clone,Copy,PartialEq,Eq)]
struct Key {
    format:&'static str, layout:&'static str, dtype:DType,
    batch:usize, n:usize, k:usize, padded:usize, blocks_k:usize,
    row_bytes:usize, scratch_col_bytes:usize, tile:usize, shared:usize, glu:bool,
}
struct Plans { encode:PrevalidatedLaunchPlan, mmq:PrevalidatedLaunchPlan }
thread_local! {
    static PLANS:RefCell<LaunchPlanCache<Key,Plans>>=RefCell::new(LaunchPlanCache::new());
}

pub(super) fn format(dt:GgmlDType)->Option<&'static str> {
    Some(match dt {
        GgmlDType::Q4_0=>"q4_0",GgmlDType::Q4_1=>"q4_1",
        GgmlDType::Q5_0=>"q5_0",GgmlDType::Q5_1=>"q5_1",
        GgmlDType::Q8_0=>"q8_0",GgmlDType::Q2K=>"q2_k",
        GgmlDType::Q3K=>"q3_k",GgmlDType::Q4K=>"q4_k",
        GgmlDType::Q5K=>"q5_k",GgmlDType::Q6K=>"q6_k",
        _=>return None,
    })
}
pub(super) fn layout(dt:GgmlDType)->&'static str {
    match ds_layout_for(dt) {DsLayout::D4=>"d4",DsLayout::DS4=>"ds4",DsLayout::D2S6=>"d2s6"}
}

// Exact historical mmq_get_dp4a_tile_x_sizes for Y=64.
// y tiles have 144-byte blocks and are rounded to 256 int lanes.
pub(super) fn shared_bytes(dt:GgmlDType,tile:usize)->usize {
    let (qs,dm,sc)=match dt {
        GgmlDType::Q4_0|GgmlDType::Q4_1=>(64*33,64*32/4+64/4,0),
        GgmlDType::Q5_0|GgmlDType::Q5_1|GgmlDType::Q8_0=>(64*65,64*64/8+64/4,0),
        GgmlDType::Q2K=>(64*65,64*33,0),
        GgmlDType::Q3K=>(64*65,64,64*32/8+64/8),
        GgmlDType::Q4K=>(64*33,64*32/32,64*32/8+64/8),
        GgmlDType::Q5K|GgmlDType::Q6K=>(64*65,64*32/32+64/32,64*32/8+64/8),
        _=>unreachable!("format checked"),
    };
    4*tile+4*(qs+dm+sc)+(tile*144).div_ceil(1024)*1024
}
fn choose_tile(dt:GgmlDType,batch:usize,limit:usize)->Option<(usize,usize)> {
    // Match historical min tiles, first (smallest) tile on ties.
    let mut best=None;
    let mut best_count=usize::MAX;
    for tile in (8..=64).step_by(8) {
        let smem=shared_bytes(dt,tile);
        if smem>limit.min(SHARED_LIMIT) {continue;}
        let count=batch.div_ceil(tile);
        if count<best_count {best=Some((tile,smem));best_count=count;}
        if best_count==1 {break;}
    }
    best
}
pub(super) fn tensor(dt:ElementType,shape:Vec<usize>,repr:Option<&str>)->Argument {
    Argument::Tensor {dtype:dt,shape,contiguous:true,start_offset:0,representation:repr.map(str::to_owned)}
}
pub(super) fn scalar()->Argument {Argument::Scalar{dtype:ElementType::I32}}
pub(super) fn prepare_one(device:&Device,operation:String,args:Vec<Argument>,
               grid:[u32;3],block:[u32;3],smem:u32)->Result<PrevalidatedLaunchPlan> {
    let mut req=LaunchRequest{operation,cc:0,args,grid,max_dynamic_shared_mem_bytes:smem};
    let prepared=catalog()?.prepare_for_device(&mut req,device)
        .map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if prepared.manifest.block!=block || prepared.manifest.shared_mem_bytes!=smem {
        candle_core::bail!("DKR GGUF MMQ geometry differs from typed contract");
    }
    prepared.prevalidate_launch(&req,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}
fn prepare(key:Key,device:&Device)->Result<Plans> {
    let dt=dtype(key.dtype)?;
    let repr=format!("gguf.q8_1.mmq.{}.v1",key.layout);
    // K-major block planes, each containing batch * 144 bytes.
    let scratch=||tensor(ElementType::U8,vec![key.padded/128,key.scratch_col_bytes],Some(&repr));
    let input=||tensor(dt,vec![key.batch,key.k],None);
    let (operation,args)=if key.glu {
        (format!("quant.gguf.glu.{}.encode.v1",key.layout),
         vec![input(),input(),scratch(),scalar(),scalar(),scalar(),scalar()])
    } else {
        (format!("quant.gguf.q8_1.mmq.{}.encode.v1",key.layout),
         vec![input(),scratch(),scalar(),scalar(),scalar()])
    };
    let encode=prepare_one(device,operation,args,[key.batch as u32,(key.padded/512) as u32,1],[128,1,1],0)?;
    let weight_repr=format!("gguf.{}.packed.v1",key.format);
    let bounds=if key.n%Y==0 {"full"} else {"tail"};
    let mmq=prepare_one(device,format!("quant.gguf.mmq.{}.x{}.{}.v1",key.format,key.tile,bounds),
        vec![tensor(ElementType::U8,vec![key.n,key.row_bytes],Some(&weight_repr)),scratch(),
             tensor(dt,vec![key.batch,key.n],None),scalar(),scalar(),scalar()],
        [key.n.div_ceil(Y) as u32,key.batch.div_ceil(key.tile) as u32,1],
        [32,8,1],key.shared as u32)?;
    Ok(Plans{encode,mmq})
}

pub(super) fn mul(a:usize,b:usize)->Result<usize> {
    a.checked_mul(b).ok_or_else(||candle_core::Error::msg("DKR MMQ size overflow"))
}
pub(super) fn i32_span(v:usize)->Result<usize> {
    if v>i32::MAX as usize {candle_core::bail!("DKR MMQ index span exceeds i32");}
    Ok(v)
}
pub(super) fn shared_lhs(weights:&[&QTensor],xs:&Tensor)->Result<Vec<Tensor>> {
    run(weights,xs,None)
}
pub(super) fn down_from_glu(w:&QTensor,gate:&Tensor,up:&Tensor,activation:crate::GluActivationType)->Result<Tensor>{
    Ok(run(&[w],gate,Some((up,activation as i32)))?.pop().expect("one projection"))
}
fn run(weights:&[&QTensor],xs:&Tensor,glu:Option<(&Tensor,i32)>)->Result<Vec<Tensor>> {
    if let Some((up,_))=glu {
        if up.shape()!=xs.shape() || up.dtype()!=xs.dtype() || !up.device().same_device(xs.device()){
            candle_core::bail!("DKR GLU encoder input mismatch");
        }
    }
    let Some(first)=weights.first() else {candle_core::bail!("DKR MMQ requires a weight");};
    let quant=first.dtype();
    let fmt=format(quant).ok_or_else(||candle_core::Error::msg("DKR MMQ unsupported weight format"))?;
    let Device::Cuda(dev)=first.device() else {candle_core::bail!("DKR MMQ requires CUDA weights");};
    let info=cuda_device_info(&dev);
    if info.capability()!=(6,1) || info.warp_size!=32 {
        candle_core::bail!("DKR MMQ conventional provider currently requires SM61");
    }
    if !xs.device().same_device(&first.device()) {candle_core::bail!("DKR MMQ same-device input required");}
    let Some((&k,dims))=xs.dims().split_last() else {candle_core::bail!("DKR MMQ input rank");};
    let batch=dims.iter().try_fold(1usize,|a,&b|a.checked_mul(b))
        .ok_or_else(||candle_core::Error::msg("DKR MMQ batch overflow"))?;
    // Historical tile_x loader consumes a complete 256-value K iteration.
    if k==0 || k%ITER_K!=0 || batch==0 {candle_core::bail!("DKR MMQ requires nonempty batch and K multiple of 256");}
    dtype(xs.dtype())?;
    let qk=qk_for(quant);
    let blocks_k=k/qk;
    let row_bytes=mul(blocks_k,quant.type_size())?;
    let padded=k.checked_add(511).ok_or_else(||candle_core::Error::msg("DKR MMQ padding overflow"))?/512*512;
    let scratch_col_bytes=mul(batch,q8_1::MMQ_BLOCK_BYTES)?;
    let main_bytes=i32_span(mul(padded/128,scratch_col_bytes)?)?;
    let workspace_bytes=main_bytes.checked_add(MMQ_X_MAX*q8_1::MMQ_BLOCK_BYTES)
        .ok_or_else(||candle_core::Error::msg("DKR MMQ scratch overflow"))?;
    i32_span(mul(batch,k)?)?;
    i32_span(padded)?;
    i32_span(batch)?;
    let (tile,shared)=choose_tile(quant,batch,info.smem_optin)
        .ok_or_else(||candle_core::Error::msg("DKR MMQ no admitted tile geometry"))?;
    if padded/512>65535 || batch.div_ceil(tile)>65535 {
        candle_core::bail!("DKR MMQ grid exceeds CUDA y dimension");
    }
    // Validate all projections before quantizing or allocating outputs.
    let mut keys=Vec::with_capacity(weights.len());
    for w in weights {
        let (n,wk)=w.shape().dims2()?;
        if n==0 || wk!=k || w.dtype()!=quant || !w.device().same_device(&first.device()) {
            candle_core::bail!("DKR MMQ shared-LHS weight contract mismatch");
        }
        i32_span(mul(n,blocks_k)?)?;
        i32_span(mul(n,row_bytes)?)?;
        i32_span(mul(n,batch)?)?;
        // Account for last row/column tile index addition in signed device math.
        i32_span(n.checked_add(Y).ok_or_else(||candle_core::Error::msg("DKR MMQ row overflow"))?)?;
        i32_span(batch.checked_add(tile).ok_or_else(||candle_core::Error::msg("DKR MMQ column overflow"))?)?;
        keys.push(Key{format:fmt,layout:layout(quant),dtype:xs.dtype(),batch,n,k,padded,
                      blocks_k,row_bytes,scratch_col_bytes,tile,shared,glu:glu.is_some()});
    }
    let xs=xs.contiguous()?;
    let (storage,l)=xs.storage_and_layout();
    let Storage::Cuda(storage)=&*storage else {candle_core::bail!("DKR MMQ requires CUDA activations");};
    let up=glu.map(|(up,_)|up.contiguous()).transpose()?;
    let up_storage=up.as_ref().map(|t|t.storage_and_layout());
    let stream=dev.cuda_stream();
    let mut workspace=workspace_ensure(&MMQ_WORKSPACE,&dev,workspace_bytes,&stream)?;
    let (scratch,_sg)=workspace.ptr_mut();
    let ki=k as i32;
    let pi=padded as i32;
    let bi=batch as i32;
    let bk=blocks_k as i32;
    macro_rules! launch {
        ($t:ty)=>{{
            let (xp,_xg)=slice_ptr_on_stream(storage.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
            // Historical float4/half2/bfloat162 activation loads require alignment.
            if xp%16!=0 {candle_core::bail!("DKR MMQ activation pointer alignment");}
            let up_ptr=if let Some((st,l))=up_storage.as_ref(){
                let Storage::Cuda(st)=&**st else{candle_core::bail!("DKR GLU CUDA storage");};
                let (p,g)=slice_ptr_on_stream(st.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
                if p%16!=0{candle_core::bail!("DKR GLU alignment");}
                Some((p,g))
            }else{None};
            let mut outputs=Vec::with_capacity(weights.len());
            for (index,(w,key)) in weights.iter().zip(keys.iter().copied()).enumerate() {
                let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
                let ni=key.n as i32;
                let mut out=unsafe{dev.alloc::<$t>(key.n*batch)?};
                {
                    let (yp,_yg)=slice_ptr_mut_on_stream(&mut out,0,&stream);
                    PLANS.with(|cache|cache.borrow_mut().with_plan(key,xs.device(),
                        ||prepare(key,xs.device()),|plans|{
                            if index==0 {
                                unsafe{plans.encode.with_prevalidated_launch(|fun,cfg|{
                                    let mut b=stream.launch_builder(fun);
                                    b.arg(&xp);
                                    if let Some((ptr,_))=&up_ptr{b.arg(ptr);}
                                    b.arg(&scratch).arg(&ki).arg(&pi).arg(&bi);
                                    let act=glu.map(|(_,a)|a).unwrap_or(0);
                                    if key.glu{b.arg(&act);}
                                    b.launch(cfg)?;Ok(())
                                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                            }
                            unsafe{plans.mmq.with_prevalidated_launch(|fun,cfg|{
                                let mut b=stream.launch_builder(fun);
                                b.arg(&wp).arg(&scratch).arg(&yp).arg(&ni).arg(&bi).arg(&bk);
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
    const FORMATS:[GgmlDType;10]=[GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,
        GgmlDType::Q8_0,GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,GgmlDType::Q5K,GgmlDType::Q6K];
    #[test]
    fn dkr_s9c_gguf_mmq_geometry() {
        for dt in FORMATS {
            for batch in [1,9,17,33,63,64,65,129,1024] {
                let (x,smem)=choose_tile(dt,batch,SHARED_LIMIT).unwrap();
                assert_eq!(smem,shared_bytes(dt,x));
                assert!(smem<=SHARED_LIMIT && x%8==0 && x<=64);
                for other in (8..=64).step_by(8) {
                    assert!(batch.div_ceil(x)<=batch.div_ceil(other));
                    if batch.div_ceil(x)==batch.div_ceil(other){assert!(x<=other);}
                }
            }
            assert!(choose_tile(dt,17,1024).is_none());
        }
    }
    #[test]
    #[ignore="requires staged GGUF MMQ SM61 catalogue and CUDA"]
    fn dkr_s9c_gguf_mmq_dense_gpu()->Result<()> {
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        for format in FORMATS {
            for n in [64usize,65] {
                let data=(0..n*256).map(|i|((i%17)as f32-8.)/8.).collect::<Vec<_>>();
                let weight=QTensor::quantize(&Tensor::from_vec(data,(n,256),&dev)?,format)?;
                let reference=weight.dequantize(&dev)?.t()?;
                for batch in [9usize,17,33,65] {
                    let values=(0..batch*256).map(|i|if i<256{0.}else{((i%9)as f32-4.)/8.}).collect::<Vec<_>>();
                    let x=Tensor::from_vec(values,(batch,256),&dev)?;
                    let expected=x.matmul(&reference)?.flatten_all()?.to_vec1::<f32>()?;
                    for dt in [DType::F16,DType::BF16,DType::F32] {
                        // One quantization for two projections. Both must match
                        // the independent dequantized dense reference.
                        let outputs=shared_lhs(&[&weight,&weight],&x.to_dtype(dt)?)?;
                        for out in outputs {
                            let actual=out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                            assert_eq!(actual.len(),expected.len());
                            for (&a,&e) in actual.iter().zip(&expected) {
                                assert!((a-e).abs()<=0.20+0.035*e.abs(),
                                        "{format:?} {dt:?} N={n} batch={batch}: {a} vs {e}");
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
