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

pub(super) fn plain(w:&QTensor, xs:&Tensor) -> Result<Tensor> {
    let fmt=format(w.dtype()).ok_or_else(|| candle_core::Error::msg("DKR GGUF format not admitted"))?;
    let Device::Cuda(dev)=w.device() else { candle_core::bail!("DKR MMVQ requires CUDA weights"); };
    if !xs.device().same_device(&w.device()) {
        candle_core::bail!("DKR MMVQ requires same-device activations");
    }
    let (n,k)=w.shape().dims2()?;
    let Some((&input_k,dims))=xs.dims().split_last() else { candle_core::bail!("DKR MMVQ input rank"); };
    let batch=dims.iter().try_fold(1usize,|a,&b|a.checked_mul(b))
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ batch overflow"))?;
    let block_elements=w.dtype().block_size();
    if input_k!=k || k==0 || n==0 || k%block_elements!=0
        || !(1..=MMVQ_MAX_BATCH).contains(&batch) {
        candle_core::bail!("DKR MMVQ shape not admitted");
    }
    dtype(xs.dtype())?;
    let weight_row=(k/block_elements).checked_mul(w.dtype().type_size())
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ weight row overflow"))?;
    let weight_bytes=weight_row.checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ weight span overflow"))?;
    let padded=k.checked_add(511).ok_or_else(|| candle_core::Error::msg("DKR MMVQ padding overflow"))? / 512 * 512;
    let scratch_row=(padded/32).checked_mul(36)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ scratch row overflow"))?;
    let scratch_bytes=scratch_row.checked_mul(batch)
        .ok_or_else(|| candle_core::Error::msg("DKR MMVQ scratch overflow"))?;
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
    let stream=dev.cuda_stream();
    let xs=xs.contiguous()?;
    let (storage,layout)=xs.storage_and_layout();
    let Storage::Cuda(storage)=&*storage else { candle_core::bail!("DKR MMVQ requires CUDA activations"); };
    let (weight_ptr,_wg)=w.device_ptr_with_guard(&stream)?;
    let mut workspace=workspace_ensure(&dev,scratch_bytes,&stream)?;
    let (scratch_ptr,_sg)=workspace.ptr_mut();
    let ki=k as i32;
    let ni=n as i32;
    let pi=padded as i32;
    let stride=(padded/32) as i32;
    PLANS.with(|plans|plans.borrow_mut().with_plan(key,xs.device(),
        ||prepare(key,xs.device()),|plans|{
            macro_rules! launch {
                ($t:ty)=>{{
                    let (xp,_xg)=slice_ptr_on_stream(storage.as_cuda_slice::<$t>()?,layout.start_offset(),&stream);
                    let mut out=unsafe{dev.alloc::<$t>(count)?};
                    {
                        let (yp,_yg)=slice_ptr_mut_on_stream(&mut out,0,&stream);
                        unsafe{plans.encode.with_prevalidated_launch(|fun,cfg|{
                            let mut b=stream.launch_builder(fun);
                            b.arg(&xp).arg(&scratch_ptr).arg(&ki).arg(&pi);
                            b.launch(cfg)?;
                            Ok(())
                        })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                        unsafe{plans.matvec.with_prevalidated_launch(|fun,cfg|{
                            let mut b=stream.launch_builder(fun);
                            b.arg(&weight_ptr).arg(&scratch_ptr).arg(&yp).arg(&ki);
                            if !prefetch {b.arg(&ni).arg(&stride).arg(&ni);}
                            b.launch(cfg)?;
                            Ok(())
                        })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                    }
                    Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),
                                    output_shape(&xs,n))))
                }}
            }
            match key.dtype {DType::F16=>launch!(half::f16),DType::BF16=>launch!(half::bf16),
                             DType::F32=>launch!(f32),_=>unreachable!()}
        }))
}

#[cfg(test)]
mod tests {
    use super::*;
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
