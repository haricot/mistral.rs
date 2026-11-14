//! Local A/B: same compiled binary, separate nextest processes.
use candle_core::{DType, Device, Module, Tensor};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use std::collections::BTreeMap;

fn save(out: &mut BTreeMap<String,Vec<f32>>, name:&str, t:Tensor)->anyhow::Result<()> {
    let values=t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    anyhow::ensure!(values.iter().all(|v|v.is_finite()),"nonfinite {name}");
    anyhow::ensure!(out.insert(name.into(),values).is_none(),"duplicate {name}");
    Ok(())
}

#[test]
#[ignore = "requires local SM61 and explicit S6 evidence output"]
fn dkr_s6_native_ab_gpu() -> anyhow::Result<()> {
    let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
    let dest=std::env::var("DKR_S6_NATIVE_OUTPUT")?;
    let dev=Device::new_cuda(0)?;
    let mut out=BTreeMap::new();
    for dtype in [DType::F32,DType::F16,DType::BF16] {
        let x=Tensor::new(&[[1f32,-2.,3.,0.5],[-1.,4.,2.,-3.]],&dev)?.to_dtype(dtype)?;
        let name=format!("{dtype:?}");
        save(&mut out,&format!("{name}/affine"),x.affine(0.5,1.)?)?;
        save(&mut out,&format!("{name}/binary"),x.broadcast_add(&x.narrow(0,0,1)?)?)?;
        save(&mut out,&format!("{name}/unary"),x.sqr()?)?;
        save(&mut out,&format!("{name}/reduce"),x.sum(1)?)?;
        save(&mut out,&format!("{name}/cast"),x.to_dtype(DType::F32)?)?;
        save(&mut out,&format!("{name}/fill"),Tensor::ones((2,4),dtype,&dev)?)?;
        let ids=Tensor::new(&[1u32,0],&dev)?;
        save(&mut out,&format!("{name}/indexing"),x.index_select(&ids,0)?)?;
        let cond=x.to_dtype(DType::F32)?.gt(0f32)?;
        save(&mut out,&format!("{name}/ternary"),cond.where_cond(&x,&x.neg()?)?)?;
        save(&mut out,&format!("{name}/sort"),x.arg_sort_last_dim(true)?)?;
        save(&mut out,&format!("{name}/softmax"),candle_nn::ops::softmax(&x,1)?)?;
    }
    let img=Tensor::from_vec(vec![1f32,2.,3.,4.,5.,6.,7.,8.,9.],(1,1,3,3),&dev)?;
    let filt=Tensor::ones((1,1,2,2),DType::F32,&dev)?;
    save(&mut out,"conv2d",img.conv2d(&filt,0,1,1,1)?)?;
    let weights=Tensor::from_vec((0..5*256).map(|i|((i%17) as f32-8.)/16.).collect::<Vec<_>>(),(5,256),&Device::Cpu)?;
    for codec in [GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,GgmlDType::Q8_0,GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,GgmlDType::Q5K,GgmlDType::Q6K] {
        let q=QTensor::quantize_onto(&weights,codec,&dev)?;
        save(&mut out,&format!("{codec:?}/dequant"),q.dequantize(&dev)?)?;
        let mat=QMatMul::from_qtensor(q)?;
        for batch in [1usize,8,9] {
            let x=Tensor::from_vec((0..batch*256).map(|i|((i%11) as f32-5.)/16.).collect::<Vec<_>>(),(batch,256),&dev)?;
            for repeat in 0..2 {
                save(&mut out,&format!("{codec:?}/mat/{batch}/{repeat}"),mat.forward(&x)?)?;
            }
        }
    }
    let ids=Tensor::new(&[0u32,1],&dev)?;
    let experts=Tensor::new(&[0u32,0],&dev)?;
    let x=Tensor::ones((2,256),DType::F32,&dev)?;
    let w=weights.reshape((1,5,256))?;
    let wf=w.to_device(&dev)?.to_dtype(DType::F16)?;
    let xf=x.to_dtype(DType::F16)?;
    for prefill in [false,true] {
        save(&mut out,&format!("moe_simt/{prefill}"),candle_nn::moe::moe_gemm(&xf,&wf,&None,&ids,&experts,1,prefill)?)?;
        let q=QTensor::quantize_onto(&w,GgmlDType::Q8_0,&dev)?;
        save(&mut out,&format!("moe_gguf/{prefill}"),candle_nn::moe::moe_gemm_gguf(&x,&q,&None,&ids,&experts,1,prefill,DType::F32)?)?;
    }
    dev.synchronize()?;
    std::fs::write(dest,serde_json::to_vec(&out)?)?;
    Ok(())
}
