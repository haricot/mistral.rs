//! S6.8.3: separate processes for static and external providers, identical inputs.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{borrow::Cow, collections::BTreeMap, sync::{Arc, Barrier}};

const CODECS: [GgmlDType;10] = [GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,GgmlDType::Q8_0,GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,GgmlDType::Q5K,GgmlDType::Q6K];
const DTYPES: [DType;3] = [DType::F32,DType::F16,DType::BF16];
const SHAPES: &[(usize,usize,&[usize])] = &[
    (5,256,&[1,8,9]), (64,512,&[8,16]), (65,2048,&[9,17]),
    (129,4096,&[9,33]), (17,11008,&[9]),
];

#[derive(Serialize)]
struct Sample {
    index: usize,
    reference: f64,
    bound: f64,
}
#[derive(Serialize)]
struct Case {
    codec: String,
    dtype: String,
    n: usize,
    k: usize,
    batch: usize,
    offset: usize,
    rank: usize,
    weights_sha256: String,
    values: Vec<f32>,
    samples: Vec<Sample>,
    repeat_exact: bool,
}
#[derive(Serialize)]
struct Dequant {
    codec: String,
    n: usize,
    k: usize,
    weights_sha256: String,
    elements: usize,
    max_abs: f64,
    max_ratio: f64,
}
#[derive(Serialize)]
struct Report {
    schema: &'static str,
    mode: String,
    dequant: BTreeMap<String,Dequant>,
    cases: BTreeMap<String,Case>,
}

fn hash(bytes: &[u8]) -> String { format!("{:x}",Sha256::digest(bytes)) }
fn mix(i: usize, seed: usize) -> u32 {
    let mut x=(i as u32).wrapping_add((seed as u32).wrapping_mul(0x9e3779b9));
    x ^= x >> 16; x=x.wrapping_mul(0x7feb352d); x ^= x >> 15;
    x=x.wrapping_mul(0x846ca68b); x ^ (x >> 16)
}
fn weight(i: usize, seed: usize) -> f32 {
    // Vary scale by block and include exact zero blocks.
    let b=i/32;
    if b%23==0 {0.} else {
        ((mix(i,seed)%129) as f32-64.)/128. * (1+(b%4)) as f32/4.
    }
}
fn activation(i: usize, seed: usize) -> f32 {
    if i/32%31==0 {0.} else { ((mix(i,seed)%65) as f32-32.)/64. }
}
fn values(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}
struct Fixture {
    mat: QMatMul,
    cpu: Vec<f32>,
    sha: String,
    dequant: Dequant,
}
fn fixture(dev: &Device, codec:GgmlDType,n:usize,k:usize,seed:usize) -> Result<Fixture> {
    let w=Tensor::from_vec((0..n*k).map(|i|weight(i,seed)).collect::<Vec<_>>(),(n,k),&Device::Cpu)?;
    let qcpu=QTensor::quantize(&w,codec)?;
    let bytes=qcpu.data()?.into_owned();
    ensure!(bytes.len()==n*k/codec.block_size()*codec.type_size(),"packed weight extent");
    let sha=hash(&bytes);
    let cpu=values(&qcpu.dequantize(&Device::Cpu)?)?;
    let qgpu=QTensor::new(QStorage::from_data(Cow::Borrowed(&bytes),dev,codec)?,(n,k))?;
    ensure!(qgpu.data()?.as_ref()==bytes.as_slice(),"GPU weight byte roundtrip");
    let actual=values(&qgpu.dequantize(dev)?)?;
    ensure!(actual.len()==cpu.len() && cpu.len()==n*k,"dequant extent");
    let mut max_abs=0f64;let mut max_ratio=0f64;
    for (i,(&a,&b)) in actual.iter().zip(&cpu).enumerate() {
        let err=(a as f64-b as f64).abs();
        let bound=2e-5*(1.+(b as f64).abs());
        ensure!(a.is_finite() && b.is_finite() && err<=bound,
            "dequant {codec:?} N={n} K={k} index={i}: GPU {a} CPU {b}");
        max_abs=max_abs.max(err);max_ratio=max_ratio.max(err/bound);
    }
    let mat=QMatMul::from_qtensor(qgpu)?;
    ensure!(matches!(&mat,QMatMul::QTensor(_)),"unexpected dequantize-all");
    Ok(Fixture{mat,cpu,sha:sha.clone(),dequant:Dequant{
        codec:format!("{codec:?}"),n,k,weights_sha256:sha,elements:n*k,max_abs,max_ratio}})
}

fn indices(len:usize,limit:usize)->Vec<usize>{
    if len<=limit {return (0..len).collect()}
    (0..limit).map(|i| i*(len-1)/(limit-1)).collect()
}

fn run_case(dev:&Device,f:&Fixture,codec:GgmlDType,n:usize,k:usize,batch:usize,
            dtype:DType,seed:usize,offset:usize,rank3:bool)->Result<Case>{
    let all=(0..offset+batch*k).map(|i|activation(i,seed)).collect::<Vec<_>>();
    let x=Tensor::from_vec(all.clone(),all.len(),dev)?.to_dtype(dtype)?
        .narrow(0,offset,batch*k)?;
    let x=if rank3 {x.reshape((2,batch/2,k))?} else {x.reshape((batch,k))?};
    let a=f.mat.forward(&x)?;
    let b=f.mat.forward(&x)?;
    // All shapes and both timings run before host readback. Do not serialize
    // every individual GPU submission by a device-wide synchronize.
    let dims=if rank3 {vec![2,batch/2,n]}else{vec![batch,n]};
    ensure!(a.dims()==dims.as_slice() && b.dims()==dims.as_slice() && a.dtype()==dtype && b.dtype()==dtype,"output layout or dtype changed");
    let a=values(&a)?;let b=values(&b)?;
    ensure!(a.len()==batch*n && a.iter().all(|v|v.is_finite()) && a==b,
        "repeat mismatch {codec:?}/{dtype:?} N={n} K={k} batch={batch}");
    let mut samples=Vec::new();
    for row in indices(batch,7) {
        let xr=&all[offset+row*k..offset+(row+1)*k];
        for col in indices(n,17){
            let wr=&f.cpu[col*k..(col+1)*k];
            let mut dot=0f64;let mut quant_bound=0f64;let mut abs_products=0f64;
            for (xb,wb) in xr.chunks_exact(32).zip(wr.chunks_exact(32)){
                let scale=xb.iter().fold(0f32,|a,b|a.max(b.abs())) as f64/127.;
                let l1=wb.iter().map(|w|(*w as f64).abs()).sum::<f64>();
                // Half-step activation quantization plus FP16 scale rounding
                // and sum compensation, conservative for MMVQ and MMQ layouts.
                quant_bound+=l1*(0.5*scale+0.001*scale*127.+0.00002);
                for (&x,&w) in xb.iter().zip(wb){
                    let term=x as f64*w as f64;dot+=term;abs_products+=term.abs();
                }
            }
            let rounding=match dtype{DType::BF16=>0.004,DType::F16=>0.0005,_=>0.};
            let bound=quant_bound+0.01+0.00005*abs_products+rounding*(dot.abs()+quant_bound+1.);
            let index=row*n+col;let err=(a[index] as f64-dot).abs();
            ensure!(dot.is_finite() && bound.is_finite() && err<=bound,
                "dot {codec:?}/{dtype:?} N={n} K={k} batch={batch} row={row} col={col}: got={} cpu={dot} bound={bound}",a[index]);
            samples.push(Sample{index,reference:dot,bound});
        }
    }
    Ok(Case{codec:format!("{codec:?}"),dtype:format!("{dtype:?}"),n,k,batch,offset,
        rank:if rank3{3}else{2},weights_sha256:f.sha.clone(),values:a,samples,repeat_exact:true})
}

// Each worker has independent inputs and repeatedly changes workspace size.
// The shared-device case uses exactly the same Candle DeviceId and QTensor,
// while explicit_stream creates separate handles/streams on the same GPU.
// Neither case claims distinct CUDA driver contexts (primary context may be shared).
fn concurrent(report:&mut Report,shared:bool)->Result<()>{
    let n=65;let k=2048;let codec=GgmlDType::Q4K;
    let common=Device::new_cuda(0)?;
    let mut jobs=Vec::new();
    let label=if shared{"shared_device"}else{"explicit_stream"};
    if shared {
        let f=Arc::new(fixture(&common,codec,n,k,411)?);
        common.synchronize()?;
        for worker in 0..4 {
            jobs.push((worker,common.clone(),f.clone()));
        }
    } else {
        for worker in 0..4{
            let dev=Device::new_cuda_with_stream(0)?;
            let f=Arc::new(fixture(&dev,codec,n,k,411+worker)?);
            dev.synchronize()?;
            jobs.push((worker,dev,f));
        }
    }
    let barrier=Arc::new(Barrier::new(jobs.len()));
    let mut handles=Vec::new();
    for (worker,dev,f) in jobs {
        let barrier=barrier.clone();
        handles.push(std::thread::spawn(move ||->Result<Vec<(String,Case)>>{
            barrier.wait();
            let mut cases=Vec::new();
            for (round,batch) in [9usize,33,17,9,33,17].into_iter().enumerate(){
                let dtype=DTYPES[(worker+round)%3];
                let c=run_case(&dev,&f,codec,n,k,batch,dtype,701+worker*19+round,1,false)?;
                cases.push((format!("concurrent/{label}/{worker}/{round}"),c));
            }
            dev.synchronize()?;
            Ok(cases)
        }));
    }
    // Join every worker even if one fails, so evidence is not emitted early.
    let joined=handles.into_iter().map(|h|h.join()).collect::<Vec<_>>();
    for result in joined{
        let cases=result.map_err(|_|anyhow::anyhow!("concurrent worker panicked"))??;
        for (key,c) in cases {ensure!(report.cases.insert(key,c).is_none(),"duplicate case");}
    }
    Ok(())
}

#[test]
#[ignore="requires local SM61 and explicit S683 output"]
fn dkr_s683_extended_gpu()->Result<()>{
    let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
    let dest=std::env::var("DKR_S683_OUTPUT").context("set DKR_S683_OUTPUT")?;
    let mode=std::env::var("DKR_S683_MODE")?;
    ensure!(mode=="legacy"||mode=="external","invalid mode");
    ensure!(std::env::var_os("CANDLE_DKR_S31_DIR").is_some()==(mode=="external"),"provider mode mismatch");
    let dev=Device::new_cuda(0)?;
    let mut report=Report{schema:"DKR-S683-V1",mode,dequant:BTreeMap::new(),cases:BTreeMap::new()};
    for codec in CODECS {
        for (shape_idx,&(n,k,batches)) in SHAPES.iter().enumerate(){
            let f=fixture(&dev,codec,n,k,101+shape_idx)?;
            let prefix=format!("{codec:?}/{n}x{k}");
            for &batch in batches {
                for dtype in DTYPES{
                    let offset=if batch<=8{4}else{1};
                    let c=run_case(&dev,&f,codec,n,k,batch,dtype,201+batch,offset,batch%2==0)?;
                    ensure!(report.cases.insert(format!("{prefix}/{batch}/{dtype:?}"),c).is_none(),"duplicate case");
                }
            }
            report.dequant.insert(prefix,f.dequant);
        }
    }
    // One full-width projection (sampled CPU dots, full A/B result vector).
    let codec=GgmlDType::Q4K;let n=2048;let k=2048;
    let f=fixture(&dev,codec,n,k,177)?;
    let c=run_case(&dev,&f,codec,n,k,9,DType::F32,777,1,false)?;
    report.cases.insert("model_width/Q4K/2048x2048/9/F32".into(),c);
    report.dequant.insert("model_width/Q4K/2048x2048".into(),f.dequant);
    dev.synchronize()?;
    concurrent(&mut report,true)?;
    concurrent(&mut report,false)?;
    ensure!(report.dequant.len()==51 && report.cases.len()==349,"matrix inventory");
    std::fs::write(dest,serde_json::to_vec(&report)?)?;
    Ok(())
}
