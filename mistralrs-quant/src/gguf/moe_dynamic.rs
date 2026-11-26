//! SM61 compact indexed MMQ. The only metadata producer is GPU dispatch.
//! Invalid expert IDs produce NaN rows, never an out-of-range weight access.
use super::{dynamic::{format,layout,shared_bytes,tensor,scalar,prepare_one,mul,i32_span},
    qk_for,q8_1,workspace_ensure,MMQ_WORKSPACE};
use crate::{cuda_device::cuda_device_info,dkr_cuda::{dtype,LaunchPlanCache},
    utils::{slice_ptr_on_stream,slice_ptr_mut_on_stream}};
use candle_core::{cuda::cudarc::driver::{CudaSlice,PushKernelArg},
    quantized::QTensor,CudaDevice,CudaStorage,DType,Device,Result,Storage,Tensor};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType},
    universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;

fn u(n:usize)->Argument {tensor(ElementType::U32,vec![n],None)}
fn tiles_arg(cap:usize)->Argument {
    tensor(ElementType::U32,vec![cap,2],Some("gguf.moe.tiles.v1"))
}
fn launch_plan(device:&Device,op:&str,args:Vec<Argument>,grid:[u32;3],block:[u32;3])->Result<PrevalidatedLaunchPlan>{
    prepare_one(device,format!("quant.gguf.moe.{op}.v1"),args,grid,block,0)
}
fn device_gate(dev:&CudaDevice)->Result<()> {
    let info=cuda_device_info(dev);
    if info.capability()!=(6,1) || info.warp_size!=32 {
        candle_core::bail!("DKR compact MMQ requires SM61");
    }
    Ok(())
}
fn capacity(routes:usize,experts:usize)->Result<usize>{
    let cap=(routes/8).checked_add(experts).ok_or_else(||candle_core::Error::msg("worklist overflow"))?.min(routes);
    if cap==0 || cap>65535 {candle_core::bail!("DKR compact worklist exceeds CUDA grid");}
    Ok(cap)
}
#[derive(Clone,Copy,PartialEq,Eq)]
struct DispatchKey {routes:usize,experts:usize}
struct DispatchPlans {init:PrevalidatedLaunchPlan,count:PrevalidatedLaunchPlan,prefix:PrevalidatedLaunchPlan,scatter:PrevalidatedLaunchPlan}
thread_local!{
    static DISPATCH:RefCell<LaunchPlanCache<DispatchKey,DispatchPlans>>=RefCell::new(LaunchPlanCache::new());
}
fn dispatch_plans(key:DispatchKey,dev:&Device)->Result<DispatchPlans>{
    let (a,e)=(key.routes,key.experts);
    let block=[256,1,1];
    let grid=[a.div_ceil(256) as u32,1,1];
    Ok(DispatchPlans{
        init:launch_plan(dev,"init",vec![u(e),u(a),u(a),scalar(),scalar()], [a.max(e).div_ceil(256) as u32,1,1],block)?,
        count:launch_plan(dev,"count",vec![u(a),u(e),scalar(),scalar()],grid,block)?,
        prefix:launch_plan(dev,"prefix",vec![u(e),u(e+1),u(e),scalar()],[1,1,1],[1,1,1])?,
        scatter:launch_plan(dev,"scatter",vec![u(a),u(e),u(a),u(a),scalar(),scalar(),scalar(),scalar()],grid,block)?,
    })
}
pub(crate) fn dispatch(ids:&CudaSlice<u32>,offset:usize,a:usize,e:usize,topk:usize,input_routes:usize,dev:&CudaDevice)
 ->Result<(CudaSlice<u32>,CudaSlice<u32>,CudaSlice<u32>)>{
    device_gate(dev)?;
    if a==0 || e==0 || topk==0 || a%topk!=0 || !(input_routes==1||input_routes==topk)
        || offset.checked_add(a).is_none_or(|end|end>ids.len()) {
        candle_core::bail!("DKR indexed dispatch shape mismatch");
    }
    i32_span(a)?;i32_span(e)?;i32_span(e+1)?;i32_span(topk)?;
    let device=Device::Cuda(dev.clone());
    let stream=dev.cuda_stream();
    let mut counts=unsafe{dev.alloc::<u32>(e)?};
    let mut cursors=unsafe{dev.alloc::<u32>(e)?};
    let mut bounds=unsafe{dev.alloc::<u32>(e+1)?};
    let mut src=unsafe{dev.alloc::<u32>(a)?};
    let mut dst=unsafe{dev.alloc::<u32>(a)?};
    {
        let (ip,_ig)=slice_ptr_on_stream(ids,offset,&stream);
        let (cp,_cg)=slice_ptr_mut_on_stream(&mut counts,0,&stream);
        let (rp,_rg)=slice_ptr_mut_on_stream(&mut cursors,0,&stream);
        let (bp,_bg)=slice_ptr_mut_on_stream(&mut bounds,0,&stream);
        let (sp,_sg)=slice_ptr_mut_on_stream(&mut src,0,&stream);
        let (dp,_dg)=slice_ptr_mut_on_stream(&mut dst,0,&stream);
        let (ai,ei,ti,ri)=(a as i32,e as i32,topk as i32,input_routes as i32);
        DISPATCH.with(|cache|cache.borrow_mut().with_plan(DispatchKey{routes:a,experts:e},&device,
            ||dispatch_plans(DispatchKey{routes:a,experts:e},&device),|p|{
                macro_rules! run {($plan:expr,$($arg:expr),+)=>{unsafe{$plan.with_prevalidated_launch(|fun,cfg|{
                    let mut b=stream.launch_builder(fun);$(b.arg(&$arg);)+ b.launch(cfg)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?}}
                run!(p.init,cp,sp,dp,ei,ai);
                run!(p.count,ip,cp,ei,ai);
                run!(p.prefix,cp,bp,rp,ei);
                run!(p.scatter,ip,rp,sp,dp,ei,ai,ti,ri);
                Ok(())
            }))?;
    }
    Ok((bounds,dst,src))
}

#[derive(Clone,Copy,PartialEq,Eq)]
struct Key {
    format:&'static str,layout:&'static str,dtype:DType,n:usize,k:usize,blocks:usize,
    packed:usize,src_rows:usize,routes:usize,experts:usize,padded:usize,cap:usize,shared:usize,glu:bool,
}
struct Plans{work:PrevalidatedLaunchPlan,encode:PrevalidatedLaunchPlan,fill:PrevalidatedLaunchPlan,mmq:PrevalidatedLaunchPlan}
thread_local!{static PLANS:RefCell<LaunchPlanCache<Key,Plans>>=RefCell::new(LaunchPlanCache::new());}
fn prepare(key:Key,device:&Device)->Result<Plans>{
    let (a,e,n,cap)=(key.routes,key.experts,key.n,key.cap);
    let repr=format!("gguf.q8_1.mmq.{}.v1",key.layout);
    let q8=||tensor(ElementType::U8,vec![key.padded/128,a*144],Some(&repr));
    let output=||tensor(ElementType::F32,vec![a,n],None);
    let work=launch_plan(device,"worklist",vec![u(e+1),tiles_arg(cap),u(1),scalar(),scalar(),scalar()],[1,1,1],[256,1,1])?;
    let input=||Ok::<Argument,candle_core::Error>(tensor(dtype(key.dtype)?,vec![key.src_rows,key.k],None));
    let (op,args)=if key.glu{
        (format!("glu.encode.{}",key.layout),vec![input()?,input()?,u(a),q8(),scalar(),scalar(),scalar(),scalar(),scalar(),scalar()])
    }else{(format!("encode.{}",key.layout),vec![input()?,u(a),q8(),scalar(),scalar(),scalar(),scalar()])};
    let encode=launch_plan(device,&op,args,[a as u32,(key.padded/512) as u32,1],[128,1,1])?;
    let fill=launch_plan(device,"fill",vec![output(),scalar()],[(a*n).div_ceil(256) as u32,1,1],[256,1,1])?;
    let weight_repr=format!("gguf.{}.packed.v1",key.format);
    let bounds=if n%64==0{"full"}else{"tail"};
    let mmq=prepare_one(device,format!("quant.gguf.moe.compact.{}.{bounds}.v1",key.format),
        vec![tensor(ElementType::U8,vec![e,n,key.packed],Some(&weight_repr)),q8(),u(a),u(e+1),tiles_arg(cap),u(1),
             output(),scalar(),scalar(),scalar(),scalar(),scalar()],
        [n.div_ceil(64) as u32,cap as u32,1],[32,8,1],key.shared as u32)?;
    Ok(Plans{work,encode,fill,mmq})
}

// For grouped callers, the metadata must originate from the typed dispatch
// producer; length checks alone do not establish uniqueness of destinations.
pub(crate) fn grouped(weights:&[&QTensor],xs:&Tensor,src:&CudaSlice<u32>,dst:&CudaSlice<u32>,
    bounds:&CudaSlice<u32>,a:usize,e:usize,dev:&CudaDevice)->Result<Vec<Tensor>>{
    grouped_run(weights,xs,Some(src),dst,bounds,a,e,dev,None)
}
pub(crate) fn grouped_glu(w:&QTensor,gate:&Tensor,up:&Tensor,src:Option<&CudaSlice<u32>>,dst:&CudaSlice<u32>,
    bounds:&CudaSlice<u32>,a:usize,e:usize,dev:&CudaDevice,activation:i32)->Result<Tensor>{
    if gate.dims2()?.0!=a || up.shape()!=gate.shape() || up.dtype()!=gate.dtype()
        || !up.device().same_device(gate.device()) || !(0..=4).contains(&activation){
        candle_core::bail!("DKR grouped GLU contract");
    }
    Ok(grouped_run(&[w],gate,src,dst,bounds,a,e,dev,Some((up,activation)))?.pop().unwrap())
}
fn grouped_run(weights:&[&QTensor],xs:&Tensor,src:Option<&CudaSlice<u32>>,dst:&CudaSlice<u32>,
    bounds:&CudaSlice<u32>,a:usize,e:usize,dev:&CudaDevice,glu:Option<(&Tensor,i32)>)->Result<Vec<Tensor>>{
    device_gate(dev)?;
    let first=weights.first().ok_or_else(||candle_core::Error::msg("DKR grouped MMQ requires weights"))?;
    let quant=first.dtype();
    let fmt=format(quant).ok_or_else(||candle_core::Error::msg("DKR compact MMQ weight format unsupported"))?;
    let (source_rows,k)=xs.dims2()?;
    let device=Device::Cuda(dev.clone());
    if k==0 || k%256!=0 || source_rows==0 || e==0 || a==0 || src.is_some_and(|s|s.len()<a) || dst.len()<a
        || bounds.len()<e.checked_add(1).ok_or_else(||candle_core::Error::msg("expert overflow"))?
        || !xs.device().same_device(&device) {
        candle_core::bail!("DKR compact MMQ shape/device/metadata span");
    }
    dtype(xs.dtype())?;
    let cap=capacity(a,e)?;
    let blocks=k/qk_for(quant);
    let packed=mul(blocks,quant.type_size())?;
    let padded=k.checked_add(511).ok_or_else(||candle_core::Error::msg("padding overflow"))?/512*512;
    let main=i32_span(mul(mul(padded/128,a)?,144)?)?;
    i32_span(mul(source_rows,k)?)?;i32_span(e)?;i32_span(a)?;i32_span(padded)?;
    if padded/512>65535 {candle_core::bail!("DKR compact encoder grid exceeds limit");}
    let shared=shared_bytes(quant,64);
    if shared>48*1024 || shared>cuda_device_info(dev).smem_optin {candle_core::bail!("DKR compact MMQ shared memory");}
    let mut keys=Vec::with_capacity(weights.len());
    for w in weights{
        let (we,n,wk)=w.shape().dims3()?;
        if we!=e || n==0 || wk!=k || w.dtype()!=quant || !w.device().same_device(&device){
            candle_core::bail!("DKR compact MMQ projection mismatch");
        }
        i32_span(mul(mul(e,n)?,packed)?)?;
        i32_span(mul(a,n)?)?;
        i32_span(n.checked_add(64).ok_or_else(||candle_core::Error::msg("row overflow"))?)?;
        keys.push(Key{format:fmt,layout:layout(quant),dtype:xs.dtype(),n,k,blocks,packed,src_rows:source_rows,
            routes:a,experts:e,padded,cap,shared,glu:glu.is_some()});
    }
    let xs=xs.contiguous()?;
    let (st,l)=xs.storage_and_layout();
    let Storage::Cuda(st)=&*st else{candle_core::bail!("DKR compact CUDA activation required");};
    let up=glu.map(|(u,_)|u.contiguous()).transpose()?;
    let up_storage=up.as_ref().map(|u|u.storage_and_layout());
    let stream=dev.cuda_stream();
    let mut workspace=workspace_ensure(&MMQ_WORKSPACE,dev,main,&stream)?;
    let (scratch,_sg)=workspace.ptr_mut();
    let mut tiles=unsafe{dev.alloc::<u32>(cap*2)?};
    let mut count=unsafe{dev.alloc::<u32>(1)?};
    let (tp,_tg)=slice_ptr_mut_on_stream(&mut tiles,0,&stream);
    let (cp,_cg)=slice_ptr_mut_on_stream(&mut count,0,&stream);
    let (sp,_srcg)=slice_ptr_on_stream(src.unwrap_or(dst),0,&stream);
    let use_src=i32::from(src.is_some());let act=glu.map(|(_,a)|a).unwrap_or(0);
    let (dp,_dstg)=slice_ptr_on_stream(dst,0,&stream);
    let (bp,_bg)=slice_ptr_on_stream(bounds,0,&stream);
    let (ki,pi,ai,ei,ci,si,bki)=(k as i32,padded as i32,a as i32,e as i32,cap as i32,source_rows as i32,blocks as i32);
    macro_rules! launch {($t:ty)=>{{
        let (xp,_xg)=slice_ptr_on_stream(st.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
        if xp%16!=0 {candle_core::bail!("DKR compact activation alignment");}
        let up_ptr=if let Some((s,l))=up_storage.as_ref(){
            let Storage::Cuda(s)=&**s else{candle_core::bail!("DKR compact GLU CUDA");};
            let (p,g)=slice_ptr_on_stream(s.as_cuda_slice::<$t>()?,l.start_offset(),&stream);
            if p%16!=0{candle_core::bail!("DKR compact GLU alignment");}
            Some((p,g))
        }else{None};
        let mut results=Vec::with_capacity(weights.len());
        for (index,(w,key)) in weights.iter().zip(keys.iter().copied()).enumerate(){
            let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
            let ni=key.n as i32;
            let elements=(a*key.n) as i32;
            let mut out=unsafe{dev.alloc::<f32>(elements as usize)?};
            {
                let (yp,_yg)=slice_ptr_mut_on_stream(&mut out,0,&stream);
                PLANS.with(|cache|cache.borrow_mut().with_plan(key,&device,||prepare(key,&device),|p|{
                    macro_rules! run {($plan:expr,$($arg:expr),+)=>{unsafe{$plan.with_prevalidated_launch(|fun,cfg|{
                        let mut b=stream.launch_builder(fun);$(b.arg(&$arg);)+ b.launch(cfg)?;Ok(())
                    })}.map_err(|e|candle_core::Error::msg(e.to_string()))?}}
                    if index==0 {
                        run!(p.work,bp,tp,cp,ei,ai,ci);
                        if let Some((p,_))=&up_ptr{
                            let up=*p;
                            run!(p.encode,xp,up,sp,scratch,ki,pi,ai,si,act,use_src);
                        }else{run!(p.encode,xp,sp,scratch,ki,pi,ai,si);}
                    }
                    run!(p.fill,yp,elements);
                    run!(p.mmq,wp,scratch,dp,bp,tp,cp,yp,ni,ai,bki,ei,ci);
                    Ok(())
                }))?;
            }
            results.push(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),(a,key.n))));
        }
        Ok(results)
    }}}
    match xs.dtype(){DType::F16=>launch!(half::f16),DType::BF16=>launch!(half::bf16),DType::F32=>launch!(f32),_=>unreachable!()}
}

pub(crate) fn indexed(w:&QTensor,x:&Tensor,ids:&Tensor)->Result<Tensor>{
    let (e,n,k)=w.shape().dims3()?;
    let (batch,input_routes,xk)=x.dims3()?;
    let (ib,topk)=ids.dims2()?;
    if batch==0 || ib!=batch || k!=xk || topk==0 || !(input_routes==1||input_routes==topk)
        || ids.dtype()!=DType::U32 || !ids.device().same_device(x.device()) || !w.device().same_device(x.device()){
        candle_core::bail!("DKR GGUF indexed MoE shape/dtype/device mismatch");
    }
    // Avoid histogram/scatter/worklist overhead for small decode batches.
    // Q8_1 weights and non-256-aligned K have no compact MMQ profile.
    if batch<=super::super::fast_mmvq::MMVQ_MAX_BATCH || k%256!=0
        || w.dtype()==candle_core::quantized::GgmlDType::Q8_1{
        return super::super::fast_mmvq::typed_indexed(w,x,ids);
    }
    let Device::Cuda(dev)=w.device() else{candle_core::bail!("DKR indexed MoE CUDA required");};
    let a=mul(batch,topk)?;
    capacity(a,e)?;
    let ids=ids.contiguous()?;
    let (st,l)=ids.storage_and_layout();
    let Storage::Cuda(st)=&*st else{candle_core::bail!("DKR indexed MoE CUDA indices required");};
    let (bounds,dst,src)=dispatch(st.as_cuda_slice::<u32>()?,l.start_offset(),a,e,topk,input_routes,&dev)?;
    let flat=x.contiguous()?.reshape((mul(batch,input_routes)?,k))?;
    let mut out=grouped(&[w],&flat,&src,&dst,&bounds,a,e,&dev)?;
    out.pop().expect("one projection").reshape((batch,topk,n))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_s9c_compact_capacity(){
        // Exhaustive small distributions, including empty experts and skew.
        for a in 1..=129usize{
            for e in 1..=32usize{
                for first in 0..=a{
                    let counts=[first,a-first];
                    let count:usize=counts.iter().map(|&c|{
                        let mut x=8;while x<64&&x<c{x*=2;}c.div_ceil(x)
                    }).sum();
                    if e>=2{assert!(count<=capacity(a,e).unwrap());}
                }
            }
        }
    }
    #[test]
    #[ignore="requires SM61 compact catalogue and CUDA"]
    fn dkr_s9c_compact_invalid_ids_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let weight=QTensor::quantize(&Tensor::ones((2,64,256),DType::F32,&dev)?,candle_core::quantized::GgmlDType::Q8_0)?;
        let x=Tensor::ones((2,1,256),DType::F32,&dev)?;
        let ids=Tensor::from_vec(vec![0u32,99,1,u32::MAX],(2,2),&dev)?;
        let actual=indexed(&weight,&x,&ids)?.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..4 {
            for &v in &actual[i*64..(i+1)*64] {
                if i%2==0 {assert!(v.is_finite() && (v-256.).abs()<2.);}
                else {assert!(v.is_nan());}
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires SM61 compact catalogue and CUDA"]
    fn dkr_s9c_compact_indexed_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        use candle_core::quantized::GgmlDType;
        for q in [GgmlDType::Q4_0,GgmlDType::Q4_1,GgmlDType::Q5_0,GgmlDType::Q5_1,GgmlDType::Q8_0,
            GgmlDType::Q2K,GgmlDType::Q3K,GgmlDType::Q4K,GgmlDType::Q5K,GgmlDType::Q6K]{
            for n in [64usize,65]{
                let w=Tensor::from_vec((0..4*n*256).map(|i|((i%17)as f32-8.)/8.).collect::<Vec<_>>(),(4,n,256),&dev)?;
                let w=QTensor::quantize(&w,q)?;
                let dense=w.dequantize(&dev)?;
                for rows in [1usize,17,65]{
                    for input_routes in [1usize,2]{
                        let ids:Vec<u32>=(0..rows*2).map(|i|if i%11==0{3}else{(i%2)as u32}).collect();
                        let indices=Tensor::from_vec(ids.clone(),(rows,2),&dev)?;
                        let x=Tensor::from_vec((0..rows*input_routes*256).map(|i|((i%9)as f32-4.)/8.).collect::<Vec<_>>(),(rows,input_routes,256),&dev)?;
                        let mut expected=Vec::new();
                        for (i,&e) in ids.iter().enumerate(){
                            let row=x.narrow(0,i/2,1)?.narrow(1,if input_routes==1{0}else{i%2},1)?.reshape((1,256))?;
                            let weight=dense.narrow(0,e as usize,1)?.reshape((n,256))?;
                            expected.extend(row.matmul(&weight.t()?)?.flatten_all()?.to_vec1::<f32>()?);
                        }
                        for dt in [DType::F16,DType::BF16,DType::F32]{
                            let actual=indexed(&w,&x.to_dtype(dt)?,&indices)?.flatten_all()?.to_vec1::<f32>()?;
                            assert_eq!(actual.len(),expected.len());
                            for (&a,&b) in actual.iter().zip(&expected){
                                assert!((a-b).abs()<=0.20+0.035*b.abs(),"{q:?} {dt:?} {a} vs {b}");
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// Explicit no-LoRA, no-bias prefill pipeline. All route metadata remains
// on device. The same dispatch is used for gate/up and down; each stage
// independently builds its bounded worklist.
pub(crate) fn fused_pipeline(gate:&QTensor,up:&QTensor,down:&QTensor,x:&Tensor,ids:&Tensor,
    routing:&Tensor,activation:i32,out_dtype:DType)->Result<Tensor>{
    let (batch,k)=x.dims2()?;let (ib,topk)=ids.dims2()?;
    let (e,inter,gk)=gate.shape().dims3()?;let (de,n,dk)=down.shape().dims3()?;
    if ib!=batch||batch==0||topk==0||e==0||gk!=k||dk!=inter||de!=e||n!=k
        || up.shape()!=gate.shape()||gate.dtype()!=up.dtype()||ids.dtype()!=DType::U32
        || routing.shape()!=ids.shape()||!routing.device().same_device(x.device())
        || !ids.device().same_device(x.device()){
        candle_core::bail!("DKR fused MoE pipeline contract");
    }
    let dev=x.device().as_cuda_device()?;
    let a=mul(batch,topk)?;capacity(a,e)?;
    let ids=ids.contiguous()?;let (st,l)=ids.storage_and_layout();
    let Storage::Cuda(st)=&*st else{candle_core::bail!("DKR fused MoE ids CUDA");};
    let (bounds,dst,src)=dispatch(st.as_cuda_slice::<u32>()?,l.start_offset(),a,e,topk,1,dev)?;
    let pair=grouped(&[gate,up],x,&src,&dst,&bounds,a,e,dev)?;
    let out=grouped_glu(down,&pair[0],&pair[1],Some(&dst),&dst,&bounds,a,e,dev,activation)?;
    // Explicit tensor reduction until the remaining fused weighted reduce
    // consumers are audited; no static GGUF FFI or full weight dequantization.
    out.reshape((batch,topk,n))?.broadcast_mul(&routing.to_dtype(DType::F32)?.unsqueeze(2)?)?
        .sum(1)?.to_dtype(out_dtype)
}

#[derive(Clone,Copy,PartialEq,Eq)]
struct RowGroupedKey{fmt:&'static str,n:usize,k:usize,padded:usize,e:usize,a:usize,topk:usize,input_rows:usize,packed:usize,smem:usize,weighted:bool}
thread_local!{static ROW_GROUPED:RefCell<LaunchPlanCache<RowGroupedKey,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());}
fn row_grouped_plan(key:RowGroupedKey,dev:&Device)->Result<PrevalidatedLaunchPlan>{
    let mode=if key.weighted{"weighted"}else{"sorted"};
    let repr=format!("gguf.{}.packed.v1",key.fmt);
    prepare_one(dev,format!("quant.gguf.grouped_row.{}.{mode}.v1",key.fmt),
        vec![tensor(ElementType::U8,vec![key.e,key.n,key.packed],Some(&repr)),
            tensor(ElementType::U8,vec![key.input_rows,key.padded/32*36],Some("gguf.q8_1.row.v1")),
            u(key.e+1),u(key.a),tensor(ElementType::F32,vec![if key.weighted{key.a}else{1}],None),
            tensor(ElementType::F32,vec![if key.weighted{key.a/key.topk}else{key.a},key.n],None),
            scalar(),scalar(),scalar(),scalar(),scalar(),scalar(),scalar()],
        [key.n.div_ceil(64) as u32,key.e as u32,1],[32,8,1],key.smem as u32)
}
pub(crate) fn grouped_row(w:&QTensor,q:&CudaSlice<u8>,k:usize,padded:usize,bounds:&CudaSlice<u32>,ids:&CudaSlice<u32>,
    routing:Option<u64>,a:usize,topk:usize,e:usize,input_dim1:usize,dev:&CudaDevice)->Result<Tensor>{
    device_gate(dev)?;
    let (we,n,wk)=w.shape().dims3()?;
    let quant=w.dtype();
    let fmt=if quant==candle_core::quantized::GgmlDType::Q8_1{"q8_1"}else{format(quant).ok_or_else(||candle_core::Error::msg("DKR grouped row format"))?};
    let device=Device::Cuda(dev.clone());
    let ordinal=dev.cuda_stream().context().ordinal();
    if a==0||topk==0||a%topk!=0||e==0||e>65535||we!=e||n==0||k==0||wk!=k||k%quant.block_size()!=0
        ||padded<k||padded%32!=0||bounds.len()<e+1||ids.len()<a||input_dim1>topk.max(2)
        ||!w.device().same_device(&device)||q.ordinal()!=ordinal||bounds.ordinal()!=ordinal||ids.ordinal()!=ordinal
        ||routing.is_some_and(|p|p==0||p%4!=0){
        candle_core::bail!("DKR grouped row contract");
    }
    let input_rows=if input_dim1==1{a/topk}else{a};
    let packed=mul(k/quant.block_size(),quant.type_size())?;
    let bytes=i32_span(mul(mul(input_rows,padded/32)?,36)?)?;
    if q.len()<bytes{candle_core::bail!("DKR grouped Q8_1 input span");}
    i32_span(mul(mul(e,n)?,packed)?)?;
    for x in [a,n,k,padded,topk,input_dim1]{i32_span(x)?;}
    let smem=if fmt=="q8_0"{2*64*64*4+2*64*8*4+256}else{64*(if fmt.ends_with("_k"){1}else{8})*quant.type_size()+64*8*36+256};
    if smem>cuda_device_info(dev).smem_optin||smem>49152{candle_core::bail!("DKR grouped row shared memory");}
    let out_rows=if routing.is_some(){a/topk}else{a};let count=i32_span(mul(out_rows,n)?)?;
    let key=RowGroupedKey{fmt,n,k,padded,e,a,topk,input_rows,packed,smem,weighted:routing.is_some()};
    let stream=dev.cuda_stream();
    let (wp,_wg)=w.device_ptr_with_guard(&stream)?;
    let (qp,_qg)=slice_ptr_on_stream(q,0,&stream);
    let (bp,_bg)=slice_ptr_on_stream(bounds,0,&stream);
    let (ip,_ig)=slice_ptr_on_stream(ids,0,&stream);
    let mut out=dev.alloc_zeros::<f32>(count)?;
    {
        let (op,_og)=slice_ptr_mut_on_stream(&mut out,0,&stream);
        // Sorted entry ignores routing, but ABI uses a non-null dummy.
        let rp=routing.unwrap_or(op);
        let (ni,ki,pi,ei,ti,ri,ai)=(n as i32,k as i32,padded as i32,e as i32,topk as i32,input_dim1 as i32,a as i32);
        ROW_GROUPED.with(|cache|cache.borrow_mut().with_plan(key,&device,||row_grouped_plan(key,&device),|p|{
            unsafe{p.with_prevalidated_launch(|fun,cfg|{
                let mut b=stream.launch_builder(fun);b.arg(&wp).arg(&qp).arg(&bp).arg(&ip).arg(&rp).arg(&op)
                    .arg(&ni).arg(&ki).arg(&pi).arg(&ei).arg(&ti).arg(&ri).arg(&ai);b.launch(cfg)?;Ok(())
            })}.map_err(|e|candle_core::Error::msg(e.to_string()))
        }))?;
    }
    Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(out,dev.clone())),(out_rows,n))))
}
