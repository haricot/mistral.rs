//! Typed prefill sinks: original tiled CUDA algorithm, catalogue-owned launch.
use super::slice_ptr_on_stream;
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CpuStorage, CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, sync::OnceLock};

const MAX_LEN: usize = 32_768;
#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode { Dense, Varlen, Packed }
impl Mode {
    fn name(self) -> &'static str {
        match self { Self::Dense=>"dense", Self::Varlen=>"varlen", Self::Packed=>"packed" }
    }
}
#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    mode: Mode, dtype: DType, sink: bool,
    batch: usize, heads: usize, kv_heads: usize, hd: usize,
    q_len: usize, kv_len: usize, total_q: usize, total_kv: usize,
}
thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache,String>> = OnceLock::new();

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| KernelExecutionCache::package_root()
        .ok_or_else(|| "DKR prefill sinks catalogue unavailable".to_owned())
        .and_then(|p| KernelExecutionCache::discover(&p).map_err(|e|e.to_string())))
        .as_ref().map_err(|e|candle_core::Error::Msg(e.clone()))
}
fn arg(dtype: ElementType, shape: Vec<usize>) -> Argument {
    Argument::Tensor { dtype, shape, contiguous:true, start_offset:0, representation:None }
}
fn qshape(k: Key) -> Vec<usize> {
    if k.mode==Mode::Packed { vec![k.total_q,k.heads,k.hd] }
    else { vec![k.batch,k.heads,k.q_len,k.hd] }
}
fn kshape(k: Key) -> Vec<usize> {
    if k.mode==Mode::Dense { vec![k.batch,k.kv_heads,k.kv_len,k.hd] }
    else { vec![k.total_kv,k.kv_heads,k.hd] }
}
fn prepare(k: Key, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dt=match k.dtype { DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,
        DType::F32=>ElementType::F32,_=>candle_core::bail!("unsupported prefill dtype") };
    let mut args=vec![arg(dt,qshape(k)),arg(dt,kshape(k)),arg(dt,kshape(k)),arg(dt,qshape(k))];
    if k.sink { args.push(arg(ElementType::F32,vec![k.heads])); }
    if k.mode!=Mode::Dense { args.extend([arg(ElementType::U32,vec![k.batch+1]),arg(ElementType::U32,vec![k.batch+1])]); }
    args.push(Argument::Scalar{dtype:ElementType::F32});
    args.extend((0..if k.mode==Mode::Dense {5} else {6}).map(|_|Argument::Scalar{dtype:ElementType::U32}));
    let mut request=LaunchRequest {
        operation:format!("attention.prefill.sinks.{}.{}.d{}.v1",k.mode.name(),if k.sink {"sink"} else {"plain"},k.hd),
        cc:0,args,grid:[k.heads as u32,k.batch as u32,k.q_len.div_ceil(8) as u32],
        max_dynamic_shared_mem_bytes:0,
    };
    let prepared=catalog()?.prepare_for_device(&mut request,device)
        .map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if prepared.manifest.block!=[256,1,1] || prepared.manifest.shared_mem_bytes!=0 {
        candle_core::bail!("unexpected prefill sinks launch geometry");
    }
    prepared.prevalidate_launch(&request,device).map_err(|e|candle_core::Error::msg(e.to_string()))
}

struct Op {
    k: Tensor, v: Tensor, sinks: Option<Tensor>,
    lengths: Option<(Tensor,Tensor)>, key: Key, scale:f32, window:u32,
}
impl candle_core::CustomOp1 for Op {
    fn name(&self)->&'static str {"dkr-prefill-sinks"}
    fn cpu_fwd(&self,_:&CpuStorage,_:&Layout)->Result<(CpuStorage,Shape)> {
        candle_core::bail!("DKR prefill sinks requires CUDA")
    }
    fn cuda_fwd(&self,q:&CudaStorage,ql:&Layout)->Result<(CudaStorage,Shape)> {
        let key=self.key;
        if !ql.is_contiguous() || ql.dims()!=qshape(key).as_slice() || q.dtype()!=key.dtype {
            candle_core::bail!("DKR prefill query layout changed");
        }
        let cuda=q.device();
        let device=Device::Cuda(cuda.clone());
        let stream=cuda.cuda_stream();
        let (ks,kl)=self.k.storage_and_layout();
        let (vs,vl)=self.v.storage_and_layout();
        let (Storage::Cuda(ks),Storage::Cuda(vs))=(&*ks,&*vs) else {
            candle_core::bail!("DKR prefill sinks requires CUDA K/V");
        };
        let sink_storage=self.sinks.as_ref().map(Tensor::storage_and_layout);
        let sink_ptr=if let Some((s,l))=&sink_storage {
            let Storage::Cuda(s)=&**s else {candle_core::bail!("DKR sinks require CUDA");};
            Some(slice_ptr_on_stream(s.as_cuda_slice::<f32>()?,l.start_offset(),&stream))
        } else {None};
        let length_storage=self.lengths.as_ref().map(|(a,b)|(a.storage_and_layout(),b.storage_and_layout()));
        let length_ptr=if let Some(((a,al),(b,bl)))=&length_storage {
            let (Storage::Cuda(a),Storage::Cuda(b))=(&**a,&**b) else {candle_core::bail!("DKR lengths require CUDA");};
            let (ap,ag)=slice_ptr_on_stream(a.as_cuda_slice::<u32>()?,al.start_offset(),&stream);
            let (bp,bg)=slice_ptr_on_stream(b.as_cuda_slice::<u32>()?,bl.start_offset(),&stream);
            Some((ap,bp,ag,bg))
        } else {None};
        let (qlen,kvlen,tq,tk,h,kh)=(key.q_len as u32,key.kv_len as u32,key.total_q as u32,key.total_kv as u32,key.heads as u32,key.kv_heads as u32);
        macro_rules! run {
            ($ty:ty)=>{{
                let (qp,_qg)=slice_ptr_on_stream(q.as_cuda_slice::<$ty>()?,ql.start_offset(),&stream);
                let (kp,_kg)=slice_ptr_on_stream(ks.as_cuda_slice::<$ty>()?,kl.start_offset(),&stream);
                let (vp,_vg)=slice_ptr_on_stream(vs.as_cuda_slice::<$ty>()?,vl.start_offset(),&stream);
                // Zero-initialize varlen output so padded rows stay zero. Valid cumulative metadata is a caller contract.
                let mut out=if key.mode==Mode::Dense {unsafe{cuda.alloc::<$ty>(ql.shape().elem_count())?}}
                    else {cuda.alloc_zeros::<$ty>(ql.shape().elem_count())?};
                {
                    let (op,_og)=out.device_ptr_mut(&stream);
                    PLANS.with(|p|p.borrow_mut().with_plan(key,&device,||prepare(key,&device),|plan|{
                        unsafe{plan.with_prevalidated_launch(|fun,config|{
                            let mut launch=stream.launch_builder(fun);
                            launch.arg(&qp).arg(&kp).arg(&vp).arg(&op);
                            if let Some((p,_))=&sink_ptr {launch.arg(p);}
                            if let Some((a,b,_,_))=&length_ptr {launch.arg(a).arg(b);}
                            launch.arg(&self.scale).arg(&qlen);
                            if key.mode==Mode::Dense {launch.arg(&kvlen);} else {launch.arg(&tq).arg(&tk);}
                            launch.arg(&h).arg(&kh).arg(&self.window);
                            launch.launch(config)?;
                            Ok(())
                        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
                    }))?;
                }
                CudaStorage::wrap_cuda_slice(out,cuda.clone())
            }}
        }
        let out=match key.dtype {DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()};
        Ok((out,ql.shape().clone()))
    }
}

#[allow(clippy::too_many_arguments)]
fn forward(q:&Tensor,k:&Tensor,v:&Tensor,sinks:Option<&Tensor>,lengths:Option<(&Tensor,&Tensor)>,
           packed_max:Option<usize>,scale:f32,window:usize)->Result<Tensor> {
    let dtype=q.dtype();
    let mode=if packed_max.is_some(){Mode::Packed}else if lengths.is_some(){Mode::Varlen}else{Mode::Dense};
    if !q.device().is_cuda() || !scale.is_finite() || window>i32::MAX as usize/4
        || !matches!(dtype,DType::F16|DType::BF16|DType::F32)
        || k.dtype()!=dtype || v.dtype()!=dtype || k.dims()!=v.dims()
        || !q.device().same_device(k.device()) || !q.device().same_device(v.device()) {
        candle_core::bail!("DKR prefill sinks unsupported dtype/device/scale/window");
    }
    let (batch,heads,q_len,hd,total_q)=if let Some(max)=packed_max {
        let (t,h,d)=q.dims3()?;
        let (a,_)=lengths.ok_or_else(||candle_core::Error::msg("packed requires cumulative lengths"))?;
        (a.elem_count().saturating_sub(1),h,max,d,t)
    } else {
        let (b,h,s,d)=q.dims4()?;
        (b,h,s,d,b.checked_mul(s).ok_or_else(||candle_core::Error::msg("Q length overflow"))?)
    };
    let (kv_heads,kv_len,total_kv,khd)=if mode==Mode::Dense {
        let (b,h,s,d)=k.dims4()?;
        if b!=batch {candle_core::bail!("DKR prefill batch mismatch");}
        (h,s,b.checked_mul(s).ok_or_else(||candle_core::Error::msg("KV length overflow"))?,d)
    } else {let(t,h,d)=k.dims3()?;(h,0,t,d)};
    if batch==0 || batch>65535 || heads==0 || heads>65535 || kv_heads==0 || heads%kv_heads!=0
        || !matches!(hd,64|80|96|112|128|192|256) || hd!=khd || q_len==0 || q_len>MAX_LEN
        || kv_len>MAX_LEN || total_q==0 || total_kv==0
        || [q,k,v].iter().any(|t|t.elem_count()>i32::MAX as usize/2) {
        candle_core::bail!("DKR prefill sinks unsupported geometry");
    }
    if let Some((a,b))=lengths {
        for t in [a,b] {
            if t.dtype()!=DType::U32 || t.dims()!=[batch+1] || !t.is_contiguous() || !q.device().same_device(t.device()) {
                candle_core::bail!("DKR prefill cumulative lengths must be contiguous U32 [B+1] on device");
            }
        }
    }
    if let Some(t)=sinks {
        if t.dims()!=[heads] || !q.device().same_device(t.device()) {candle_core::bail!("DKR prefill sinks must be [H] on device");}
    }
    let key=Key{mode,dtype,sink:sinks.is_some(),batch,heads,kv_heads,hd,q_len,kv_len,total_q,total_kv};
    // Historical public APIs materialize contiguous Q/K/V when needed.
    let q=q.contiguous()?;
    let op=Op{k:k.contiguous()?,v:v.contiguous()?,
        sinks:sinks.map(|s|s.to_dtype(DType::F32)?.contiguous()).transpose()?,
        lengths:lengths.map(|(a,b)|(a.clone(),b.clone())),key,scale,window:window as u32};
    q.apply_op1(op)
}

pub fn flash_attn_sinks(q:&Tensor,k:&Tensor,v:&Tensor,sinks:Option<&Tensor>,softmax_scale:f32,window_size:usize)->Result<Tensor>{
    forward(q,k,v,sinks,None,None,softmax_scale,window_size)
}
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen(q:&Tensor,k:&Tensor,v:&Tensor,sinks:Option<&Tensor>,cu_seqlens_q:&Tensor,cu_seqlens_k:&Tensor,softmax_scale:f32,window_size:usize)->Result<Tensor>{
    forward(q,k,v,sinks,Some((cu_seqlens_q,cu_seqlens_k)),None,softmax_scale,window_size)
}
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen_packed(q:&Tensor,k:&Tensor,v:&Tensor,sinks:Option<&Tensor>,cu_seqlens_q:&Tensor,cu_seqlens_k:&Tensor,max_q_len:usize,softmax_scale:f32,window_size:usize)->Result<Tensor>{
    forward(q,k,v,sinks,Some((cu_seqlens_q,cu_seqlens_k)),Some(max_q_len),softmax_scale,window_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Binary-exact inputs so the independent F64 oracle does not depend
    // on a particular GPU conversion or reduction implementation.
    fn query(h: usize, row: usize, d: usize) -> f32 {
        if d==0 { (if h%2==0 {1.0} else {-1.0}) * (1.0 + (row%3) as f32*0.25) }
        else if d==1 {0.5} else {0.0}
    }
    fn key(seq: usize, token: usize, d: usize) -> f32 {
        (((seq+token+d)%5) as f32-2.0)*0.5
    }
    fn value(seq: usize, token: usize, d: usize) -> f32 {
        (((seq+3*token+d)%13) as f32-6.0)*0.125
    }
    fn oracle(seq:usize,h:usize,row:usize,d:usize,ql:usize,kl:usize,window:usize,sink:bool) -> f64 {
        let end=(kl as i64-ql as i64+row as i64+1).clamp(0,kl as i64) as usize;
        let start=if window==0 {0} else {end.saturating_sub(window)};
        let logit=|t:usize| 0.5*(query(h,row,0) as f64*key(seq,t,0) as f64+
                                      query(h,row,1) as f64*key(seq,t,1) as f64);
        let sl=if h==0 {1.0f64} else {-0.5};
        let m=(start..end).map(logit).fold(if sink {sl}else {f64::NEG_INFINITY},f64::max);
        if m==f64::NEG_INFINITY {return 0.0;}
        let mut den=if sink {(sl-m).exp()}else{0.0};
        let mut num=0.0;
        for t in start..end {
            let w=(logit(t)-m).exp();
            den+=w;
            num+=w*value(seq,t,d) as f64;
        }
        if den>0.0 {num/den}else{0.0}
    }

    #[test]
    #[ignore = "requires physical CUDA and staged prefill-sinks catalogue"]
    fn dkr_prefill_sinks_gpu() -> Result<()> {
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let (b,h,kh,qmax,kmax)=(3usize,2usize,1usize,9usize,17usize);
        let qlens=[9usize,3,0];
        let klens=[17usize,2,1];
        let cuq=Tensor::from_vec(vec![0u32,9,12,12],4,&dev)?;
        let cuk=Tensor::from_vec(vec![0u32,17,19,20],4,&dev)?;
        let sinks=Tensor::from_vec(vec![1.0f32,-0.5],h,&dev)?;
        for dtype in [DType::F16,DType::BF16,DType::F32] {
            for hd in [64usize,80,96,112,128,192,256] {
                let mut q=vec![0f32;b*h*qmax*hd];
                let mut k=vec![0f32;b*kh*kmax*hd];
                let mut v=k.clone();
                let mut qp=Vec::new();
                let mut kp=Vec::new();
                let mut vp=Vec::new();
                for seq in 0..b {
                    for row in 0..qmax {
                        for head in 0..h {
                            for d in 0..hd {
                                let a=query(head,row,d);
                                q[((seq*h+head)*qmax+row)*hd+d]=a;
                                if row<qlens[seq] {qp.push(a);}
                            }
                        }
                    }
                    for t in 0..kmax {
                        for d in 0..hd {
                            let a=key(seq,t,d);
                            let z=value(seq,t,d);
                            k[(seq*kmax+t)*hd+d]=a;
                            v[(seq*kmax+t)*hd+d]=z;
                            if t<klens[seq] {kp.push(a);vp.push(z);}
                        }
                    }
                }
                let q=Tensor::from_vec(q,(b,h,qmax,hd),&dev)?.to_dtype(dtype)?;
                let k=Tensor::from_vec(k,(b,kh,kmax,hd),&dev)?.to_dtype(dtype)?;
                let v=Tensor::from_vec(v,(b,kh,kmax,hd),&dev)?.to_dtype(dtype)?;
                let qp=Tensor::from_vec(qp,(12,h,hd),&dev)?.to_dtype(dtype)?;
                let kp=Tensor::from_vec(kp,(20,kh,hd),&dev)?.to_dtype(dtype)?;
                let vp=Tensor::from_vec(vp,(20,kh,hd),&dev)?.to_dtype(dtype)?;
                for sink in [false,true] {
                    let st=sink.then_some(&sinks);
                    for window in [0usize,3] {
                        for mode in [Mode::Dense,Mode::Varlen,Mode::Packed] {
                            let run=||match mode {
                                Mode::Dense=>super::super::flash_attn_sinks(&q,&k,&v,st,0.5,window),
                                Mode::Varlen=>super::super::flash_attn_sinks_varlen(&q,&kp,&vp,st,&cuq,&cuk,0.5,window),
                                Mode::Packed=>super::super::flash_attn_sinks_varlen_packed(&qp,&kp,&vp,st,&cuq,&cuk,qmax,0.5,window),
                            };
                            let out=run()?;
                            assert_eq!(out.dtype(),dtype);
                            assert_eq!(out.dims(),if mode==Mode::Packed {qp.dims()}else{q.dims()});
                            let actual=out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                            let mut err=0f64;
                            let mut base=0usize;
                            for seq in 0..b {
                                let (ql,kl)=if mode==Mode::Dense {(qmax,kmax)}else{(qlens[seq],klens[seq])};
                                let rows=if mode==Mode::Packed {ql}else{qmax};
                                for row in 0..rows {
                                    for head in 0..h {
                                        for d in 0..hd {
                                            let idx=if mode==Mode::Packed {((base+row)*h+head)*hd+d}
                                                else {((seq*h+head)*qmax+row)*hd+d};
                                            let got=actual[idx] as f64;
                                            assert!(got.is_finite());
                                            let expected=if row<ql {oracle(seq,head,row,d,ql,kl,window,sink)}else{0.0};
                                            if row>=ql {assert_eq!(got,0.0);}
                                            err=err.max((got-expected).abs());
                                        }
                                    }
                                }
                                base+=ql;
                            }
                            let tol=match dtype {DType::F32=>3e-5,DType::F16=>0.001,_=>0.006};
                            assert!(err<=tol,"prefill sinks mode={} dtype={dtype:?} hd={hd} sink={sink} window={window} err={err}",mode.name());
                            let count=PLANS.with(|p|p.borrow().len());
                            assert!(count>0);
                            let again=run()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                            assert_eq!(actual,again);
                            assert_eq!(count,PLANS.with(|p|p.borrow().len()));
                            eprintln!("DKR_R126_PREFILL_SINKS=PASS mode={} dtype={dtype:?} hd={hd} sink={sink} window={window} max_abs={err:.8}",mode.name());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dkr_prefill_sinks_cpu_rejected() -> Result<()> {
        let t=Tensor::zeros((1,1,1,64),DType::F32,&Device::Cpu)?;
        assert!(flash_attn_sinks(&t,&t,&t,None,1.0,0).is_err());
        Ok(())
    }
}
