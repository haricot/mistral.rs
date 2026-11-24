//! Ordinary FlashInfer decode and split-KV merge, using only typed DKR entries.
//! The existing planner owns device metadata values (page IDs, indptr, active
//! request/tile mapping, chunk size and mask). No device-to-host readback here.
//! Callers must supply trusted planner tensors, not arbitrary index contents.
use super::{flashinfer_fastdiv::fastdiv, slice_ptr_on_stream, FlashInferDecodeScratch};
use crate::{KvCacheScales, DEFAULT_FP8_KV_CACHE_SCALES};
use candle_core::{cuda::cudarc::driver::{DeviceSlice, PushKernelArg},
    DType, Device, Result, Storage, Tensor};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
};
use std::{cell::RefCell, sync::OnceLock};

#[derive(Clone, Copy, Eq, PartialEq)]
struct Key {
    dtype: DType, fp8: bool, batch: usize, padded: usize, heads: usize, kv_heads: usize,
    hd: usize, pages: usize, page: usize, indices: usize, q_span: usize,
    window: bool, softcap: bool,
}
struct Plan {
    decode: PrevalidatedLaunchPlan,
    merge: Option<PrevalidatedLaunchPlan>,
    magic: u32, shift: u32, add: u32,
}
thread_local! {
    static PLANS: RefCell<LaunchPlanCache<Key, Plan>> = RefCell::new(LaunchPlanCache::new());
}
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG.get_or_init(|| KernelExecutionCache::package_root()
        .ok_or_else(|| "DKR FlashInfer decode catalogue unavailable".to_owned())
        .and_then(|p| KernelExecutionCache::discover(&p).map_err(|e| e.to_string())))
        .as_ref().map_err(|e| candle_core::Error::Msg(e.clone()))
}
fn dtype(dt: DType) -> Result<(ElementType, &'static str)> {
    Ok(match dt {
        DType::F16 => (ElementType::F16, "f16"),
        DType::BF16 => (ElementType::Bf16, "bf16"),
        DType::F32 => (ElementType::F32, "f32"),
        _ => candle_core::bail!("DKR FlashInfer unsupported activation dtype"),
    })
}
fn arg(dtype: ElementType, shape: Vec<usize>, repr: Option<&str>) -> Argument {
    Argument::Tensor {dtype, shape, contiguous: true, start_offset: 0,
        representation: repr.map(str::to_owned)}
}
fn scalar(dtype: ElementType) -> Argument { Argument::Scalar {dtype} }
fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or_else(|| candle_core::Error::msg("DKR FlashInfer size overflow"))
}
fn q_span(batch: usize, heads: usize, hd: usize, sn: usize, sh: usize) -> Result<usize> {
    if batch == 0 || heads == 0 || hd == 0 || sh < hd {
        candle_core::bail!("DKR FlashInfer invalid query strides");
    }
    let row = (heads-1).checked_mul(sh).and_then(|n| n.checked_add(hd))
        .ok_or_else(|| candle_core::Error::msg("DKR FlashInfer query row overflow"))?;
    if batch > 1 && sn < row { candle_core::bail!("DKR FlashInfer overlapping query rows"); }
    (batch-1).checked_mul(sn).and_then(|n| n.checked_add(row))
        .ok_or_else(|| candle_core::Error::msg("DKR FlashInfer query span overflow"))
}
fn range(ptr: u64, n: usize, bytes: usize) -> Result<(u64,u64)> {
    let end = ptr.checked_add(product(&[n,bytes])? as u64)
        .ok_or_else(|| candle_core::Error::msg("DKR FlashInfer pointer range overflow"))?;
    Ok((ptr,end))
}
fn disjoint(writes: &[(u64,u64)], reads: &[(u64,u64)]) -> Result<()> {
    let overlaps = |a: (u64,u64), b: (u64,u64)| a.0 < b.1 && b.0 < a.1;
    for (i,&w) in writes.iter().enumerate() {
        if reads.iter().chain(writes[..i].iter()).any(|&r| overlaps(w,r)) {
            candle_core::bail!("DKR FlashInfer writable span aliases another argument");
        }
    }
    Ok(())
}
fn decode_geometry(k: Key, cc: u32) -> Result<([u32;3],u32)> {
    let width = if k.fp8 {1} else {k.dtype.size_in_bytes()};
    let g = k.heads/k.kv_heads;
    let stages = if cc >= 80 {2} else {1};
    let vec = (16/width).max(k.hd/32);
    let x = k.hd/vec;
    let intended = 128.max(x*g);
    let z = intended/(x*g);
    let tile = if g == 1 {if width == 1 {2} else {4}} else {1};
    let shared = 2*stages*tile*g*z*k.hd*width + (tile*intended*8).max(2*g*z*4);
    if x*g*z%32 != 0 || (z>1 && 2*stages*tile*width<4) || shared>48*1024 {
        candle_core::bail!("DKR FlashInfer geometry quarantined");
    }
    Ok(([x as u32,g as u32,z as u32],shared as u32))
}
fn prepare(k: Key, dev: &Device) -> Result<Plan> {
    let (dt, name) = dtype(k.dtype)?;
    let cd = if k.fp8 {ElementType::Fp8E4m3} else {dt};
    let cache = || arg(cd,vec![k.pages,k.kv_heads,k.page,k.hd],Some("flashinfer.hnd.v1"));
    let partition = k.padded > k.batch;
    let partial = || arg(dt,vec![k.padded,k.heads,k.hd],Some("flashinfer.splitkv.partial.v1"));
    let lse = || arg(ElementType::F32,vec![k.padded,k.heads],Some("flashinfer.splitkv.log2_lse.v1"));
    let output = || arg(dt,vec![k.batch,k.heads,k.hd],None);
    let mut args = vec![arg(dt,vec![k.q_span],Some("flashinfer.q.strided_span.v1")),cache(),cache()];
    for n in [k.batch+1,k.indices,k.batch,k.padded,k.padded,k.batch+1,1] {
        args.push(arg(ElementType::I32,vec![n],None));
    }
    args.push(arg(ElementType::U8,vec![k.padded],None));
    if partition {args.extend([partial(),lse()]);} else {args.push(output());}
    args.extend((0..8).map(|_|scalar(ElementType::U32)));
    args.extend([scalar(ElementType::I32),scalar(ElementType::I32),scalar(ElementType::U32),scalar(ElementType::I32)]);
    args.extend((0..4).map(|_|scalar(ElementType::F32)));
    let suffix = format!("d{}.g{}.w{}.s{}.{}.v1",k.hd,k.heads/k.kv_heads,
        k.window as u8,k.softcap as u8,if partition {"partition"} else {"direct"});
    let implementation = format!("cuda.flashinfer_decode.{name}.{}.{suffix}",if k.fp8 {"fp8"} else {"plain"});
    let mut req = LaunchRequest {operation:format!("attention.flashinfer.decode.{suffix}"),
        cc:0,args,grid:[k.padded as u32,k.kv_heads as u32,1],max_dynamic_shared_mem_bytes:48*1024};
    let p = catalog()?.prepare_for_device(&mut req,dev).map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let (block,shared) = decode_geometry(k,req.cc)?;
    if p.manifest.implementation_id != implementation || p.manifest.block != block || p.manifest.shared_mem_bytes != shared {
        candle_core::bail!("DKR FlashInfer decode contract mismatch");
    }
    let decode = p.prevalidate_launch(&req,dev).map_err(|e| candle_core::Error::msg(e.to_string()))?;
    let merge = if partition {
        let mut req = LaunchRequest {operation:format!("attention.flashinfer.merge.d{}.v1",k.hd),
            cc:0,args:vec![partial(),lse(),arg(ElementType::I32,vec![k.batch+1],None),
                output(),scalar(ElementType::U32),scalar(ElementType::U32)],
            grid:[1,1,1],max_dynamic_shared_mem_bytes:48*1024};
        let p = catalog()?.prepare_for_device(&mut req,dev).map_err(|e| candle_core::Error::msg(e.to_string()))?;
        let width = k.dtype.size_in_bytes();
        let x = k.hd/(16/width).max(k.hd/32);
        let y = 128/x;
        let shared = 4*y*k.hd*width+128*4;
        if p.manifest.implementation_id != format!("cuda.flashinfer_merge.{name}.d{}.v1",k.hd)
            || p.manifest.block != [x as u32,y as u32,1] || p.manifest.shared_mem_bytes != shared as u32 {
            candle_core::bail!("DKR FlashInfer merge contract mismatch");
        }
        req.grid[0] = p.persistent_grid_x((k.batch*k.heads) as u32,dev)
            .map_err(|e| candle_core::Error::msg(e.to_string()))?;
        Some(p.prevalidate_launch(&req,dev).map_err(|e| candle_core::Error::msg(e.to_string()))?)
    } else {None};
    let (magic,shift,add) = fastdiv(k.page as u32);
    if shift>=32 {candle_core::bail!("DKR FlashInfer page divisor unsupported");}
    Ok(Plan{decode,merge,magic,shift,add})
}

#[allow(clippy::too_many_arguments)]
pub fn flashinfer_decode(query:&Tensor,key_cache:&Tensor,value_cache:&Tensor,scales:KvCacheScales,
    paged_kv_indptr:&Tensor,paged_kv_indices:&Tensor,paged_kv_last_page_len:&Tensor,
    request_indices:&Tensor,kv_tile_indices:&Tensor,o_indptr:&Tensor,kv_chunk_size:&Tensor,
    block_valid_mask:&Tensor,sm_scale:f32,window_left:Option<usize>,logits_soft_cap:Option<f32>,
    scratch:Option<FlashInferDecodeScratch<'_>>) -> Result<Tensor> {
    let dev=query.device(); let dt=query.dtype(); dtype(dt)?;
    let cd=key_cache.dtype(); let fp8=cd==DType::F8E4M3;
    let (batch,heads,hd)=query.dims3()?;
    let (pages,kv_heads,page,cache_hd)=key_cache.dims4()?;
    let padded=request_indices.dims1()?; let indices=paged_kv_indices.dims1()?;
    let window=window_left.map(i32::try_from).transpose()
        .map_err(|_|candle_core::Error::msg("DKR FlashInfer window overflow"))?.unwrap_or(-1);
    let softcap=logits_soft_cap.unwrap_or(0.);
    if !dev.is_cuda() || (!fp8 && cd!=dt) || value_cache.dtype()!=cd ||
        value_cache.dims()!=key_cache.dims() || hd!=cache_hd ||
        !matches!(hd,64|128|256|512) || [batch,heads,kv_heads,pages,page,padded,indices].contains(&0)
        || padded<batch || heads%kv_heads!=0 || !matches!(heads/kv_heads,1|2|3|4|5|6|7|8|16)
        || kv_heads>65535 || page>65536 || !sm_scale.is_finite() || !softcap.is_finite()
        || !scales.k.is_finite() || !scales.v.is_finite() || scales.k<=0. || scales.v<=0.
        || !(sm_scale*scales.k).is_finite() || (!fp8 && scales!=DEFAULT_FP8_KV_CACHE_SCALES)
        || [batch,padded,heads,pages,indices].iter().any(|&n|n>i32::MAX as usize)
        || product(&[indices,page])? > i32::MAX as usize-4096
        || product(&[pages,kv_heads,page,hd])? > i32::MAX as usize
        || product(&[padded,heads,hd])? > i32::MAX as usize {
        candle_core::bail!("DKR FlashInfer incompatible geometry or scales");
    }
    let metadata=[(paged_kv_indptr,batch+1),(paged_kv_indices,indices),(paged_kv_last_page_len,batch),
        (request_indices,padded),(kv_tile_indices,padded),(o_indptr,batch+1),(kv_chunk_size,1)];
    for (t,n) in metadata {
        if t.dtype()!=DType::I32 || t.dims()!=[n] || !t.is_contiguous() || !dev.same_device(t.device()) {
            candle_core::bail!("DKR FlashInfer invalid planner metadata layout");
        }
    }
    if block_valid_mask.dtype()!=DType::U8 || block_valid_mask.dims()!=[padded] ||
        !block_valid_mask.is_contiguous() || !dev.same_device(block_valid_mask.device()) {
        candle_core::bail!("DKR FlashInfer invalid mask layout");
    }
    for t in [key_cache,value_cache] {
        if !t.is_contiguous() || !dev.same_device(t.device()) {
            candle_core::bail!("DKR FlashInfer requires dense same-device HND cache");
        }
    }
    let (qs,ql)=query.storage_and_layout();
    let sn=ql.stride()[0]; let sh=ql.stride()[1];
    if ql.stride()[2]!=1 || sn>i32::MAX as usize || sh>i32::MAX as usize {
        candle_core::bail!("DKR FlashInfer invalid query strides");
    }
    let span=q_span(batch,heads,hd,sn,sh)?;
    let align=16/dt.size_in_bytes();
    if span>i32::MAX as usize || sn%align!=0 || sh%align!=0 {
        candle_core::bail!("DKR FlashInfer query span or vector alignment unsupported");
    }
    let key=Key{dtype:dt,fp8,batch,padded,heads,kv_heads,hd,pages,page,indices,q_span:span,
        window:window>=0,softcap:softcap>0.};
    let split=padded>batch;
    let output=unsafe{Tensor::empty((batch,heads,hd),dt,dev)?};
    // Preserve caller-provided capacity (the kernel uses packed rows, not
    // scratch tensor strides). Allocate only when the legacy API supplies none.
    let owned=if split && scratch.is_none(){Some((
        unsafe{Tensor::empty((padded,heads,hd),dt,dev)?},
        unsafe{Tensor::empty((padded,heads),DType::F32,dev)?}
    ))}else{None};
    let (partial,lse)=if split {
        let (v,s)=if let Some(s)=scratch {(s.tmp_v,s.tmp_s)}
            else {let (v,s)=owned.as_ref().unwrap();(v,s)};
        let (vr,vh,vd)=v.dims3()?; let (sr,sh)=s.dims2()?;
        if v.dtype()!=dt || s.dtype()!=DType::F32 || !v.is_contiguous() || !s.is_contiguous()
            || !dev.same_device(v.device()) || !dev.same_device(s.device())
            || vr<padded || vh<heads || vd<hd || sr<padded || sh<heads {
            candle_core::bail!("DKR FlashInfer invalid split-KV scratch");
        }
        (v,s)
    } else {(&output,&output)};
    let (ks,kl)=key_cache.storage_and_layout(); let (vs,vl)=value_cache.storage_and_layout();
    let (os,ol)=output.storage_and_layout();
    let (ps,pl)=partial.storage_and_layout(); let (ss,sl)=lse.storage_and_layout();
    let (is,il)=paged_kv_indptr.storage_and_layout(); let (ns,nl)=paged_kv_indices.storage_and_layout();
    let (ls,ll)=paged_kv_last_page_len.storage_and_layout(); let (rs,rl)=request_indices.storage_and_layout();
    let (ts,tl)=kv_tile_indices.storage_and_layout(); let (us,ul)=o_indptr.storage_and_layout();
    let (cs,cl)=kv_chunk_size.storage_and_layout(); let (ms,ml)=block_valid_mask.storage_and_layout();
    let stream=dev.as_cuda_device()?.cuda_stream();
    let (batch32,padded32,heads32,kv32,page32)=(batch as u32,padded as u32,heads as u32,kv_heads as u32,page as u32);
    let (sn32,sh32)=(sn as i32,sh as i32);
    let use_mask=1u32;
    let cache_len=product(&[pages,kv_heads,page,hd])?;
    let out_len=product(&[batch,heads,hd])?;
    let partial_len=product(&[padded,heads,hd])?;
    let lse_len=product(&[padded,heads])?;
    macro_rules! ptr {
        ($s:ident,$l:ident,$ty:ty,$n:expr,$a:expr) => {{
            let Storage::Cuda(s)=&*$s else {candle_core::bail!("DKR FlashInfer requires CUDA storage");};
            let slice=s.as_cuda_slice::<$ty>()?;
            let start=$l.start_offset();
            if start.checked_add($n).is_none_or(|end|end>slice.len()) ||
                start%$a!=0 {
                candle_core::bail!("DKR FlashInfer invalid physical span or alignment");
            }
            slice_ptr_on_stream(slice,start,&stream)
        }};
    }
    let (ip,_ig)=ptr!(is,il,i32,batch+1,1);
    let (np,_ng)=ptr!(ns,nl,i32,indices,1);
    let (lp,_lg)=ptr!(ls,ll,i32,batch,1);
    let (rp,_rg)=ptr!(rs,rl,i32,padded,1);
    let (tp,_tg)=ptr!(ts,tl,i32,padded,1);
    let (up,_ug)=ptr!(us,ul,i32,batch+1,1);
    let (cp,_cg)=ptr!(cs,cl,i32,1,1);
    let (mp,_mg)=ptr!(ms,ml,u8,padded,1);
    macro_rules! run {
        ($ty:ty,$ct:ty)=>{{
            let (qp,_qg)=ptr!(qs,ql,$ty,span,align);
            let (kp,_kg)=ptr!(ks,kl,$ct,cache_len,16/cd.size_in_bytes());
            let (vp,_vg)=ptr!(vs,vl,$ct,cache_len,16/cd.size_in_bytes());
            let (op,_og)=ptr!(os,ol,$ty,out_len,align);
            let (pp,_pg)=if split {let (p,g)=ptr!(ps,pl,$ty,partial_len,align);(p,Some(g))}else{(op,None)};
            let (sp,_sg)=if split {let (p,g)=ptr!(ss,sl,f32,lse_len,1);(p,Some(g))}else{(0,None)};
            let reads=[range(qp,span,dt.size_in_bytes())?,range(kp,cache_len,cd.size_in_bytes())?,
                range(vp,cache_len,cd.size_in_bytes())?,range(ip,batch+1,4)?,range(np,indices,4)?,
                range(lp,batch,4)?,range(rp,padded,4)?,range(tp,padded,4)?,range(up,batch+1,4)?,
                range(cp,1,4)?,range(mp,padded,1)?];
            let output_range=range(op,out_len,dt.size_in_bytes())?;
            if split {
                disjoint(&[output_range,range(pp,partial_len,dt.size_in_bytes())?,range(sp,lse_len,4)?],&reads)?;
            }else{disjoint(&[output_range],&reads)?;}
            PLANS.with(|cache| cache.borrow_mut().with_plan(key,dev,||prepare(key,dev),|plan|{
                // Both plans are resolved before the first enqueue. No host wait
                // between decode and merge: same stream, ordered launches.
                unsafe{plan.decode.with_prevalidated_launch(|fun,config|{
                    let mut l=stream.launch_builder(fun);
                    l.arg(&qp).arg(&kp).arg(&vp).arg(&ip).arg(&np).arg(&lp)
                        .arg(&rp).arg(&tp).arg(&up).arg(&cp).arg(&mp).arg(&pp);
                    if split {l.arg(&sp);}
                    l.arg(&batch32).arg(&padded32).arg(&heads32).arg(&kv32).arg(&page32)
                        .arg(&plan.magic).arg(&plan.shift).arg(&plan.add)
                        .arg(&sn32).arg(&sh32).arg(&use_mask).arg(&window)
                        .arg(&softcap).arg(&sm_scale).arg(&scales.k).arg(&scales.v);
                    l.launch(config)?; Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                if let Some(merge)=&plan.merge {
                    unsafe{merge.with_prevalidated_launch(|fun,config|{
                        let mut l=stream.launch_builder(fun);
                        l.arg(&pp).arg(&sp).arg(&up).arg(&op).arg(&batch32).arg(&heads32);
                        l.launch(config)?;Ok(())
                    })}.map_err(|e|candle_core::Error::msg(e.to_string()))?;
                }
                Ok(())
            }))
        }};
    }
    match (dt,fp8) {
        (DType::F16,false)=>run!(half::f16,half::f16),
        (DType::BF16,false)=>run!(half::bf16,half::bf16),
        (DType::F32,false)=>run!(f32,f32),
        (DType::F16,true)=>run!(half::f16,float8::F8E4M3),
        (DType::BF16,true)=>run!(half::bf16,float8::F8E4M3),
        (DType::F32,true)=>run!(f32,float8::F8E4M3),
        _=>unreachable!(),
    }?;
    // Storage and stream guards are released in reverse declaration order.
    Ok(output.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_flashinfer_decode_span_and_alias_cpu() {
        assert_eq!(q_span(2,4,64,320,64).unwrap(),576);
        assert!(q_span(2,4,64,255,64).is_err());
        assert!(q_span(2,4,64,usize::MAX,64).is_err());
        assert!(disjoint(&[(100,200),(200,300)],&[(0,100)]).is_ok());
        assert!(disjoint(&[(100,200),(199,300)],&[]).is_err());
        assert!(disjoint(&[(100,200)],&[(150,160)]).is_err());
    }
    #[test]
    #[ignore="requires real flashinfer-decode and flashinfer-merge images plus CUDA"]
    fn dkr_flashinfer_direct_split_gpu() -> Result<()> {
        let dev=Device::new_cuda(0)?;
        let q=Tensor::zeros((1,1,64),DType::F32,&dev)?;
        let k=Tensor::zeros((2,1,4,64),DType::F32,&dev)?;
        let v=Tensor::from_vec((0..512).map(|i|if i<256{1f32}else{3f32}).collect::<Vec<_>>(),
            (2,1,4,64),&dev)?;
        let ip=Tensor::from_vec(vec![0i32,2],2,&dev)?;
        let ids=Tensor::from_vec(vec![0i32,1],2,&dev)?;
        let last=Tensor::from_vec(vec![3i32],1,&dev)?;
        let direct_r=Tensor::from_vec(vec![0i32],1,&dev)?;
        let direct_t=Tensor::from_vec(vec![0i32],1,&dev)?;
        let direct_o=Tensor::from_vec(vec![0i32,1],2,&dev)?;
        let direct_m=Tensor::from_vec(vec![1u8],1,&dev)?;
        let split_r=Tensor::from_vec(vec![0i32,0,-1],3,&dev)?;
        let split_t=Tensor::from_vec(vec![0i32,1,-1],3,&dev)?;
        let split_o=Tensor::from_vec(vec![0i32,2],2,&dev)?;
        let split_m=Tensor::from_vec(vec![1u8,1,0],3,&dev)?;
        let chunk=Tensor::from_vec(vec![4i32],1,&dev)?;
        let tmp_v=Tensor::zeros((4,1,64),DType::F32,&dev)?;
        let tmp_s=Tensor::zeros((4,1),DType::F32,&dev)?;
        for (window,expected) in [(None,13f32/7.),(Some(2),3.)] {
            for cap in [None,Some(2.)] {
                let direct=flashinfer_decode(&q,&k,&v,DEFAULT_FP8_KV_CACHE_SCALES,&ip,&ids,&last,
                    &direct_r,&direct_t,&direct_o,&chunk,&direct_m,0.125,window,cap,None)?;
                let split=||flashinfer_decode(&q,&k,&v,DEFAULT_FP8_KV_CACHE_SCALES,&ip,&ids,&last,
                    &split_r,&split_t,&split_o,&chunk,&split_m,0.125,window,cap,
                    Some(FlashInferDecodeScratch{tmp_v:&tmp_v,tmp_s:&tmp_s}));
                let actual=split()?.flatten_all()?.to_vec1::<f32>()?;
                let direct=direct.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(actual.len(),64);
                for (&a,&d) in actual.iter().zip(&direct) {
                    assert!(a.is_finite() && d.is_finite());
                    assert!((a-expected).abs()<1e-4 && (d-expected).abs()<1e-4,
                        "window={window:?} cap={cap:?} actual={a} direct={d} expected={expected}");
                }
                let count=PLANS.with(|p|p.borrow().len());
                assert_eq!(split()?.flatten_all()?.to_vec1::<f32>()?,actual);
                assert_eq!(PLANS.with(|p|p.borrow().len()),count);
            }
        }
        Ok(())
    }

    #[test]
    fn dkr_flashinfer_decode_geometry_cpu() {
        let mut eligible=0;
        for dt in [DType::F16,DType::BF16,DType::F32] {
            for fp8 in [false,true] {
                for hd in [64,128,256,512] {
                    for heads in [1,2,3,4,5,6,7,8,16] {
                        let key=Key{dtype:dt,fp8,batch:1,padded:1,heads,kv_heads:1,hd,pages:1,page:16,
                            indices:1,q_span:heads*hd,window:false,softcap:false};
                        if decode_geometry(key,61).is_ok(){eligible+=1;}
                    }
                }
            }
        }
        assert_eq!(eligible,139);
    }
}
