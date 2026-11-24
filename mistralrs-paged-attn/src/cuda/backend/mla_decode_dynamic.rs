//! Typed nonpartitioned FlashInfer MLA decode; same device algorithm as legacy.
use super::slice_ptr_on_stream;
use candle_core::{cuda::cudarc::driver::PushKernelArg,DType,Device,Result,Storage,Tensor};
use mistralrs_kernel_runtime::{
    launch_plan_cache::LaunchPlanCache,
    universal::{Argument,ElementType,LaunchRequest},
    universal_cuda::{KernelExecutionCache,PrevalidatedLaunchPlan},
};
use std::{cell::RefCell,sync::OnceLock};

#[derive(Clone,Copy,Eq,PartialEq)]
struct Key {dtype:DType,batch:usize,heads:usize,blocks:usize,page:usize,indices:usize}
struct Plan {launch:PrevalidatedLaunchPlan,magic:u32,shift:u32,add:u32}
thread_local! {
    static PLANS:RefCell<LaunchPlanCache<Key,Plan>>=RefCell::new(LaunchPlanCache::new());
}
static CATALOG:OnceLock<std::result::Result<KernelExecutionCache,String>>=OnceLock::new();

fn catalog()->Result<&'static KernelExecutionCache>{
    CATALOG.get_or_init(||KernelExecutionCache::package_root()
        .ok_or_else(||"DKR MLA decode catalogue unavailable".to_owned())
        .and_then(|p|KernelExecutionCache::discover(&p).map_err(|e|e.to_string())))
        .as_ref().map_err(|e|candle_core::Error::Msg(e.clone()))
}

use super::flashinfer_fastdiv::fastdiv;
fn arg(dtype:ElementType,shape:Vec<usize>,repr:Option<&str>)->Argument{
    Argument::Tensor{dtype,shape,contiguous:true,start_offset:0,representation:repr.map(str::to_owned)}
}
fn prepare(k:Key,dev:&Device)->Result<Plan>{
    let dt=match k.dtype{DType::F16=>ElementType::F16,DType::BF16=>ElementType::Bf16,
        DType::F32=>ElementType::F32,_=>candle_core::bail!("unsupported MLA decode dtype")};
    let mut args=vec![arg(dt,vec![k.batch,k.heads,512],None),arg(dt,vec![k.batch,k.heads,64],None),
        arg(dt,vec![k.blocks,k.page,512],Some("mla.ckv.paged.v1")),
        arg(dt,vec![k.blocks,k.page,64],Some("mla.kpe.paged.v1"))];
    for n in [k.batch+1,k.indices,k.batch,k.batch,k.batch,k.batch+1,1]{
        args.push(arg(ElementType::I32,vec![n],None));
    }
    args.push(arg(dt,vec![k.batch,k.heads,512],None));
    args.extend((0..6).map(|_|Argument::Scalar{dtype:ElementType::U32}));
    args.push(Argument::Scalar{dtype:ElementType::F32});
    let mut req=LaunchRequest{operation:"attention.mla.decode.ckv512.kpe64.v1".to_owned(),
        cc:0,args,grid:[k.batch as u32,k.heads.div_ceil(16) as u32,1],max_dynamic_shared_mem_bytes:48*1024};
    let p=catalog()?.prepare_for_device(&mut req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=[32,8,1] || p.manifest.shared_mem_bytes==0{
        candle_core::bail!("DKR MLA decode launch geometry mismatch");
    }
    let launch=p.prevalidate_launch(&req,dev).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    let (magic,shift,add)=fastdiv(k.page as u32);
    if shift>=32{candle_core::bail!("DKR MLA decode unsupported page divider");}
    Ok(Plan{launch,magic,shift,add})
}

// The historical API represents one full-context CTA group per request.
// No scratch buffers are exposed; split-KV is deliberately not enabled.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_mla_decode(q_nope:&Tensor,q_pe:&Tensor,ckv_cache:&Tensor,kpe_cache:&Tensor,
    paged_kv_indptr:&Tensor,paged_kv_indices:&Tensor,paged_kv_last_page_len:&Tensor,
    request_indices:&Tensor,kv_tile_indices:&Tensor,o_indptr:&Tensor,kv_chunk_size:&Tensor,
    sm_scale:f32)->Result<Tensor>{
    let dev=q_nope.device();let dtype=q_nope.dtype();
    let (batch,heads,rank)=q_nope.dims3()?;
    let (blocks,page,cache_rank)=ckv_cache.dims3()?;
    let indices=paged_kv_indices.dims1()?;
    if !dev.is_cuda() || !matches!(dtype,DType::F16|DType::BF16|DType::F32) || !sm_scale.is_finite()
        || rank!=512 || cache_rank!=512 || q_pe.dims()!=[batch,heads,64] ||
        kpe_cache.dims()!=[blocks,page,64] || batch==0 || heads==0 || heads.div_ceil(16)>65535
        || blocks==0 || page==0 || page>65536 || indices==0 || indices>i32::MAX as usize
        || indices.checked_mul(page).is_none_or(|n|n>i32::MAX as usize-256){
        candle_core::bail!("DKR MLA decode incompatible geometry");
    }
    for t in [q_nope,q_pe,ckv_cache,kpe_cache]{
        // Historical vector loads require aligned addresses and dense rows.
        let (_,l)=t.storage_and_layout();
        if !dev.same_device(t.device()) || t.dtype()!=dtype || !t.is_contiguous() ||
            t.elem_count()>i32::MAX as usize || l.start_offset()%(16/dtype.size_in_bytes())!=0{
            candle_core::bail!("DKR MLA decode requires aligned contiguous same-device tensors");
        }
    }
    for (t,n) in [(paged_kv_indptr,batch+1),(paged_kv_indices,indices),(paged_kv_last_page_len,batch),
        (request_indices,batch),(kv_tile_indices,batch),(o_indptr,batch+1),(kv_chunk_size,1)]{
        if t.dtype()!=DType::I32 || t.dims()!=[n] || !t.is_contiguous() || !dev.same_device(t.device()){
            candle_core::bail!("DKR MLA decode requires contiguous I32 planner metadata on the same device");
        }
    }
    // Valid page IDs, monotone indptr, nonempty contexts and valid request
    // mapping remain planner contracts, as for the historical device body.
    let key=Key{dtype,batch,heads,blocks,page,indices};
    let output=Tensor::zeros((batch,heads,512),dtype,dev)?;
    {
        let (qs,ql)=q_nope.storage_and_layout();let (ps,pl)=q_pe.storage_and_layout();
        let (cs,cl)=ckv_cache.storage_and_layout();let (ks,kl)=kpe_cache.storage_and_layout();
        let (is,il)=paged_kv_indptr.storage_and_layout();let (ns,nl)=paged_kv_indices.storage_and_layout();
        let (ls,ll)=paged_kv_last_page_len.storage_and_layout();let (rs,rl)=request_indices.storage_and_layout();
        let (ts,tl)=kv_tile_indices.storage_and_layout();let (os,ol)=o_indptr.storage_and_layout();
        let (hs,hl)=kv_chunk_size.storage_and_layout();let (ys,yl)=output.storage_and_layout();
        let (Storage::Cuda(qs),Storage::Cuda(ps),Storage::Cuda(cs),Storage::Cuda(ks),
            Storage::Cuda(is),Storage::Cuda(ns),Storage::Cuda(ls),Storage::Cuda(rs),
            Storage::Cuda(ts),Storage::Cuda(os),Storage::Cuda(hs),Storage::Cuda(ys))=
            (&*qs,&*ps,&*cs,&*ks,&*is,&*ns,&*ls,&*rs,&*ts,&*os,&*hs,&*ys)
            else{candle_core::bail!("DKR MLA decode requires CUDA storage");};
        let stream=dev.as_cuda_device()?.cuda_stream();
        let (ip,_ig)=slice_ptr_on_stream(is.as_cuda_slice::<i32>()?,il.start_offset(),&stream);
        let (np,_ng)=slice_ptr_on_stream(ns.as_cuda_slice::<i32>()?,nl.start_offset(),&stream);
        let (lp,_lg)=slice_ptr_on_stream(ls.as_cuda_slice::<i32>()?,ll.start_offset(),&stream);
        let (rp,_rg)=slice_ptr_on_stream(rs.as_cuda_slice::<i32>()?,rl.start_offset(),&stream);
        let (tp,_tg)=slice_ptr_on_stream(ts.as_cuda_slice::<i32>()?,tl.start_offset(),&stream);
        let (op,_og)=slice_ptr_on_stream(os.as_cuda_slice::<i32>()?,ol.start_offset(),&stream);
        let (hp,_hg)=slice_ptr_on_stream(hs.as_cuda_slice::<i32>()?,hl.start_offset(),&stream);
        let (batch,heads,page)=(batch as u32,heads as u32,page as u32);
        macro_rules! run {($ty:ty)=>{{
            let (qp,_qg)=slice_ptr_on_stream(qs.as_cuda_slice::<$ty>()?,ql.start_offset(),&stream);
            let (pp,_pg)=slice_ptr_on_stream(ps.as_cuda_slice::<$ty>()?,pl.start_offset(),&stream);
            let (cp,_cg)=slice_ptr_on_stream(cs.as_cuda_slice::<$ty>()?,cl.start_offset(),&stream);
            let (kp,_kg)=slice_ptr_on_stream(ks.as_cuda_slice::<$ty>()?,kl.start_offset(),&stream);
            let (yp,_yg)=slice_ptr_on_stream(ys.as_cuda_slice::<$ty>()?,yl.start_offset(),&stream);
            PLANS.with(|cache|cache.borrow_mut().with_plan(key,dev,||prepare(key,dev),|plan|{
                unsafe{plan.launch.with_prevalidated_launch(|fun,config|{
                    let mut l=stream.launch_builder(fun);
                    l.arg(&qp).arg(&pp).arg(&cp).arg(&kp).arg(&ip).arg(&np).arg(&lp)
                        .arg(&rp).arg(&tp).arg(&op).arg(&hp).arg(&yp)
                        .arg(&batch).arg(&heads).arg(&page).arg(&plan.magic)
                        .arg(&plan.shift).arg(&plan.add).arg(&sm_scale);
                    l.launch(config)?;Ok(())
                })}.map_err(|e|candle_core::Error::msg(e.to_string()))
            }))
        }}}
        match dtype{DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),_=>unreachable!()}?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dkr_mla_fastdiv_cpu(){
        // Covers the entire admitted range of divisors, plus dense samples
        // near page boundaries and the maximum supported packed offset.
        for d in 1..=65536u32{
            let (m,s,a)=fastdiv(d);
            assert!(s<32);
            for n in [0,1,d-1,d,d+1,2*d-1,2*d,65535,1234567,0x7fff_feff]{
                let q=if d==1{n}else{((((m as u64*n as u64)>>32) as u32)
                    .wrapping_add(a.wrapping_mul(n)))>>s};
                assert_eq!(q,n/d,"d={d}, n={n}");
                assert_eq!(n-q*d,n%d);
            }
        }
    }

    #[test]
    #[ignore="requires staged mla-decode CUBIN and physical CUDA"]
    fn dkr_mla_decode_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let (batch,heads,blocks)=(2usize,17usize,3usize);
        let qnval=|b:usize,h:usize,d:usize|->f32{
            if d==0 {(if h%2==0{1.0}else{-1.0})*(1.0+b as f32*0.25)}
            else if d==31 {0.5} else {0.0}
        };
        let qpval=|b:usize,h:usize,d:usize|->f32{
            if d==0 {0.25*(1+(b+h)%3) as f32}else if d==63{-0.5}else{0.0}
        };
        let ckval=|p:usize,t:usize,d:usize|->f32{(((p*3+t*5+d)%13) as f32-6.0)*0.125};
        let pkval=|p:usize,t:usize,d:usize|->f32{(((p+t*3+d)%9) as f32-4.0)*0.25};
        let qn=(0..batch*heads*512).map(|i|qnval(i/(heads*512),i/512%heads,i%512)).collect::<Vec<_>>();
        let qp=(0..batch*heads*64).map(|i|qpval(i/(heads*64),i/64%heads,i%64)).collect::<Vec<_>>();
        for dtype in [DType::F16,DType::BF16,DType::F32]{
            let qn=Tensor::from_vec(qn.clone(),(batch,heads,512),&dev)?.to_dtype(dtype)?;
            let qp=Tensor::from_vec(qp.clone(),(batch,heads,64),&dev)?.to_dtype(dtype)?;
            for page in [7usize,8,16]{
                let ck=Tensor::from_vec((0..blocks*page*512).map(|i|ckval(i/(page*512),i/512%page,i%512))
                    .collect::<Vec<_>>(),(blocks,page,512),&dev)?.to_dtype(dtype)?;
                let pk=Tensor::from_vec((0..blocks*page*64).map(|i|pkval(i/(page*64),i/64%page,i%64))
                    .collect::<Vec<_>>(),(blocks,page,64),&dev)?.to_dtype(dtype)?;
                let indptr=Tensor::from_vec(vec![0i32,2,3],3,&dev)?;
                let indices=Tensor::from_vec(vec![2i32,0,1],3,&dev)?;
                let last=Tensor::from_vec(vec![3i32,(page-2) as i32],2,&dev)?;
                let tile=Tensor::from_vec(vec![0i32,0],2,&dev)?;
                let outptr=Tensor::from_vec(vec![0i32,1,2],3,&dev)?;
                let chunk=Tensor::from_vec(vec![(2*page) as i32],1,&dev)?;
                for mapping in [[0usize,1],[1,0]]{
                    let requests=Tensor::from_vec(mapping.map(|x|x as i32).to_vec(),batch,&dev)?;
                    for scale in [0.125f32,0.5]{
                        let run=||super::super::flashinfer_mla_decode(
                            &qn,&qp,&ck,&pk,&indptr,&indices,&last,&requests,&tile,&outptr,&chunk,scale);
                        let out=run()?;
                        assert_eq!(out.dims(),[batch,heads,512]);
                        assert_eq!(out.dtype(),dtype);
                        let actual=out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        let mut max_err=0f64;
                        for (out_b,&b) in mapping.iter().enumerate(){
                            let len=if b==0 {page+3}else{page-2};
                            let physical=|t:usize|if b==0 {if t<page {2}else{0}}else{1};
                            for h in 0..heads{
                                let logits=(0..len).map(|t|{
                                    let p=physical(t);let s=t%page;
                                    let a=qnval(b,h,0) as f64*ckval(p,s,0) as f64+
                                          qnval(b,h,31) as f64*ckval(p,s,31) as f64+
                                          qpval(b,h,0) as f64*pkval(p,s,0) as f64+
                                          qpval(b,h,63) as f64*pkval(p,s,63) as f64;
                                    a*scale as f64
                                }).collect::<Vec<_>>();
                                let m=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
                                let w=logits.iter().map(|v|(v-m).exp()).collect::<Vec<_>>();
                                let den=w.iter().sum::<f64>();
                                for d in 0..512{
                                    let want=w.iter().enumerate().map(|(t,w)|w*ckval(physical(t),t%page,d) as f64)
                                        .sum::<f64>()/den;
                                    let got=actual[(out_b*heads+h)*512+d] as f64;
                                    assert!(got.is_finite());
                                    max_err=max_err.max((got-want).abs());
                                }
                            }
                        }
                        let tol=match dtype{DType::F32=>5e-5,DType::F16=>0.001,_=>0.006};
                        assert!(max_err<=tol,"MLA decode dtype={dtype:?} page={page} mapping={mapping:?} scale={scale} err={max_err}");
                        let count=PLANS.with(|c|c.borrow().len());
                        assert!(count>0);
                        let again=run()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                        assert_eq!(actual,again);
                        assert_eq!(count,PLANS.with(|c|c.borrow().len()));
                        eprintln!("DKR_R126_MLA_DECODE=PASS dtype={dtype:?} page={page} mapping={mapping:?} scale={scale} max_abs={max_err:.8}");
                    }
                }
            }
        }
        Ok(())
    }
}
