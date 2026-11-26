//! Typed GDN recurrence and causal convolution. Device-specific resources
//! are discovered on cache miss; no host FFI launcher is used.
use super::*;
use crate::cuda::{attention_prep_dynamic::span, kernel_runtime::{kernel_cache,LaunchPlanCache}};
use candle_core::cuda_backend::cudarc::driver::{DevicePtrMut,PushKernelArg,sys::CUdevice_attribute};
use mistralrs_kernel_runtime::{universal::{Argument,ElementType,LaunchRequest},universal_cuda::PrevalidatedLaunchPlan};
use std::cell::RefCell;

#[derive(Clone,Copy,Eq,PartialEq)]
struct Key {
    kind: &'static str,
    variant: &'static str,
    dtype: ElementType,
    spans: [usize;8],
    grid:[u32;3],
    block:[u32;3],
    shared:u32,
    ceiling:u32,
}
thread_local! {
    static PLANS:RefCell<LaunchPlanCache<Key,PrevalidatedLaunchPlan>>=RefCell::new(LaunchPlanCache::new());
    static DUMMY:RefCell<Option<(Device,Tensor)>>=const{RefCell::new(None)};
    static RESOURCE:RefCell<Option<(Device,u32)>>=const{RefCell::new(None)};
}
fn dtype(dt:DType)->Result<ElementType>{
    match dt{DType::F16=>Ok(ElementType::F16),DType::BF16=>Ok(ElementType::Bf16),
        DType::F32=>Ok(ElementType::F32),_=>candle_core::bail!("unsupported GDN typed dtype")}
}
fn capacity(device:&Device)->Result<u32>{
    RESOURCE.with(|cache|{
        let mut cache=cache.borrow_mut();
        if let Some((d,n))=&*cache{if d.same_device(device){return Ok(*n);}}
        let stream=device.as_cuda_device()?.cuda_stream();
        let ctx=stream.context();
        let normal=ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
            .map_err(candle_core::Error::wrap)?;
        let optin=ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .unwrap_or(normal);
        let n=u32::try_from(normal.max(optin)).map_err(candle_core::Error::wrap)?;
        *cache=Some((device.clone(),n));
        Ok(n)
    })
}
fn slot_tensor(device:&Device,slots:GdnStateSlots<'_>,batch:usize)->Result<(Tensor,i32)>{
    match slots{
        GdnStateSlots::Pooled(t)=>{
            if t.dtype()!=DType::U32||!t.is_contiguous()||t.dims()!=[batch]||!t.device().same_device(device){
                candle_core::bail!("invalid GDN slot table");
            }
            Ok((t.clone(),1))
        }
        GdnStateSlots::Gathered=>DUMMY.with(|cache|{
            let mut cache=cache.borrow_mut();
            if let Some((d,t))=&*cache{if d.same_device(device){return Ok((t.clone(),0));}}
            // A real typed allocation, never a fabricated tensor or null pointer.
            // Cold-path setup must finish before CUDA graph capture.
            let t=Tensor::from_vec(vec![0u32],1,device)?;
            *cache=Some((device.clone(),t.clone()));
            Ok((t,0))
        })
    }
}
fn shared(bt:u32,bk:u32)->u32{(bt*bk+bt*bt+2*bt+bk)*4}
fn recurrence_variant(kernel:RecurrenceKernel,k:usize,v:usize,cap:u32)->Result<(&'static str,[u32;3],u32,u32,usize)>{
    if k==0||k>256||v==0{candle_core::bail!("invalid GDN recurrence geometry");}
    let scalar=||match k{64=>("scalar64",[64,1,1],0,0,64),
        128=>("scalar128",[64,1,1],0,0,64),
        _=>("fallback",[64,1,1],(8*k)as u32,2048,64)};
    let chunk=|vm:bool|{
        let large=shared(64,k as u32)<=cap;
        let bt=if large{64}else{32};
        let bytes=shared(bt,k as u32);
        let tag=match (vm,k,large){
            (true,128,true)=>"vmchunk64",(true,128,false)=>"vmchunk32",
            (false,64,true)=>"chunk64k64",(false,64,false)=>"chunk32k64",
            (false,128,true)=>"chunk64k128",(false,128,false)=>"chunk32k128",
            _=>unreachable!(),
        };
        (tag,[64,1,1],bytes,bytes,64)
    };
    let vm=matches!(kernel,RecurrenceKernel::ValueMajorWarp|RecurrenceKernel::ValueMajorWarp2|
        RecurrenceKernel::ValueMajorWarp4|RecurrenceKernel::ValueMajorWarp8|RecurrenceKernel::ValueMajorChunked);
    if vm&&(k!=128||v!=128){candle_core::bail!("value-major GDN requires K=V=128");}
    let result=match kernel{
        RecurrenceKernel::Scalar=>scalar(),
        RecurrenceKernel::Warp=>match k{64=>("warp64",[32,4,1],0,0,4),128=>("warp128",[32,4,1],0,0,4),_=>scalar()},
        RecurrenceKernel::ValueMajorWarp=>("vm1",[32,4,1],0,0,4),
        RecurrenceKernel::ValueMajorWarp2=>("vm2",[32,4,1],0,0,8),
        RecurrenceKernel::ValueMajorWarp4=>("vm4",[32,4,1],0,0,16),
        RecurrenceKernel::ValueMajorWarp8=>("vm8",[32,4,1],0,0,32),
        RecurrenceKernel::Chunked if k==64||k==128=>chunk(false),
        RecurrenceKernel::Chunked=>scalar(),
        RecurrenceKernel::ValueMajorChunked=>chunk(true),
    };
    if result.3>cap{candle_core::bail!("GDN recurrence exceeds shared memory capacity");}
    Ok(result)
}
fn grid(x:usize,y:usize,z:usize)->Result<[u32;3]>{
    if x==0||x>i32::MAX as usize||y==0||y>65535||z==0||z>65535{
        candle_core::bail!("GDN grid exceeds CUDA bounds");
    }
    Ok([x as u32,y as u32,z as u32])
}
fn i32n(n:usize)->Result<i32>{i32::try_from(n).map_err(candle_core::Error::wrap)}
fn mul(a:usize,b:usize)->Result<usize>{a.checked_mul(b).ok_or_else(||candle_core::Error::msg("GDN size overflow"))}
fn prepare(key:Key,device:&Device)->Result<PrevalidatedLaunchPlan>{
    let tensor=|i:usize,dt|Argument::Tensor{dtype:dt,shape:vec![key.spans[i]],contiguous:true,start_offset:0,representation:None};
    let scalar=|dt|Argument::Scalar{dtype:dt};
    let mut args=Vec::new();
    if key.kind=="recurrence"{
        args.extend((0..5).map(|i|tensor(i,ElementType::F32)));
        args.extend([tensor(5,key.dtype),tensor(6,ElementType::F32),tensor(7,ElementType::U32)]);
        args.extend((0..5).map(|_|scalar(ElementType::I32)));
    }else{
        let count=if key.variant=="save"{4}else{5};
        args.extend((0..count-1).map(|i|tensor(i,key.dtype)));
        args.push(tensor(count-1,ElementType::U32));
        let dims=if key.variant.starts_with("update"){3}else{4};
        args.extend((0..dims).map(|_|scalar(ElementType::I32)));
        args.extend((0..3).map(|_|scalar(ElementType::I64)));
        args.push(scalar(ElementType::I32));
    }
    let mut req=LaunchRequest{operation:format!("gdn.{}.{}.v1",key.kind,key.variant),cc:0,args,
        grid:key.grid,max_dynamic_shared_mem_bytes:key.ceiling};
    let p=kernel_cache()?.prepare_for_device(&mut req,device).map_err(|e|candle_core::Error::msg(e.to_string()))?;
    if p.manifest.block!=key.block||p.manifest.shared_mem_bytes!=key.ceiling{
        candle_core::bail!("GDN catalogue launch geometry mismatch");
    }
    if key.shared!=key.ceiling{
        // Runtime K<=256, exactly two K-wide f32 arrays.
        unsafe{p.prevalidate_launch_with_shared(&req,device,key.shared)}
    }else{p.prevalidate_launch_with_optin(&req,device)}
    .map_err(|e|candle_core::Error::msg(e.to_string()))
}
fn launch(key:Key,device:&Device,ptrs:&[u64],dims:&[i32],strides:&[i64],has_slots:Option<i32>)->Result<()>{
    if ptrs.contains(&0)||key.spans.contains(&0){candle_core::bail!("invalid GDN pointer/span");}
    let stream=device.as_cuda_device()?.cuda_stream();
    PLANS.with(|cache|cache.borrow_mut().with_plan(key,device,||prepare(key,device),|p|{
        unsafe{p.with_prevalidated_launch(|fun,cfg|{
            let mut b=stream.launch_builder(fun);
            for ptr in ptrs{b.arg(ptr);}
            for dim in dims{b.arg(dim);}
            for stride in strides{b.arg(stride);}
            if let Some(ref has)=has_slots{b.arg(has);}
            b.launch(cfg)?;Ok(())
        })}.map_err(|e|candle_core::Error::msg(e.to_string()))
    }))
}
pub(super) fn recurrence(kernel:RecurrenceKernel,inputs:RecurrenceInputs<'_>,state:&mut Tensor,slots:GdnStateSlots<'_>)->Result<Tensor>{
    let RecurrenceInputs{q,k,v,g,beta}=inputs;
    let (bh,seq,kd)=q.dims3()?;
    let vd=v.dim(2)?;
    let device=q.device();
    if bh==0||seq==0||k.dims()!=[bh,seq,kd]||v.dims()!=[bh,seq,vd]||
        g.dims()!=[bh,seq]||beta.dims()!=[bh,seq]{
        candle_core::bail!("GDN recurrence input shape mismatch");
    }
    for t in [q,k,v,g,beta]{
        if t.dtype()!=DType::F32||!t.is_contiguous()||!t.device().same_device(device){
            candle_core::bail!("GDN recurrence requires contiguous F32 inputs");
        }
    }
    let batch=match slots{GdnStateSlots::Gathered=>bh,GdnStateSlots::Pooled(t)=>t.dim(0)?};
    if batch==0||bh%batch!=0{candle_core::bail!("GDN slot/head geometry mismatch");}
    let heads=bh/batch;
    // The producer owns slot bounds and uniqueness; no device-to-host slot read.
    let vm=matches!(kernel,RecurrenceKernel::ValueMajorWarp|RecurrenceKernel::ValueMajorWarp2|
        RecurrenceKernel::ValueMajorWarp4|RecurrenceKernel::ValueMajorWarp8|RecurrenceKernel::ValueMajorChunked);
    let tail=if vm{[vd,kd]}else{[kd,vd]};
    let sd=state.dims();
    let per_head=mul(kd,vd)?;
    if sd.len()<3||sd[sd.len()-2..]!=tail||!state.is_contiguous()||!state.device().same_device(device)||
        per_head==0||state.elem_count()%per_head!=0||state.elem_count()/per_head%heads!=0||
        (matches!(slots,GdnStateSlots::Gathered)&&state.elem_count()/per_head!=bh){
        candle_core::bail!("GDN recurrence state layout mismatch");
    }
    let (variant,block,bytes,ceiling,tile)=recurrence_variant(kernel,kd,vd,capacity(device)?)?;
    let key=Key{kind:"recurrence",variant,dtype:dtype(state.dtype())?,
        spans:[q.elem_count(),k.elem_count(),v.elem_count(),g.elem_count(),beta.elem_count(),
            state.elem_count(),v.elem_count(),if matches!(slots,GdnStateSlots::Gathered){1}else{batch}],
        grid:grid(vd.div_ceil(tile),bh,1)?,block,shared:bytes,ceiling};
    let (slot,has)=slot_tensor(device,slots,batch)?;
    let dev=device.as_cuda_device()?;
    let stream=dev.cuda_stream();
    let (qs,ql)=q.storage_and_layout();let (ks,kl)=k.storage_and_layout();
    let (vs,vl)=v.storage_and_layout();let (gs,gl)=g.storage_and_layout();
    let (bs,bl)=beta.storage_and_layout();let (ss,sl)=state.storage_and_layout();
    let (is,il)=slot.storage_and_layout();
    let (qp,_qg)=cuda_read_ptr_with_guard::<f32>(&qs,ql,&stream,"q")?;
    let (kp,_kg)=cuda_read_ptr_with_guard::<f32>(&ks,kl,&stream,"k")?;
    let (vp,_vg)=cuda_read_ptr_with_guard::<f32>(&vs,vl,&stream,"v")?;
    let (gp,_gg)=cuda_read_ptr_with_guard::<f32>(&gs,gl,&stream,"g")?;
    let (bp,_bg)=cuda_read_ptr_with_guard::<f32>(&bs,bl,&stream,"beta")?;
    let (sp,_,_sg)=cuda_recurrent_state_inplace_ptr_with_guard(state.dtype(),&ss,sl,&stream,"state")?;
    let (ip,_ig)=cuda_read_ptr_with_guard::<u32>(&is,il,&stream,"slots")?;
    let mut output=unsafe{dev.alloc::<f32>(v.elem_count())}?;
    let (op,og)=output.device_ptr_mut(&stream);
    launch(key,device,&[qp,kp,vp,gp,bp,sp,op,ip],&[i32n(seq)?,i32n(kd)?,i32n(vd)?,i32n(heads)?,has],&[],None)?;
    drop(og);
    Ok(Tensor::from((candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(output,dev.clone())),(bh,seq,vd))))
}

pub(super) fn convolution(x:&Tensor,weight:&Tensor,state:&Tensor,width:usize,update:bool,slots:GdnStateSlots<'_>)->Result<(Tensor,Tensor)>{
    let (batch,seq,channels)=x.dims3()?;
    if batch==0||seq==0||channels==0||width==0||(update&&seq!=1)||weight.dims()!=[channels,width]{
        candle_core::bail!("GDN convolution geometry mismatch");
    }
    let device=x.device();
    let pooled=matches!(slots,GdnStateSlots::Pooled(_));
    let (capacity,sc,sw)=state.dims3()?;
    if capacity==0||sc!=channels||sw!=width||(!pooled&&capacity!=batch)||
        !state.is_contiguous()||state.dtype()!=x.dtype()||weight.dtype()!=x.dtype()||
        !weight.device().same_device(device)||!state.device().same_device(device){
        candle_core::bail!("GDN convolution state/weight mismatch");
    }
    let weight=weight.contiguous()?;
    let (slot,has)=slot_tensor(device,slots,batch)?;
    let dt=dtype(x.dtype())?;
    let dims=[i32n(batch)?,i32n(channels)?,i32n(seq)?,i32n(width)?];
    let strides=x.stride().iter().map(|&v|i64::try_from(v).map_err(candle_core::Error::wrap)).collect::<Result<Vec<_>>>()?;
    let dev=device.as_cuda_device()?;let stream=dev.cuda_stream();
    let (xs,xl)=x.storage_and_layout();let (ws,wl)=weight.storage_and_layout();
    let (ss,sl)=state.storage_and_layout();let (is,il)=slot.storage_and_layout();
    let xspan=span(xl)?;
    let (ip,_ig)=cuda_read_ptr_with_guard::<u32>(&is,il,&stream,"slots")?;
    let elements=mul(mul(batch,seq)?,channels)?;
    let state_elements=mul(mul(batch,channels)?,width)?;
    macro_rules! run{($ty:ty)=>{{
        let (xp,_xg)=cuda_read_ptr_with_guard::<$ty>(&xs,xl,&stream,"conv input")?;
        let (wp,_wg)=cuda_read_ptr_with_guard::<$ty>(&ws,wl,&stream,"conv weight")?;
        let (sp,_sg)=cuda_inplace_ptr_with_guard::<$ty>(&ss,sl,&stream,"conv state")?;
        let mut output=unsafe{dev.alloc::<$ty>(elements)}?;
        let (op,og)=output.device_ptr_mut(&stream);
        let conv_key=|variant,spans,grid,block|Key{kind:"conv",variant,dtype:dt,spans,grid,block,shared:0,ceiling:0};
        let spans=[xspan,weight.elem_count(),state.elem_count(),elements,slot.elem_count(),1,1,1];
        let next=if update{
            // Vectorized width4 reads 8-byte aligned state/weights.
            let variant=if width==4&&sp%8==0&&wp%8==0{"update4"}else{"update"};
            let key=conv_key(variant,spans,grid(channels.div_ceil(256),batch,1)?,[256,1,1]);
            launch(key,device,&[xp,wp,sp,op,ip],&[dims[0],dims[1],dims[3]],&strides,Some(has))?;
            state.clone()
        }else{
            let tiled=width==4&&strides[2]==1;
            let (variant,gr,block)=if tiled{("full4",grid(channels.div_ceil(128),seq.div_ceil(64),batch)?,[128,1,1])}
                else{("full",grid(mul(channels,seq)?.div_ceil(256),batch,1)?,[256,1,1])};
            let key=conv_key(variant,spans,gr,block);
            let mut next=if pooled{None}else{Some(unsafe{dev.alloc::<$ty>(state_elements)}?)};
            let save=|np:u64|->Result<()>{
                let key=conv_key("save",[xspan,state.elem_count(),if pooled{state.elem_count()}else{state_elements},slot.elem_count(),1,1,1,1],
                    grid(channels.div_ceil(256),batch,1)?,[256,1,1]);
                launch(key,device,&[xp,sp,np,ip],&dims,&strides,Some(has))
            };
            // Convolution reads old state before the second launch mutates it.
            launch(key,device,&[xp,wp,sp,op,ip],&dims,&strides,Some(has))?;
            if let Some(buf)=next.as_mut(){let (np,ng)=buf.device_ptr_mut(&stream);save(np)?;drop(ng);}
            else{save(sp)?;}
            match next{Some(buf)=>Tensor::from((candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(buf,dev.clone())),(batch,channels,width))),
                None=>state.clone()}
        };
        drop(og);
        Ok((Tensor::from((candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(output,dev.clone())),(batch,seq,channels))),next))
    }}}
    match x.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),_=>candle_core::bail!("unsupported GDN conv dtype")}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_partial_tiles_recurrence_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let seq=65usize;
        for kd in [17usize,64,128]{
            let q=Tensor::from_vec((0..seq*kd).map(|i|if i%kd==kd-1{1f32}else{0f32}).collect::<Vec<_>>(),(1,seq,kd),&dev)?;
            let v=Tensor::ones((1,seq,65),DType::F32,&dev)?;
            let g=Tensor::zeros((1,seq),DType::F32,&dev)?;
            let beta=Tensor::ones((1,seq),DType::F32,&dev)?.affine(0.5,0.)?;
            let inputs=RecurrenceInputs{q:&q,k:&q,v:&v,g:&g,beta:&beta};
            for kernel in [RecurrenceKernel::Scalar,RecurrenceKernel::Warp,RecurrenceKernel::Chunked]{
                for dt in [DType::F16,DType::BF16,DType::F32]{
                    let mut state=Tensor::zeros((1,kd,65),dt,&dev)?;
                    let out=recurrence(kernel,inputs,&mut state,GdnStateSlots::Gathered)?;
                    let rows=out.to_vec3::<f32>()?;
                    for (t,row) in rows[0].iter().enumerate(){
                        let want=1.0-0.5f32.powi((t+1) as i32);
                        assert!(row.iter().all(|&x|(x-want).abs()<2e-4),"{kernel:?} kd={kd} t={t}");
                    }
                    let data=state.to_dtype(DType::F32)?.to_vec3::<f32>()?;
                    for (k,row) in data[0].iter().enumerate(){
                        let want=if k==kd-1{1.0}else{0.0};
                        assert!(row.iter().all(|&x|(x-want).abs()<2e-4),"{kernel:?} state kd={kd}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_pooled_padding_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        let q=Tensor::from_vec((0..3*17).map(|i|if i%17==16{1f32}else{0f32}).collect::<Vec<_>>(),(3,1,17),&dev)?;
        let v=Tensor::ones((3,1,65),DType::F32,&dev)?;
        let g=Tensor::zeros((3,1),DType::F32,&dev)?;
        let beta=Tensor::ones((3,1),DType::F32,&dev)?;
        let slots=Tensor::from_vec(vec![1u32,u32::MAX,0],3,&dev)?;
        let mut state=Tensor::zeros((3,1,17,65),DType::F32,&dev)?;
        let inputs=RecurrenceInputs{q:&q,k:&q,v:&v,g:&g,beta:&beta};
        let out=recurrence(RecurrenceKernel::Scalar,inputs,&mut state,GdnStateSlots::Pooled(&slots))?;
        let rows=out.to_vec3::<f32>()?;
        assert!(rows[0][0].iter().all(|&x|x==1.));
        assert!(rows[1][0].iter().all(|&x|x==0.));
        assert!(rows[2][0].iter().all(|&x|x==1.));
        let state=state.reshape((3,17,65))?.to_vec3::<f32>()?;
        assert!(state[0][16].iter().all(|&x|x==1.));
        assert!(state[1][16].iter().all(|&x|x==1.));
        assert!(state[2].iter().flatten().all(|&x|x==0.));
        Ok(())
    }

    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_causal_conv_full_vs_update_gpu()->Result<()>{
        let _evidence=mistralrs_kernel_runtime::RuntimeEvidenceGuard;
        let dev=Device::new_cuda(0)?;
        for dt in [DType::F16,DType::BF16]{
            for width in [3usize,4]{
                for seq in [2usize,65]{
                    let x=Tensor::from_vec((0..2*seq*9).map(|i|((i%11)as f32-5.)/16.).collect::<Vec<_>>(),(2,seq,9),&dev)?.to_dtype(dt)?;
                    let w=Tensor::ones((9,width),dt,&dev)?.affine(0.25,0.)?;
                    let initial=Tensor::zeros((2,9,width),dt,&dev)?;
                    let (full,final_state)=convolution(&x,&w,&initial,width,false,GdnStateSlots::Gathered)?;
                    let mut state=Tensor::zeros((2,9,width),dt,&dev)?;
                    let mut outputs=Vec::new();
                    for t in 0..seq{
                        let (o,next)=convolution(&x.narrow(1,t,1)?,&w,&state,width,true,GdnStateSlots::Gathered)?;
                        outputs.push(o);state=next;
                    }
                    let refs=outputs.iter().collect::<Vec<_>>();
                    let step=Tensor::cat(&refs,1)?;
                    let err=(full.to_dtype(DType::F32)?-step.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(err<0.004,"dt={dt:?} width={width} seq={seq} err={err}");
                    let err=(state.to_dtype(DType::F32)?-final_state.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert_eq!(err,0.);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn dkr_gdn_resource_policy(){
        let a=recurrence_variant(RecurrenceKernel::Chunked,128,65,49152).unwrap();
        assert_eq!((a.0,a.2),("chunk32k128",21248));
        let b=recurrence_variant(RecurrenceKernel::Chunked,128,65,98304).unwrap();
        assert_eq!((b.0,b.2),("chunk64k128",50176));
        assert_eq!(recurrence_variant(RecurrenceKernel::Chunked,64,65,49152).unwrap().0,"chunk64k64");
        assert_eq!(recurrence_variant(RecurrenceKernel::ValueMajorChunked,128,128,49152).unwrap().0,"vmchunk32");
        assert!(recurrence_variant(RecurrenceKernel::Scalar,257,1,49152).is_err());
        assert!(recurrence_variant(RecurrenceKernel::ValueMajorWarp,64,128,49152).is_err());
    }
}
