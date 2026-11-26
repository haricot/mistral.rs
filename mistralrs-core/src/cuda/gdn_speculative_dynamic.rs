//! Typed GDN speculative launches. Only scalar dimensions/strides are packed.
//! All CUDA tensor addresses remain individual ABI parameters.
use super::*;
use super::preparation_dynamic as p;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use std::sync::Arc;

#[derive(Clone,Copy)]
pub(super) struct Bound {
    pub ptr:u64,pub span:usize,pub dtype:p::E,pub write:bool,
}
impl Bound{
    pub(super) fn new(ptr:u64,span:usize,dtype:p::E,write:bool)->Self{Self{ptr,span,dtype,write}}
    fn tuple(self)->(u64,usize,p::E){(self.ptr,self.span,self.dtype)}
    fn end(self)->Result<u64>{
        self.ptr.checked_add(p::mul(self.span,self.dtype.size_bytes())? as u64)
            .ok_or_else(||candle_core::Error::msg("GDN speculative address range overflow"))
    }
}
fn check_bindings(bindings:&[Bound])->Result<()>{
    for (i,a) in bindings.iter().copied().enumerate(){
        if a.ptr==0||a.span==0||a.ptr%(a.dtype.size_bytes() as u64)!=0{candle_core::bail!("invalid GDN speculative pointer");}
        let ae=a.end()?;
        for b in bindings[i+1..].iter().copied(){
            if (a.write||b.write)&&a.ptr<b.end()?&&b.ptr<ae{
                candle_core::bail!("GDN speculative writable tensor overlaps another binding");
            }
        }
    }
    Ok(())
}
/// Nested InplaceOp1 callbacks need distinct storage locks, even if two
/// tensor views occupy disjoint subranges of a single allocation.
pub(super) fn reject_storage_aliases(inputs:&[(&Tensor,bool)])->Result<()>{
    let layouts=inputs.iter().map(|(t,_)|t.storage_and_layout()).collect::<Vec<_>>();
    for i in 0..inputs.len(){
        for j in i+1..inputs.len(){
            if (inputs[i].1||inputs[j].1)&&std::ptr::eq::<candle_core::Storage>(&*layouts[i].0,&*layouts[j].0){
                candle_core::bail!("GDN deferred writable storage is shared with another binding");
            }
        }
    }
    Ok(())
}

/// Collect tensor storage and same-stream guards before exposing addresses.
/// Mutable tensors use the existing stream-owned GDN in-place contract.
pub(super) fn with_bindings<R>(device:&Device,stream:&CudaStream,inputs:&[(&Tensor,bool)],run:impl FnOnce(&[Bound])->Result<R>)->Result<R>{
    for (t,write) in inputs{
        if !t.device().same_device(device)||(*write&&!t.is_contiguous()){
            candle_core::bail!("GDN speculative tensor device/layout mismatch");
        }
    }
    let layouts=inputs.iter().map(|(t,_)|t.storage_and_layout()).collect::<Vec<_>>();
    let mut guards=Vec::with_capacity(inputs.len());
    let mut bound=Vec::with_capacity(inputs.len());
    for ((tensor,write),(storage,layout)) in inputs.iter().zip(&layouts){
        macro_rules! ptr{($ty:ty)=>{{
            if *write{cuda_inplace_ptr_with_guard::<$ty>(storage,layout,stream,"speculative writable tensor")?}
            else{cuda_read_ptr_with_guard::<$ty>(storage,layout,stream,"speculative input tensor")?}
        }}}
        let (addr,guard)=match tensor.dtype(){
            DType::F16=>ptr!(half::f16),DType::BF16=>ptr!(half::bf16),
            DType::F32=>ptr!(f32),DType::U32=>ptr!(u32),
            _=>candle_core::bail!("unsupported GDN speculative tensor dtype"),
        };
        let dtype=if tensor.dtype()==DType::U32{p::E::U32}else{p::dt(tensor.dtype())?};
        bound.push(Bound::new(addr,p::span(layout)?,dtype,*write));
        guards.push(guard);
    }
    check_bindings(&bound)?;
    let result=run(&bound);
    drop(guards);
    result
}

pub(super) fn pack(a:usize,b:usize)->Result<p::S>{
    let a=u32::try_from(a).map_err(candle_core::Error::wrap)?;
    let b=u32::try_from(b).map_err(candle_core::Error::wrap)?;
    Ok(p::S::U64(u64::from(a)|(u64::from(b)<<32)))
}
fn checked_dims(values:&[usize])->Result<()>{
    for &v in values{p::n(v)?;}
    Ok(())
}
fn count(v:&[usize])->Result<usize>{
    v.iter().try_fold(1usize,|n,&x|p::mul(n,x))
}
#[derive(Clone,Copy)]
pub(super) struct Geometry{
    pub batch:usize,pub seq:usize,pub kh:usize,pub vh:usize,pub kd:usize,pub vd:usize,
    pub lanes:usize,pub cap:usize,pub tiled:bool,pub vm:bool,
}
impl Geometry{
    fn validate(self)->Result<()>{
        let Self{batch,seq,kh,vh,kd,vd,lanes,cap,..}=self;
        checked_dims(&[batch,seq,kh,vh,kd,vd,lanes,cap])?;
        if batch==0||seq==0||kh==0||vh==0||vh%kh!=0||kd==0||kd>256||vd==0||lanes==0||
            cap==0||cap%lanes!=0{candle_core::bail!("invalid GDN speculative geometry");}
        let width=count(&[2,kh,kd])?.checked_add(count(&[vh,vd])?)
            .ok_or_else(||candle_core::Error::msg("GDN speculative width overflow"))?;
        p::n(width)?;p::n(count(&[batch,seq,vh])?)?;
        count(&[cap,vh,kd,vd])?;
        Ok(())
    }
}
#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub(super) enum Mode{Checkpoint,Batch,Direct}
#[derive(Clone,Copy)]
pub(super) struct Post{
    pub mode:Mode,pub gate_strides:[usize;4],pub eps:f32,pub sigmoid:bool,
    pub fp8:Option<(usize,i32)>,pub pending:Option<(usize,usize)>,
}
fn fused_operation(mode:Mode,fp8:bool,paired:bool)->&'static str{
    match (mode,fp8,paired){
        (Mode::Checkpoint,false,true)=>"gdn.spec.fused.checkpoint.native.paired.v1",
        (Mode::Checkpoint,false,false)=>"gdn.spec.fused.checkpoint.native.serial.v1",
        (Mode::Checkpoint,true,true)=>"gdn.spec.fused.checkpoint.fp8.paired.v1",
        (Mode::Checkpoint,true,false)=>"gdn.spec.fused.checkpoint.fp8.serial.v1",
        (Mode::Batch,false,true)=>"gdn.spec.fused.batch.native.paired.v1",
        (Mode::Batch,false,false)=>"gdn.spec.fused.batch.native.serial.v1",
        (Mode::Batch,true,true)=>"gdn.spec.fused.batch.fp8.paired.v1",
        (Mode::Batch,true,false)=>"gdn.spec.fused.batch.fp8.serial.v1",
        (Mode::Direct,false,true)=>"gdn.spec.fused.direct.native.paired.v1",
        (Mode::Direct,false,false)=>"gdn.spec.fused.direct.native.serial.v1",
        (Mode::Direct,true,true)=>"gdn.spec.fused.direct.fp8.paired.v1",
        (Mode::Direct,true,false)=>"gdn.spec.fused.direct.fp8.serial.v1",
    }
}
fn scale_arg(scale:(usize,i32),rows:usize)->Result<p::S>{
    let (stride,layout)=scale;
    if stride<rows||!matches!(layout,0|1){candle_core::bail!("invalid GDN FP8 scale contract");}
    p::n(stride)?;
    pack(stride,layout as usize)
}
pub(super) fn launch_recurrence(device:&Device,stream:&Arc<CudaStream>,g:Geometry,b:[usize;3],a:[usize;3],
    post:Option<Post>,bindings:&[Bound])->Result<()>{
    g.validate()?;
    check_bindings(bindings)?;
    let Geometry{batch,seq,kh,vh,kd,vd,lanes,cap,tiled,vm}=g;
    let ptrs=bindings.iter().copied().map(Bound::tuple).collect::<Vec<_>>();
    let (op,grid,block,args)=if let Some(post)=post{
        if !vm||kd!=128||vd!=128||seq>8||!post.eps.is_finite()||post.eps<0.||
            (post.mode==Mode::Checkpoint&&seq>lanes)||(post.mode==Mode::Direct&&lanes!=1)||
            (post.mode==Mode::Direct)!=post.pending.is_some(){
            candle_core::bail!("invalid fused GDN speculative geometry");
        }
        if bindings.len()!=10+usize::from(post.fp8.is_some())+match post.mode{Mode::Checkpoint=>0,Mode::Batch=>3,Mode::Direct=>7}{
            candle_core::bail!("invalid fused GDN binding arity");
        }
        // Vectorized state copies in the shared body require 16-byte alignment.
        if bindings[5].ptr%16!=0{candle_core::bail!("unaligned fused GDN state");}
        let nblocks=p::mul(batch,vh)?;
        let paired=bindings[5].dtype!=p::E::F32||nblocks<=144||nblocks>=768;
        let s=post.gate_strides;
        let mut args=vec![pack(b[0],b[1])?,pack(b[2],a[0])?,pack(a[1],a[2])?,
            pack(s[0],s[1])?,pack(s[2],s[3])?,pack(batch,seq)?,pack(kh,vh)?,
            pack(lanes,usize::from(tiled))?,p::n(cap)?,p::S::F32(post.eps),p::S::I32(i32::from(post.sigmoid))];
        if let Some(scale)=post.fp8{args.push(scale_arg(scale,p::mul(batch,seq)?)?);}
        if let Some((rows,capacity))=post.pending{
            if rows<seq||rows>8||capacity!=cap{candle_core::bail!("invalid direct GDN pending dimensions");}
            args.push(pack(rows,capacity)?);
        }
        (fused_operation(post.mode,post.fp8.is_some(),paired),p::grid(batch,vh)?,[256,1,1],args)
    }else{
        if seq>lanes||bindings.len()!=8{candle_core::bail!("invalid plain GDN checkpoint ABI");}
        let mut args=b.into_iter().chain(a).map(p::stride).collect::<Result<Vec<_>>>()?;
        args.extend([p::n(batch)?,p::n(seq)?,p::n(kh)?,p::n(vh)?,p::n(kd)?,p::n(vd)?,
            p::n(lanes)?,p::S::I32(i32::from(tiled)),p::n(cap)?]);
        let (op,block)=if vm&&kd==128&&vd==128{("gdn.spec.plain.vm128.v1",[32,4,1])}
            else if vm{("gdn.spec.plain.vm.v1",[32,1,1])}else{("gdn.spec.plain.km.v1",[32,1,1])};
        (op,p::grid(vd.div_ceil(32),p::mul(batch,vh)?)?,block,args)
    };
    unsafe{p::launch_on_stream(device,stream,op,&ptrs,&args,grid,block,0,0)}
}

pub(super) fn launch_deferred(device:&Device,stream:&Arc<CudaStream>,bindings:&[Bound],g:Geometry,
    strides:[usize;7],eps:f32,sigmoid:bool,fp8:Option<(usize,i32)>)->Result<()>{
    g.validate()?;
    if g.seq!=1||g.kd!=128||g.vd!=128||!g.vm||g.lanes!=1||!eps.is_finite()||eps<0.||
        bindings.len()!=14+usize::from(fp8.is_some())||bindings[5].ptr%16!=0{
        candle_core::bail!("invalid deferred GDN decode ABI");
    }
    check_bindings(bindings)?;
    let ptrs=bindings.iter().copied().map(Bound::tuple).collect::<Vec<_>>();
    let mut args=strides.into_iter().map(p::stride).collect::<Result<Vec<_>>>()?;
    args.extend([p::n(g.batch)?,p::n(g.cap)?,p::n(g.kh)?,p::n(g.vh)?,p::S::I32(i32::from(g.tiled)),
        p::S::F32(eps),p::S::I32(i32::from(sigmoid))]);
    if let Some(scale)=fp8{args.push(scale_arg(scale,g.batch)?);}
    let name=if fp8.is_some(){"gdn.spec.deferred.fp8.v1"}else{"gdn.spec.deferred.native.v1"};
    let grid=p::grid(g.batch,g.vh)?;
    let finalize_grid=p::grid(g.batch.div_ceil(256),1)?;
    let finalize=[bindings[12].tuple(),bindings[6].tuple()];
    let finalize_args=[p::n(g.batch)?,p::n(g.cap)?];
    p::preflight(device,name,&ptrs,&args,grid,[256,1,1])?;
    p::preflight(device,"gdn.spec.deferred.advance.v1",&finalize,&finalize_args,finalize_grid,[256,1,1])?;
    unsafe{
        p::launch_on_stream(device,stream,name,&ptrs,&args,grid,[256,1,1],0,0)?;
        p::launch_on_stream(device,stream,"gdn.spec.deferred.advance.v1",&finalize,&finalize_args,finalize_grid,[256,1,1],0,0)
    }
}
pub(super) fn launch_flush(device:&Device,stream:&Arc<CudaStream>,bindings:&[Bound],g:Geometry)->Result<()>{
    g.validate()?;
    if g.seq!=1||g.kd!=128||g.vd!=128||!g.vm||g.lanes!=1||bindings.len()!=6||bindings[0].ptr%16!=0{
        candle_core::bail!("invalid GDN deferred flush ABI");
    }
    check_bindings(bindings)?;
    let ptrs=bindings.iter().copied().map(Bound::tuple).collect::<Vec<_>>();
    let args=[p::n(g.batch)?,p::n(g.cap)?,p::n(g.kh)?,p::n(g.vh)?,p::S::I32(i32::from(g.tiled))];
    let grid=p::grid(g.batch,g.vh)?;
    let finalize_grid=p::grid(g.batch.div_ceil(256),1)?;
    let finalize=[bindings[5].tuple(),bindings[1].tuple()];
    let finalize_args=[p::n(g.batch)?,p::n(g.cap)?];
    p::preflight(device,"gdn.spec.deferred.flush.v1",&ptrs,&args,grid,[256,1,1])?;
    p::preflight(device,"gdn.spec.deferred.clear.v1",&finalize,&finalize_args,finalize_grid,[256,1,1])?;
    unsafe{
        p::launch_on_stream(device,stream,"gdn.spec.deferred.flush.v1",&ptrs,&args,grid,[256,1,1],0,0)?;
        p::launch_on_stream(device,stream,"gdn.spec.deferred.clear.v1",&finalize,&finalize_args,finalize_grid,[256,1,1],0,0)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_gdn_speculative_scalar_pairs(){
        match pack(0x12345678,0xfedcba98).unwrap(){
            p::S::U64(v)=>assert_eq!(v,0xfedcba9812345678),_=>panic!("wrong scalar type"),
        }
        assert!(pack(u32::MAX as usize+1,0).is_err());
        assert!(pack(0,u32::MAX as usize+1).is_err());
        assert!(scale_arg((2,0),3).is_err());
        assert!(scale_arg((3,2),3).is_err());
        let a=Bound::new(16,4,p::E::F32,true);
        assert!(check_bindings(&[a,Bound::new(28,4,p::E::F32,false)]).is_err());
        assert!(check_bindings(&[a,Bound::new(32,4,p::E::F32,false)]).is_ok());
        assert_ne!(fused_operation(Mode::Direct,true,false),fused_operation(Mode::Batch,true,false));
        let a=Tensor::zeros(4,DType::F32,&Device::Cpu).unwrap();
        let b=a.narrow(0,1,1).unwrap();
        assert!(reject_storage_aliases(&[(&a,true),(&b,false)]).is_err());
        assert!(reject_storage_aliases(&[(&a,false),(&b,false)]).is_ok());
    }
}
