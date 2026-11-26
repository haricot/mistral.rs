//! Typed speculative state commits and convolution checkpoints. Keep the
//! historical two-launch commit order, but admit both plans before mutation.
use super::*;
use super::preparation_dynamic as p;
use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;

fn add(a:usize,b:usize)->Result<usize>{a.checked_add(b).ok_or_else(||candle_core::Error::msg("GDN checkpoint size overflow"))}
fn commit_geometry(batch:usize,seq:usize,kh:usize,vh:usize,kd:usize,vd:usize,cd:usize,cw:usize,cap:usize)->Result<([u32;3],[u32;3])>{
    for n in [batch,seq,kh,vh,kd,vd,cd,cw,cap]{p::n(n)?;}
    if batch==0||seq==0||kh==0||vh==0||vh%kh!=0||kd==0||kd>256||kd%32!=0||
        vd==0||vd%4!=0||cw==0||cap==0||cd!=add(p::mul(p::mul(kh,kd)?,2)?,p::mul(vh,vd)?)?{
        candle_core::bail!("invalid GDN speculative commit geometry");
    }
    Ok((p::grid(cd.div_ceil(256),batch)?,p::grid(vd.div_ceil(4),p::mul(batch,vh)?)?))
}
fn check_device(device:&Device,tensors:&[&Tensor])->Result<()>{
    if !device.is_cuda()||tensors.iter().any(|t|!t.device().same_device(device)||!t.is_contiguous()){
        candle_core::bail!("GDN checkpoint requires contiguous buffers on one CUDA device");
    }
    Ok(())
}
/// The initial snapshots and mutable pools must be disjoint. This prevents
/// an inter-block read/write race if the caller supplies overlapping views.
fn disjoint(a:(u64,usize,usize),b:(u64,usize,usize))->Result<()>{
    let end=|(ptr,n,bytes):(u64,usize,usize)|->Result<u64>{
        ptr.checked_add(p::mul(n,bytes)? as u64).ok_or_else(||candle_core::Error::msg("GDN address range overflow"))
    };
    let ae=end(a)?;let be=end(b)?;
    if a.0<be&&b.0<ae{candle_core::bail!("GDN commit snapshot overlaps mutable pool");}
    Ok(())
}
// Caller must ensure non-padding slots are unique. Device-side bounds checks
// deliberately do not read the slot table back to the host.
pub(super) fn commit<T:candle_core::cuda_backend::CudaDType+candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    mixed:&Tensor,convolved:&Tensor,b:&Tensor,a:&Tensor,initial_conv:&Tensor,initial_rec:&Tensor,
    a_log:&Tensor,dt_bias:&Tensor,conv_pool:&Tensor,rec_pool:&Tensor,keep:&Tensor,slots:&Tensor,
    kh:usize,vh:usize,kd:usize,vd:usize,tiled:bool,vm:bool,
)->Result<()>{
    let device=mixed.device();
    let (batch,seq,cd)=mixed.dims3()?;
    let (cap,pc,pw)=conv_pool.dims3()?;
    let cw=initial_conv.dim(candle_core::D::Minus1)?;
    let (cg,rg)=commit_geometry(batch,seq,kh,vh,kd,vd,cd,cw,cap)?;
    let tail=if vm{[vd,kd]}else{[kd,vd]};
    if pc!=cd||pw!=cw||rec_pool.dims()!=[cap,vh,tail[0],tail[1]]||
        initial_rec.elem_count()!=p::mul(p::mul(batch,vh)?,p::mul(kd,vd)?)?||
        a_log.elem_count()!=vh||dt_bias.elem_count()!=vh||
        keep.dtype()!=DType::U32||slots.dtype()!=DType::U32||keep.dims()!=[batch]||slots.dims()!=[batch]{
        candle_core::bail!("invalid GDN speculative commit pool, bias or index shape");
    }
    check_device(device,&[mixed,convolved,b,a,initial_conv,initial_rec,a_log,dt_bias,conv_pool,rec_pool,keep,slots])?;
    let dt=p::dt(mixed.dtype())?;
    let sd=p::dt(initial_rec.dtype())?;
    let stream=device.as_cuda_device()?.cuda_stream();
    let (ms,ml)=mixed.storage_and_layout();let (qs,ql)=convolved.storage_and_layout();
    let (bs,bl)=b.storage_and_layout();let (as_,al)=a.storage_and_layout();
    let (ics,icl)=initial_conv.storage_and_layout();let (irs,irl)=initial_rec.storage_and_layout();
    let (ls,ll)=a_log.storage_and_layout();let (ds,dl)=dt_bias.storage_and_layout();
    let (cps,cpl)=conv_pool.storage_and_layout();let (rps,rpl)=rec_pool.storage_and_layout();
    let (ks,kl)=keep.storage_and_layout();let (ss,sl)=slots.storage_and_layout();
    let (mp,_mg)=cuda_read_ptr_with_guard::<T>(&ms,ml,&stream,"mixed")?;
    let (qp,_qg)=cuda_read_ptr_with_guard::<T>(&qs,ql,&stream,"convolved")?;
    let (bp,_bg)=cuda_read_ptr_with_guard::<T>(&bs,bl,&stream,"b")?;
    let (ap,_ag)=cuda_read_ptr_with_guard::<T>(&as_,al,&stream,"a")?;
    let (icp,_icg)=cuda_read_ptr_with_guard::<T>(&ics,icl,&stream,"initial_conv")?;
    let (lp,_lg)=cuda_read_ptr_with_guard::<f32>(&ls,ll,&stream,"a_log")?;
    let (dp,_dg)=cuda_read_ptr_with_guard::<f32>(&ds,dl,&stream,"dt_bias")?;
    let (cp,_cg)=cuda_inplace_ptr_with_guard::<T>(&cps,cpl,&stream,"conv_pool")?;
    let (kp,_kg)=cuda_read_ptr_with_guard::<u32>(&ks,kl,&stream,"keep")?;
    let (sp,_sg)=cuda_read_ptr_with_guard::<u32>(&ss,sl,&stream,"slots")?;
    macro_rules! run{($ty:ty)=>{{
        let (irp,_irg)=cuda_read_ptr_with_guard::<$ty>(&irs,irl,&stream,"initial_rec")?;
        let (rp,_rpg)=cuda_inplace_ptr_with_guard::<$ty>(&rps,rpl,&stream,"rec_pool")?;
        let cb=std::mem::size_of::<T>();let rb=std::mem::size_of::<$ty>();
        let outputs=[(cp,conv_pool.elem_count(),cb),(rp,rec_pool.elem_count(),rb)];
        disjoint(outputs[0],outputs[1])?;
        for output in outputs{
            for input in [(mp,mixed.elem_count(),cb),(qp,convolved.elem_count(),cb),(bp,b.elem_count(),cb),
                (ap,a.elem_count(),cb),(icp,initial_conv.elem_count(),cb),(irp,initial_rec.elem_count(),rb),
                (lp,vh,4),(dp,vh,4),(kp,batch,4),(sp,batch,4)]{disjoint(output,input)?;}
        }
        let conv=[(mp,mixed.elem_count(),dt),(icp,initial_conv.elem_count(),dt),(cp,conv_pool.elem_count(),dt),
            (kp,batch,p::E::U32),(sp,batch,p::E::U32)];
        let rec=[(qp,convolved.elem_count(),dt),(bp,b.elem_count(),dt),(ap,a.elem_count(),dt),
            (irp,initial_rec.elem_count(),sd),(lp,vh,p::E::F32),(dp,vh,p::E::F32),(rp,rec_pool.elem_count(),sd),
            (kp,batch,p::E::U32),(sp,batch,p::E::U32)];
        let ca=[p::n(batch)?,p::n(seq)?,p::n(cd)?,p::n(cw)?,p::n(cap)?];
        let ra=[p::n(batch)?,p::n(seq)?,p::n(kh)?,p::n(vh)?,p::n(kd)?,p::n(vd)?,p::S::I32(i32::from(tiled)),p::n(cap)?];
        let rop=if vm{"gdn.commit.rec.vm.v1"}else{"gdn.commit.rec.km.v1"};
        p::preflight(device,"gdn.commit.conv.v1",&conv,&ca,cg,[256,1,1])?;
        p::preflight(device,rop,&rec,&ra,rg,[32,4,1])?;
        unsafe{
            p::launch(device,"gdn.commit.conv.v1",&conv,&ca,cg,[256,1,1])?;
            p::launch(device,rop,&rec,&ra,rg,[32,4,1])?;
        }
        Ok(())
    }}}
    match initial_rec.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),DType::F32=>run!(f32),
        _=>candle_core::bail!("invalid GDN recurrent state dtype")}
}

// For write=true, active slots must belong to disjoint checkpoint groups,
// not merely be distinct slots. Pending metadata and payload are stream-owned.
pub(super) fn conv<T:candle_core::cuda_backend::CudaDType+candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
    x:&Tensor,weight:&Tensor,state:&Tensor,slots:&Tensor,lanes:usize,write:bool,
    pending:Option<(GdnPendingSpeculativeConv<'_>,usize,usize,usize)>,
)->Result<Tensor>{
    let device=x.device();let (batch,seq,cd)=x.dims3()?;let (cap,_,cw)=state.dims3()?;
    for n in [batch,seq,cd,cap,cw,lanes]{p::n(n)?;}
    if cap==0||lanes==0||cap%lanes!=0||cw==0||cw>16||(write&&seq>lanes){
        candle_core::bail!("invalid GDN convolution checkpoint geometry");
    }
    let grid=p::grid(cd.div_ceil(256),batch)?;
    let count=p::mul(p::mul(batch,seq)?,cd)?;
    let dt=p::dt(x.dtype())?;
    let dev=device.as_cuda_device()?;let stream=dev.cuda_stream();
    let (xs,xl)=x.storage_and_layout();let (ws,wl)=weight.storage_and_layout();
    let (ss,sl)=state.storage_and_layout();let (is,il)=slots.storage_and_layout();
    let (xp,_xg)=cuda_read_ptr_with_guard::<T>(&xs,xl,&stream,"x")?;
    let (wp,_wg)=cuda_read_ptr_with_guard::<T>(&ws,wl,&stream,"weight")?;
    let (sp,_sg)=cuda_inplace_ptr_with_guard::<T>(&ss,sl,&stream,"state_pool")?;
    let (ip,_ig)=cuda_read_ptr_with_guard::<u32>(&is,il,&stream,"slots")?;
    let mut out=unsafe{dev.alloc::<T>(count)}?;
    let (op,og)=out.device_ptr_mut(&stream);
    let bytes=std::mem::size_of::<T>();
    let inputs=[(xp,p::span(xl)?,bytes),(wp,weight.elem_count(),bytes),(ip,batch,4)];
    let state_range=(sp,state.elem_count(),bytes);
    for input in inputs{disjoint(state_range,input)?;}
    let width4=cw==4&&sp%8==0&&wp%8==0;
    let mut ptrs=vec![(xp,p::span(xl)?,dt),(wp,weight.elem_count(),dt),(sp,state.elem_count(),dt),(op,count,dt),(ip,batch,p::E::U32)];
    let mut args=vec![p::n(batch)?,p::n(seq)?,p::n(cd)?,p::n(cw)?,p::n(lanes)?,p::S::I32(i32::from(write)),p::n(cap)?];
    let strides=[p::stride(xl.stride()[0])?,p::stride(xl.stride()[1])?,p::stride(xl.stride()[2])?];
    if let Some((pending,max_rows,pcap,blocks))=pending{
        let (cs,cl)=pending.conv_input.storage_and_layout();let (ks,kl)=pending.keep_rows.storage_and_layout();
        let (es,el)=pending.pending_epochs.storage_and_layout();let (aps,apl)=pending.applied_epochs.storage_and_layout();
        let (cp,_cg)=cuda_inplace_ptr_with_guard::<T>(&cs,cl,&stream,"pending_conv")?;
        let (kp,_kg)=cuda_read_ptr_with_guard::<u32>(&ks,kl,&stream,"keep")?;
        let (ep,_eg)=cuda_read_ptr_with_guard::<u32>(&es,el,&stream,"epoch")?;
        let (app,_apg)=cuda_inplace_ptr_with_guard::<u32>(&aps,apl,&stream,"applied_epoch")?;
        let writes=[state_range,(cp,pending.conv_input.elem_count(),bytes),(app,pending.applied_epochs.elem_count(),4)];
        let reads=[(kp,pcap,4),(ep,pcap,4)];
        for i in 0..writes.len(){
            for j in i+1..writes.len(){disjoint(writes[i],writes[j])?;}
            for input in inputs.into_iter().chain(reads){disjoint(writes[i],input)?;}
        }
        ptrs.extend([(cp,pending.conv_input.elem_count(),dt),(kp,pcap,p::E::U32),(ep,pcap,p::E::U32),
            (app,pending.applied_epochs.elem_count(),p::E::U32)]);
        args.extend([p::n(max_rows)?,p::n(pcap)?,p::n(blocks)?]);args.extend(strides);
        let name=if width4{"gdn.checkpoint.pending4.v1"}else{"gdn.checkpoint.pending.v1"};
        unsafe{p::launch(device,name,&ptrs,&args,grid,[256,1,1])?;}
    }else{
        args.extend(strides);
        let name=if width4{"gdn.checkpoint.conv4.v1"}else{"gdn.checkpoint.conv.v1"};
        unsafe{p::launch(device,name,&ptrs,&args,grid,[256,1,1])?;}
    }
    drop(og);
    Ok(Tensor::from((candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(out,dev.clone())),(batch,seq,cd))))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_snapshot_commit_fences_gpu()->Result<()>{
        let device=Device::new_cuda(0)?;
        let (batch,seq,kd,vd,cd,cw,cap)=(4usize,2usize,32usize,4usize,68usize,2usize,2usize);
        let keep=Tensor::from_vec(vec![1u32,3,1,1],batch,&device)?;
        let slots=Tensor::from_vec(vec![0u32,1,2,u32::MAX],batch,&device)?;
        let bias=Tensor::zeros(1,DType::F32,&device)?;
        for dt in [DType::F16,DType::BF16]{
            let mixed=Tensor::ones((batch,seq,cd),dt,&device)?;
            let mut qkv=vec![0f32;batch*seq*cd];
            for row in qkv.chunks_mut(cd){row[kd]=1.;row[2*kd..].fill(1.);}
            let qkv=Tensor::from_vec(qkv,(batch,seq,cd),&device)?.to_dtype(dt)?;
            let gates=Tensor::zeros((batch,seq,1),dt,&device)?;
            let initial_conv=Tensor::zeros((batch,cd,cw),dt,&device)?;
            for sd in [DType::F16,DType::BF16,DType::F32]{
                let initial=Tensor::zeros((batch,1,kd,vd),sd,&device)?;
                let conv=Tensor::zeros((cap,cd,cw),dt,&device)?;
                let state=Tensor::zeros((cap,1,kd,vd),sd,&device)?;
                speculative_state_commit_cuda(GdnSpeculativeStateCommit{mixed_qkv:&mixed,convolved_qkv:&qkv,b:&gates,a:&gates,
                    initial_conv_state:&initial_conv,initial_recurrent_state:&initial,a_log:&bias,dt_bias:&bias,
                    conv_state_pool:&conv,recurrent_state_pool:&state,keep_rows:&keep,slot_indices:&slots,
                    num_k_heads:1,num_v_heads:1,head_k_dim:kd,head_v_dim:vd,tiled_v_heads:false,
                    state_layout:RecurrentStateLayout::GdnKeyMajor})?;
                let cv=conv.to_dtype(DType::F32)?.reshape((cap,cd*cw))?.to_vec2::<f32>()?;
                assert!(cv[0].chunks_exact(2).all(|v|v==[0.,1.]));
                assert!(cv[1].iter().all(|&v|v==0.));
                let r=state.to_dtype(DType::F32)?.reshape((cap,kd*vd))?.to_vec2::<f32>()?;
                assert!(r[0][..vd].iter().all(|&v|(v-0.5).abs()<0.001));
                assert!(r[0][vd..].iter().chain(&r[1]).all(|&v|v==0.));
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_conv_checkpoint_slot_fences_gpu()->Result<()>{
        let device=Device::new_cuda(0)?;
        let slots=Tensor::from_vec(vec![1u32,4,u32::MAX],3,&device)?;
        for dt in [DType::F16,DType::BF16]{
            for width in [3usize,4]{
                let input=Tensor::ones((3,2,3),dt,&device)?;
                let weight=Tensor::zeros((3,width),dt,&device)?;
                let pool=Tensor::zeros((4,3,width),dt,&device)?;
                let out=speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints{x:&input,weight:&weight,
                    state_pool:&pool,active_slots:&slots,checkpoint_lanes:2,write_checkpoints:true,pending:None})?;
                assert!(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v|v==0.));
                let data=pool.to_dtype(DType::F32)?.reshape((4,3*width))?.to_vec2::<f32>()?;
                for (row,ones) in [(0,1),(1,2),(2,0),(3,0)]{
                    assert!(data[row].chunks_exact(width).all(|c|c[..width-ones].iter().all(|&v|v==0.)&&c[width-ones..].iter().all(|&v|v==1.)));
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_conv_pending_epoch_gpu()->Result<()>{
        let device=Device::new_cuda(0)?;
        let (cap,cd,cw)=(2usize,257usize,4usize);
        let slots=Tensor::from_vec(vec![0u32,u32::MAX],2,&device)?;
        for dt in [DType::F16,DType::BF16]{
            let input=Tensor::ones((2,1,cd),dt,&device)?.affine(3.,0.)?;
            let weight=Tensor::zeros((cd,cw),dt,&device)?;
            let pool=Tensor::zeros((cap,cd,cw),dt,&device)?;
            let pending=Tensor::ones((cap,2,cd),dt,&device)?;
            let keep=Tensor::from_vec(vec![1u32,1],cap,&device)?;
            let epoch=Tensor::from_vec(vec![7u32,9],cap,&device)?;
            let applied=Tensor::zeros((cap,2),DType::U32,&device)?;
            let run=||speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints{x:&input,weight:&weight,
                state_pool:&pool,active_slots:&slots,checkpoint_lanes:1,write_checkpoints:false,
                pending:Some(GdnPendingSpeculativeConv{conv_input:&pending,keep_rows:&keep,pending_epochs:&epoch,applied_epochs:&applied})});
            for _ in 0..2{
                let out=run()?;
                assert!(out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().all(|&v|v==0.));
                assert_eq!(applied.flatten_all()?.to_vec1::<u32>()?,vec![7,7,0,0]);
                let states=pool.to_dtype(DType::F32)?.reshape((cap,cd*cw))?.to_vec2::<f32>()?;
                assert!(states[0].chunks_exact(cw).all(|v|v==[0.,0.,0.,1.]));
                assert!(states[1].iter().all(|&v|v==0.));
                let logged=pending.to_dtype(DType::F32)?.reshape((cap,2*cd))?.to_vec2::<f32>()?;
                assert!(logged[0][..cd].iter().all(|&v|v==3.));
                assert!(logged[1].iter().all(|&v|v==1.));
            }
        }
        Ok(())
    }
    #[test]
    fn dkr_gdn_checkpoint_geometry(){
        assert!(commit_geometry(1,2,1,1,32,5,69,4,2).is_err());
        assert_eq!(commit_geometry(1,2,1,1,32,4,68,4,2).unwrap(),([1,1,1],[1,1,1]));
        assert!(disjoint((100,4,4),(115,1,1)).is_err());
        assert!(disjoint((100,4,4),(116,1,1)).is_ok());
        assert!(disjoint((u64::MAX,1,4),(0,1,1)).is_err());
    }
}
