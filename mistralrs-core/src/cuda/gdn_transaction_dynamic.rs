//! Batched GDN state transactions. Only the internal tensor adapter may form
//! a pointer table. The catalogue checks its I64 storage, not the addresses
//! stored inside it. Callers validate every pointee and retain its guard.
use super::preparation_dynamic as p;
use candle_core::{Device,DType,Result,Tensor};

#[derive(Clone,Copy,Debug,Eq,PartialEq)]
pub(super) enum Kind{Commit,Stage,Publish,Apply}

fn operation(kind:Kind,activation:DType,state:DType)->Result<&'static str>{
    use Kind::*;
    Ok(match (kind,activation,state){
        (Commit,DType::F16,DType::F16)=>"gdn.tx.commit.f16_f16.v1",
        (Commit,DType::F16,DType::BF16)=>"gdn.tx.commit.f16_bf16.v1",
        (Commit,DType::F16,DType::F32)=>"gdn.tx.commit.f16_f32.v1",
        (Commit,DType::BF16,DType::F16)=>"gdn.tx.commit.bf16_f16.v1",
        (Commit,DType::BF16,DType::BF16)=>"gdn.tx.commit.bf16_bf16.v1",
        (Commit,DType::BF16,DType::F32)=>"gdn.tx.commit.bf16_f32.v1",
        (Apply,DType::F16,DType::F16)=>"gdn.tx.apply.f16_f16.v1",
        (Apply,DType::F16,DType::BF16)=>"gdn.tx.apply.f16_bf16.v1",
        (Apply,DType::F16,DType::F32)=>"gdn.tx.apply.f16_f32.v1",
        (Apply,DType::BF16,DType::F16)=>"gdn.tx.apply.bf16_f16.v1",
        (Apply,DType::BF16,DType::BF16)=>"gdn.tx.apply.bf16_bf16.v1",
        (Apply,DType::BF16,DType::F32)=>"gdn.tx.apply.bf16_f32.v1",
        (Stage,DType::F16,_)=>"gdn.tx.stage.f16.v1",
        (Stage,DType::BF16,_)=>"gdn.tx.stage.bf16.v1",
        (Publish,_,_)=>"gdn.tx.publish.v1",
        _=>candle_core::bail!("unsupported GDN transaction dtype pair"),
    })
}

fn table_segments(kind:Kind)->usize{
    match kind{Kind::Commit=>6,Kind::Stage=>10,Kind::Publish=>3,Kind::Apply=>11}
}
fn geometry(kind:Kind,dims:&[usize])->Result<([u32;3],[u32;3])>{
    use Kind::*;
    let arity=match kind{Commit=>12,Stage=>10,Publish=>4,Apply=>12};
    if dims.len()!=arity{candle_core::bail!("invalid GDN transaction ABI arity");}
    for &n in dims{p::n(n)?;}
    let layers=dims[0];let batch=dims[1];
    if layers==0||layers>65535||batch==0||batch>65535{
        candle_core::bail!("GDN transaction grid dimensions out of range");
    }
    let grid=match kind{
        Commit|Apply=>{
            let (rows,capacity,kh,vh,kd,vd,cd,cw,tiled)=if kind==Commit{
                (dims[2],dims[11],dims[3],dims[4],dims[5],dims[6],dims[7],dims[8],dims[9])
            }else{
                (dims[2],dims[3],dims[4],dims[5],dims[6],dims[7],dims[8],dims[9],dims[10])
            };
            if rows==0||rows>8||capacity==0||kh==0||vh==0||vh%kh!=0||kd!=128||vd!=128||
                cd==0||cw==0||cw>16||tiled>1{
                candle_core::bail!("GDN transaction recurrent geometry");
            }
            let conv_blocks=cd.div_ceil(256);
            if (kind==Commit&&dims[10]>1)||(kind==Apply&&dims[11]!=conv_blocks){
                candle_core::bail!("GDN transaction layout or block count");
            }
            let work=conv_blocks.checked_add(vh).ok_or_else(||candle_core::Error::msg("GDN transaction grid overflow"))?;
            let mut grid=p::grid(work,batch)?;
            grid[2]=layers as u32;
            grid
        }
        Stage=>{
            let (seq,max_rows,cap,kh,vh,kd,vd,cd)=(dims[2],dims[3],dims[4],dims[5],dims[6],dims[7],dims[8],dims[9]);
            if seq==0||seq>max_rows||max_rows>8||cap==0||kh==0||vh==0||vh%kh!=0||kd==0||vd==0||cd==0{
                candle_core::bail!("GDN transaction stage geometry");
            }
            let row=cd.checked_add(p::mul(kh,kd)?)
                .and_then(|n|vh.checked_mul(vd).and_then(|v|n.checked_add(v)))
                .and_then(|n|n.checked_add(vh)).ok_or_else(||candle_core::Error::msg("GDN stage size overflow"))?;
            let copies=p::mul(seq,row)?.div_ceil(256).min(4);
            let mut grid=p::grid(copies,batch)?;
            grid[2]=layers as u32;
            grid
        }
        Publish=>{
            if dims[2]==0||dims[2]>8||dims[3]==0{candle_core::bail!("GDN publish geometry");}
            p::grid(batch,layers)?
        }
    };
    Ok((grid,[if kind==Publish{1}else{256},1,1]))
}

/// Safety: every address stored in table must come from a validated tensor
/// on this device. The caller owns the table and all pointee guards across
/// the launch. Indices must have unique non-padding destination slots.
/// The transaction producer and all consumers are ordered on the same stream.
pub(super) unsafe fn launch(kind:Kind,device:&Device,activation:DType,state:DType,
    table:&Tensor,table_ptr:u64,keep:Option<(&Tensor,u64)>,slots:&Tensor,slots_ptr:u64,dims:&[usize])->Result<()>{
    let (grid,block)=geometry(kind,dims)?;
    let layers=dims[0];let batch=dims[1];
    if table.dtype()!=DType::I64||table.dims()!=[p::mul(table_segments(kind),layers)?]||
        !table.is_contiguous()||!table.device().same_device(device){
        candle_core::bail!("invalid internal GDN transaction pointer table");
    }
    let index_ok=|t:&Tensor|t.dtype()==DType::U32&&t.dims()==[batch]&&t.is_contiguous()&&t.device().same_device(device);
    if !index_ok(slots)||keep.is_some_and(|(t,_)|!index_ok(t))||keep.is_some()!=(kind!=Kind::Apply){
        candle_core::bail!("invalid GDN transaction indices");
    }
    let mut pointers=vec![(table_ptr,table.elem_count(),p::E::I64)];
    if let Some((_,ptr))=keep{pointers.push((ptr,batch,p::E::U32));}
    pointers.push((slots_ptr,batch,p::E::U32));
    let args=dims.iter().map(|&n|p::n(n)).collect::<Result<Vec<_>>>()?;
    unsafe{p::launch(device,operation(kind,activation,state)?,&pointers,&args,grid,block)}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_transaction_publish_apply_once_gpu()->Result<()>{
        use super::super::{GdnPendingTransitionPublish,GdnPendingTransitionPublishLayer,
            GdnPendingTransitionApply,GdnPendingTransitionApplyLayer,
            pending_transition_publish_batched_cuda,pending_transition_apply_batched_cuda,
            RecurrentStateLayout};
        let device=Device::new_cuda(0)?;
        let (capacity,max_rows,conv_dim,conv_width)=(3usize,2usize,3usize,2usize);
        for activation in [DType::F16,DType::BF16]{
            for sd in [DType::F16,DType::BF16,DType::F32]{
                let pending_conv=Tensor::from_vec((1..=18).map(|v|v as f32).collect::<Vec<_>>(),(capacity,max_rows,conv_dim),&device)?.to_dtype(activation)?;
                // Bank 0 is all zeros, bank 1 contains the accepted key.
                let bank0=Tensor::zeros((1,capacity,max_rows,1,128),DType::F32,&device)?;
                let bank1=Tensor::ones((1,capacity,max_rows,1,128),DType::F32,&device)?;
                let keys=Tensor::cat(&[&bank0,&bank1],0)?;
                let delta=Tensor::ones((capacity,max_rows,1,128),DType::F32,&device)?.affine(2.,0.)?;
                let decay=Tensor::ones((capacity,max_rows,1),DType::F32,&device)?.affine(0.5,0.)?;
                let keep=Tensor::zeros(capacity,DType::U32,&device)?;
                let epoch=Tensor::from_vec(vec![u32::MAX,9,7],capacity,&device)?;
                let bank=Tensor::zeros(capacity,DType::U32,&device)?;
                let ce=Tensor::zeros((capacity,1),DType::U32,&device)?;
                let re=Tensor::zeros((capacity,1),DType::U32,&device)?;
                let conv=Tensor::zeros((capacity,conv_dim,conv_width),activation,&device)?;
                let state=Tensor::ones((capacity,1,128,128),sd,&device)?;
                let publish_keep=Tensor::from_vec(vec![1u32,0,3,1],4,&device)?;
                let publish_slots=Tensor::from_vec(vec![0u32,1,2,u32::MAX],4,&device)?;
                pending_transition_publish_batched_cuda(GdnPendingTransitionPublish{
                    layers:&[GdnPendingTransitionPublishLayer{pending_keep_rows:&keep,pending_epochs:&epoch,pending_key_bank:&bank}],
                    keep_rows:&publish_keep,destination_slots:&publish_slots,max_rows,destination_capacity:capacity})?;
                assert_eq!(keep.to_vec1::<u32>()?,vec![1,0,0]);
                assert_eq!(epoch.to_vec1::<u32>()?,vec![1,10,8]);
                assert_eq!(bank.to_vec1::<u32>()?,vec![1,1,1]);
                let slots=Tensor::from_vec(vec![0u32,1,2,u32::MAX,3],5,&device)?;
                let apply=||pending_transition_apply_batched_cuda(GdnPendingTransitionApply{
                    layers:&[GdnPendingTransitionApplyLayer{pending_conv_input:&pending_conv,pending_key_banks:&keys,
                        pending_key_bank:&bank,pending_delta:&delta,pending_decay:&decay,pending_keep_rows:&keep,
                        pending_epochs:&epoch,conv_applied_epochs:&ce,recurrent_applied_epochs:&re,conv_state:&conv,recurrent_state:&state}],
                    active_slots:&slots,num_k_heads:1,num_v_heads:1,head_k_dim:128,head_v_dim:128,
                    conv_dim,conv_width,tiled_v_heads:false,state_layout:RecurrentStateLayout::GdnValueMajor});
                apply()?;
                let conv_expected=vec![0.,1.,0.,2.,0.,3.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.];
                assert_eq!(conv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,conv_expected);
                let check_state=||->Result<()>{
                    let rows=state.to_dtype(DType::F32)?.reshape((capacity,128*128))?.to_vec2::<f32>()?;
                    assert!(rows[0].iter().all(|&v|v==2.5));
                    assert!(rows[1..].iter().flatten().all(|&v|v==1.));
                    assert_eq!(ce.flatten_all()?.to_vec1::<u32>()?,vec![1,0,0]);
                    assert_eq!(re.flatten_all()?.to_vec1::<u32>()?,vec![1,0,0]);
                    Ok(())
                };
                check_state()?;
                apply()?;
                check_state()?;
                assert_eq!(conv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,conv_expected);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore="requires CUDA and GDN typed catalogue"]
    fn dkr_gdn_transaction_commit_invalid_prefix_gpu()->Result<()>{
        use super::super::{GdnSpeculativeTransitionCommit,GdnSpeculativeTransitionLayer,
            speculative_transition_commit_batched_cuda,RecurrentStateLayout};
        let device=Device::new_cuda(0)?;
        for activation in [DType::F16,DType::BF16]{
            let input=Tensor::ones((4,2,3),activation,&device)?;
            let key=Tensor::ones((4,2,1,128),DType::F32,&device)?;
            let delta=Tensor::ones((4,2,1,128),DType::F32,&device)?;
            let decay=Tensor::ones((4,2,1),DType::F32,&device)?;
            let keep=Tensor::from_vec(vec![1u32,3,1,1],4,&device)?;
            // Row 1 exceeds seq_len; row 2 is out of capacity; row 3 is padding.
            let slots=Tensor::from_vec(vec![0u32,1,2,u32::MAX],4,&device)?;
            let conv=Tensor::zeros((2,3,2),activation,&device)?;
            let state=Tensor::zeros((2,1,128,128),DType::F32,&device)?;
            speculative_transition_commit_batched_cuda(GdnSpeculativeTransitionCommit{
                layers:&[GdnSpeculativeTransitionLayer{conv_input:&input,key:&key,delta:&delta,decay:&decay,
                    conv_state:&conv,recurrent_state:&state}],keep_rows:&keep,active_slots:&slots,
                num_k_heads:1,num_v_heads:1,head_k_dim:128,head_v_dim:128,conv_dim:3,conv_width:2,
                tiled_v_heads:false,state_layout:RecurrentStateLayout::GdnKeyMajor})?;
            assert_eq!(conv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,vec![0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,0.,0.]);
            let data=state.reshape((2,128*128))?.to_vec2::<f32>()?;
            assert!(data[0].iter().all(|&v|v==1.));
            assert!(data[1].iter().all(|&v|v==0.));
        }
        Ok(())
    }

    #[test]
    fn dkr_gdn_transaction_geometry(){
        assert_eq!(geometry(Kind::Commit,&[2,3,8,1,4,128,128,768,4,0,1,16]).unwrap(),([7,3,2],[256,1,1]));
        assert_eq!(geometry(Kind::Apply,&[2,3,8,16,1,4,128,128,768,4,0,3]).unwrap(),([7,3,2],[256,1,1]));
        assert_eq!(geometry(Kind::Stage,&[2,3,8,8,16,1,4,128,128,768]).unwrap(),([4,3,2],[256,1,1]));
        assert_eq!(geometry(Kind::Publish,&[2,3,8,16]).unwrap(),([3,2,1],[1,1,1]));
        assert!(geometry(Kind::Commit,&[2,3,9,1,4,128,128,768,4,0,1,16]).is_err());
        assert!(geometry(Kind::Apply,&[2,3,8,16,1,4,128,128,768,4,0,2]).is_err());
        assert!(geometry(Kind::Publish,&[65536,1,8,16]).is_err());
        assert!(geometry(Kind::Stage,&[2,3,8,7,16,1,4,128,128,768]).is_err());
        assert_ne!(operation(Kind::Apply,DType::F16,DType::F32).unwrap(),
            operation(Kind::Apply,DType::BF16,DType::F32).unwrap());
    }
}
