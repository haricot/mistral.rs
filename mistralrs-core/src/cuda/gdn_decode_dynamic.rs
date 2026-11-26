//! Typed decode route. Keep the historical policy and state layout, but no host FFI.
use super::*;
use super::preparation_dynamic as p;
use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;

fn variant(kernel:GdnDecodeKernel,kd:usize,vd:usize,bh:usize)->Result<(&'static str,[u32;3],[u32;3],u32,u32)>{
    if kd==0||kd>GDN_DECODE_FALLBACK_MAX_K||vd==0||bh==0{candle_core::bail!("invalid GDN decode geometry");}
    let (op,grid,block,shared,ceiling)=match kernel{
        GdnDecodeKernel::Baseline=>match kd{
            64=>("gdn.decode.k64.v1",p::grid(vd.div_ceil(64),bh)?,[64,1,1],0,0),
            128=>("gdn.decode.k128.v1",p::grid(vd.div_ceil(64),bh)?,[64,1,1],0,0),
            _=>("gdn.decode.fallback.v1",p::grid(vd.div_ceil(64),bh)?,[64,1,1],(8*(64+kd)) as u32,2560),
        },
        GdnDecodeKernel::Cooperative if kd==128&&vd%16==0=>
            ("gdn.decode.cooperative.v1",p::grid(vd/16,bh)?,[128,1,1],0,0),
        GdnDecodeKernel::Pipelined if kd==128&&vd%32==0=>
            ("gdn.decode.pipelined.v1",p::grid(bh,1)?,[256,1,1],0,0),
        GdnDecodeKernel::ValueMajor4 if kd==128&&vd==128=>
            ("gdn.decode.vm4.v1",p::grid(p::mul(bh,32)?,1)?,[32,1,1],0,0),
        GdnDecodeKernel::ValueMajor32 if kd==128&&vd==128=>
            ("gdn.decode.vm32.v1",p::grid(p::mul(bh,4)?,1)?,[32,1,1],0,0),
        _=>candle_core::bail!("invalid GDN optimized decode geometry"),
    };
    Ok((op,grid,block,shared,ceiling))
}
pub(super) fn decode(launch:GdnDecodeLaunch<'_>)->Result<Tensor>{
    let GdnDecodeLaunch{mixed_qkv,b,a,a_log,dt_bias,state,batch_size,num_k_heads,num_v_heads,
        head_k_dim,head_v_dim,tiled_v_heads,state_layout,slots,requested_kernel}=launch;
    let device=mixed_qkv.device();
    let dt=p::dt(mixed_qkv.dtype())?;
    if !matches!(mixed_qkv.dtype(),DType::F16|DType::BF16)||batch_size==0||num_k_heads==0||
        num_v_heads==0||num_v_heads%num_k_heads!=0||head_k_dim==0||head_v_dim==0||
        head_k_dim>GDN_DECODE_FALLBACK_MAX_K{candle_core::bail!("invalid GDN decode dimensions");}
    let kh=p::mul(num_k_heads,head_k_dim)?;
    let vh=p::mul(num_v_heads,head_v_dim)?;
    let conv=p::mul(kh,2)?.checked_add(vh).ok_or_else(||candle_core::Error::msg("GDN decode width overflow"))?;
    p::n(conv)?;
    let bh=p::mul(batch_size,num_v_heads)?;
    let count=p::mul(bh,head_v_dim)?;
    p::n(bh)?;
    if !mixed_qkv.is_contiguous()||mixed_qkv.elem_count()!=p::mul(batch_size,conv)?{
        candle_core::bail!("GDN decode mixed QKV geometry");
    }
    for t in [b,a]{
        if t.dtype()!=mixed_qkv.dtype()||!t.device().same_device(device)||t.rank()<2||
            t.dim(0)?!=batch_size||t.dims().last()!=Some(&num_v_heads)||t.elem_count()!=bh{
            candle_core::bail!("GDN decode gating geometry");
        }
    }
    for t in [a_log,dt_bias]{
        if t.dtype()!=DType::F32||!t.device().same_device(device)||!t.is_contiguous()||t.elem_count()!=num_v_heads{
            candle_core::bail!("GDN decode bias geometry");
        }
    }
    let tail=match state_layout{
        RecurrentStateLayout::GdnValueMajor=>[head_v_dim,head_k_dim],
        RecurrentStateLayout::GdnKeyMajor=>[head_k_dim,head_v_dim],
        _=>candle_core::bail!("unsupported GDN state layout"),
    };
    let sd=state.dims();
    let per_head=p::mul(head_k_dim,head_v_dim)?;
    let state_heads=state.elem_count()/per_head;
    if sd.len()<3||sd[sd.len()-2..]!=tail||!state.is_contiguous()||!state.device().same_device(device)||
        state.elem_count()%per_head!=0||state_heads==0||state_heads%num_v_heads!=0||
        (matches!(slots,GdnStateSlots::Gathered)&&state_heads!=bh){
        candle_core::bail!("GDN decode recurrent state geometry");
    }
    let (slot,has)=super::recurrence_dynamic::slot_tensor(device,slots,batch_size)?;
    let dev=device.as_cuda_device()?;
    let stream=dev.cuda_stream();
    let (ms,ml)=mixed_qkv.storage_and_layout();
    let (bs,bl)=b.storage_and_layout();let (as_,al)=a.storage_and_layout();
    let (ls,ll)=a_log.storage_and_layout();let (ds,dl)=dt_bias.storage_and_layout();
    let (ss,sl)=state.storage_and_layout();let (is,il)=slot.storage_and_layout();
    let (lp,_lg)=cuda_read_ptr_with_guard::<f32>(&ls,ll,&stream,"a_log")?;
    let (dp,_dg)=cuda_read_ptr_with_guard::<f32>(&ds,dl,&stream,"dt_bias")?;
    let (sp,_,_sg)=cuda_recurrent_state_inplace_ptr_with_guard(state.dtype(),&ss,sl,&stream,"decode state")?;
    let (ip,_ig)=cuda_read_ptr_with_guard::<u32>(&is,il,&stream,"decode slots")?;
    let props=gdn_cuda_device_properties(dev)?;
    let bstr=b.stride();let astr=a.stride();
    let args=[p::n(batch_size)?,p::n(num_k_heads)?,p::n(num_v_heads)?,p::n(head_k_dim)?,p::n(head_v_dim)?,
        p::S::I32(i32::from(tiled_v_heads)),p::stride(bstr[0])?,p::stride(bstr[bstr.len()-1])?,
        p::stride(astr[0])?,p::stride(astr[astr.len()-1])?,p::S::I32(has)];
    macro_rules! run{($ty:ty)=>{{
        let (mp,_mg)=cuda_read_ptr_with_guard::<$ty>(&ms,ml,&stream,"mixed_qkv")?;
        let (bp,_bg)=cuda_read_ptr_with_guard::<$ty>(&bs,bl,&stream,"b")?;
        let (ap,_ag)=cuda_read_ptr_with_guard::<$ty>(&as_,al,&stream,"a")?;
        let policy=GdnDecodePolicy{compute_major:props.compute_major,multiprocessor_count:props.multiprocessor_count,
            state_blocks:bh,head_k_dim,head_v_dim,vector_aligned:(sp as usize).is_multiple_of(GDN_DECODE_VECTOR_ALIGNMENT)&&
                (mp as usize).is_multiple_of(GDN_DECODE_INPUT_ALIGNMENT),bf16:mixed_qkv.dtype()==DType::BF16,state_layout};
        let kernel=select_decode_kernel(policy,requested_kernel).map_err(|k|candle_core::Error::msg(format!("GDN decode kernel {k:?} not admitted")))?;
        let (op,grid,block,shared,ceiling)=variant(kernel,head_k_dim,head_v_dim,bh)?;
        let mut output=unsafe{dev.alloc::<$ty>(count)}?;
        let (out,og)=output.device_ptr_mut(&stream);
        let pointers=[(mp,mixed_qkv.elem_count(),dt),(bp,p::span(bl)?,dt),(ap,p::span(al)?,dt),
            (lp,num_v_heads,p::E::F32),(dp,num_v_heads,p::E::F32),(sp,state.elem_count(),p::dt(state.dtype())?),
            (out,count,dt),(ip,slot.elem_count(),p::E::U32)];
        unsafe{p::launch_shared(device,op,&pointers,&args,grid,block,shared,ceiling)?;}
        drop(og);
        Ok(Tensor::from((candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(output,dev.clone())),(bh,1,head_v_dim))))
    }}}
    match mixed_qkv.dtype(){DType::F16=>run!(half::f16),DType::BF16=>run!(half::bf16),_=>unreachable!()}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn dkr_gdn_decode_geometry(){
        let a=variant(GdnDecodeKernel::Baseline,17,65,3).unwrap();
        assert_eq!((a.0,a.1,a.3,a.4),("gdn.decode.fallback.v1",[2,3,1],648,2560));
        assert_eq!(variant(GdnDecodeKernel::Baseline,128,65,3).unwrap().1,[2,3,1]);
        assert_eq!(variant(GdnDecodeKernel::ValueMajor4,128,128,3).unwrap().1,[96,1,1]);
        assert_eq!(variant(GdnDecodeKernel::ValueMajor32,128,128,3).unwrap().1,[12,1,1]);
        assert!(variant(GdnDecodeKernel::Pipelined,128,65,3).is_err());
    }
}
