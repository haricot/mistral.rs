#[cfg(not(feature = "dkr-static-closure-probe"))]
use std::ffi::c_void;

use candle_core::{backend::BackendStorage, CudaStorage};
#[cfg(not(feature = "dkr-static-closure-probe"))]
use candle_core::cuda::cudarc::driver::sys::CUstream;

use super::*;
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

#[cfg(not(feature = "dkr-static-closure-probe"))]
extern "C" {
    fn lfm3_conv_f32(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
    fn lfm3_conv_f16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
    fn lfm3_conv_bf16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
}

fn cuda_storage(storage: &Storage) -> Result<&CudaStorage> {
    match storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => candle_core::bail!("LFM3 convolution expects CUDA storage"),
    }
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
impl ConvInputs<'_> {
    pub(super) fn cuda(&self) -> Result<(Tensor, Tensor)> {
        let (projected, projected_layout) = self.projected.storage_and_layout();
        let (weight, weight_layout) = self.weight.storage_and_layout();
        let (state, state_layout) = self.state.storage_and_layout();
        let bias = self.bias.map(Tensor::storage_and_layout);
        let projected = cuda_storage(&projected)?;
        let weight = cuda_storage(&weight)?;
        let state = cuda_storage(&state)?;
        let (bias, bias_offset) = match &bias {
            Some((storage, layout)) => (cuda_storage(storage)?, layout.start_offset()),
            None => (weight, weight_layout.start_offset()),
        };
        let device = projected.device();
        let stream = device.cuda_stream();

        macro_rules! run {
            ($ty:ty, $kernel:ident) => {{
                let mut output = unsafe { device.alloc::<$ty>(self.output_len()) }?;
                let mut next_state = unsafe { device.alloc::<$ty>(self.state_len()) }?;
                let (x_ptr, _x_guard) = slice_ptr_on_stream(
                    projected.as_cuda_slice::<$ty>()?,
                    projected_layout.start_offset(),
                    &stream,
                );
                let (weight_ptr, _weight_guard) = slice_ptr_on_stream(
                    weight.as_cuda_slice::<$ty>()?,
                    weight_layout.start_offset(),
                    &stream,
                );
                let (bias_ptr, _bias_guard) =
                    slice_ptr_on_stream(bias.as_cuda_slice::<$ty>()?, bias_offset, &stream);
                let (state_ptr, _state_guard) = slice_ptr_on_stream(
                    state.as_cuda_slice::<$ty>()?,
                    state_layout.start_offset(),
                    &stream,
                );
                let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
                let (next_state_ptr, next_state_guard) =
                    slice_ptr_mut_on_stream(&mut next_state, 0, &stream);
                let status = unsafe {
                    $kernel(
                        x_ptr as *const c_void,
                        weight_ptr as *const c_void,
                        bias_ptr as *const c_void,
                        state_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        next_state_ptr as *mut c_void,
                        self.params.as_ptr(),
                        stream.cu_stream(),
                    )
                };
                if status != 0 {
                    candle_core::bail!("LFM3 CUDA convolution launch failed with status {status}");
                }
                drop(output_guard);
                drop(next_state_guard);
                self.tensors(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(next_state, device.clone())),
                )
            }};
        }
        Ok(match self.projected.dtype() {
            DType::F32 => run!(f32, lfm3_conv_f32),
            DType::F16 => run!(half::f16, lfm3_conv_f16),
            DType::BF16 => run!(half::bf16, lfm3_conv_bf16),
            dtype => candle_core::bail!("Unsupported LFM3 convolution dtype {dtype:?}"),
        })
    }
}

#[cfg(feature = "dkr-static-closure-probe")]
impl ConvInputs<'_> {
    pub(super) fn cuda(&self) -> Result<(Tensor, Tensor)> {
        use crate::lfm3_dynamic as dkr;
        use candle_core::cuda::cudarc::driver::DeviceSlice;
        let (projected, projected_layout)=self.projected.storage_and_layout();
        let (weight, weight_layout)=self.weight.storage_and_layout();
        let (state, state_layout)=self.state.storage_and_layout();
        let bias=self.bias.map(Tensor::storage_and_layout);
        let projected=cuda_storage(&projected)?;
        let weight=cuda_storage(&weight)?;
        let state=cuda_storage(&state)?;
        let bias=bias.as_ref().map(|(s,l)|->Result<_>{Ok((cuda_storage(s)?,*l))}).transpose()?;
        let device=projected.device();
        let stream=device.cuda_stream();
        let out_len=dkr::checked_product(&[self.batch,self.seq,self.hidden])?;
        let state_len=dkr::checked_product(&[self.batch,self.hidden,self.width])?;
        macro_rules! run {
            ($ty:ty) => {{
                let x=projected.as_cuda_slice::<$ty>()?;
                let w=weight.as_cuda_slice::<$ty>()?;
                let st=state.as_cuda_slice::<$ty>()?;
                let mut spans=[dkr::span(projected_layout,x.len())?,
                               dkr::span(weight_layout,w.len())?,
                               dkr::span(state_layout,st.len())?,0,0,0];
                let (xp,xg)=slice_ptr_on_stream(x,projected_layout.start_offset(),&stream);
                let (wp,wg)=slice_ptr_on_stream(w,weight_layout.start_offset(),&stream);
                let (sp,sg)=slice_ptr_on_stream(st,state_layout.start_offset(),&stream);
                let mut pointers=[xp,wp,sp,0,0,0];
                let mut count=3;
                let bias_guard=if let Some((bias,layout))=bias{
                    let values=bias.as_cuda_slice::<$ty>()?;
                    spans[count]=dkr::span(layout,values.len())?;
                    let (ptr,guard)=slice_ptr_on_stream(values,layout.start_offset(),&stream);
                    pointers[count]=ptr;
                    count+=1;
                    Some(guard)
                }else{None};
                spans[count]=out_len;
                spans[count+1]=state_len;
                let key=dkr::Key::causal(self.projected.dtype(),&spans[..count+2],out_len,self.width,bias.is_some())?;
                let mut output=unsafe{device.alloc::<$ty>(out_len)}?;
                let mut next_state=unsafe{device.alloc::<$ty>(state_len)}?;
                let (op,og)=slice_ptr_mut_on_stream(&mut output,0,&stream);
                let (np,ng)=slice_ptr_mut_on_stream(&mut next_state,0,&stream);
                pointers[count]=op;
                pointers[count+1]=np;
                unsafe{dkr::causal(key,device,&pointers[..count+2],&self.params)?;}
                drop((xg,wg,sg,bias_guard,og,ng));
                self.tensors(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(output,device.clone())),
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(next_state,device.clone())),
                )
            }};
        }
        Ok(match self.projected.dtype(){
            DType::F32=>run!(f32),
            DType::F16=>run!(half::f16),
            DType::BF16=>run!(half::bf16),
            dtype=>candle_core::bail!("Unsupported LFM3 convolution dtype {dtype:?}"),
        })
    }
}
