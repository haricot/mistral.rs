use std::ffi::c_void;

use candle_core::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{DeviceRepr, ValidAsZeroBits},
        CudaDType,
    },
    CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape, Tensor,
};
use half::f16;

use crate::utils::slice_ptr;

pub(super) const HAVE_HQZ4_DP4A_KERNELS: bool = cfg!(has_hqz4_dp4a_kernels);

const HQZ4_CUDA_F16: u32 = 0;
const HQZ4_CUDA_F32: u32 = 1;
const CUDA_MAX_BLOCK_THREADS: usize = 1024;

#[repr(C)]
struct Hqz4Dp4aLaunch {
    input: *const c_void,
    weight: *const u8,
    weight_scales: *const f32,
    quantized_activation: *mut i8,
    activation_scales: *mut f32,
    output: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    group_offset: u64,
    seed: u64,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[cfg(has_hqz4_dp4a_kernels)]
extern "C" {
    fn launch_hqz4_dp4a(params: *const Hqz4Dp4aLaunch) -> i32;
}

struct Hqz4Dp4aMatmul {
    rows: usize,
    cols: usize,
    group_size: usize,
    group_offset: usize,
    seed: u64,
}

struct Hqz4CudaInputs<'a> {
    input: &'a CudaStorage,
    input_layout: &'a Layout,
    weight: &'a CudaStorage,
    weight_layout: &'a Layout,
    scales: &'a CudaStorage,
    scales_layout: &'a Layout,
}

impl Hqz4Dp4aMatmul {
    fn checked_u32(label: &str, value: usize) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("HQZ4 CUDA {label} exceeds the U32 launch limit."))
        })
    }

    fn validate_layouts(&self, inputs: &Hqz4CudaInputs<'_>) -> Result<(usize, Shape)> {
        if !(inputs.input_layout.is_contiguous()
            && inputs.weight_layout.is_contiguous()
            && inputs.scales_layout.is_contiguous())
        {
            candle_core::bail!("HQZ4 CUDA inputs must be contiguous.");
        }
        if inputs.weight.dtype() != DType::U8 || inputs.scales.dtype() != DType::F32 {
            candle_core::bail!(
                "HQZ4 CUDA expects U8 packed weights and F32 scales, got {:?} and {:?}.",
                inputs.weight.dtype(),
                inputs.scales.dtype()
            );
        }
        if self.group_size < 4
            || !self.group_size.is_power_of_two()
            || self.group_size > CUDA_MAX_BLOCK_THREADS
        {
            candle_core::bail!(
                "HQZ4 CUDA group size {} must be a power of two in 4..={CUDA_MAX_BLOCK_THREADS}.",
                self.group_size
            );
        }
        if !self.cols.is_multiple_of(self.group_size) {
            candle_core::bail!(
                "HQZ4 CUDA input width {} is not divisible by group size {}.",
                self.cols,
                self.group_size
            );
        }

        let input_dims = inputs.input_layout.dims();
        let Some(&input_width) = input_dims.last() else {
            candle_core::bail!("HQZ4 CUDA input must have rank at least one.");
        };
        if input_width != self.cols {
            candle_core::bail!(
                "HQZ4 CUDA input width {input_width} does not match weight width {}.",
                self.cols
            );
        }
        if inputs.weight_layout.dims() != [self.rows, self.cols / 2] {
            candle_core::bail!(
                "HQZ4 CUDA packed weight shape {:?} does not match [{}, {}].",
                inputs.weight_layout.dims(),
                self.rows,
                self.cols / 2
            );
        }
        if inputs.scales_layout.dims() != [self.rows, self.cols / self.group_size] {
            candle_core::bail!(
                "HQZ4 CUDA scale shape {:?} does not match [{}, {}].",
                inputs.scales_layout.dims(),
                self.rows,
                self.cols / self.group_size
            );
        }

        let elements = inputs.input_layout.shape().elem_count();
        if !elements.is_multiple_of(self.cols) {
            candle_core::bail!("HQZ4 CUDA input element count is not row aligned.");
        }
        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().expect("input rank checked") = self.rows;
        Ok((elements / self.cols, Shape::from_dims(&output_dims)))
    }

    fn cuda_fwd_t<T>(
        &self,
        dtype_code: u32,
        inputs: &Hqz4CudaInputs<'_>,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr + ValidAsZeroBits,
    {
        let (m, output_shape) = self.validate_layouts(inputs)?;
        let device = inputs.input.device();
        let input_slice = inputs.input.as_cuda_slice::<T>()?;
        let weight_slice = inputs.weight.as_cuda_slice::<u8>()?;
        let scale_slice = inputs.scales.as_cuda_slice::<f32>()?;
        let quantized_elements = m
            .checked_mul(self.cols)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 activation size overflow.".into()))?;
        let activation_scale_elements = m
            .checked_mul(self.cols / self.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale size overflow.".into()))?;
        let output_elements = m
            .checked_mul(self.rows)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 output size overflow.".into()))?;

        let quantized_activation = device.alloc_zeros::<i8>(quantized_elements)?;
        let activation_scales = device.alloc_zeros::<f32>(activation_scale_elements)?;
        let output = device.alloc_zeros::<T>(output_elements)?;

        let (input_ptr, _input_guard) =
            slice_ptr(input_slice, inputs.input_layout.start_offset());
        let (weight_ptr, _weight_guard) =
            slice_ptr(weight_slice, inputs.weight_layout.start_offset());
        let (scale_ptr, _scale_guard) =
            slice_ptr(scale_slice, inputs.scales_layout.start_offset());
        let (quantized_ptr, _quantized_guard) = slice_ptr(&quantized_activation, 0);
        let (activation_scale_ptr, _activation_scale_guard) =
            slice_ptr(&activation_scales, 0);
        let (output_ptr, output_guard) = slice_ptr(&output, 0);

        let params = Hqz4Dp4aLaunch {
            input: input_ptr as *const c_void,
            weight: weight_ptr as *const u8,
            weight_scales: scale_ptr as *const f32,
            quantized_activation: quantized_ptr as *mut i8,
            activation_scales: activation_scale_ptr as *mut f32,
            output: output_ptr as *mut c_void,
            m: Self::checked_u32("batch", m)?,
            n: Self::checked_u32("row count", self.rows)?,
            k: Self::checked_u32("column count", self.cols)?,
            group_size: Self::checked_u32("group size", self.group_size)?,
            group_offset: self.group_offset as u64,
            seed: self.seed,
            dtype: dtype_code,
            stream: device.cuda_stream().cu_stream(),
        };
        let status = {
            #[cfg(has_hqz4_dp4a_kernels)]
            {
                unsafe { launch_hqz4_dp4a(&params) }
            }
            #[cfg(not(has_hqz4_dp4a_kernels))]
            {
                let _ = &params;
                unreachable!("HQZ4 DP4A availability was checked before dispatch")
            }
        };
        if status != 0 {
            candle_core::bail!("HQZ4 DP4A CUDA launch failed with status {status}.");
        }

        drop(output_guard);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            output_shape,
        ))
    }
}

impl CustomOp3 for Hqz4Dp4aMatmul {
    fn name(&self) -> &'static str {
        "hqz4-dp4a-matmul"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _scales: &CpuStorage,
        _scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("HQZ4 DP4A requires CUDA storage.")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if !HAVE_HQZ4_DP4A_KERNELS {
            candle_core::bail!(
                "HQZ4 DP4A was not compiled; set CUDA_COMPUTE_CAP to at least 61."
            );
        }
        let inputs = Hqz4CudaInputs {
            input,
            input_layout,
            weight,
            weight_layout,
            scales,
            scales_layout,
        };
        match input.dtype() {
            DType::F16 => self.cuda_fwd_t::<f16>(HQZ4_CUDA_F16, &inputs),
            DType::F32 => self.cuda_fwd_t::<f32>(HQZ4_CUDA_F32, &inputs),
            dtype => candle_core::bail!(
                "HQZ4 DP4A supports F16 and F32 activations, got {dtype:?}."
            ),
        }
    }
}

pub(super) fn dp4a_matmul(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    encoded: &super::Hqz4Tensor,
) -> Result<Tensor> {
    input.apply_op3_no_bwd(
        weight,
        scales,
        &Hqz4Dp4aMatmul {
            rows: encoded.rows(),
            cols: encoded.cols(),
            group_size: encoded.group_size(),
            group_offset: encoded.group_offset(),
            seed: encoded.seed(),
        },
    )
}
