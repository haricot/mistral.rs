use crate::cuda::kernel_runtime::kernel_cache;
use candle_core::{CpuStorage, CudaStorage, DType, InplaceOp3, Layout, Result, Tensor};
use mistralrs_kernel_runtime::universal::{Argument, ElementType, LaunchRequest};

const OPERATION: &str = "tensor.indexed_row_copy.v1";

struct IndexedRowCopy {
    rows: u32,
    row_elements: u64,
}

fn tensor(dtype: ElementType, shape: Vec<usize>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

impl InplaceOp3 for IndexedRowCopy {
    fn name(&self) -> &'static str {
        "indexed-row-copy"
    }

    fn cpu_fwd(
        &self,
        _: &mut CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<()> {
        candle_core::bail!("indexed row copy requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dst_layout: &Layout,
        src: &CudaStorage,
        src_layout: &Layout,
        rows: &CudaStorage,
        rows_layout: &Layout,
    ) -> Result<()> {
        use candle_core::{
            backend::BackendStorage,
            cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
            Device,
        };

        let dtype = match dst.dtype() {
            DType::F16 => ElementType::F16,
            DType::BF16 => ElementType::Bf16,
            DType::F32 => ElementType::F32,
            dtype => candle_core::bail!("indexed row copy does not support {dtype:?}"),
        };
        let device = Device::Cuda(dst.device().clone());
        let source_rows = self.rows as usize;
        let row_elements = usize::try_from(self.row_elements)
            .map_err(|_| candle_core::Error::msg("indexed row copy row size exceeds usize"))?;
        let total = source_rows
            .checked_mul(row_elements)
            .ok_or_else(|| candle_core::Error::msg("indexed row copy element count overflow"))?;
        let dest_rows = dst_layout.dims()[0];
        let mut request = LaunchRequest {
            operation: OPERATION.into(),
            cc: 0,
            args: vec![
                tensor(dtype, vec![source_rows, row_elements]),
                tensor(dtype, vec![dest_rows, row_elements]),
                tensor(ElementType::U32, vec![source_rows]),
                Argument::Scalar {
                    dtype: ElementType::U32,
                },
                Argument::Scalar {
                    dtype: ElementType::U32,
                },
                Argument::Scalar {
                    dtype: ElementType::U64,
                },
            ],
            grid: [1, 1, 1],
            max_dynamic_shared_mem_bytes: 0,
        };
        let prepared = kernel_cache()?
            .prepare_for_device(&mut request, &device)
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        if prepared.manifest.block[1..] != [1, 1] {
            candle_core::bail!("indexed row copy requires a one-dimensional launch block")
        }
        let blocks = total.div_ceil(prepared.manifest.block[0] as usize);
        request.grid[0] = u32::try_from(blocks)
            .map_err(|_| candle_core::Error::msg("indexed row copy launch grid overflow"))?;
        let dest_rows_u32 = u32::try_from(dest_rows)
            .map_err(|_| candle_core::Error::msg("indexed row copy destination exceeds u32"))?;

        let cuda = dst.device();
        let stream = cuda.cuda_stream();
        let row_slice = rows.as_cuda_slice::<u32>()?;
        let row_view = row_slice.slice(rows_layout.start_offset()..);
        let (row_ptr, _row_guard) = row_view.device_ptr(&stream);

        macro_rules! launch {
            ($type:ty) => {{
                let src_slice = src.as_cuda_slice::<$type>()?;
                let dst_slice = dst.as_cuda_slice_mut::<$type>()?;
                let src_view = src_slice.slice(src_layout.start_offset()..);
                let mut dst_view = dst_slice.slice_mut(dst_layout.start_offset()..);
                let (src_ptr, _src_guard) = src_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_view.device_ptr_mut(&stream);
                unsafe {
                    prepared.with_verified_launch(&request, &device, |function, config| {
                        let mut builder = stream.launch_builder(function);
                        builder
                            .arg(&src_ptr)
                            .arg(&dst_ptr)
                            .arg(&row_ptr)
                            .arg(&self.rows)
                            .arg(&dest_rows_u32)
                            .arg(&self.row_elements);
                        builder.launch(config)?;
                        Ok(())
                    })
                }
                .map_err(|error| candle_core::Error::msg(error.to_string()))?
            }};
        }
        match dst.dtype() {
            DType::F16 => launch!(half::f16),
            DType::BF16 => launch!(half::bf16),
            DType::F32 => launch!(f32),
            _ => unreachable!(),
        }
        Ok(())
    }
}

pub(crate) fn copy_rows(src: &Tensor, dst: &Tensor, dst_rows: &Tensor) -> Result<()> {
    if !src.device().is_cuda()
        || !src.device().same_device(dst.device())
        || !src.device().same_device(dst_rows.device())
    {
        candle_core::bail!("indexed row copy requires CUDA tensors on one device");
    }
    if src.dtype() != dst.dtype() || dst_rows.dtype() != DType::U32 {
        candle_core::bail!(
            "indexed row copy dtype mismatch: src={:?}, dst={:?}, rows={:?}",
            src.dtype(), dst.dtype(), dst_rows.dtype()
        );
    }
    if !src.is_contiguous() || !dst.is_contiguous() || !dst_rows.is_contiguous() {
        candle_core::bail!("indexed row copy requires contiguous tensors");
    }
    let src_dims = src.dims();
    let dst_dims = dst.dims();
    let Some((&src_rows, src_row_dims)) = src_dims.split_first() else {
        candle_core::bail!("indexed row copy source must have at least one dimension");
    };
    let Some((_, dst_row_dims)) = dst_dims.split_first() else {
        candle_core::bail!("indexed row copy destination must have at least one dimension");
    };
    if src_rows != dst_rows.dim(0)? || src_row_dims != dst_row_dims {
        candle_core::bail!(
            "indexed row copy shape mismatch: src={src_dims:?}, dst={dst_dims:?}, rows={:?}",
            dst_rows.dims()
        );
    }
    if src_rows == 0 {
        return Ok(());
    }
    let row_elements = src_row_dims.iter().try_fold(1usize, |elements, dim| {
        elements
            .checked_mul(*dim)
            .ok_or_else(|| candle_core::Error::msg("indexed row copy size overflow"))
    })?;
    if row_elements == 0 {
        return Ok(());
    }
    let rows = u32::try_from(src_rows)
        .map_err(|_| candle_core::Error::msg("indexed row copy row count exceeds u32"))?;
    let row_elements = u64::try_from(row_elements)
        .map_err(|_| candle_core::Error::msg("indexed row copy row size exceeds u64"))?;
    dst.inplace_op3(src, dst_rows, &IndexedRowCopy { rows, row_elements })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires CUDA and separately built external indexed copy packages"]
    fn dkr_indexed_copy_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let expected = vec![
            vec![4., 5., 6.],
            vec![0., 0., 0.],
            vec![1., 2., 3.],
            vec![0., 0., 0.],
        ];
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let src = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device)?
                .to_dtype(dtype)?;
            let dst = Tensor::zeros((4, 3), dtype, &device)?;
            let rows = Tensor::from_vec(vec![2u32, 0], (2,), &device)?;
            copy_rows(&src, &dst, &rows)?;
            let prepared_before = kernel_cache()?
                .prepared_count()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            copy_rows(&src, &dst, &rows)?;
            let prepared_after = kernel_cache()?
                .prepared_count()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            assert_eq!(prepared_before, prepared_after);
            device.synchronize()?;
            assert_eq!(dst.to_dtype(DType::F32)?.to_vec2::<f32>()?, expected);
        }
        Ok(())
    }
}
