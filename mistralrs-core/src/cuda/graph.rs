use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Layout, Result, Storage, Tensor};

#[cfg(not(feature = "dkr-static-closure-probe"))]
fn launch(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n: usize,
    stream: i64,
) -> Result<()> {
    let status = unsafe { crate::cuda::ffi::cuda_graph_copy_bytes(src, dst, n as i64, stream) };
    if status != 0 {
        candle_core::bail!("cuda_graph_copy_bytes failed with status {status}");
    }
    Ok(())
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
fn launch_2d(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    width: usize,
    height: usize,
    src_pitch: usize,
    dst_pitch: usize,
    stream: i64,
) -> Result<()> {
    let status = unsafe {
        crate::cuda::ffi::cuda_graph_copy_2d_bytes(
            src,
            dst,
            width as i64,
            height as i64,
            src_pitch as i64,
            dst_pitch as i64,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("cuda_graph_copy_2d_bytes failed with status {status}");
    }
    Ok(())
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
fn dense_row_geometry(layout: &Layout) -> Option<(usize, usize, usize)> {
    let dims = layout.dims();
    let strides = layout.stride();
    let (&width, outer_dims) = dims.split_last()?;
    let (&last_stride, outer_strides) = strides.split_last()?;
    if outer_dims.is_empty() || last_stride != 1 || width == 0 {
        return None;
    }
    let &pitch = outer_strides.last()?;
    if pitch < width {
        return None;
    }
    let mut expected = pitch;
    for (&dim, &stride) in outer_dims.iter().zip(outer_strides).rev() {
        if dim > 1 && stride != expected {
            return None;
        }
        expected = expected.checked_mul(dim)?;
    }
    Some((width, layout.shape().elem_count() / width, pitch))
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
fn launch_layout(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    src_layout: &Layout,
    dst_layout: &Layout,
    elem_size: usize,
    stream: i64,
) -> Result<()> {
    if !dst_layout.is_contiguous() {
        candle_core::bail!("CUDA graph copy expected a contiguous destination tensor");
    }
    let n = src_layout.shape().elem_count();
    if src_layout.is_contiguous() {
        return launch(src, dst, n * elem_size, stream);
    }
    let Some((width, height, src_pitch)) = dense_row_geometry(src_layout) else {
        candle_core::bail!("CUDA graph copy source layout is not dense by row");
    };
    launch_2d(
        src,
        dst,
        width * elem_size,
        height,
        src_pitch * elem_size,
        width * elem_size,
        stream,
    )
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
macro_rules! copy_dtype {
    ($storage:expr, $layout:expr, $out_storage:expr, $out_layout:expr, $ty:ty, $stream:expr) => {{
        let src_slice = $storage.as_cuda_slice::<$ty>()?;
        let dst_slice = $out_storage.as_cuda_slice::<$ty>()?;
        let src_view = src_slice.slice($layout.start_offset()..);
        let dst_view = dst_slice.slice($out_layout.start_offset()..);
        let (src_ptr, _src_guard) = src_view.device_ptr(src_slice.stream());
        let (dst_ptr, _dst_guard) = dst_view.device_ptr(dst_slice.stream());
        launch_layout(
            src_ptr as *const core::ffi::c_void,
            dst_ptr as *mut core::ffi::c_void,
            &$layout,
            &$out_layout,
            std::mem::size_of::<$ty>(),
            $stream,
        )
    }};
}

#[cfg(not(feature = "dkr-static-closure-probe"))]
pub fn copy_tensor(src: &Tensor, dst: &Tensor) -> Result<()> {
    if src.shape() != dst.shape()
        || src.dtype() != dst.dtype()
        || src.device().location() != dst.device().location()
    {
        candle_core::bail!("CUDA graph copy expected matching tensors");
    }
    if !src.device().is_cuda() {
        candle_core::bail!("CUDA graph copy expected CUDA tensors");
    }

    let (src_storage, src_layout) = src.storage_and_layout();
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let Storage::Cuda(src_storage) = &*src_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA source storage");
    };
    let Storage::Cuda(dst_storage) = &*dst_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA destination storage");
    };

    let stream = src.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    match src.dtype() {
        DType::U8 => copy_dtype!(src_storage, src_layout, dst_storage, dst_layout, u8, stream)?,
        DType::U32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            u32,
            stream
        )?,
        DType::I16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i16,
            stream
        )?,
        DType::I32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i32,
            stream
        )?,
        DType::I64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i64,
            stream
        )?,
        DType::BF16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::bf16,
            stream
        )?,
        DType::F16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::f16,
            stream
        )?,
        DType::F32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f32,
            stream
        )?,
        DType::F64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f64,
            stream
        )?,
        dtype => candle_core::bail!("CUDA graph copy unsupported dtype {dtype:?}"),
    }
    Ok(())
}


// The strict route intentionally uses only DtoD driver copies. Arbitrary
// Candle pitches do not carry a cuMemAllocPitch provenance guarantee.
// No attempt/retry with cuMemcpy2DAsync inside a live graph capture.
#[cfg(feature = "dkr-static-closure-probe")]
pub fn copy_tensor(src: &Tensor, dst: &Tensor) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::{result, DeviceSlice};
    use mistralrs_kernel_runtime::graph_copy_geometry::CopyGeometry;

    if src.shape() != dst.shape() || src.dtype() != dst.dtype() ||
        !src.device().is_cuda() || !src.device().same_device(dst.device()) {
        candle_core::bail!("CUDA graph copy expected matching tensors on the same CUDA device");
    }
    let stream = src.device().as_cuda_device()?.cuda_stream();
    let dst_stream = dst.device().as_cuda_device()?.cuda_stream();
    if !std::sync::Arc::ptr_eq(stream.context(), dst_stream.context()) ||
        stream.cu_stream() != dst_stream.cu_stream() {
        candle_core::bail!("CUDA graph copy requires the same context and stream");
    }
    let (src_storage, src_layout) = src.storage_and_layout();
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    let Storage::Cuda(src_storage) = &*src_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA source storage");
    };
    let Storage::Cuda(dst_storage) = &*dst_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA destination storage");
    };
    macro_rules! copy_checked {
        ($ty:ty) => {{
            let s = src_storage.as_cuda_slice::<$ty>()?;
            let d = dst_storage.as_cuda_slice::<$ty>()?;
            let geometry = CopyGeometry::new(
                src_layout.dims(), src_layout.stride(), dst_layout.stride(),
                src_layout.start_offset(), dst_layout.start_offset(),
                s.len(), d.len(), std::mem::size_of::<$ty>(),
            ).map_err(|e| candle_core::Error::msg(e.to_string()))?;
            if geometry.rows == 0 { return Ok(()); }
            // The graph buffer owner retains the destination allocation and
            // exclusively schedules writes on this stream, including replay.
            // DevicePtr guards remain live through submission. In capture,
            // the existing caller disables event tracking and owns lifetimes.
            if !std::sync::Arc::ptr_eq(s.stream().context(), stream.context()) ||
                !std::sync::Arc::ptr_eq(d.stream().context(), stream.context()) ||
                s.stream().cu_stream() != stream.cu_stream() ||
                d.stream().cu_stream() != stream.cu_stream() {
                candle_core::bail!("CUDA graph copy allocation stream/context mismatch");
            }
            stream.context().bind_to_thread().map_err(candle_core::Error::wrap)?;
            let (src_ptr, src_guard) = s.device_ptr(&stream);
            let (dst_ptr, dst_guard) = d.device_ptr(&stream);
            let bound = geometry.bind(src_ptr, dst_ptr)
                .map_err(|e| candle_core::Error::msg(e.to_string()))?;
            if let Some(bound) = bound {
                for (src_addr, dst_addr, bytes) in bound.rows() {
                    // SAFETY: checked complete allocation/address ranges,
                    // non-overlap, same context/stream, and caller-owned graph
                    // buffer lifetime. On error, return immediately; capture
                    // owner must discard the graph, never retry a partial copy.
                    unsafe { result::memcpy_dtod_async(dst_addr, src_addr, bytes, stream.cu_stream()) }
                        .map_err(candle_core::Error::wrap)?;
                }
            }
            drop(dst_guard);
            drop(src_guard);
        }};
    }
    match src.dtype() {
        DType::U8 => copy_checked!(u8),
        DType::U32 => copy_checked!(u32),
        DType::I16 => copy_checked!(i16),
        DType::I32 => copy_checked!(i32),
        DType::I64 => copy_checked!(i64),
        DType::BF16 => copy_checked!(half::bf16),
        DType::F16 => copy_checked!(half::f16),
        DType::F32 => copy_checked!(f32),
        DType::F64 => copy_checked!(f64),
        dtype => candle_core::bail!("CUDA graph copy unsupported dtype {dtype:?}"),
    }
    Ok(())
}
