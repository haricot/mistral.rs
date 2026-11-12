//! DKR-R1.20 FP4 packed-format dequantization fixtures.
//! This is a parity/reference operation, deliberately NOT a GEMM hot-path
//! conversion. SM61 inference keeps its existing direct SIMT/DP4A kernels.

use crate::dkr_cuda::{catalog, LaunchPlanCache};
use crate::utils::slice_ptr_on_stream;
use candle_core::{
    cuda::cudarc::driver::{DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const BLOCK: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Nvfp4,
    Mxfp4,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp4 => "mxfp4",
        }
    }
    fn group(self) -> usize {
        match self {
            Self::Nvfp4 => 16,
            Self::Mxfp4 => 32,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    format: Format,
    rows: usize,
    k: usize,
}

thread_local! {
    static HOT: RefCell<LaunchPlanCache<Key, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn tensor(
    dtype: ElementType,
    shape: Vec<usize>,
    representation: Option<&str>,
) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: representation.map(str::to_owned),
    }
}

fn prepare(key: Key, device: &candle_core::Device) -> Result<PrevalidatedLaunchPlan> {
    let fmt = key.format;
    let weights_repr = match fmt {
        Format::Nvfp4 => "nvfp4_e2m1_group16.v1",
        Format::Mxfp4 => "mxfp4_e2m1_group32.v1",
    };
    let (scale_type, scale_repr) = match fmt {
        Format::Nvfp4 => (ElementType::Fp8E4m3, "nvfp4_e4m3_group16.v1"),
        Format::Mxfp4 => (ElementType::U8, "mxfp4_e8m0_group32.v1"),
    };
    let mut args = vec![
        tensor(ElementType::U8, vec![key.rows, key.k / 2], Some(weights_repr)),
        tensor(scale_type, vec![key.rows, key.k / fmt.group()], Some(scale_repr)),
    ];
    if fmt == Format::Nvfp4 {
        args.push(tensor(ElementType::F32, vec![key.rows], None));
    }
    args.extend([
        tensor(ElementType::F32, vec![key.rows, key.k], None),
        Argument::Scalar { dtype: ElementType::U32 },
        Argument::Scalar { dtype: ElementType::U32 },
    ]);
    let elements = key.rows.checked_mul(key.k)
        .ok_or_else(|| candle_core::Error::msg("FP4 output size overflow"))?;
    let mut request = LaunchRequest {
        operation: format!("quant.fp4.dequant.{}.v1", fmt.name()),
        cc: 0, args,
        grid: [u32::try_from(elements.div_ceil(BLOCK))
            .map_err(candle_core::Error::wrap)?, 1, 1],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK as u32, 1, 1] {
        candle_core::bail!("DKR FP4 dequant block geometry mismatch");
    }
    prepared.prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Debug/parity dequantization only; no conversion is inserted into inference.
fn dequant(
    fmt: Format, weights: &Tensor, scales: &Tensor, global: Option<&Tensor>,
) -> Result<Tensor> {
    let (rows, packed_k) = weights.dims2()?;
    let k = packed_k.checked_mul(2)
        .ok_or_else(|| candle_core::Error::msg("FP4 K overflow"))?;
    let groups = k / fmt.group();
    if rows == 0 || k == 0 || k % fmt.group() != 0
        || weights.dtype() != DType::U8 || !weights.is_contiguous()
        || scales.dims() != [rows, groups] || !scales.is_contiguous()
        || (fmt == Format::Nvfp4) != global.is_some()
        || (fmt == Format::Nvfp4 && scales.dtype() != DType::F8E4M3)
        || (fmt == Format::Mxfp4 && scales.dtype() != DType::U8)
        || !weights.device().same_device(scales.device())
        || [rows, k].iter().any(|&v| v > u32::MAX as usize)
    {
        candle_core::bail!("DKR FP4 format/layout contract not met");
    }
    if let Some(g) = global {
        if g.dtype() != DType::F32 || g.dims() != [rows]
            || !g.is_contiguous() || !weights.device().same_device(g.device())
        {
            candle_core::bail!("DKR NVFP4 requires F32 per-row global multipliers");
        }
    }
    let count = rows.checked_mul(k)
        .ok_or_else(|| candle_core::Error::msg("FP4 output overflow"))?;
    if count.div_ceil(BLOCK) > u32::MAX as usize {
        candle_core::bail!("DKR FP4 launch grid overflow");
    }
    let cuda = weights.device().as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let (ws, wl) = weights.storage_and_layout();
    let (ss, sl) = scales.storage_and_layout();
    let (Storage::Cuda(ws), Storage::Cuda(ss)) = (&*ws, &*ss) else {
        candle_core::bail!("DKR FP4 requires CUDA tensors");
    };
    let global_storage = global.map(|tensor| tensor.storage_and_layout());
    let (wp, _wg) = slice_ptr_on_stream(ws.as_cuda_slice::<u8>()?,
        wl.start_offset(), &stream);
    let (sp, _sg) = match fmt {
        Format::Nvfp4 => slice_ptr_on_stream(
            ss.as_cuda_slice::<float8::F8E4M3>()?, sl.start_offset(), &stream),
        Format::Mxfp4 => slice_ptr_on_stream(
            ss.as_cuda_slice::<u8>()?, sl.start_offset(), &stream),
    };
    let (gp, _gg) = if let Some((ref storage, layout)) = global_storage {
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("NVFP4 global multipliers must be CUDA");
        };
        let (pointer, guard) = slice_ptr_on_stream(
            storage.as_cuda_slice::<f32>()?, layout.start_offset(), &stream);
        (pointer, Some(guard))
    } else {
        (0u64, None)
    };
    let mut output = unsafe { cuda.alloc::<f32>(count)? };
    let row_count = u32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let width = u32::try_from(k).map_err(candle_core::Error::wrap)?;
    {
        let (op, _og) = output.device_ptr_mut(&stream);
        let key = Key { format: fmt, rows, k };
        HOT.with(|hot| {
            hot.borrow_mut().with_plan(key, weights.device(),
                || prepare(key, weights.device()),
                |plan| {
                    unsafe {
                        plan.with_prevalidated_launch(|function, config| {
                            let mut launch = stream.launch_builder(function);
                            launch.arg(&wp).arg(&sp);
                            if fmt == Format::Nvfp4 {
                                launch.arg(&gp);
                            }
                            launch.arg(&op).arg(&row_count).arg(&width);
                            launch.launch(config)?;
                            Ok(())
                        })
                    }.map_err(|err| candle_core::Error::msg(err.to_string()))
                },
            )
        })?;
    }
    let storage = CudaStorage::wrap_cuda_slice(output, cuda.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (rows, k))))
}

pub(crate) fn dequant_nvfp4(
    weight: &Tensor, e4m3_scales: &Tensor, global: &Tensor,
) -> Result<Tensor> {
    dequant(Format::Nvfp4, weight, e4m3_scales, Some(global))
}

pub(crate) fn dequant_mxfp4(
    weight: &Tensor, e8m0_scales: &Tensor,
) -> Result<Tensor> {
    dequant(Format::Mxfp4, weight, e8m0_scales, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires prebuilt FP4 dequant PTX on an SM61 GPU"]
    fn dkr_fp4_dequant_gpu() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let rows = 2usize;
        let k = 32usize;
        // Low nibble 4 = 2.0; high nibble 2 = 1.0.
        let weight = Tensor::from_vec(vec![0x24u8; rows * k/2], (rows, k/2), &dev)?;
        let nv_scales = Tensor::from_vec(
            vec![float8::F8E4M3::from_f32(1.0); rows * (k/16)],
            (rows, k/16), &dev,
        )?;
        let global = Tensor::from_vec(vec![2.0f32, 3.0], rows, &dev)?;
        let mx_scales = Tensor::from_vec(vec![127u8; rows * (k/32)],
            (rows, k/32), &dev)?;
        let nv = dequant_nvfp4(&weight, &nv_scales, &global)?
            .to_vec2::<f32>()?;
        let mx = dequant_mxfp4(&weight, &mx_scales)?
            .to_vec2::<f32>()?;
        for row in 0..rows {
            let factor = if row == 0 { 2.0 } else { 3.0 };
            for col in 0..k {
                let code = if col % 2 == 0 { 2.0 } else { 1.0 };
                assert_eq!(nv[row][col], code * factor);
                assert_eq!(mx[row][col], code);
            }
        }
        let count = HOT.with(|hot| hot.borrow().len());
        dequant_nvfp4(&weight, &nv_scales, &global)?;
        dequant_mxfp4(&weight, &mx_scales)?;
        assert_eq!(count, HOT.with(|hot| hot.borrow().len()));
        Ok(())
    }
}
