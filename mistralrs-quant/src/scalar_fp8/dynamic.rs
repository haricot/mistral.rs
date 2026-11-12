//! DKR-R1.17 scalar E4M3 conversions, with separate fixed typed ABIs.
//! No conversions/repacking/cuda module load on the prevalidated hot path.

use crate::dkr_cuda::{catalog, tensor, LaunchPlanCache};
use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut, PushKernelArg},
    CudaStorage, DType, Device, Layout, Result, Shape,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::PrevalidatedLaunchPlan,
};
use std::cell::RefCell;

const BLOCK: usize = 256;

#[derive(Clone, Copy, Eq, PartialEq)]
struct CastKey {
    source: DType,
    destination: DType,
    elements: usize,
}

thread_local! {
    static HOT_PLANS: RefCell<LaunchPlanCache<CastKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn element(dtype: DType) -> Result<ElementType> {
    match dtype {
        DType::F16 => Ok(ElementType::F16),
        DType::BF16 => Ok(ElementType::Bf16),
        DType::F32 => Ok(ElementType::F32),
        DType::F8E4M3 => Ok(ElementType::Fp8E4m3),
        _ => candle_core::bail!("unsupported DKR FP8 scalar cast {dtype:?}"),
    }
}

fn prepare(key: CastKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let direction = if key.destination == DType::F8E4M3 {
        "encode"
    } else {
        "decode"
    };
    let mut request = LaunchRequest {
        operation: format!("quant.fp8_e4m3.{direction}.v1"),
        cc: 0,
        args: vec![
            tensor(element(key.source)?, vec![key.elements]),
            tensor(element(key.destination)?, vec![key.elements]),
            Argument::Scalar { dtype: ElementType::U64 },
        ],
        grid: [
            u32::try_from(key.elements.div_ceil(BLOCK))
                .map_err(candle_core::Error::wrap)?,
            1, 1,
        ],
        max_dynamic_shared_mem_bytes: 0,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK as u32, 1, 1] {
        candle_core::bail!("invalid DKR FP8 scalar block geometry");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// A non-DKR layout returns None for the original static CUDA implementation.
pub(super) fn try_cast(
    storage: &CudaStorage,
    layout: &Layout,
    source: DType,
    destination: DType,
) -> Result<Option<(CudaStorage, Shape)>> {
    if !layout.is_contiguous()
        || layout.start_offset() != 0
        || layout.shape().elem_count() == 0
        || source != storage.dtype()
    {
        return Ok(None);
    }
    let supported = matches!(
        (source, destination),
        (DType::F8E4M3, DType::F16 | DType::BF16 | DType::F32)
            | (DType::F16 | DType::BF16 | DType::F32, DType::F8E4M3)
    );
    if !supported {
        return Ok(None);
    }
    let elements = layout.shape().elem_count();
    if elements.div_ceil(BLOCK) > u32::MAX as usize {
        return Ok(None);
    }
    let shape = layout.shape().clone();
    let device_cuda = storage.device();
    let device = Device::Cuda(device_cuda.clone());
    let stream = device_cuda.cuda_stream();
    let key = CastKey { source, destination, elements };
    let count = u64::try_from(elements).map_err(candle_core::Error::wrap)?;

    HOT_PLANS.with(|plans| {
        plans.borrow_mut().with_plan(
            key, &device,
            || prepare(key, &device),
            |plan| {
                macro_rules! launch {
                    ($input:ty, $output:ty) => {{
                        let (input_ptr, _input_guard) =
                            storage.as_cuda_slice::<$input>()?.device_ptr(&stream);
                        let mut output = unsafe { device_cuda.alloc::<$output>(elements)? };
                        {
                            let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);
                            unsafe {
                                plan.with_prevalidated_launch(|function, config| {
                                    let mut builder = stream.launch_builder(function);
                                    builder.arg(&input_ptr).arg(&output_ptr).arg(&count);
                                    builder.launch(config)?;
                                    Ok(())
                                })
                            }
                            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        }
                        Ok(Some((CudaStorage::wrap_cuda_slice(output, device_cuda.clone()), shape)))
                    }};
                }
                match (source, destination) {
                    (DType::F8E4M3, DType::F16) => launch!(float8::F8E4M3, half::f16),
                    (DType::F8E4M3, DType::BF16) => launch!(float8::F8E4M3, half::bf16),
                    (DType::F8E4M3, DType::F32) => launch!(float8::F8E4M3, f32),
                    (DType::F16, DType::F8E4M3) => launch!(half::f16, float8::F8E4M3),
                    (DType::BF16, DType::F8E4M3) => launch!(half::bf16, float8::F8E4M3),
                    (DType::F32, DType::F8E4M3) => launch!(f32, float8::F8E4M3),
                    _ => unreachable!(),
                }
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    #[ignore = "requires staged scalar FP8 E4M3 PTX on an SM61 GPU"]
    fn dkr_fp8_cast_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        // Exactly representable values and saturation endpoints.
        let values = vec![
            -448.0f32, -240.0, -32.0, -6.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 6.0, 32.0, 240.0, 448.0, 512.0,
        ];
        for source in [DType::F16, DType::BF16, DType::F32] {
            let x = Tensor::from_vec(values.clone(), values.len(), &device)?.to_dtype(source)?;
            let encoded = super::ops::dtype_to_fp8(&x)?;
            let decoded = super::ops::fp8_to_dtype(&encoded, source)?;
            let actual = decoded.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            for (original, output) in values.iter().zip(actual.iter()) {
                let expected = (*original).clamp(-448.0, 448.0);
                assert!((expected - *output).abs() < 0.001,
                    "FP8 E4M3 cast dtype={source:?}: {output} != {expected}");
            }
            let cached = HOT_PLANS.with(|hot| hot.borrow().len());
            let again = super::ops::dtype_to_fp8(&x)?;
            assert_eq!(cached, HOT_PLANS.with(|hot| hot.borrow().len()));
            let roundtrip = super::ops::fp8_to_dtype(&again, source)?;
            assert_eq!(
                actual,
                roundtrip.to_dtype(DType::F32)?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }
}
