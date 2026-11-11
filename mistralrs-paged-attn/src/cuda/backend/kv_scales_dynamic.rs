//! DKR-R1.16: FP8 KV-cache scalar scale updates on Pascal (SM61).
//! This module only handles contiguous zero-offset inputs; all other
//! layouts retain the existing static paged-attention implementation.

use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::driver::{DevicePtr, PushKernelArg},
    CudaStorage, DType, Device, Layout, Result, Storage, Tensor,
};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType, LaunchRequest},
    universal_cuda::{KernelExecutionCache, PrevalidatedLaunchPlan},
    launch_plan_cache::LaunchPlanCache,
};
use std::{cell::RefCell, path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
const BLOCK: usize = 512;
const SHARED_BYTES: u32 = 2 * BLOCK as u32 * std::mem::size_of::<f32>() as u32;
const MAX_BLOCKS: usize = 65_535;
static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

#[derive(Clone, Copy, Eq, PartialEq)]
struct ScaleKey {
    dtype: DType,
    elements: usize,
}

thread_local! {
    static THREAD_LAUNCH_PLANS: RefCell<LaunchPlanCache<ScaleKey, PrevalidatedLaunchPlan>> =
        RefCell::new(LaunchPlanCache::new());
}

fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG
        .get_or_init(|| {
            std::env::var_os(PACKAGE_ROOT_ENV)
                .map(PathBuf::from)
                .ok_or_else(|| format!("{PACKAGE_ROOT_ENV} must point to external CUDA packages"))
                .and_then(|root| KernelExecutionCache::discover(&root)
                    .map_err(|error| error.to_string()))
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
}

fn tensor(dtype: ElementType, count: usize) -> Argument {
    Argument::Tensor {
        dtype,
        // All input buffers are normalized to flat views without copying.
        shape: vec![count],
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

fn prepare(key: ScaleKey, device: &Device) -> Result<PrevalidatedLaunchPlan> {
    let dtype = match key.dtype {
        DType::F16 => ElementType::F16,
        DType::BF16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
        _ => candle_core::bail!("DKR KV-scale update supports F16/BF16/F32 only"),
    };
    let blocks = key.elements.div_ceil(BLOCK).min(MAX_BLOCKS);
    let mut request = LaunchRequest {
        operation: "kv_cache.scale_update.atomic_max.v1".into(),
        cc: 0,
        args: vec![
            tensor(dtype, key.elements),
            tensor(dtype, key.elements),
            tensor(ElementType::F32, 1),
            tensor(ElementType::F32, 1),
            Argument::Scalar { dtype: ElementType::U64 },
        ],
        grid: [u32::try_from(blocks).map_err(candle_core::Error::wrap)?, 1, 1],
        max_dynamic_shared_mem_bytes: SHARED_BYTES,
    };
    let prepared = catalog()?
        .prepare_for_device(&mut request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
    if prepared.manifest.block != [BLOCK as u32, 1, 1]
        || prepared.manifest.shared_mem_bytes != SHARED_BYTES
    {
        candle_core::bail!("DKR KV-scale update block/shared-memory mismatch");
    }
    prepared
        .prevalidate_launch(&request, device)
        .map_err(|error| candle_core::Error::msg(error.to_string()))
}

/// Returns false only for layouts handled by the original CUDA FFI.
/// The scales are modified in-place; the external kernel allocates no GPU data.
pub(super) fn update_if_supported(
    key: &CudaStorage,
    key_layout: &Layout,
    value: &CudaStorage,
    value_layout: &Layout,
    key_scale: &Tensor,
    value_scale: &Tensor,
) -> Result<bool> {
    let dtype = key.dtype();
    let elements = key_layout.shape().elem_count();
    if elements == 0
        || key_layout.shape() != value_layout.shape()
        || !key_layout.is_contiguous()
        || !value_layout.is_contiguous()
        || key_layout.start_offset() != 0
        || value_layout.start_offset() != 0
        || dtype != value.dtype()
        || !matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
        || key_scale.dtype() != DType::F32
        || value_scale.dtype() != DType::F32
        || key_scale.elem_count() != 1
        || value_scale.elem_count() != 1
        || !key_scale.is_contiguous()
        || !value_scale.is_contiguous()
        || key_scale.layout().start_offset() != 0
        || value_scale.layout().start_offset() != 0
    {
        return Ok(false);
    }

    let cuda = key.device();
    let device = Device::Cuda(cuda.clone());
    if !device.same_device(&Device::Cuda(value.device().clone()))
        || !device.same_device(key_scale.device())
        || !device.same_device(value_scale.device())
    {
        return Ok(false);
    }

    let (k_scales_storage, _) = key_scale.storage_and_layout();
    let (v_scales_storage, _) = value_scale.storage_and_layout();
    let (Storage::Cuda(k_scales_storage), Storage::Cuda(v_scales_storage)) =
        (&*k_scales_storage, &*v_scales_storage)
    else {
        candle_core::bail!("DKR KV-scale update requires CUDA F32 scalar scales");
    };
    let stream = cuda.cuda_stream();
    let count = u64::try_from(elements).map_err(candle_core::Error::wrap)?;
    let cache_key = ScaleKey { dtype, elements };
    THREAD_LAUNCH_PLANS.with(|plans| {
        plans.borrow_mut().with_plan(
            cache_key,
            &device,
            || prepare(cache_key, &device),
            |plan| {
                macro_rules! launch {
                    ($ty:ty) => {{
                        let (key_ptr, _kg) = key.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        let (value_ptr, _vg) = value.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                        let (ks_ptr, _ksg) = k_scales_storage
                            .as_cuda_slice::<f32>()?.device_ptr(&stream);
                        let (vs_ptr, _vsg) = v_scales_storage
                            .as_cuda_slice::<f32>()?.device_ptr(&stream);
                        unsafe {
                            plan.with_prevalidated_launch(|function, config| {
                                let mut builder = stream.launch_builder(function);
                                builder.arg(&key_ptr).arg(&value_ptr)
                                    .arg(&ks_ptr).arg(&vs_ptr).arg(&count);
                                builder.launch(config)?;
                                Ok(())
                            })
                        }
                        .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                        Ok(true)
                    }};
                }
                match dtype {
                    DType::F16 => launch!(half::f16),
                    DType::BF16 => launch!(half::bf16),
                    DType::F32 => launch!(f32),
                    _ => unreachable!(),
                }
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires externally built SM61 DKR KV-scale CUDA PTX"]
    fn dkr_kv_scales_gpu() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let key = Tensor::from_vec(
                vec![-480.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                (2, 4),
                &device,
            )?.to_dtype(dtype)?;
            let value = Tensor::from_vec(
                vec![120.0f32, -60.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0],
                (2, 4),
                &device,
            )?.to_dtype(dtype)?;
            let k_scale = Tensor::from_vec(vec![0.01f32], (1,), &device)?;
            let v_scale = Tensor::from_vec(vec![0.25f32], (1,), &device)?;
            super::super::scale_update::kv_scale_update(&key, &value, &k_scale, &v_scale)?;
            let cached = THREAD_LAUNCH_PLANS.with(|plans| plans.borrow().len());
            assert!((k_scale.to_vec1::<f32>()?[0] - 2.0).abs() < 1e-6);
            assert!((v_scale.to_vec1::<f32>()?[0] - 0.5).abs() < 1e-6);
            super::super::scale_update::kv_scale_update(&key, &value, &k_scale, &v_scale)?;
            assert_eq!(cached, THREAD_LAUNCH_PLANS.with(|plans| plans.borrow().len()));
            assert!((k_scale.to_vec1::<f32>()?[0] - 2.0).abs() < 1e-6);
            assert!((v_scale.to_vec1::<f32>()?[0] - 0.5).abs() < 1e-6);
        }
        Ok(())
    }
}
