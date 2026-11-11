use candle_core::{DType, Result};
#[cfg(test)]
use candle_core::Device;
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType},
    universal_cuda::KernelExecutionCache,
};
use std::{path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
pub(crate) use mistralrs_kernel_runtime::launch_plan_cache::LaunchPlanCache;

static CATALOG: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

pub(crate) fn catalog() -> Result<&'static KernelExecutionCache> {
    CATALOG
        .get_or_init(|| {
            std::env::var_os(PACKAGE_ROOT_ENV)
                .map(PathBuf::from)
                .ok_or_else(|| format!("{PACKAGE_ROOT_ENV} must point to trusted CUDA packages"))
                .and_then(|root| {
                    KernelExecutionCache::discover(&root).map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
}

pub(crate) fn dtype(dtype: DType) -> Result<ElementType> {
    match dtype {
        DType::F16 => Ok(ElementType::F16),
        DType::BF16 => Ok(ElementType::Bf16),
        DType::F32 => Ok(ElementType::F32),
        _ => candle_core::bail!("DKR reduction requires F16, BF16 or F32"),
    }
}

pub(crate) fn tensor(dtype: ElementType, shape: Vec<usize>) -> Argument {
    Argument::Tensor {
        dtype,
        shape,
        contiguous: true,
        start_offset: 0,
        representation: None,
    }
}

#[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn dkr_reduction_hot_plan_cache() -> Result<()> {
        let prepares = Cell::new(0);
        let mut cache = LaunchPlanCache::<u32, u32>::new();
        for _ in 0..3 {
            assert_eq!(
                cache.with_plan(
                    7,
                    &Device::Cpu,
                    || {
                        prepares.set(prepares.get() + 1);
                        Ok(11)
                    },
                    |plan| Ok(*plan),
                )?,
                11
            );
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(prepares.get(), 1);
        Ok(())
    }
}
