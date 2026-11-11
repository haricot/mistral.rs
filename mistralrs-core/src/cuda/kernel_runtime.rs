use candle_core::{Device, Result};
use mistralrs_kernel_runtime::universal_cuda::KernelExecutionCache;
use std::{path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
pub(crate) use mistralrs_kernel_runtime::launch_plan_cache::LaunchPlanCache;

static KERNEL_CACHE: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

pub(crate) fn kernel_cache() -> Result<&'static KernelExecutionCache> {
    KERNEL_CACHE
        .get_or_init(|| {
            std::env::var_os(PACKAGE_ROOT_ENV)
                .map(PathBuf::from)
                .ok_or_else(|| {
                    format!("{PACKAGE_ROOT_ENV} must point to trusted CUDA kernel packages")
                })
                .and_then(|root| {
                    KernelExecutionCache::discover(&root).map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .map_err(|error| candle_core::Error::Msg(error.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn hot_plan_reuses_preparation() -> Result<()> {
        let device = Device::Cpu;
        let prepares = Cell::new(0);
        let mut plans = LaunchPlanCache::<[u32; 2], u32>::new();
        for _ in 0..4 {
            let value = plans.with_plan(
                [2, 3],
                &device,
                || {
                    prepares.set(prepares.get() + 1);
                    Ok(17)
                },
                |plan| Ok(*plan),
            )?;
            assert_eq!(value, 17);
        }
        assert_eq!(prepares.get(), 1);
        assert_eq!(plans.len(), 1);
        plans.with_plan([3, 2], &device, || Ok(18), |plan| Ok(*plan))?;
        assert_eq!(plans.len(), 2);
        Ok(())
    }
}
