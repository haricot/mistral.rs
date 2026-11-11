use candle_core::Result;
use mistralrs_kernel_runtime::universal_cuda::KernelExecutionCache;
use std::{path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
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
