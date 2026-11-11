use candle_core::{DType, Device, Result};
use mistralrs_kernel_runtime::{
    universal::{Argument, ElementType},
    universal_cuda::KernelExecutionCache,
};
use std::{path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
const MAX_HOT_PLANS_PER_THREAD: usize = 64;
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

struct HotEntry<K, P> {
    key: K,
    device: Device,
    plan: P,
}

pub(crate) struct HotPlanCache<K, P> {
    entries: Vec<HotEntry<K, P>>,
}

impl<K: Copy + Eq, P> HotPlanCache<K, P> {
    pub(crate) const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub(crate) fn with_plan<T>(
        &mut self,
        key: K,
        device: &Device,
        prepare: impl FnOnce() -> Result<P>,
        launch: impl FnOnce(&P) -> Result<T>,
    ) -> Result<T> {
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key == key && entry.device.same_device(device))
        {
            let last = self.entries.len() - 1;
            self.entries.swap(index, last);
            return launch(&self.entries[last].plan);
        }
        let plan = prepare()?;
        if self.entries.len() == MAX_HOT_PLANS_PER_THREAD {
            self.entries.swap_remove(0);
        }
        self.entries.push(HotEntry {
            key,
            device: device.clone(),
            plan,
        });
        launch(&self.entries.last().expect("prepared plan missing").plan)
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
        let mut cache = HotPlanCache::<u32, u32>::new();
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
