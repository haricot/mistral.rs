use candle_core::{Device, Result};
use mistralrs_kernel_runtime::universal_cuda::KernelExecutionCache;
use std::{path::PathBuf, sync::OnceLock};

const PACKAGE_ROOT_ENV: &str = "DKR_KERNEL_PACKAGES";
const MAX_HOT_PLANS_PER_THREAD: usize = 64;
static KERNEL_CACHE: OnceLock<std::result::Result<KernelExecutionCache, String>> = OnceLock::new();

struct HotPlanEntry<K, P> {
    key: K,
    device: Device,
    plan: P,
}

pub(crate) struct HotPlanCache<K, P> {
    entries: Vec<HotPlanEntry<K, P>>,
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
        self.entries.push(HotPlanEntry {
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
        let mut plans = HotPlanCache::<[u32; 2], u32>::new();
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
