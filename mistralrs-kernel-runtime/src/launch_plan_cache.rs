//! Validated CUDA launch plans keyed by workload signature and device.
//!
//! No CUDA interaction during cache hits. The caller supplies a preparation
//! closure only when a (signature, device) pair is not resident.

use candle_core::{Device, Result};

const MAX_LAUNCH_PLANS_PER_THREAD: usize = 64;

struct LaunchPlanEntry<K, P> {
    key: K,
    device: Device,
    plan: P,
}

pub struct LaunchPlanCache<K, P> {
    entries: Vec<LaunchPlanEntry<K, P>>,
}

impl<K: Copy + Eq, P> LaunchPlanCache<K, P> {
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn with_plan<T>(
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
        if self.entries.len() == MAX_LAUNCH_PLANS_PER_THREAD {
            self.entries.swap_remove(0);
        }
        self.entries.push(LaunchPlanEntry {
            key,
            device: device.clone(),
            plan,
        });
        launch(&self.entries.last().expect("prepared plan missing").plan)
    }

    /// Number of resident prevalidated plans, useful for cache diagnostics.
    /// Available to downstream crates even when this crate is not built as a test.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn reuses_prevalidated_plan_for_same_signature_and_device() -> Result<()> {
        let prepares = Cell::new(0);
        let mut cache = LaunchPlanCache::<[u32; 2], u32>::new();
        for _ in 0..8 {
            let value = cache.with_plan(
                [7, 13],
                &Device::Cpu,
                || {
                    prepares.set(prepares.get() + 1);
                    Ok(42)
                },
                |plan| Ok(*plan),
            )?;
            assert_eq!(value, 42);
        }
        assert_eq!(prepares.get(), 1);
        assert_eq!(cache.len(), 1);
        Ok(())
    }

    #[test]
    fn caps_number_of_resident_plans() -> Result<()> {
        let mut cache = LaunchPlanCache::<u32, u32>::new();
        for key in 0..(MAX_LAUNCH_PLANS_PER_THREAD as u32 + 3) {
            cache.with_plan(key, &Device::Cpu, || Ok(key), |plan| {
                assert_eq!(*plan, key);
                Ok(())
            })?;
        }
        assert_eq!(cache.len(), MAX_LAUNCH_PLANS_PER_THREAD);
        Ok(())
    }
}
