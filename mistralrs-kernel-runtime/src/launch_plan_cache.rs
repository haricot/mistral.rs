//! Device-aware adapter to the neutral bounded LRU plan cache.
use candle_core::{Device, Result};
use dkr_kernel_registry::PlanCache;

const MAX_LAUNCH_PLANS_PER_THREAD: usize = 64;

struct DeviceKey<K>{key:K,device:Device}
impl<K:Eq> PartialEq for DeviceKey<K>{
    fn eq(&self,other:&Self)->bool{self.key==other.key && self.device.same_device(&other.device)}
}
impl<K:Eq> Eq for DeviceKey<K>{}

pub struct LaunchPlanCache<K,P>{
    inner:PlanCache<DeviceKey<K>,P,MAX_LAUNCH_PLANS_PER_THREAD>,
}
impl<K:Copy+Eq,P> LaunchPlanCache<K,P>{
    pub const fn new()->Self{Self{inner:PlanCache::new()}}
    pub fn with_plan<T>(&mut self,key:K,device:&Device,prepare:impl FnOnce()->Result<P>,
        launch:impl FnOnce(&P)->Result<T>)->Result<T>{
        self.inner.with_plan(DeviceKey{key,device:device.clone()},prepare,launch)
    }
    pub fn len(&self)->usize{self.inner.len()}
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
