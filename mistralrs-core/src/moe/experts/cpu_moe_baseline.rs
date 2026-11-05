use candle_core::{Device, Result, Tensor};
use std::sync::OnceLock;

const ENABLE_FLAG: &str = "MISTRALRS_CPU_MOE_BASELINE";

#[derive(Clone, Copy)]
pub(super) struct CpuMoeBaseline {
    enabled: bool,
}

impl CpuMoeBaseline {
    fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub(super) fn current() -> &'static Self {
        static POLICY: OnceLock<CpuMoeBaseline> = OnceLock::new();
        POLICY.get_or_init(|| {
            let enabled = std::env::var(ENABLE_FLAG)
                .ok()
                .is_some_and(|value| Self::flag_enabled(&value));
            Self::new(enabled)
        })
    }

    fn flag_enabled(value: &str) -> bool {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    }

    pub(super) fn enabled(self) -> bool {
        self.enabled
    }

    pub(super) fn expert_device(self, layer_device: &Device) -> Device {
        if self.enabled {
            Device::Cpu
        } else {
            layer_device.clone()
        }
    }

    pub(super) fn validate(self, world_size: usize) -> Result<()> {
        if self.enabled && world_size != 1 {
            candle_core::bail!("CPU MoE baseline supports only a single rank");
        }
        Ok(())
    }

    pub(super) fn prepare(
        self,
        xs: &Tensor,
        routing: Tensor,
        indices: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if self.enabled {
            Ok((
                xs.to_device(&Device::Cpu)?,
                routing.to_device(&Device::Cpu)?,
                indices.to_device(&Device::Cpu)?,
            ))
        } else {
            Ok((xs.clone(), routing, indices.clone()))
        }
    }

    pub(super) fn finish(self, ys: Tensor, output_device: &Device) -> Result<Tensor> {
        if self.enabled {
            ys.to_device(output_device)
        } else {
            Ok(ys)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opt_in_is_explicit() {
        assert!(!CpuMoeBaseline::flag_enabled("0"));
        assert!(!CpuMoeBaseline::flag_enabled("false"));
        assert!(!CpuMoeBaseline::flag_enabled(""));
        assert!(CpuMoeBaseline::flag_enabled("1"));
        assert!(CpuMoeBaseline::flag_enabled("TRUE"));
    }

    #[test]
    fn rejects_multi_rank_when_enabled() {
        assert!(CpuMoeBaseline::new(false).validate(2).is_ok());
        assert!(CpuMoeBaseline::new(true).validate(1).is_ok());
        assert!(CpuMoeBaseline::new(true).validate(2).is_err());
    }

    #[test]
    fn cpu_baseline_preserves_route_data() -> Result<()> {
        let xs = Tensor::new(&[[1f32, -2.0]], &Device::Cpu)?;
        let weights = Tensor::new(&[[0.75f32, 0.25]], &Device::Cpu)?;
        let ids = Tensor::new(&[[2u32, 0]], &Device::Cpu)?;
        let policy = CpuMoeBaseline::new(true);
        assert!(policy.expert_device(&Device::Cpu).is_cpu());
        let (xs_cpu, weights_cpu, ids_cpu) = policy.prepare(&xs, weights, &ids)?;
        assert_eq!(xs_cpu.to_vec2::<f32>()?, vec![vec![1.0, -2.0]]);
        assert_eq!(weights_cpu.to_vec2::<f32>()?, vec![vec![0.75, 0.25]]);
        assert_eq!(ids_cpu.to_vec2::<u32>()?, vec![vec![2, 0]]);
        assert_eq!(policy.finish(xs_cpu, &Device::Cpu)?.dims(), xs.dims());
        Ok(())
    }
}
