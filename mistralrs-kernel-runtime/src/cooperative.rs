//! Device-neutral arithmetic for a resident grouped cooperative grid.
//! No CUDA initialization is needed for these contract tests.

/// The operation namespace is part of the signed-by-content manifest identity.
/// Ordinary launch APIs must reject every operation in this namespace.
pub const COOPERATIVE_PREFIX: &str = "cuda.cooperative.";

pub fn requires_cooperative_launch(operation: &str) -> bool {
    operation.starts_with(COOPERATIVE_PREFIX)
}

/// Return the largest number of *complete* groups that can be simultaneously
/// resident. Zero residency is a normal pre-launch algorithm-selection outcome.
pub fn resident_group_grid(
    rows: u32,
    ctas_per_group: u32,
    active_blocks_per_sm: u32,
    sms: u32,
) -> anyhow::Result<Option<u32>> {
    anyhow::ensure!(rows > 0 && ctas_per_group > 0 && sms > 0,
        "invalid cooperative grid inputs");
    let resident = u64::from(active_blocks_per_sm) * u64::from(sms);
    let groups = u64::from(rows).min(resident / u64::from(ctas_per_group));
    if groups == 0 { return Ok(None); }
    Ok(Some(u32::try_from(groups * u64::from(ctas_per_group))?))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dkr_cooperative_grid_residency() -> anyhow::Result<()> {
        assert_eq!(resident_group_grid(17, 3, 2, 10)?, Some(18));
        assert_eq!(resident_group_grid(2, 3, 2, 10)?, Some(6));
        assert_eq!(resident_group_grid(1, 21, 2, 10)?, None);
        assert_eq!(resident_group_grid(8, 2, 0, 10)?, None);
        assert!(resident_group_grid(0, 1, 2, 10).is_err());
        assert!(resident_group_grid(1, 0, 2, 10).is_err());
        assert!(resident_group_grid(u32::MAX, u32::MAX, u32::MAX, u32::MAX).is_err());
        Ok(())
    }
    #[test]
    fn dkr_cooperative_namespace_is_not_ordinary() {
        assert!(requires_cooperative_launch("cuda.cooperative.ranked_radix.vec4.v1"));
        assert!(!requires_cooperative_launch("sampling.ranked_radix.sort_pack.v1"));
    }
}
