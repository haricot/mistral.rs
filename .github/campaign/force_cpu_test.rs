#[cfg(test)]
mod campaign_runtime_backend_tests {
    #[test]
    fn campaign_force_cpu_selects_cpu() {
        let device = super::init_device(true, None).unwrap();
        assert!(device.is_cpu());
    }
}
