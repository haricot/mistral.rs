#[cfg(test)]
mod campaign_cpu_flag_tests {
    use clap::Parser;

    #[test]
    fn campaign_cpu_flag_is_available() {
        assert!(
            super::Cli::try_parse_from(["mistralrs", "run", "-m", "local-fixture", "--cpu",])
                .is_ok()
        );
    }
}
