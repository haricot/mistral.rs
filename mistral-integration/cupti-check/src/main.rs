#[path = "../../../mistralrs-cli/src/commands/flow_tuner_cupti.rs"]
mod flow_tuner_cupti;
fn main() -> anyhow::Result<()> {
    #[cfg(feature = "d7-trace")]
    let capture = if std::env::var_os("D7_FAKE_CORRELATED").is_some() {
        Some(candle_core::diagnostics::Capture::start(
            std::env::var("D7_TRACE_RUN_ID")?,
            std::env::var("D7_TRACE_NONCE")?,
            4096,
            candle_core::diagnostics::Payload::new(),
            false,
        )?)
    } else {
        None
    };
    let result = (|| -> anyhow::Result<()> {
        if let Some(trace) = flow_tuner_cupti::BenchCuptiTrace::start_from_env()? {
            #[cfg(feature = "d7-trace")]
            let held = if std::env::var_os("D7_FAKE_HELD_SCOPE").is_some() {
                Some(candle_core::diagnostics::scope(
                    "op",
                    candle_core::diagnostics::Payload::new,
                ))
            } else {
                None
            };
            #[cfg(feature = "d7-trace")]
            if capture.is_some() {
                let _scope = candle_core::diagnostics::scope("op", || {
                    candle_core::diagnostics::Payload::new().text("operation", "mock_launch")
                });
                cudarc::driver::result::init()?;
                std::thread::spawn(|| {
                    let _scope = candle_core::diagnostics::scope("op", || {
                        candle_core::diagnostics::Payload::new()
                            .text("operation", "worker_mock_launch")
                    });
                    cudarc::driver::result::init().unwrap();
                })
                .join()
                .unwrap();
            }
            if std::env::var_os("D7_CUPTI_CHECK_DROP").is_some() {
                drop(trace);
            } else {
                trace.finish()?;
            }
            #[cfg(feature = "d7-trace")]
            drop(held);
        }
        Ok(())
    })();
    #[cfg(feature = "d7-trace")]
    if let Some(capture) = capture {
        capture.finish(&std::path::PathBuf::from(std::env::var("D7_TRACE_DIR")?))?;
    }
    result
}
