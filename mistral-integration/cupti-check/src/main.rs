#[path = "../../../mistralrs-cli/src/commands/flow_tuner_cupti.rs"]
mod flow_tuner_cupti;
fn main() -> anyhow::Result<()> {
    if let Some(trace) = flow_tuner_cupti::BenchCuptiTrace::start_from_env()? {
        if std::env::var_os("D7_CUPTI_CHECK_DROP").is_some() {
            drop(trace);
        } else {
            trace.finish()?;
        }
    }
    Ok(())
}
