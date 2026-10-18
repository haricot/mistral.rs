#[path = "../../../mistralrs-cli/src/commands/flow_tuner_cupti.rs"]
mod flow_tuner_cupti;
fn main() -> anyhow::Result<()> {
    if let Some(trace) = flow_tuner_cupti::BenchCuptiTrace::start_from_env()? { trace.finish()?; }
    Ok(())
}
