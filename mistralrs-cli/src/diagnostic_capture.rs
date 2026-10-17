use anyhow::Result;
use candle_core::diagnostics::{Capture, Payload};
use std::path::PathBuf;

const DEFAULT_CAPACITY: usize = 500_000;

pub struct DiagnosticCapture {
    capture: Option<Capture>,
    path: PathBuf,
}

impl DiagnosticCapture {
    pub fn from_env() -> Result<Option<Self>> {
        let Some(path) = std::env::var_os("D7_TRACE_DIR") else {
            return Ok(None);
        };
        let run_id = std::env::var("D7_TRACE_RUN_ID")?;
        let nonce = std::env::var("D7_TRACE_NONCE")?;
        let capacity = std::env::var("D7_TRACE_CAPACITY")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(DEFAULT_CAPACITY);
        let metadata = Payload::new()
            .text("engine", "mistralrs")
            .text("build", env!("CARGO_PKG_VERSION"))
            .text("request_mapping", "unobserved")
            .text("precision", "per_tensor_descriptors");
        let nvtx = std::env::var("D7_NVTX").is_ok_and(|v| v == "1");
        let capture = Capture::start(run_id, nonce, capacity, metadata, nvtx)?;
        Ok(Some(Self {
            capture: Some(capture),
            path: path.into(),
        }))
    }
}

impl Drop for DiagnosticCapture {
    fn drop(&mut self) {
        if let Some(capture) = self.capture.take() {
            if let Err(err) = capture.finish(&self.path) {
                eprintln!("D7_TRACE_EXPORT_ERROR={err}");
            }
        }
    }
}
