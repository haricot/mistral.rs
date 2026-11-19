use metrics::{
    Counter, Gauge, Histogram, HistogramFn, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

const EVENT_METRIC: &str = "mistralrs_cuda_graph_events_total";
const CAPTURE_SECONDS_METRIC: &str = "mistralrs_cuda_decode_graph_capture_seconds";
const HOST_SECONDS_METRIC: &str = "mistralrs_dkr_decode_host_seconds";

#[derive(Default)]
struct MillisecondSamples(Mutex<Vec<f64>>);

impl HistogramFn for MillisecondSamples {
    fn record(&self, seconds: f64) {
        self.0.lock().unwrap().push(seconds * 1000.0);
    }
}

#[derive(Clone, Default)]
pub(crate) struct GraphMetrics {
    captured: Arc<AtomicU64>,
    replayed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
    fallback: Arc<AtomicU64>,
    capture_times: Arc<MillisecondSamples>,
    host_times: Arc<MillisecondSamples>,
}

#[derive(Default)]
pub(crate) struct Snapshot {
    captured: u64,
    replayed: u64,
    failed: u64,
    fallback: u64,
    capture_count: usize,
    host_count: usize,
}

impl GraphMetrics {
    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            captured: self.captured.load(Ordering::Relaxed),
            replayed: self.replayed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            fallback: self.fallback.load(Ordering::Relaxed),
            capture_count: self.capture_times.0.lock().unwrap().len(),
            host_count: self.host_times.0.lock().unwrap().len(),
        }
    }

    pub(crate) fn since(&self, before: &Snapshot) -> Value {
        let after = self.snapshot();
        let times = self.capture_times.0.lock().unwrap();
        let host_times = self.host_times.0.lock().unwrap();
        json!({
            "captures": after.captured - before.captured,
            "replays": after.replayed - before.replayed,
            "failures": after.failed - before.failed,
            "eager_fallbacks": after.fallback - before.fallback,
            "capture_ms": &times[before.capture_count..after.capture_count],
            "decode_host_ms": &host_times[before.host_count..after.host_count],
        })
    }
}

impl Recorder for GraphMetrics {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}

    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        if key.name() != EVENT_METRIC {
            return Counter::noop();
        }
        let mut component = "";
        let mut event = "";
        let mut outcome = "";
        for label in key.labels() {
            match label.key() {
                "component" => component = label.value(),
                "event" => event = label.value(),
                "outcome" => outcome = label.value(),
                _ => {}
            }
        }
        if component != "target" {
            return Counter::noop();
        }
        match (event, outcome) {
            ("capture", "success") => Counter::from_arc(self.captured.clone()),
            ("replay", "success") => Counter::from_arc(self.replayed.clone()),
            ("capture" | "replay", "failure") => Counter::from_arc(self.failed.clone()),
            ("eager_fallback", _) => Counter::from_arc(self.fallback.clone()),
            _ => Counter::noop(),
        }
    }

    fn register_gauge(&self, _: &Key, _: &Metadata<'_>) -> Gauge {
        Gauge::noop()
    }

    fn register_histogram(&self, key: &Key, _: &Metadata<'_>) -> Histogram {
        match key.name() {
            CAPTURE_SECONDS_METRIC => Histogram::from_arc(self.capture_times.clone()),
            HOST_SECONDS_METRIC => Histogram::from_arc(self.host_times.clone()),
            _ => Histogram::noop(),
        }
    }
}
