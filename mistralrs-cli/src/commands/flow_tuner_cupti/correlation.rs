use super::{sys, CUptiResult};
use anyhow::{bail, Result};
use candle_core::diagnostics::{self, ScopeObserver};
use serde::Serialize;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};

const DRAIN_TIMEOUT: Duration = Duration::from_secs(1);
const DRAIN_POLL: Duration = Duration::from_millis(1);
pub(super) const KIND: sys::CUpti_ExternalCorrelationKind =
    sys::CUpti_ExternalCorrelationKind::CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;

#[derive(Clone, Debug, Serialize)]
pub(super) struct ClockAnchor {
    host_before_ns: u64,
    cupti_ns: u64,
    host_after_ns: u64,
}

#[derive(Debug, Serialize)]
pub(super) struct Metadata {
    pub complete: bool,
    entered_scopes: u64,
    exited_scopes: u64,
    external_kind_id: u32,
    scope_namespace: &'static str,
    host_clock: &'static str,
    api_thread_ids: &'static str,
    pub clock_anchors: Vec<ClockAnchor>,
    errors: Vec<String>,
}

pub(super) fn configure() -> Result<()> {
    if !diagnostics::enabled() || diagnostics::cpu_clock_sample().monotonic_ns.is_none() {
        bail!("correlated collection requires an active Linux semantic capture");
    }
    let code = unsafe {
        sys::cuptiSetThreadIdType(
            sys::CUpti_ActivityThreadIdType::CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM,
        )
    };
    if code != CUptiResult::CUPTI_SUCCESS {
        bail!("CUPTI system thread identity configuration failed: {code:?}");
    }
    Ok(())
}

pub(super) struct Bridge {
    accepting: AtomicBool,
    live: RwLock<bool>,
    entered: AtomicU64,
    exited: AtomicU64,
    open: AtomicU64,
    error: Mutex<Option<String>>,
    anchors: Mutex<Vec<ClockAnchor>>,
}

impl Bridge {
    pub(super) fn start() -> Result<Arc<Self>> {
        let bridge = Arc::new(Self {
            accepting: AtomicBool::new(true),
            live: RwLock::new(true),
            entered: AtomicU64::new(0),
            exited: AtomicU64::new(0),
            open: AtomicU64::new(0),
            error: Mutex::new(None),
            anchors: Mutex::new(Vec::new()),
        });
        bridge.anchor();
        diagnostics::set_scope_observer(Some(bridge.clone()))?;
        Ok(bridge)
    }

    fn fail(&self, message: impl Into<String>) {
        if let Ok(mut error) = self.error.lock() {
            if error.is_none() {
                *error = Some(message.into());
            }
        }
    }

    fn anchor(&self) {
        let before = diagnostics::cpu_clock_sample().monotonic_ns;
        let mut timestamp = 0;
        let code = unsafe { sys::cuptiGetTimestamp(&mut timestamp) };
        let after = diagnostics::cpu_clock_sample().monotonic_ns;
        if let (Some(host_before_ns), Some(host_after_ns), CUptiResult::CUPTI_SUCCESS) =
            (before, after, code)
        {
            if let Ok(mut anchors) = self.anchors.lock() {
                anchors.push(ClockAnchor {
                    host_before_ns,
                    cupti_ns: timestamp,
                    host_after_ns,
                });
            }
        } else {
            self.fail("CUPTI/host clock calibration failed");
        }
    }

    pub(super) fn stop(&self) -> Result<()> {
        self.accepting.store(false, Ordering::Release);
        if let Err(error) = diagnostics::set_scope_observer(None) {
            self.fail(error.to_string());
        }
        // Fence admitted pushes before checking the drain count.
        drop(
            self.live
                .write()
                .map_err(|_| anyhow::anyhow!("correlation gate poisoned"))?,
        );
        let deadline = Instant::now() + DRAIN_TIMEOUT;
        while self.open.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
            std::thread::sleep(DRAIN_POLL);
        }
        self.anchor();
        // Late scope destructors must never call CUPTI after finalization.
        *self
            .live
            .write()
            .map_err(|_| anyhow::anyhow!("correlation gate poisoned"))? = false;
        if self.open.load(Ordering::Acquire) != 0 {
            self.fail("external correlation scopes did not drain");
        }
        if !self.metadata().complete {
            bail!(
                "CUPTI external correlation incomplete: {:?}",
                self.metadata().errors
            );
        }
        Ok(())
    }

    pub(super) fn metadata(&self) -> Metadata {
        let errors: Vec<String> = self
            .error
            .lock()
            .ok()
            .and_then(|e| e.clone())
            .into_iter()
            .collect();
        let anchors = self.anchors.lock().map(|a| a.clone()).unwrap_or_default();
        let entered = self.entered.load(Ordering::Acquire);
        let exited = self.exited.load(Ordering::Acquire);
        Metadata {
            complete: errors.is_empty()
                && entered == exited
                && self.open.load(Ordering::Acquire) == 0
                && anchors.len() == 2,
            entered_scopes: entered,
            exited_scopes: exited,
            external_kind_id: KIND as u32,
            scope_namespace: "flow-adaptive-tuner.trace-event.v1.scope_id",
            host_clock: "linux:CLOCK_MONOTONIC",
            api_thread_ids: "system_gettid",
            clock_anchors: anchors,
            errors,
        }
    }
}

impl ScopeObserver for Bridge {
    fn enter(&self, scope_id: u64) -> bool {
        let Ok(live) = self.live.read() else {
            self.fail("correlation gate poisoned");
            return false;
        };
        if !*live || !self.accepting.load(Ordering::Acquire) {
            return false;
        }
        let code = unsafe { sys::cuptiActivityPushExternalCorrelationId(KIND, scope_id) };
        if code != CUptiResult::CUPTI_SUCCESS {
            self.fail(format!("external correlation push failed: {code:?}"));
            return false;
        }
        self.entered.fetch_add(1, Ordering::Relaxed);
        self.open.fetch_add(1, Ordering::Release);
        true
    }

    fn exit(&self, scope_id: u64) {
        if let Ok(live) = self.live.read() {
            if *live {
                let mut actual = 0;
                let code = unsafe { sys::cuptiActivityPopExternalCorrelationId(KIND, &mut actual) };
                if code != CUptiResult::CUPTI_SUCCESS || actual != scope_id {
                    self.fail(format!("external correlation pop mismatch: {code:?}, {actual}, expected {scope_id}"));
                } else {
                    self.exited.fetch_add(1, Ordering::Relaxed);
                }
            }
        } else {
            self.fail("correlation gate poisoned");
        }
        self.open.fetch_sub(1, Ordering::Release);
    }
}
