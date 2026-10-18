use anyhow::{anyhow, bail, Context, Result};
use cudarc::{
    cupti::{
        result::{
            activity, compute_capability_supported, enable_callback, subscribe, unsubscribe,
            CuptiError,
        },
        sys::{self, CUptiResult},
    },
    driver,
};
use serde::Serialize;
use std::{
    alloc::{alloc, dealloc, Layout},
    ffi::CStr,
    fs,
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Mutex, OnceLock,
    },
};

const TRACE_SCHEMA: &str = "mistralrs.flow-tuner-cupti.v2";
const TRACE_ENV: &str = "FLOW_TUNER_CUPTI_OUT";
const TRACE_MODE_ENV: &str = "FLOW_TUNER_CUPTI_MODE";
const GPU_TIMELINE_BACKEND: &str = "cudarc-0.19.10-cupti-activity-cuda12.9-gpu-timeline-only";
const RUNTIME_DRIVER_BACKEND: &str =
    "cudarc-0.19.10-cupti-activity-cuda12.9-runtime-driver-diagnostic";
const CUPTI_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const CUPTI_BUFFER_ALIGN: usize = 8;
const FINALIZE_PENDING: u32 = u32::MAX;
const FINALIZE_CBID: u32 = sys::CUpti_driver_api_trace_cbid::CUPTI_DRIVER_TRACE_CBID_cuInit as u32;

static ACTIVITIES: OnceLock<Mutex<Vec<TraceActivity>>> = OnceLock::new();
static CALLBACK_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static DROPPED_RECORDS: AtomicU64 = AtomicU64::new(0);
static FINALIZE_REQUESTED: AtomicBool = AtomicBool::new(false);
static FINALIZE_RESULT: AtomicU32 = AtomicU32::new(FINALIZE_PENDING);

#[derive(Debug, Clone, Copy)]
enum TraceMode {
    GpuTimeline,
    RuntimeDriverDiagnostic,
}

impl TraceMode {
    fn from_env() -> Result<Self> {
        match std::env::var(TRACE_MODE_ENV)
            .unwrap_or_else(|_| "gpu-timeline".into())
            .as_str()
        {
            "gpu-timeline" => Ok(Self::GpuTimeline),
            "runtime-driver-diagnostic" => Ok(Self::RuntimeDriverDiagnostic),
            value => bail!(
                "unsupported {TRACE_MODE_ENV}={value}; expected \
                 gpu-timeline or runtime-driver-diagnostic"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GpuTimeline => "gpu-timeline",
            Self::RuntimeDriverDiagnostic => "runtime-driver-diagnostic",
        }
    }

    fn backend(self) -> &'static str {
        match self {
            Self::GpuTimeline => GPU_TIMELINE_BACKEND,
            Self::RuntimeDriverDiagnostic => RUNTIME_DRIVER_BACKEND,
        }
    }

    fn activity_kinds(self) -> Vec<sys::CUpti_ActivityKind> {
        let mut kinds = vec![
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY,
            sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET,
        ];
        if matches!(self, Self::RuntimeDriverDiagnostic) {
            kinds.extend([
                sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME,
                sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER,
            ]);
        }
        kinds
    }
}

#[derive(Debug, Serialize)]
struct TraceFile {
    schema: &'static str,
    nonce: Option<String>,
    run_id: Option<String>,
    flush_complete: bool,
    finalize_complete: bool,
    cuda_synchronized: bool,
    backend: &'static str,
    mode: &'static str,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    dropped_records: u64,
    activities: Vec<TraceActivity>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceActivity {
    kind: &'static str,
    name: String,
    start_ns: u64,
    end_ns: u64,
    device_id: Option<u32>,
    context_id: Option<u32>,
    stream_id: Option<u32>,
    correlation_id: Option<u32>,
    bytes: Option<u64>,
    callback_id: Option<u32>,
}

fn activities() -> &'static Mutex<Vec<TraceActivity>> {
    ACTIVITIES.get_or_init(|| Mutex::new(Vec::new()))
}

fn callback_error() -> &'static Mutex<Option<String>> {
    CALLBACK_ERROR.get_or_init(|| Mutex::new(None))
}

fn synchronize_primary_contexts() -> Result<()> {
    let current =
        driver::result::ctx::get_current().context("CUDA current context query failed")?;
    let count = driver::result::device::get_count().context("CUDA device count query failed")?;
    for ordinal in 0..count {
        let device = driver::result::device::get(ordinal).context("CUDA device query failed")?;
        let context = unsafe { driver::result::primary_ctx::retain(device) }
            .context("CUDA primary context retain failed")?;
        let synchronized = unsafe { driver::result::ctx::set_current(context) }
            .and_then(|()| driver::result::ctx::synchronize());
        let restored =
            unsafe { driver::result::ctx::set_current(current.unwrap_or(ptr::null_mut())) };
        let released = unsafe { driver::result::primary_ctx::release(device) };
        synchronized.context("CUDA primary context synchronization failed")?;
        restored.context("CUDA original context restore failed")?;
        released.context("CUDA primary context release failed")?;
    }
    Ok(())
}

unsafe extern "C" fn finalize_callback(
    _userdata: *mut std::ffi::c_void,
    domain: sys::CUpti_CallbackDomain,
    cbid: sys::CUpti_CallbackId,
    cbdata: *const std::ffi::c_void,
) {
    if domain != sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API
        || cbid != FINALIZE_CBID
        || cbdata.is_null()
    {
        return;
    }
    let data = unsafe { &*cbdata.cast::<sys::CUpti_CallbackData>() };
    if data.callbackSite == sys::CUpti_ApiCallbackSite::CUPTI_API_EXIT
        && FINALIZE_REQUESTED.swap(false, Ordering::AcqRel)
    {
        // CUPTI requires API-exit detachment to serialize teardown with other CUDA threads.
        let result = unsafe { sys::cuptiFinalize() };
        FINALIZE_RESULT.store(result as u32, Ordering::Release);
    }
}

fn set_callback_error(error: impl Into<String>) {
    if let Ok(mut slot) = callback_error().lock() {
        if slot.is_none() {
            *slot = Some(error.into());
        }
    }
}

fn c_string(value: *const std::ffi::c_char) -> String {
    if value.is_null() {
        return "<unnamed>".into();
    }
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

unsafe fn read_field<T: Copy>(field: *const T) -> T {
    unsafe { field.read_unaligned() }
}

unsafe fn activity_from_record(record: *mut sys::CUpti_Activity) -> Option<TraceActivity> {
    let kind = unsafe { read_field(ptr::addr_of!((*record).kind)) };

    match kind {
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL => {
            let kernel = record.cast::<sys::CUpti_ActivityKernel9>();
            let name = c_string(unsafe { read_field(ptr::addr_of!((*kernel).name)) });
            Some(TraceActivity {
                kind: "kernel",
                name,
                start_ns: unsafe { read_field(ptr::addr_of!((*kernel).start)) },
                end_ns: unsafe { read_field(ptr::addr_of!((*kernel).end)) },
                device_id: Some(unsafe { read_field(ptr::addr_of!((*kernel).deviceId)) }),
                context_id: Some(unsafe { read_field(ptr::addr_of!((*kernel).contextId)) }),
                stream_id: Some(unsafe { read_field(ptr::addr_of!((*kernel).streamId)) }),
                correlation_id: Some(unsafe { read_field(ptr::addr_of!((*kernel).correlationId)) }),
                bytes: None,
                callback_id: None,
            })
        }
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMCPY => {
            let copy = record.cast::<sys::CUpti_ActivityMemcpy6>();
            let copy_kind = unsafe { read_field(ptr::addr_of!((*copy).copyKind)) };
            Some(TraceActivity {
                kind: "memcpy",
                name: format!("memcpy_kind_{copy_kind}"),
                start_ns: unsafe { read_field(ptr::addr_of!((*copy).start)) },
                end_ns: unsafe { read_field(ptr::addr_of!((*copy).end)) },
                device_id: Some(unsafe { read_field(ptr::addr_of!((*copy).deviceId)) }),
                context_id: Some(unsafe { read_field(ptr::addr_of!((*copy).contextId)) }),
                stream_id: Some(unsafe { read_field(ptr::addr_of!((*copy).streamId)) }),
                correlation_id: Some(unsafe { read_field(ptr::addr_of!((*copy).correlationId)) }),
                bytes: Some(unsafe { read_field(ptr::addr_of!((*copy).bytes)) }),
                callback_id: None,
            })
        }
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_MEMSET => {
            let memset = record.cast::<sys::CUpti_ActivityMemset4>();
            Some(TraceActivity {
                kind: "memset",
                name: "memset".into(),
                start_ns: unsafe { read_field(ptr::addr_of!((*memset).start)) },
                end_ns: unsafe { read_field(ptr::addr_of!((*memset).end)) },
                device_id: Some(unsafe { read_field(ptr::addr_of!((*memset).deviceId)) }),
                context_id: Some(unsafe { read_field(ptr::addr_of!((*memset).contextId)) }),
                stream_id: Some(unsafe { read_field(ptr::addr_of!((*memset).streamId)) }),
                correlation_id: Some(unsafe { read_field(ptr::addr_of!((*memset).correlationId)) }),
                bytes: Some(unsafe { read_field(ptr::addr_of!((*memset).bytes)) }),
                callback_id: None,
            })
        }
        sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME
        | sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_DRIVER => {
            let api = record.cast::<sys::CUpti_ActivityAPI>();
            let callback_id: u32 = unsafe { read_field(ptr::addr_of!((*api).cbid)) };
            let is_runtime = kind == sys::CUpti_ActivityKind::CUPTI_ACTIVITY_KIND_RUNTIME;
            Some(TraceActivity {
                kind: if is_runtime { "runtime" } else { "driver" },
                name: if is_runtime {
                    format!("runtime_cbid_{callback_id}")
                } else {
                    format!("driver_cbid_{callback_id}")
                },
                start_ns: unsafe { read_field(ptr::addr_of!((*api).start)) },
                end_ns: unsafe { read_field(ptr::addr_of!((*api).end)) },
                device_id: None,
                context_id: None,
                stream_id: None,
                correlation_id: Some(unsafe { read_field(ptr::addr_of!((*api).correlationId)) }),
                bytes: None,
                callback_id: Some(callback_id),
            })
        }
        _ => None,
    }
}

fn process_buffer(buffer: *mut u8, valid_size: usize) -> std::result::Result<(), CuptiError> {
    let mut record: *mut sys::CUpti_Activity = ptr::null_mut();

    loop {
        match unsafe { activity::get_next_record(buffer, valid_size, &mut record) } {
            Ok(()) => {
                if let Some(activity_record) = unsafe { activity_from_record(record) } {
                    if let Ok(mut records) = activities().lock() {
                        records.push(activity_record);
                    } else {
                        set_callback_error("CUPTI activity record lock poisoned");
                    }
                }
            }
            Err(CuptiError(
                CUptiResult::CUPTI_ERROR_MAX_LIMIT_REACHED | CUptiResult::CUPTI_ERROR_INVALID_KIND,
            )) => break,
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

extern "C" fn buffer_requested_callback(
    buffer: *mut *mut u8,
    size: *mut usize,
    max_num_records: *mut usize,
) {
    let Ok(layout) = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN) else {
        set_callback_error("invalid CUPTI activity buffer layout");
        return;
    };

    unsafe {
        let allocated = alloc(layout);
        if allocated.is_null() {
            set_callback_error("CUPTI activity buffer allocation failed");
            return;
        }
        *buffer = allocated;
        *size = CUPTI_BUFFER_SIZE;
        *max_num_records = 0;
    }
}

extern "C" fn buffer_complete_callback(
    context: driver::sys::CUcontext,
    stream_id: u32,
    buffer: *mut u8,
    _size: usize,
    valid_size: usize,
) {
    if valid_size > 0 {
        if let Err(error) = process_buffer(buffer, valid_size) {
            set_callback_error(format!("CUPTI activity buffer parse failed: {error}"));
        }
    }

    let mut dropped = 0usize;
    if unsafe { activity::get_num_dropped_records(context, stream_id, &mut dropped) }.is_ok() {
        DROPPED_RECORDS.fetch_add(dropped as u64, Ordering::Relaxed);
    }

    if let Ok(layout) = Layout::from_size_align(CUPTI_BUFFER_SIZE, CUPTI_BUFFER_ALIGN) {
        unsafe {
            dealloc(buffer, layout);
        }
    }
}

pub(crate) struct BenchCuptiTrace {
    subscriber: usize,
    output: PathBuf,
    mode: TraceMode,
    enabled_kinds: Vec<sys::CUpti_ActivityKind>,
    active: bool,
}

impl BenchCuptiTrace {
    pub(crate) fn start_from_env() -> Result<Option<Self>> {
        let Some(output) = std::env::var_os(TRACE_ENV) else {
            return Ok(None);
        };
        let mode = TraceMode::from_env()?;

        if std::env::var_os("MISTRALRS_BENCH_CUDA_PROFILER_RANGE").is_some() {
            bail!(
                "{TRACE_ENV} cannot be combined with \
                 MISTRALRS_BENCH_CUDA_PROFILER_RANGE"
            );
        }

        let mut supported = 0;
        unsafe {
            compute_capability_supported(6, 1, &mut supported)
                .context("CUPTI compute-capability support query failed")?;
        }
        if supported == 0 {
            bail!(
                "CUPTI reports compute capability 6.1 unsupported; \
                 build/run the SM61 collector with CUDA/CUPTI 12.9, not 13.x"
            );
        }

        if let Ok(mut records) = activities().lock() {
            records.clear();
        }
        if let Ok(mut error) = callback_error().lock() {
            *error = None;
        }
        DROPPED_RECORDS.store(0, Ordering::Relaxed);

        let mut subscriber: sys::CUpti_SubscriberHandle = ptr::null_mut();
        unsafe {
            subscribe(&mut subscriber, Some(finalize_callback), ptr::null_mut())
                .context("CUPTI subscribe failed")?;
        }

        if let Err(error) = activity::register_callbacks(
            Some(buffer_requested_callback),
            Some(buffer_complete_callback),
        ) {
            unsafe {
                let _ = unsubscribe(subscriber);
            }
            return Err(anyhow!("CUPTI register callbacks failed: {error}"));
        }

        let kinds = mode.activity_kinds();

        let mut enabled = Vec::new();
        for kind in kinds {
            if let Err(error) = activity::enable(kind) {
                for previous in enabled {
                    let _ = activity::disable(previous);
                }
                unsafe {
                    let _ = unsubscribe(subscriber);
                }
                return Err(anyhow!(
                    "CUPTI activity enable failed for {kind:?}: {error}"
                ));
            }
            enabled.push(kind);
        }

        Ok(Some(Self {
            subscriber: subscriber as usize,
            output: PathBuf::from(output),
            mode,
            enabled_kinds: enabled,
            active: true,
        }))
    }

    pub(crate) fn finish(mut self) -> Result<()> {
        self.flush_disable_and_write()?;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.active = false;
        let flush_flag = unsafe {
            std::mem::transmute::<sys::CUpti_ActivityFlag, u32>(
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
            )
        };

        let mut stopped = Ok(());
        for kind in self.enabled_kinds.iter().rev().copied() {
            let disabled = activity::disable(kind)
                .with_context(|| format!("CUPTI disable failed for {kind:?}"));
            if stopped.is_ok() {
                stopped = disabled;
            }
        }
        let synchronized = synchronize_primary_contexts();
        let flushed = activity::flush_all(flush_flag).context("CUPTI activity flush failed");
        let finalize = self.finalize_at_api_exit();
        stopped.and(synchronized).and(flushed).and(finalize)
    }

    fn finalize_at_api_exit(&self) -> Result<()> {
        FINALIZE_RESULT.store(FINALIZE_PENDING, Ordering::Release);
        unsafe {
            enable_callback(
                1,
                self.subscriber as sys::CUpti_SubscriberHandle,
                sys::CUpti_CallbackDomain::CUPTI_CB_DOMAIN_DRIVER_API,
                FINALIZE_CBID,
            )
            .context("CUPTI finalize callback enable failed")?;
        }
        FINALIZE_REQUESTED.store(true, Ordering::Release);
        let trigger = driver::result::init();
        FINALIZE_REQUESTED.store(false, Ordering::Release);
        trigger.context("CUDA API exit for CUPTI finalize failed")?;
        let result = FINALIZE_RESULT.load(Ordering::Acquire);
        if result == FINALIZE_PENDING {
            bail!("CUPTI finalize callback was not invoked");
        }
        if result != CUptiResult::CUPTI_SUCCESS as u32 {
            bail!("CUPTI finalize failed with result {result}");
        }
        Ok(())
    }

    fn flush_disable_and_write(&mut self) -> Result<()> {
        self.stop()?;

        if let Some(error) = callback_error()
            .lock()
            .map_err(|_| anyhow!("CUPTI callback error lock poisoned"))?
            .take()
        {
            bail!("{error}");
        }

        let records = activities()
            .lock()
            .map_err(|_| anyhow!("CUPTI activity record lock poisoned"))?
            .clone();

        let trace = TraceFile {
            schema: TRACE_SCHEMA,
            nonce: std::env::var("D7_TRACE_NONCE").ok(),
            run_id: std::env::var("D7_TRACE_RUN_ID").ok(),
            flush_complete: true,
            finalize_complete: true,
            cuda_synchronized: true,
            backend: self.mode.backend(),
            mode: self.mode.label(),
            compute_capability_major: 6,
            compute_capability_minor: 1,
            dropped_records: DROPPED_RECORDS.load(Ordering::Relaxed),
            activities: records,
        };

        if let Some(parent) = self.output.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        fs::write(&self.output, serde_json::to_vec_pretty(&trace)?)
            .with_context(|| format!("write Flow Tuner CUPTI trace {}", self.output.display()))?;

        Ok(())
    }
}

impl Drop for BenchCuptiTrace {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let _ = self.stop();
    }
}
