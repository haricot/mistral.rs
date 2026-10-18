use anyhow::{anyhow, bail, Context, Result};
use cudarc::{
    cupti::{
        result::{activity, compute_capability_supported, subscribe, unsubscribe, CuptiError},
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
        atomic::{AtomicU64, Ordering},
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

static ACTIVITIES: OnceLock<Mutex<Vec<TraceActivity>>> = OnceLock::new();
static CALLBACK_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();
static DROPPED_RECORDS: AtomicU64 = AtomicU64::new(0);

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
            subscribe(&mut subscriber, None, ptr::null_mut()).context("CUPTI subscribe failed")?;
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
        self.active = false;
        Ok(())
    }

    fn flush_disable_and_write(&mut self) -> Result<()> {
        let flush_flag = unsafe {
            std::mem::transmute::<sys::CUpti_ActivityFlag, u32>(
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
            )
        };

        activity::flush_all(flush_flag).context("CUPTI activity flush failed")?;

        for kind in self.enabled_kinds.iter().rev().copied() {
            activity::disable(kind)
                .with_context(|| format!("CUPTI disable failed for {kind:?}"))?;
        }

        unsafe {
            unsubscribe(self.subscriber as sys::CUpti_SubscriberHandle)
                .context("CUPTI unsubscribe failed")?;
        }

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

        let _ = activity::flush_all(unsafe {
            std::mem::transmute::<sys::CUpti_ActivityFlag, u32>(
                sys::CUpti_ActivityFlag::CUPTI_ACTIVITY_FLAG_FLUSH_FORCED,
            )
        });

        for kind in self.enabled_kinds.iter().rev().copied() {
            let _ = activity::disable(kind);
        }

        unsafe {
            let _ = unsubscribe(self.subscriber as sys::CUpti_SubscriberHandle);
        }
        self.active = false;
    }
}
