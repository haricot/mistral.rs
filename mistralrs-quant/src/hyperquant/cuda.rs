use std::{
    ffi::c_void,
    sync::{LazyLock, Mutex},
};

#[cfg(has_hqz4_dp4a_kernels)]
use std::sync::Once;

use candle_core::{
    backend::BackendStorage,
    cuda_backend::{
        cudarc::driver::{sys, CudaStream, DeviceRepr},
        CudaDType,
    },
    CpuStorage, CudaStorage, CustomOp3, DType, Device, Layout, Result, Shape, Storage,
    Tensor,
};
use half::f16;

use crate::utils::{slice_ptr, slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub(super) const HAVE_HQZ4_DP4A_KERNELS: bool = cfg!(has_hqz4_dp4a_kernels);

const HQZ4_CUDA_F16: u32 = 0;
const HQZ4_CUDA_F32: u32 = 1;
const CUDA_MAX_BLOCK_THREADS: usize = 1024;
const HQZ4_WARP_RHT_GROUP_SIZE: usize = 128;
const HQZ4_PROFILE_ENV: &str = "MISTRALRS_HQZ4_PROFILE";
const HQZ4_PROFILE_DEFAULT_INTERVAL: u64 = 256;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_RHT_A8_128: u32 = 0;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_LINEAR_DECODE: u32 = 1;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_LINEAR_PREFILL: u32 = 2;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_QKV_DECODE: u32 = 3;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_QKV_PREFILL_128: u32 = 4;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_GATE_UP_DECODE: u32 = 5;
#[cfg(has_hqz4_dp4a_kernels)]
const HQZ4_RESOURCE_GATE_UP_PREFILL: u32 = 6;

static HQZ4_PROFILE_INTERVAL: LazyLock<Option<u64>> = LazyLock::new(|| {
    let Ok(value) = std::env::var(HQZ4_PROFILE_ENV) else {
        return None;
    };
    if value == "0" || value.eq_ignore_ascii_case("off") {
        return None;
    }
    Some(
        value
            .parse()
            .ok()
            .filter(|interval| *interval != 0)
            .unwrap_or(HQZ4_PROFILE_DEFAULT_INTERVAL),
    )
});

#[derive(Clone, Copy)]
enum Hqz4ProfilePhase {
    RhtA8,
    Dp4aLinear,
    Dp4aQkv,
    Dp4aGateUp,
}

impl Hqz4ProfilePhase {
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy)]
enum Hqz4ProfileMode {
    Decode,
    Prefill,
}

impl Hqz4ProfileMode {
    fn for_rows(rows: u32) -> Self {
        if rows >= 4 {
            Self::Prefill
        } else {
            Self::Decode
        }
    }

    fn index(self) -> usize {
        self as usize
    }

    fn label(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Clone, Copy)]
struct Hqz4ProfileWork {
    mode: Hqz4ProfileMode,
    logical_bytes: u128,
    macs: u128,
    dtype: u32,
}

impl Hqz4ProfileWork {
    fn rht(rows: u32, width: u32, group_size: u32, dtype: u32) -> Self {
        let mode = Hqz4ProfileMode::for_rows(rows);
        let rows = u128::from(rows);
        let width = u128::from(width);
        let groups = width / u128::from(group_size);
        let dtype_bytes = u128::from(hqz4_dtype_bytes(dtype));
        Self {
            mode,
            logical_bytes: rows * width * (dtype_bytes + 1) + rows * groups * 4,
            macs: 0,
            dtype,
        }
    }

    fn linear(
        rows: u32,
        output_rows: u32,
        width: u32,
        group_size: u32,
        dtype: u32,
    ) -> Self {
        Self::projections(rows, output_rows, width, group_size, dtype, 1)
    }

    fn qkv(
        rows: u32,
        output_rows: u32,
        width: u32,
        group_size: u32,
        dtype: u32,
    ) -> Self {
        Self::projections(rows, output_rows, width, group_size, dtype, 1)
    }

    fn gate_up(
        rows: u32,
        output_rows: u32,
        width: u32,
        group_size: u32,
        dtype: u32,
    ) -> Self {
        Self::projections(rows, output_rows, width, group_size, dtype, 2)
    }

    fn projections(
        rows: u32,
        output_rows: u32,
        width: u32,
        group_size: u32,
        dtype: u32,
        weight_sets: u32,
    ) -> Self {
        let rows_u128 = u128::from(rows);
        let output_rows = u128::from(output_rows);
        let width = u128::from(width);
        let groups = width / u128::from(group_size);
        let weight_sets = u128::from(weight_sets);
        let dtype_bytes = u128::from(hqz4_dtype_bytes(dtype));
        let activation = rows_u128 * width + rows_u128 * groups * 4;
        let weights = weight_sets * (output_rows * width / 2 + output_rows * groups * 4);
        Self {
            mode: Hqz4ProfileMode::for_rows(rows),
            logical_bytes: activation + weights + rows_u128 * output_rows * dtype_bytes,
            macs: weight_sets * rows_u128 * output_rows * width,
            dtype,
        }
    }
}

fn hqz4_dtype_bytes(dtype: u32) -> u32 {
    if dtype == HQZ4_CUDA_F32 {
        4
    } else {
        2
    }
}

#[derive(Clone, Copy, Default)]
struct Hqz4PhaseStats {
    calls: u64,
    total_ms: f64,
    logical_bytes: u128,
    macs: u128,
}

impl Hqz4PhaseStats {
    fn average_us(self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ms * 1_000.0 / self.calls as f64
        }
    }

    fn logical_gb_s(self) -> f64 {
        if self.total_ms == 0.0 {
            0.0
        } else {
            self.logical_bytes as f64 / (self.total_ms * 1_000_000.0)
        }
    }

    fn gmac_s(self) -> f64 {
        if self.total_ms == 0.0 {
            0.0
        } else {
            self.macs as f64 / (self.total_ms * 1_000_000.0)
        }
    }
}

#[derive(Default)]
struct Hqz4ProfileWindow {
    phases: [[Hqz4PhaseStats; 4]; 2],
    mode_launches: [u64; 2],
    launches: u64,
}

static HQZ4_PROFILE_WINDOW: LazyLock<Mutex<Hqz4ProfileWindow>> =
    LazyLock::new(|| Mutex::new(Hqz4ProfileWindow::default()));
#[cfg(has_hqz4_dp4a_kernels)]
static HQZ4_RESOURCE_REPORT_F16: Once = Once::new();
#[cfg(has_hqz4_dp4a_kernels)]
static HQZ4_RESOURCE_REPORT_F32: Once = Once::new();

fn hqz4_profile_enabled() -> bool {
    HQZ4_PROFILE_INTERVAL.is_some()
}

fn hqz4_profiled_launch(
    stream: &CudaStream,
    phase: Hqz4ProfilePhase,
    work: Hqz4ProfileWork,
    launch: impl FnOnce() -> i32,
) -> i32 {
    let Some(report_interval) = *HQZ4_PROFILE_INTERVAL else {
        return launch();
    };
    crate::utils::log::once_log_info(format!(
        "HQZ4 CUDA profiling enabled via {HQZ4_PROFILE_ENV}; event timing synchronizes profiled launches."
    ));
    hqz4_report_kernel_resources(work.dtype);
    let start = match stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)) {
        Ok(event) => event,
        Err(error) => {
            crate::utils::log::once_log_warn(format!(
                "HQZ4 CUDA profiling could not create a start event: {error}"
            ));
            return launch();
        }
    };
    let status = launch();
    if status != 0 {
        return status;
    }
    let end = match stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT)) {
        Ok(event) => event,
        Err(error) => {
            crate::utils::log::once_log_warn(format!(
                "HQZ4 CUDA profiling could not create an end event: {error}"
            ));
            return status;
        }
    };
    let elapsed_ms = match start.elapsed_ms(&end) {
        Ok(elapsed_ms) => f64::from(elapsed_ms),
        Err(error) => {
            crate::utils::log::once_log_warn(format!(
                "HQZ4 CUDA profiling could not measure an event pair: {error}"
            ));
            return status;
        }
    };

    let mut window = HQZ4_PROFILE_WINDOW.lock().expect("HQZ4 profile lock poisoned");
    let mode_index = work.mode.index();
    let stats = &mut window.phases[mode_index][phase.index()];
    stats.calls += 1;
    stats.total_ms += elapsed_ms;
    stats.logical_bytes = stats.logical_bytes.saturating_add(work.logical_bytes);
    stats.macs = stats.macs.saturating_add(work.macs);
    window.mode_launches[mode_index] += 1;
    window.launches += 1;
    if window.launches >= report_interval {
        let phases = window.phases;
        let mode_launches = window.mode_launches;
        *window = Hqz4ProfileWindow::default();
        drop(window);
        for mode in [Hqz4ProfileMode::Decode, Hqz4ProfileMode::Prefill] {
            let mode_index = mode.index();
            if mode_launches[mode_index] == 0 {
                continue;
            }
            let [rht, linear, qkv, gate_up] = phases[mode_index];
            tracing::info!(
                "HQZ4 CUDA profile: mode={} launches={} rht_a8={:.3}ms/{}({:.2}us) dp4a_linear={:.3}ms/{}({:.2}us) dp4a_qkv={:.3}ms/{}({:.2}us) dp4a_gate_up={:.3}ms/{}({:.2}us)",
                mode.label(),
                mode_launches[mode_index],
                rht.total_ms,
                rht.calls,
                rht.average_us(),
                linear.total_ms,
                linear.calls,
                linear.average_us(),
                qkv.total_ms,
                qkv.calls,
                qkv.average_us(),
                gate_up.total_ms,
                gate_up.calls,
                gate_up.average_us(),
            );
            tracing::info!(
                "HQZ4 CUDA profile throughput: mode={} logical_gb_s[rht_a8={:.2},linear={:.2},qkv={:.2},gate_up={:.2}] gmac_s[linear={:.2},qkv={:.2},gate_up={:.2}]",
                mode.label(),
                rht.logical_gb_s(),
                linear.logical_gb_s(),
                qkv.logical_gb_s(),
                gate_up.logical_gb_s(),
                linear.gmac_s(),
                qkv.gmac_s(),
                gate_up.gmac_s(),
            );
        }
    }
    status
}

#[repr(C)]
#[derive(Default)]
#[cfg(has_hqz4_dp4a_kernels)]
struct Hqz4CudaKernelResources {
    registers_per_thread: u32,
    static_shared_bytes: u32,
    local_bytes_per_thread: u32,
    max_threads_per_block: u32,
    launch_threads_per_block: u32,
    active_blocks_per_sm: u32,
    max_threads_per_sm: u32,
}

fn hqz4_report_kernel_resources(dtype: u32) {
    #[cfg(has_hqz4_dp4a_kernels)]
    {
        let report = if dtype == HQZ4_CUDA_F32 {
            &HQZ4_RESOURCE_REPORT_F32
        } else {
            &HQZ4_RESOURCE_REPORT_F16
        };
        report.call_once(|| {
            let kernels = [
                ("rht_a8_128", HQZ4_RESOURCE_RHT_A8_128),
                ("linear_decode", HQZ4_RESOURCE_LINEAR_DECODE),
                ("linear_prefill_4x4", HQZ4_RESOURCE_LINEAR_PREFILL),
                ("qkv_decode", HQZ4_RESOURCE_QKV_DECODE),
                ("qkv_prefill_shared_a8_128", HQZ4_RESOURCE_QKV_PREFILL_128),
                ("gate_up_decode", HQZ4_RESOURCE_GATE_UP_DECODE),
                ("gate_up_prefill_4x4", HQZ4_RESOURCE_GATE_UP_PREFILL),
            ];
            for (name, kind) in kernels {
                let mut resources = Hqz4CudaKernelResources::default();
                let status = unsafe { hqz4_query_kernel_resources(kind, dtype, &mut resources) };
                if status != 0 {
                    tracing::warn!(
                        "HQZ4 CUDA resources: kernel={name} query failed with status {status}"
                    );
                    continue;
                }
                let resident_threads = resources
                    .active_blocks_per_sm
                    .saturating_mul(resources.launch_threads_per_block);
                let occupancy = if resources.max_threads_per_sm == 0 {
                    0.0
                } else {
                    100.0 * f64::from(resident_threads)
                        / f64::from(resources.max_threads_per_sm)
                };
                tracing::info!(
                    "HQZ4 CUDA resources: kernel={name} dtype={} threads={} max_threads_per_block={} registers_per_thread={} static_shared_bytes={} local_or_spill_bytes_per_thread={} active_blocks_per_sm={} theoretical_occupancy={occupancy:.1}%",
                    if dtype == HQZ4_CUDA_F32 { "f32" } else { "f16" },
                    resources.launch_threads_per_block,
                    resources.max_threads_per_block,
                    resources.registers_per_thread,
                    resources.static_shared_bytes,
                    resources.local_bytes_per_thread,
                    resources.active_blocks_per_sm,
                );
            }
        });
    }
    #[cfg(not(has_hqz4_dp4a_kernels))]
    let _ = dtype;
}

#[repr(C)]
struct Hqz4Dp4aLaunch {
    input: *const c_void,
    weight: *const u8,
    weight_scales: *const f32,
    quantized_activation: *mut i8,
    activation_scales: *mut f32,
    output: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    group_offset: u64,
    seed: u64,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[repr(C)]
struct Hqz4QuantizeLaunch {
    input: *const c_void,
    quantized_activation: *mut i8,
    activation_scales: *mut f32,
    m: u32,
    k: u32,
    group_size: u32,
    group_offset: u64,
    seed: u64,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[repr(C)]
struct Hqz4QuantizedMatmulLaunch {
    quantized_activation: *const i8,
    activation_scales: *const f32,
    weight: *const u8,
    weight_scales: *const f32,
    output: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[repr(C)]
struct Hqz4QkvLaunch {
    quantized_activation: *const i8,
    activation_scales: *const f32,
    q_weight: *const u8,
    q_weight_scales: *const f32,
    q_output: *mut c_void,
    q_rows: u32,
    k_weight: *const u8,
    k_weight_scales: *const f32,
    k_output: *mut c_void,
    k_rows: u32,
    v_weight: *const u8,
    v_weight_scales: *const f32,
    v_output: *mut c_void,
    v_rows: u32,
    m: u32,
    k: u32,
    group_size: u32,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[repr(C)]
struct Hqz4SiluGateUpLaunch {
    quantized_activation: *const i8,
    activation_scales: *const f32,
    gate_weight: *const u8,
    gate_weight_scales: *const f32,
    up_weight: *const u8,
    up_weight_scales: *const f32,
    output: *mut c_void,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[repr(C)]
struct Hqz4EmbeddingLaunch {
    ids: *const u32,
    weight: *const u8,
    weight_scales: *const f32,
    output: *mut c_void,
    id_count: u32,
    rows: u32,
    cols: u32,
    group_size: u32,
    group_offset: u64,
    seed: u64,
    dtype: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
}

#[cfg(has_hqz4_dp4a_kernels)]
extern "C" {
    fn launch_hqz4_dp4a(params: *const Hqz4Dp4aLaunch) -> i32;
    fn launch_hqz4_quantize(params: *const Hqz4QuantizeLaunch) -> i32;
    fn launch_hqz4_dp4a_quantized(params: *const Hqz4QuantizedMatmulLaunch) -> i32;
    fn launch_hqz4_qkv_quantized(params: *const Hqz4QkvLaunch) -> i32;
    fn launch_hqz4_silu_gate_up_quantized(params: *const Hqz4SiluGateUpLaunch) -> i32;
    fn launch_hqz4_embedding(params: *const Hqz4EmbeddingLaunch) -> i32;
    fn hqz4_query_kernel_resources(
        kind: u32,
        dtype: u32,
        resources: *mut Hqz4CudaKernelResources,
    ) -> i32;
}

struct Hqz4Dp4aMatmul {
    rows: usize,
    cols: usize,
    group_size: usize,
    group_offset: usize,
    seed: u64,
}

struct Hqz4CudaInputs<'a> {
    input: &'a CudaStorage,
    input_layout: &'a Layout,
    weight: &'a CudaStorage,
    weight_layout: &'a Layout,
    scales: &'a CudaStorage,
    scales_layout: &'a Layout,
}

struct Hqz4Embedding {
    rows: usize,
    cols: usize,
    group_size: usize,
    group_offset: usize,
    seed: u64,
    output_dtype: DType,
}

impl Hqz4Dp4aMatmul {
    fn checked_u32(label: &str, value: usize) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            candle_core::Error::Msg(format!("HQZ4 CUDA {label} exceeds the U32 launch limit."))
        })
    }

    fn validate_layouts(&self, inputs: &Hqz4CudaInputs<'_>) -> Result<(usize, Shape)> {
        if !(inputs.input_layout.is_contiguous()
            && inputs.weight_layout.is_contiguous()
            && inputs.scales_layout.is_contiguous())
        {
            candle_core::bail!("HQZ4 CUDA inputs must be contiguous.");
        }
        if inputs.weight.dtype() != DType::U8 || inputs.scales.dtype() != DType::F32 {
            candle_core::bail!(
                "HQZ4 CUDA expects U8 packed weights and F32 scales, got {:?} and {:?}.",
                inputs.weight.dtype(),
                inputs.scales.dtype()
            );
        }
        if self.group_size < 4
            || !self.group_size.is_power_of_two()
            || self.group_size > CUDA_MAX_BLOCK_THREADS
        {
            candle_core::bail!(
                "HQZ4 CUDA group size {} must be a power of two in 4..={CUDA_MAX_BLOCK_THREADS}.",
                self.group_size
            );
        }
        if !self.cols.is_multiple_of(self.group_size) {
            candle_core::bail!(
                "HQZ4 CUDA input width {} is not divisible by group size {}.",
                self.cols,
                self.group_size
            );
        }

        let input_dims = inputs.input_layout.dims();
        let Some(&input_width) = input_dims.last() else {
            candle_core::bail!("HQZ4 CUDA input must have rank at least one.");
        };
        if input_width != self.cols {
            candle_core::bail!(
                "HQZ4 CUDA input width {input_width} does not match weight width {}.",
                self.cols
            );
        }
        if inputs.weight_layout.dims() != [self.rows, self.cols / 2] {
            candle_core::bail!(
                "HQZ4 CUDA packed weight shape {:?} does not match [{}, {}].",
                inputs.weight_layout.dims(),
                self.rows,
                self.cols / 2
            );
        }
        if inputs.scales_layout.dims() != [self.rows, self.cols / self.group_size] {
            candle_core::bail!(
                "HQZ4 CUDA scale shape {:?} does not match [{}, {}].",
                inputs.scales_layout.dims(),
                self.rows,
                self.cols / self.group_size
            );
        }

        let elements = inputs.input_layout.shape().elem_count();
        if !elements.is_multiple_of(self.cols) {
            candle_core::bail!("HQZ4 CUDA input element count is not row aligned.");
        }
        let mut output_dims = input_dims.to_vec();
        *output_dims.last_mut().expect("input rank checked") = self.rows;
        Ok((elements / self.cols, Shape::from_dims(&output_dims)))
    }

    fn cuda_fwd_t<T>(
        &self,
        dtype_code: u32,
        inputs: &Hqz4CudaInputs<'_>,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr,
    {
        let (m, output_shape) = self.validate_layouts(inputs)?;
        let device = inputs.input.device();
        let input_slice = inputs.input.as_cuda_slice::<T>()?;
        let weight_slice = inputs.weight.as_cuda_slice::<u8>()?;
        let scale_slice = inputs.scales.as_cuda_slice::<f32>()?;
        let quantized_elements = m
            .checked_mul(self.cols)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 activation size overflow.".into()))?;
        let activation_scale_elements = m
            .checked_mul(self.cols / self.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale size overflow.".into()))?;
        let output_elements = m
            .checked_mul(self.rows)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 output size overflow.".into()))?;

        // Both kernels overwrite every element. Avoid three redundant memset
        // launches for every HQZ4 projection.
        let mut quantized_activation = unsafe { device.alloc::<i8>(quantized_elements)? };
        let mut activation_scales = unsafe { device.alloc::<f32>(activation_scale_elements)? };
        let mut output = unsafe { device.alloc::<T>(output_elements)? };

        let (input_ptr, _input_guard) =
            slice_ptr(input_slice, inputs.input_layout.start_offset());
        let (weight_ptr, _weight_guard) =
            slice_ptr(weight_slice, inputs.weight_layout.start_offset());
        let (scale_ptr, _scale_guard) =
            slice_ptr(scale_slice, inputs.scales_layout.start_offset());
        let stream = device.cuda_stream();
        let (quantized_ptr, _quantized_guard) =
            slice_ptr_mut_on_stream(&mut quantized_activation, 0, &stream);
        let (activation_scale_ptr, _activation_scale_guard) =
            slice_ptr_mut_on_stream(&mut activation_scales, 0, &stream);
        let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);

        let params = Hqz4Dp4aLaunch {
            input: input_ptr as *const c_void,
            weight: weight_ptr as *const u8,
            weight_scales: scale_ptr as *const f32,
            quantized_activation: quantized_ptr as *mut i8,
            activation_scales: activation_scale_ptr as *mut f32,
            output: output_ptr as *mut c_void,
            m: Self::checked_u32("batch", m)?,
            n: Self::checked_u32("row count", self.rows)?,
            k: Self::checked_u32("column count", self.cols)?,
            group_size: Self::checked_u32("group size", self.group_size)?,
            group_offset: self.group_offset as u64,
            seed: self.seed,
            dtype: dtype_code,
            stream: stream.cu_stream(),
        };
        let status = {
            #[cfg(has_hqz4_dp4a_kernels)]
            {
                if hqz4_profile_enabled() {
                    let quantize = Hqz4QuantizeLaunch {
                        input: params.input,
                        quantized_activation: params.quantized_activation,
                        activation_scales: params.activation_scales,
                        m: params.m,
                        k: params.k,
                        group_size: params.group_size,
                        group_offset: params.group_offset,
                        seed: params.seed,
                        dtype: params.dtype,
                        stream: params.stream,
                    };
                    let quantize_status = hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::RhtA8,
                        Hqz4ProfileWork::rht(
                            quantize.m,
                            quantize.k,
                            quantize.group_size,
                            quantize.dtype,
                        ),
                        || unsafe { launch_hqz4_quantize(&quantize) },
                    );
                    if quantize_status != 0 {
                        quantize_status
                    } else {
                        let matmul = Hqz4QuantizedMatmulLaunch {
                            quantized_activation: params.quantized_activation,
                            activation_scales: params.activation_scales,
                            weight: params.weight,
                            weight_scales: params.weight_scales,
                            output: params.output,
                            m: params.m,
                            n: params.n,
                            k: params.k,
                            group_size: params.group_size,
                            dtype: params.dtype,
                            stream: params.stream,
                        };
                        hqz4_profiled_launch(
                            &stream,
                            Hqz4ProfilePhase::Dp4aLinear,
                            Hqz4ProfileWork::linear(
                                matmul.m,
                                matmul.n,
                                matmul.k,
                                matmul.group_size,
                                matmul.dtype,
                            ),
                            || unsafe { launch_hqz4_dp4a_quantized(&matmul) },
                        )
                    }
                } else {
                    unsafe { launch_hqz4_dp4a(&params) }
                }
            }
            #[cfg(not(has_hqz4_dp4a_kernels))]
            {
                let _ = &params;
                unreachable!("HQZ4 DP4A availability was checked before dispatch")
            }
        };
        if status != 0 {
            candle_core::bail!("HQZ4 DP4A CUDA launch failed with status {status}.");
        }

        drop(output_guard);
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            output_shape,
        ))
    }
}

impl CustomOp3 for Hqz4Dp4aMatmul {
    fn name(&self) -> &'static str {
        "hqz4-dp4a-matmul"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _scales: &CpuStorage,
        _scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("HQZ4 DP4A requires CUDA storage.")
    }

    fn cuda_fwd(
        &self,
        input: &CudaStorage,
        input_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if !HAVE_HQZ4_DP4A_KERNELS {
            candle_core::bail!(
                "HQZ4 DP4A was not compiled; set CUDA_COMPUTE_CAP to at least 61."
            );
        }
        let inputs = Hqz4CudaInputs {
            input,
            input_layout,
            weight,
            weight_layout,
            scales,
            scales_layout,
        };
        match input.dtype() {
            DType::F16 => self.cuda_fwd_t::<f16>(HQZ4_CUDA_F16, &inputs),
            DType::F32 => self.cuda_fwd_t::<f32>(HQZ4_CUDA_F32, &inputs),
            dtype => candle_core::bail!(
                "HQZ4 DP4A supports F16 and F32 activations, got {dtype:?}."
            ),
        }
    }
}

impl Hqz4Embedding {
    fn cuda_fwd_t<T>(
        &self,
        dtype_code: u32,
        ids: &CudaStorage,
        ids_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)>
    where
        T: CudaDType + DeviceRepr,
    {
        if !(ids_layout.is_contiguous()
            && weight_layout.is_contiguous()
            && scales_layout.is_contiguous())
        {
            candle_core::bail!("HQZ4 CUDA embedding inputs must be contiguous.");
        }
        if ids.dtype() != DType::U32 {
            candle_core::bail!(
                "HQZ4 CUDA embedding expects U32 ids, got {:?}.",
                ids.dtype()
            );
        }
        if weight.dtype() != DType::U8 || scales.dtype() != DType::F32 {
            candle_core::bail!(
                "HQZ4 CUDA embedding expects U8 weights and F32 scales, got {:?} and {:?}.",
                weight.dtype(),
                scales.dtype()
            );
        }
        if self.group_size < 4
            || !self.group_size.is_power_of_two()
            || self.group_size > CUDA_MAX_BLOCK_THREADS
            || !self.cols.is_multiple_of(self.group_size)
        {
            candle_core::bail!("Invalid HQZ4 CUDA embedding group layout.");
        }
        if weight_layout.dims() != [self.rows, self.cols / 2]
            || scales_layout.dims() != [self.rows, self.cols / self.group_size]
        {
            candle_core::bail!("HQZ4 CUDA embedding weight layout mismatch.");
        }

        let id_count = ids_layout.shape().elem_count();
        let groups = id_count
            .checked_mul(self.cols / self.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 embedding grid overflow.".into()))?;
        Hqz4Dp4aMatmul::checked_u32("embedding grid", groups)?;
        let output_elements = id_count
            .checked_mul(self.cols)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 embedding size overflow.".into()))?;
        let device = ids.device();
        let ids_slice = ids.as_cuda_slice::<u32>()?;
        let weight_slice = weight.as_cuda_slice::<u8>()?;
        let scales_slice = scales.as_cuda_slice::<f32>()?;
        let mut output = unsafe { device.alloc::<T>(output_elements)? };
        if id_count == 0 {
            return Ok((
                CudaStorage::wrap_cuda_slice(output, device.clone()),
                Shape::from_dims(&[0, self.cols]),
            ));
        }

        let (ids_ptr, _ids_guard) = slice_ptr(ids_slice, ids_layout.start_offset());
        let (weight_ptr, _weight_guard) =
            slice_ptr(weight_slice, weight_layout.start_offset());
        let (scales_ptr, _scales_guard) =
            slice_ptr(scales_slice, scales_layout.start_offset());
        let stream = device.cuda_stream();
        let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
        let params = Hqz4EmbeddingLaunch {
            ids: ids_ptr as *const u32,
            weight: weight_ptr as *const u8,
            weight_scales: scales_ptr as *const f32,
            output: output_ptr as *mut c_void,
            id_count: Hqz4Dp4aMatmul::checked_u32("embedding id count", id_count)?,
            rows: Hqz4Dp4aMatmul::checked_u32("embedding row count", self.rows)?,
            cols: Hqz4Dp4aMatmul::checked_u32("embedding column count", self.cols)?,
            group_size: Hqz4Dp4aMatmul::checked_u32(
                "embedding group size",
                self.group_size,
            )?,
            group_offset: self.group_offset as u64,
            seed: self.seed,
            dtype: dtype_code,
            stream: stream.cu_stream(),
        };
        let status = {
            #[cfg(has_hqz4_dp4a_kernels)]
            {
                unsafe { launch_hqz4_embedding(&params) }
            }
            #[cfg(not(has_hqz4_dp4a_kernels))]
            {
                let _ = &params;
                unreachable!("HQZ4 DP4A availability was checked before dispatch")
            }
        };
        if status != 0 {
            candle_core::bail!("HQZ4 embedding CUDA launch failed with status {status}.");
        }
        drop(output_guard);

        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            Shape::from_dims(&[id_count, self.cols]),
        ))
    }
}

impl CustomOp3 for Hqz4Embedding {
    fn name(&self) -> &'static str {
        "hqz4-embedding"
    }

    fn cpu_fwd(
        &self,
        _ids: &CpuStorage,
        _ids_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _scales: &CpuStorage,
        _scales_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("HQZ4 CUDA embedding requires CUDA storage.")
    }

    fn cuda_fwd(
        &self,
        ids: &CudaStorage,
        ids_layout: &Layout,
        weight: &CudaStorage,
        weight_layout: &Layout,
        scales: &CudaStorage,
        scales_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if !HAVE_HQZ4_DP4A_KERNELS {
            candle_core::bail!(
                "HQZ4 embedding CUDA kernel was not compiled; set CUDA_COMPUTE_CAP to at least 61."
            );
        }
        match self.output_dtype {
            DType::F16 => self.cuda_fwd_t::<f16>(
                HQZ4_CUDA_F16,
                ids,
                ids_layout,
                weight,
                weight_layout,
                scales,
                scales_layout,
            ),
            DType::F32 => self.cuda_fwd_t::<f32>(
                HQZ4_CUDA_F32,
                ids,
                ids_layout,
                weight,
                weight_layout,
                scales,
                scales_layout,
            ),
            dtype => candle_core::bail!(
                "HQZ4 CUDA embedding supports F16 and F32 output, got {dtype:?}."
            ),
        }
    }
}

pub(super) fn dp4a_matmul(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    encoded: &super::Hqz4Tensor,
) -> Result<Tensor> {
    if encoded.cols() != 0 && input.elem_count() / encoded.cols() >= 4 {
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using tiled 4x4 A8/W4 DP4A prefill backend (SM61+).",
        );
    }
    input.apply_op3_no_bwd(
        weight,
        scales,
        &Hqz4Dp4aMatmul {
            rows: encoded.rows(),
            cols: encoded.cols(),
            group_size: encoded.group_size(),
            group_offset: encoded.group_offset(),
            seed: encoded.seed(),
        },
    )
}

pub(super) fn quantize_activation(
    input: &Tensor,
    encoded: &super::Hqz4Tensor,
) -> Result<(Tensor, Tensor)> {
    if !HAVE_HQZ4_DP4A_KERNELS {
        candle_core::bail!(
            "HQZ4 activation quantization was not compiled; set CUDA_COMPUTE_CAP to at least 61."
        );
    }
    let input = input.contiguous()?;
    let Device::Cuda(device) = input.device() else {
        candle_core::bail!("HQZ4 activation quantization requires CUDA storage.");
    };
    let Some(&k) = input.dims().last() else {
        candle_core::bail!("HQZ4 activation input must have rank at least one.");
    };
    if k != encoded.cols() || !k.is_multiple_of(encoded.group_size()) {
        candle_core::bail!(
            "HQZ4 activation width {k} does not match encoded width {} and group size {}.",
            encoded.cols(),
            encoded.group_size()
        );
    }
    let elements = input.elem_count();
    if elements == 0 || !elements.is_multiple_of(k) {
        candle_core::bail!("HQZ4 activation input must contain complete nonempty rows.");
    }
    let m = elements / k;
    let scale_count = m
        .checked_mul(k / encoded.group_size())
        .ok_or_else(|| candle_core::Error::msg("HQZ4 activation scale count overflow."))?;
    let stream = device.cuda_stream();
    // Candle has no signed 8-bit tensor dtype. The kernel writes two's-
    // complement i8 values into U8 storage and the DP4A consumer reinterprets
    // the same bytes as signed values.
    let mut quantized = unsafe { device.alloc::<u8>(elements)? };
    let mut scales = unsafe { device.alloc::<f32>(scale_count)? };
    let (quantized_ptr, quantized_guard) =
        slice_ptr_mut_on_stream(&mut quantized, 0, &stream);
    let (scales_ptr, scales_guard) =
        slice_ptr_mut_on_stream(&mut scales, 0, &stream);
    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    if encoded.group_size() == HQZ4_WARP_RHT_GROUP_SIZE {
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using the register-vectorized warp-shuffle RHT/A8 kernel.",
        );
    }

    let status = match input.dtype() {
        DType::F16 => {
            let input = input_storage.as_cuda_slice::<f16>()?;
            let (input_ptr, input_guard) =
                slice_ptr_on_stream(input, input_layout.start_offset(), &stream);
            let params = Hqz4QuantizeLaunch {
                input: input_ptr as *const c_void,
                quantized_activation: quantized_ptr as *mut i8,
                activation_scales: scales_ptr as *mut f32,
                m: Hqz4Dp4aMatmul::checked_u32("activation row count", m)?,
                k: Hqz4Dp4aMatmul::checked_u32("activation width", k)?,
                group_size: Hqz4Dp4aMatmul::checked_u32(
                    "activation group size",
                    encoded.group_size(),
                )?,
                group_offset: encoded.group_offset() as u64,
                seed: encoded.seed(),
                dtype: HQZ4_CUDA_F16,
                stream: stream.cu_stream(),
            };
            let status = {
                #[cfg(has_hqz4_dp4a_kernels)]
                {
                    hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::RhtA8,
                        Hqz4ProfileWork::rht(
                            params.m,
                            params.k,
                            params.group_size,
                            params.dtype,
                        ),
                        || unsafe { launch_hqz4_quantize(&params) },
                    )
                }
                #[cfg(not(has_hqz4_dp4a_kernels))]
                {
                    let _ = &params;
                    unreachable!("HQZ4 DP4A availability was checked before dispatch")
                }
            };
            drop(input_guard);
            status
        }
        DType::F32 => {
            let input = input_storage.as_cuda_slice::<f32>()?;
            let (input_ptr, input_guard) =
                slice_ptr_on_stream(input, input_layout.start_offset(), &stream);
            let params = Hqz4QuantizeLaunch {
                input: input_ptr as *const c_void,
                quantized_activation: quantized_ptr as *mut i8,
                activation_scales: scales_ptr as *mut f32,
                m: Hqz4Dp4aMatmul::checked_u32("activation row count", m)?,
                k: Hqz4Dp4aMatmul::checked_u32("activation width", k)?,
                group_size: Hqz4Dp4aMatmul::checked_u32(
                    "activation group size",
                    encoded.group_size(),
                )?,
                group_offset: encoded.group_offset() as u64,
                seed: encoded.seed(),
                dtype: HQZ4_CUDA_F32,
                stream: stream.cu_stream(),
            };
            let status = {
                #[cfg(has_hqz4_dp4a_kernels)]
                {
                    hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::RhtA8,
                        Hqz4ProfileWork::rht(
                            params.m,
                            params.k,
                            params.group_size,
                            params.dtype,
                        ),
                        || unsafe { launch_hqz4_quantize(&params) },
                    )
                }
                #[cfg(not(has_hqz4_dp4a_kernels))]
                {
                    let _ = &params;
                    unreachable!("HQZ4 DP4A availability was checked before dispatch")
                }
            };
            drop(input_guard);
            status
        }
        dtype => candle_core::bail!(
            "HQZ4 activation quantization supports F16 and F32 inputs, got {dtype:?}."
        ),
    };
    if status != 0 {
        candle_core::bail!(
            "HQZ4 activation quantization CUDA launch failed with status {status}."
        );
    }
    drop((quantized_guard, scales_guard));

    let quantized = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(quantized, device.clone())),
        Shape::from_dims(&[m, k]),
    ));
    let scales = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, device.clone())),
        Shape::from_dims(&[m, k / encoded.group_size()]),
    ));
    Ok((quantized, scales))
}

pub(super) fn dp4a_matmul_quantized(
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    encoded: &super::Hqz4Tensor,
    source_shape: &[usize],
    source_dtype: DType,
) -> Result<Tensor> {
    if !HAVE_HQZ4_DP4A_KERNELS {
        candle_core::bail!(
            "HQZ4 quantized matmul was not compiled; set CUDA_COMPUTE_CAP to at least 61."
        );
    }
    if activation.dtype() != DType::U8
        || activation_scales.dtype() != DType::F32
        || weight.dtype() != DType::U8
        || weight_scales.dtype() != DType::F32
    {
        candle_core::bail!(
            "HQZ4 quantized matmul expects signed-byte U8/F32 activations and U8/F32 weights."
        );
    }
    if !(activation.is_contiguous()
        && activation_scales.is_contiguous()
        && weight.is_contiguous()
        && weight_scales.is_contiguous())
    {
        candle_core::bail!("HQZ4 quantized matmul inputs must be contiguous.");
    }
    if !activation.device().same_device(activation_scales.device())
        || !activation.device().same_device(weight.device())
        || !activation.device().same_device(weight_scales.device())
    {
        candle_core::bail!(
            "HQZ4 quantized matmul inputs must use the same CUDA device."
        );
    }
    let Device::Cuda(device) = activation.device() else {
        candle_core::bail!("HQZ4 quantized matmul requires CUDA storage.");
    };
    let (m, k) = activation.dims2()?;
    let n = encoded.rows();
    let groups_per_row = k / encoded.group_size();
    if k != encoded.cols()
        || activation_scales.dims() != [m, groups_per_row]
        || weight.dims() != [n, k / 2]
        || weight_scales.dims() != [n, groups_per_row]
    {
        candle_core::bail!("HQZ4 quantized matmul shape mismatch.");
    }
    let Some((&source_k, source_batch)) = source_shape.split_last() else {
        candle_core::bail!("HQZ4 source activation shape cannot be empty.");
    };
    let source_rows = source_batch
        .iter()
        .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
        .ok_or_else(|| candle_core::Error::msg("HQZ4 source activation shape overflow."))?;
    if source_k != k || source_rows != m {
        candle_core::bail!(
            "HQZ4 source shape {:?} does not match quantized activation [{m}, {k}].",
            source_shape
        );
    }

    let stream = device.cuda_stream();
    let (activation_storage, activation_layout) = activation.storage_and_layout();
    let Storage::Cuda(activation_storage) = &*activation_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let activation = activation_storage.as_cuda_slice::<u8>()?;
    let (activation_ptr, activation_guard) =
        slice_ptr_on_stream(activation, activation_layout.start_offset(), &stream);
    let (scale_storage, scale_layout) = activation_scales.storage_and_layout();
    let Storage::Cuda(scale_storage) = &*scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let scales = scale_storage.as_cuda_slice::<f32>()?;
    let (activation_scale_ptr, activation_scale_guard) =
        slice_ptr_on_stream(scales, scale_layout.start_offset(), &stream);
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Cuda(weight_storage) = &*weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let weights = weight_storage.as_cuda_slice::<u8>()?;
    let (weight_ptr, weight_guard) =
        slice_ptr_on_stream(weights, weight_layout.start_offset(), &stream);
    let (weight_scale_storage, weight_scale_layout) = weight_scales.storage_and_layout();
    let Storage::Cuda(weight_scale_storage) = &*weight_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let scales = weight_scale_storage.as_cuda_slice::<f32>()?;
    let (weight_scale_ptr, weight_scale_guard) =
        slice_ptr_on_stream(scales, weight_scale_layout.start_offset(), &stream);

    let mut output_shape = source_shape.to_vec();
    *output_shape
        .last_mut()
        .expect("source activation rank checked") = n;
    let output_elements = m
        .checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("HQZ4 quantized output size overflow."))?;
    if m >= 4 {
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using tiled 4x4 A8/W4 DP4A prefill backend (SM61+).",
        );
    }

    macro_rules! launch_output {
        ($ty:ty, $dtype_code:expr) => {{
            let mut output = unsafe { device.alloc::<$ty>(output_elements)? };
            let (output_ptr, output_guard) =
                slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let params = Hqz4QuantizedMatmulLaunch {
                quantized_activation: activation_ptr as *const i8,
                activation_scales: activation_scale_ptr as *const f32,
                weight: weight_ptr as *const u8,
                weight_scales: weight_scale_ptr as *const f32,
                output: output_ptr as *mut c_void,
                m: Hqz4Dp4aMatmul::checked_u32("quantized row count", m)?,
                n: Hqz4Dp4aMatmul::checked_u32("quantized output width", n)?,
                k: Hqz4Dp4aMatmul::checked_u32("quantized input width", k)?,
                group_size: Hqz4Dp4aMatmul::checked_u32(
                    "quantized group size",
                    encoded.group_size(),
                )?,
                dtype: $dtype_code,
                stream: stream.cu_stream(),
            };
            let status = {
                #[cfg(has_hqz4_dp4a_kernels)]
                {
                    hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::Dp4aLinear,
                        Hqz4ProfileWork::linear(
                            params.m,
                            params.n,
                            params.k,
                            params.group_size,
                            params.dtype,
                        ),
                        || unsafe { launch_hqz4_dp4a_quantized(&params) },
                    )
                }
                #[cfg(not(has_hqz4_dp4a_kernels))]
                {
                    let _ = &params;
                    unreachable!("HQZ4 DP4A availability was checked before dispatch")
                }
            };
            if status != 0 {
                candle_core::bail!(
                    "HQZ4 prequantized DP4A CUDA launch failed with status {status}."
                );
            }
            drop(output_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                Shape::from_dims(&output_shape),
            ))
        }};
    }

    let output = match source_dtype {
        DType::F16 | DType::BF16 => launch_output!(f16, HQZ4_CUDA_F16),
        DType::F32 => launch_output!(f32, HQZ4_CUDA_F32),
        dtype => candle_core::bail!(
            "HQZ4 prequantized output supports F16, BF16, and F32, got {dtype:?}."
        ),
    };
    drop((
        activation_guard,
        activation_scale_guard,
        weight_guard,
        weight_scale_guard,
    ));
    if output.dtype() == source_dtype {
        Ok(output)
    } else {
        output.to_dtype(source_dtype)
    }
}

fn validate_multi_projection_inputs(
    activation: &Tensor,
    activation_scales: &Tensor,
    weights: &[&super::Hqz4CudaInner],
    source_shape: &[usize],
) -> Result<(usize, usize, usize)> {
    if !HAVE_HQZ4_DP4A_KERNELS {
        candle_core::bail!(
            "HQZ4 multi-projection matmul was not compiled; set CUDA_COMPUTE_CAP to at least 61."
        );
    }
    if activation.dtype() != DType::U8 || activation_scales.dtype() != DType::F32 {
        candle_core::bail!(
            "HQZ4 multi-projection matmul expects signed-byte U8/F32 activations."
        );
    }
    if !activation.is_contiguous() || !activation_scales.is_contiguous() {
        candle_core::bail!("HQZ4 multi-projection activations must be contiguous.");
    }
    let (m, k) = activation.dims2()?;
    let Some(first) = weights.first() else {
        candle_core::bail!("HQZ4 multi-projection matmul requires at least one weight.");
    };
    let group_size = first.group_size;
    if m == 0 || k == 0 || group_size == 0 || !k.is_multiple_of(group_size) {
        candle_core::bail!("HQZ4 multi-projection activation shape is invalid.");
    }
    let groups_per_row = k / group_size;
    if activation_scales.dims() != [m, groups_per_row] {
        candle_core::bail!("HQZ4 multi-projection activation scale shape mismatch.");
    }
    if weights.iter().any(|weight| {
        weight.cols != k
            || weight.group_size != group_size
            || weight.codes.dtype() != DType::U8
            || weight.scales.dtype() != DType::F32
            || !weight.codes.is_contiguous()
            || !weight.scales.is_contiguous()
            || weight.codes.dims() != [weight.rows, k / 2]
            || weight.scales.dims() != [weight.rows, groups_per_row]
            || !activation.device().same_device(weight.codes.device())
            || !activation.device().same_device(weight.scales.device())
    }) {
        candle_core::bail!("HQZ4 multi-projection weight metadata mismatch.");
    }
    if !activation.device().same_device(activation_scales.device()) {
        candle_core::bail!("HQZ4 multi-projection inputs must use the same CUDA device.");
    }
    let Some((&source_k, source_batch)) = source_shape.split_last() else {
        candle_core::bail!("HQZ4 source activation shape cannot be empty.");
    };
    let source_rows = source_batch
        .iter()
        .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
        .ok_or_else(|| candle_core::Error::msg("HQZ4 source activation shape overflow."))?;
    if source_k != k || source_rows != m {
        candle_core::bail!(
            "HQZ4 source shape {:?} does not match quantized activation [{m}, {k}].",
            source_shape
        );
    }
    Ok((m, k, group_size))
}

pub(super) fn qkv_matmul_quantized(
    activation: &Tensor,
    activation_scales: &Tensor,
    q: &super::Hqz4CudaInner,
    key: &super::Hqz4CudaInner,
    value: &super::Hqz4CudaInner,
    source_shape: &[usize],
    source_dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (m, k, group_size) = validate_multi_projection_inputs(
        activation,
        activation_scales,
        &[q, key, value],
        source_shape,
    )?;
    if !matches!(source_dtype, DType::F16 | DType::BF16 | DType::F32) {
        candle_core::bail!(
            "HQZ4 Q/K/V output supports F16, BF16, and F32, got {source_dtype:?}."
        );
    }
    let Device::Cuda(device) = activation.device() else {
        candle_core::bail!("HQZ4 Q/K/V fusion requires CUDA storage.");
    };
    let stream = device.cuda_stream();

    let (activation_storage, activation_layout) = activation.storage_and_layout();
    let Storage::Cuda(activation_storage) = &*activation_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let activation_slice = activation_storage.as_cuda_slice::<u8>()?;
    let (activation_ptr, activation_guard) =
        slice_ptr_on_stream(activation_slice, activation_layout.start_offset(), &stream);
    let (scale_storage, scale_layout) = activation_scales.storage_and_layout();
    let Storage::Cuda(scale_storage) = &*scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let activation_scale_slice = scale_storage.as_cuda_slice::<f32>()?;
    let (activation_scale_ptr, activation_scale_guard) = slice_ptr_on_stream(
        activation_scale_slice,
        scale_layout.start_offset(),
        &stream,
    );

    let (q_weight_storage, q_weight_layout) = q.codes.storage_and_layout();
    let Storage::Cuda(q_weight_storage) = &*q_weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let q_weight_slice = q_weight_storage.as_cuda_slice::<u8>()?;
    let (q_weight_ptr, q_weight_guard) =
        slice_ptr_on_stream(q_weight_slice, q_weight_layout.start_offset(), &stream);
    let (q_scale_storage, q_scale_layout) = q.scales.storage_and_layout();
    let Storage::Cuda(q_scale_storage) = &*q_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let q_scale_slice = q_scale_storage.as_cuda_slice::<f32>()?;
    let (q_scale_ptr, q_scale_guard) =
        slice_ptr_on_stream(q_scale_slice, q_scale_layout.start_offset(), &stream);

    let (k_weight_storage, k_weight_layout) = key.codes.storage_and_layout();
    let Storage::Cuda(k_weight_storage) = &*k_weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let k_weight_slice = k_weight_storage.as_cuda_slice::<u8>()?;
    let (k_weight_ptr, k_weight_guard) =
        slice_ptr_on_stream(k_weight_slice, k_weight_layout.start_offset(), &stream);
    let (k_scale_storage, k_scale_layout) = key.scales.storage_and_layout();
    let Storage::Cuda(k_scale_storage) = &*k_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let k_scale_slice = k_scale_storage.as_cuda_slice::<f32>()?;
    let (k_scale_ptr, k_scale_guard) =
        slice_ptr_on_stream(k_scale_slice, k_scale_layout.start_offset(), &stream);

    let (v_weight_storage, v_weight_layout) = value.codes.storage_and_layout();
    let Storage::Cuda(v_weight_storage) = &*v_weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let v_weight_slice = v_weight_storage.as_cuda_slice::<u8>()?;
    let (v_weight_ptr, v_weight_guard) =
        slice_ptr_on_stream(v_weight_slice, v_weight_layout.start_offset(), &stream);
    let (v_scale_storage, v_scale_layout) = value.scales.storage_and_layout();
    let Storage::Cuda(v_scale_storage) = &*v_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let v_scale_slice = v_scale_storage.as_cuda_slice::<f32>()?;
    let (v_scale_ptr, v_scale_guard) =
        slice_ptr_on_stream(v_scale_slice, v_scale_layout.start_offset(), &stream);

    let q_elements = m
        .checked_mul(q.rows)
        .ok_or_else(|| candle_core::Error::msg("HQZ4 Q output size overflow."))?;
    let k_elements = m
        .checked_mul(key.rows)
        .ok_or_else(|| candle_core::Error::msg("HQZ4 K output size overflow."))?;
    let v_elements = m
        .checked_mul(value.rows)
        .ok_or_else(|| candle_core::Error::msg("HQZ4 V output size overflow."))?;
    let mut q_shape = source_shape.to_vec();
    *q_shape.last_mut().expect("source rank checked") = q.rows;
    let mut k_shape = source_shape.to_vec();
    *k_shape.last_mut().expect("source rank checked") = key.rows;
    let mut v_shape = source_shape.to_vec();
    *v_shape.last_mut().expect("source rank checked") = value.rows;

    crate::utils::log::once_log_info(
        "HQZ4 CUDA: fusing Q/K/V into one multi-projection DP4A launch.",
    );
    if m >= 4 {
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using tiled 4x4 A8/W4 DP4A prefill backend (SM61+).",
        );
        if group_size == HQZ4_WARP_RHT_GROUP_SIZE {
            crate::utils::log::once_log_info(
                "HQZ4 CUDA: Q/K/V prefill shares each A8 tile across four projection rows.",
            );
        }
    }

    macro_rules! launch_qkv {
        ($ty:ty, $dtype_code:expr) => {{
            let mut q_output = unsafe { device.alloc::<$ty>(q_elements)? };
            let mut k_output = unsafe { device.alloc::<$ty>(k_elements)? };
            let mut v_output = unsafe { device.alloc::<$ty>(v_elements)? };
            let (q_output_ptr, q_output_guard) =
                slice_ptr_mut_on_stream(&mut q_output, 0, &stream);
            let (k_output_ptr, k_output_guard) =
                slice_ptr_mut_on_stream(&mut k_output, 0, &stream);
            let (v_output_ptr, v_output_guard) =
                slice_ptr_mut_on_stream(&mut v_output, 0, &stream);
            let params = Hqz4QkvLaunch {
                quantized_activation: activation_ptr as *const i8,
                activation_scales: activation_scale_ptr as *const f32,
                q_weight: q_weight_ptr as *const u8,
                q_weight_scales: q_scale_ptr as *const f32,
                q_output: q_output_ptr as *mut c_void,
                q_rows: Hqz4Dp4aMatmul::checked_u32("Q output width", q.rows)?,
                k_weight: k_weight_ptr as *const u8,
                k_weight_scales: k_scale_ptr as *const f32,
                k_output: k_output_ptr as *mut c_void,
                k_rows: Hqz4Dp4aMatmul::checked_u32("K output width", key.rows)?,
                v_weight: v_weight_ptr as *const u8,
                v_weight_scales: v_scale_ptr as *const f32,
                v_output: v_output_ptr as *mut c_void,
                v_rows: Hqz4Dp4aMatmul::checked_u32("V output width", value.rows)?,
                m: Hqz4Dp4aMatmul::checked_u32("Q/K/V row count", m)?,
                k: Hqz4Dp4aMatmul::checked_u32("Q/K/V input width", k)?,
                group_size: Hqz4Dp4aMatmul::checked_u32(
                    "Q/K/V group size",
                    group_size,
                )?,
                dtype: $dtype_code,
                stream: stream.cu_stream(),
            };
            let status = {
                #[cfg(has_hqz4_dp4a_kernels)]
                {
                    let output_rows = params
                        .q_rows
                        .saturating_add(params.k_rows)
                        .saturating_add(params.v_rows);
                    hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::Dp4aQkv,
                        Hqz4ProfileWork::qkv(
                            params.m,
                            output_rows,
                            params.k,
                            params.group_size,
                            params.dtype,
                        ),
                        || unsafe { launch_hqz4_qkv_quantized(&params) },
                    )
                }
                #[cfg(not(has_hqz4_dp4a_kernels))]
                {
                    let _ = &params;
                    unreachable!("HQZ4 DP4A availability was checked before dispatch")
                }
            };
            if status != 0 {
                candle_core::bail!(
                    "HQZ4 Q/K/V DP4A CUDA launch failed with status {status}."
                );
            }
            drop((q_output_guard, k_output_guard, v_output_guard));
            (
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_output, device.clone())),
                    Shape::from_dims(&q_shape),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_output, device.clone())),
                    Shape::from_dims(&k_shape),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_output, device.clone())),
                    Shape::from_dims(&v_shape),
                )),
            )
        }};
    }

    let outputs = match source_dtype {
        DType::F16 | DType::BF16 => launch_qkv!(f16, HQZ4_CUDA_F16),
        DType::F32 => launch_qkv!(f32, HQZ4_CUDA_F32),
        _ => unreachable!("source dtype checked"),
    };
    drop((
        activation_guard,
        activation_scale_guard,
        q_weight_guard,
        q_scale_guard,
        k_weight_guard,
        k_scale_guard,
        v_weight_guard,
        v_scale_guard,
    ));
    if source_dtype == DType::BF16 {
        Ok((
            outputs.0.to_dtype(DType::BF16)?,
            outputs.1.to_dtype(DType::BF16)?,
            outputs.2.to_dtype(DType::BF16)?,
        ))
    } else {
        Ok(outputs)
    }
}

pub(super) fn silu_gate_up_matmul_quantized(
    activation: &Tensor,
    activation_scales: &Tensor,
    gate: &super::Hqz4CudaInner,
    up: &super::Hqz4CudaInner,
    source_shape: &[usize],
    source_dtype: DType,
) -> Result<Tensor> {
    let (m, k, group_size) = validate_multi_projection_inputs(
        activation,
        activation_scales,
        &[gate, up],
        source_shape,
    )?;
    if gate.rows != up.rows {
        candle_core::bail!("HQZ4 fused gate/up projections must have equal output widths.");
    }
    if !matches!(source_dtype, DType::F16 | DType::F32) {
        candle_core::bail!("HQZ4 fused gate/up supports F16 and F32 outputs.");
    }
    let Device::Cuda(device) = activation.device() else {
        candle_core::bail!("HQZ4 gate/up fusion requires CUDA storage.");
    };
    let stream = device.cuda_stream();

    let (activation_storage, activation_layout) = activation.storage_and_layout();
    let Storage::Cuda(activation_storage) = &*activation_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let activation_slice = activation_storage.as_cuda_slice::<u8>()?;
    let (activation_ptr, activation_guard) =
        slice_ptr_on_stream(activation_slice, activation_layout.start_offset(), &stream);
    let (scale_storage, scale_layout) = activation_scales.storage_and_layout();
    let Storage::Cuda(scale_storage) = &*scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let activation_scale_slice = scale_storage.as_cuda_slice::<f32>()?;
    let (activation_scale_ptr, activation_scale_guard) = slice_ptr_on_stream(
        activation_scale_slice,
        scale_layout.start_offset(),
        &stream,
    );

    let (gate_weight_storage, gate_weight_layout) = gate.codes.storage_and_layout();
    let Storage::Cuda(gate_weight_storage) = &*gate_weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let gate_weight_slice = gate_weight_storage.as_cuda_slice::<u8>()?;
    let (gate_weight_ptr, gate_weight_guard) = slice_ptr_on_stream(
        gate_weight_slice,
        gate_weight_layout.start_offset(),
        &stream,
    );
    let (gate_scale_storage, gate_scale_layout) = gate.scales.storage_and_layout();
    let Storage::Cuda(gate_scale_storage) = &*gate_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let gate_scale_slice = gate_scale_storage.as_cuda_slice::<f32>()?;
    let (gate_scale_ptr, gate_scale_guard) = slice_ptr_on_stream(
        gate_scale_slice,
        gate_scale_layout.start_offset(),
        &stream,
    );

    let (up_weight_storage, up_weight_layout) = up.codes.storage_and_layout();
    let Storage::Cuda(up_weight_storage) = &*up_weight_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let up_weight_slice = up_weight_storage.as_cuda_slice::<u8>()?;
    let (up_weight_ptr, up_weight_guard) =
        slice_ptr_on_stream(up_weight_slice, up_weight_layout.start_offset(), &stream);
    let (up_scale_storage, up_scale_layout) = up.scales.storage_and_layout();
    let Storage::Cuda(up_scale_storage) = &*up_scale_storage else {
        unreachable!("CUDA device returned non-CUDA storage")
    };
    let up_scale_slice = up_scale_storage.as_cuda_slice::<f32>()?;
    let (up_scale_ptr, up_scale_guard) =
        slice_ptr_on_stream(up_scale_slice, up_scale_layout.start_offset(), &stream);

    let output_elements = m
        .checked_mul(gate.rows)
        .ok_or_else(|| candle_core::Error::msg("HQZ4 gate/up output size overflow."))?;
    let mut output_shape = source_shape.to_vec();
    *output_shape.last_mut().expect("source rank checked") = gate.rows;
    crate::utils::log::once_log_info(
        "HQZ4 CUDA: fusing gate/up and SiLU*up into one DP4A launch.",
    );
    if m >= 4 {
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using tiled 4x4 A8/W4 DP4A prefill backend (SM61+).",
        );
    }

    macro_rules! launch_gate_up {
        ($ty:ty, $dtype_code:expr) => {{
            let mut output = unsafe { device.alloc::<$ty>(output_elements)? };
            let (output_ptr, output_guard) =
                slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let params = Hqz4SiluGateUpLaunch {
                quantized_activation: activation_ptr as *const i8,
                activation_scales: activation_scale_ptr as *const f32,
                gate_weight: gate_weight_ptr as *const u8,
                gate_weight_scales: gate_scale_ptr as *const f32,
                up_weight: up_weight_ptr as *const u8,
                up_weight_scales: up_scale_ptr as *const f32,
                output: output_ptr as *mut c_void,
                m: Hqz4Dp4aMatmul::checked_u32("gate/up row count", m)?,
                n: Hqz4Dp4aMatmul::checked_u32("gate/up output width", gate.rows)?,
                k: Hqz4Dp4aMatmul::checked_u32("gate/up input width", k)?,
                group_size: Hqz4Dp4aMatmul::checked_u32(
                    "gate/up group size",
                    group_size,
                )?,
                dtype: $dtype_code,
                stream: stream.cu_stream(),
            };
            let status = {
                #[cfg(has_hqz4_dp4a_kernels)]
                {
                    hqz4_profiled_launch(
                        &stream,
                        Hqz4ProfilePhase::Dp4aGateUp,
                        Hqz4ProfileWork::gate_up(
                            params.m,
                            params.n,
                            params.k,
                            params.group_size,
                            params.dtype,
                        ),
                        || unsafe { launch_hqz4_silu_gate_up_quantized(&params) },
                    )
                }
                #[cfg(not(has_hqz4_dp4a_kernels))]
                {
                    let _ = &params;
                    unreachable!("HQZ4 DP4A availability was checked before dispatch")
                }
            };
            if status != 0 {
                candle_core::bail!(
                    "HQZ4 gate/up SiLU DP4A CUDA launch failed with status {status}."
                );
            }
            drop(output_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                Shape::from_dims(&output_shape),
            ))
        }};
    }

    let output = match source_dtype {
        DType::F16 => launch_gate_up!(f16, HQZ4_CUDA_F16),
        DType::F32 => launch_gate_up!(f32, HQZ4_CUDA_F32),
        _ => unreachable!("source dtype checked"),
    };
    drop((
        activation_guard,
        activation_scale_guard,
        gate_weight_guard,
        gate_scale_guard,
        up_weight_guard,
        up_scale_guard,
    ));
    Ok(output)
}

pub(super) fn embedding(
    ids: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    encoded: &super::Hqz4Tensor,
    output_dtype: DType,
) -> Result<Tensor> {
    ids.apply_op3_no_bwd(
        weight,
        scales,
        &Hqz4Embedding {
            rows: encoded.rows(),
            cols: encoded.cols(),
            group_size: encoded.group_size(),
            group_offset: encoded.group_offset(),
            seed: encoded.seed(),
            output_dtype,
        },
    )
}
