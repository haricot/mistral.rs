use anyhow::{ensure, Context, Result};
use candle_core::{
    cuda::cudarc::driver::{sys, CudaGraph},
    DType, Device, Tensor, Var,
};
use mistralrs_kernel_runtime::runtime_evidence;
use mistralrs_paged_attn::{paged_attention, reshape_and_cache};
use serde_json::{json, Value};
use std::time::Instant;

const HEAD: usize = 64;
const PAGE: usize = 16;
const PAGES: usize = 2;
const VECTOR_WIDTH: usize = 4;
const ROUTES: [&str; 2] = [
    "kv_cache.reshape_store.vllm.b256.v1",
    "attention.paged.decode.online.v1",
];

struct Inputs {
    query: Var,
    key: Var,
    value: Var,
    slot: Var,
    table: Var,
    length: Var,
}

impl Inputs {
    fn new(device: &Device) -> Result<Self> {
        let shape = (1, 1, HEAD);
        Ok(Self {
            query: Var::zeros(shape, DType::F32, device)?,
            key: Var::zeros(shape, DType::F32, device)?,
            value: Var::zeros(shape, DType::F32, device)?,
            slot: Var::zeros(1, DType::I64, device)?,
            table: Var::zeros((1, 1), DType::U32, device)?,
            length: Var::zeros(1, DType::U32, device)?,
        })
    }

    fn set(&self, page: u32, length: u32, query: f32, key: f32, value: f32) -> Result<()> {
        let device = self.query.device();
        let slot = i64::from(page) * PAGE as i64 + i64::from(length) - 1;
        for (var, value) in [(&self.query, query), (&self.key, key), (&self.value, value)] {
            var.set(&Tensor::full(value, (1, 1, HEAD), device)?)?;
        }
        self.slot.set(&Tensor::new(&[slot], device)?)?;
        self.table.set(&Tensor::new(&[[page]], device)?)?;
        self.length.set(&Tensor::new(&[length], device)?)?;
        Ok(())
    }
}

struct Cache {
    key: Tensor,
    value: Tensor,
}

impl Cache {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            key: Tensor::zeros(
                (PAGES, 1, HEAD / VECTOR_WIDTH, PAGE, VECTOR_WIDTH),
                DType::F32,
                device,
            )?,
            value: Tensor::zeros((PAGES, 1, HEAD, PAGE), DType::F32, device)?,
        })
    }

    fn decode(&self, input: &Inputs) -> candle_core::Result<Tensor> {
        reshape_and_cache(
            &input.key,
            &input.value,
            None,
            None,
            &self.key,
            &self.value,
            &input.slot,
        )?;
        paged_attention(
            &input.query,
            None,
            None,
            &self.key,
            &self.value,
            &input.table,
            &input.length,
            None,
            PAGE,
            1.0 / (HEAD as f32).sqrt(),
            1.0,
            None,
        )
    }
}

fn capture(
    device: &Device,
    call: impl FnOnce() -> candle_core::Result<Tensor>,
) -> Result<(CudaGraph, Tensor)> {
    device.synchronize()?;
    let stream = device.as_cuda_device()?.cuda_stream();
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    let result = (|| -> Result<_> {
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)?;
        let output = call();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        Ok((graph?.context("capture returned an empty graph")?, output?))
    })();
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    result
}

fn route_submissions() -> Result<[u64; 2]> {
    let evidence = runtime_evidence();
    ensure!(
        evidence["dropped_rows"] == 0,
        "enable complete DKR telemetry for the graph proof"
    );
    let rows = evidence["rows"]
        .as_array()
        .context("no DKR route evidence")?;
    Ok(ROUTES.map(|op| {
        rows.iter()
            .filter(|row| {
                row["abi"] == "dkr.typed.v1" && row["target_cc"] == 61 && row["operation"] == op
            })
            .filter_map(|row| row["adapter_ok"].as_u64())
            .sum()
    }))
}

fn values(tensor: &Tensor) -> Result<Vec<u32>> {
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;
    ensure!(
        data.iter().all(|value| value.is_finite()),
        "nonfinite decode output"
    );
    Ok(data.into_iter().map(f32::to_bits).collect())
}

pub(crate) fn run() -> Result<Value> {
    let device = Device::new_cuda_with_stream(0)?;
    let graph_input = Inputs::new(&device)?;
    let eager_input = Inputs::new(&device)?;
    let graph_cache = Cache::new(&device)?;
    let eager_cache = Cache::new(&device)?;
    graph_input.set(0, 1, 0.25, 0.125, 0.5)?;
    drop(graph_cache.decode(&graph_input)?);
    let before = route_submissions()?;
    let capture_start = Instant::now();
    let (graph, output) = capture(&device, || graph_cache.decode(&graph_input))?;
    let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;
    let after = route_submissions()?;
    ensure!(
        after == before.map(|n| n + 1),
        "both qualified DKR routes must be submitted during capture"
    );
    let mut samples = Vec::new();
    for (page, length, query, key, value) in [
        (0, 1, 0.5, 0.125, 1.0),
        (0, 2, -0.5, 0.375, 3.0),
        (1, 1, 0.75, 0.25, -4.0),
        (1, 2, -0.75, 0.5, -2.0),
    ] {
        graph_input.set(page, length, query, key, value)?;
        eager_input.set(page, length, query, key, value)?;
        graph.launch()?;
        device.synchronize()?;
        let expected = eager_cache.decode(&eager_input)?;
        let actual = values(&output)?;
        ensure!(
            actual == values(&expected)?,
            "replay differs for page {page}, length {length}"
        );
        let first = f32::from_bits(actual[0]);
        ensure!(
            samples
                .last()
                .is_none_or(|sample: &Value| sample["output0"] != first),
            "stale replay output"
        );
        samples.push(json!({"page": page, "length": length, "output0": first}));
    }
    ensure!(
        values(&graph_cache.key)? == values(&eager_cache.key)?
            && values(&graph_cache.value)? == values(&eager_cache.value)?,
        "replayed KV state differs from eager"
    );
    ensure!(
        route_submissions()? == after.map(|n| n + samples.len() as u64),
        "eager reference must use the same two DKR routes"
    );
    Ok(json!({
        "schema": "DKR-SM61-PAGED-GRAPH-PROOF-V1",
        "batch": 1,
        "capture_ms": capture_ms,
        "captured_routes": ROUTES,
        "replays": samples,
        "exact_output_and_kv": true
    }))
}
