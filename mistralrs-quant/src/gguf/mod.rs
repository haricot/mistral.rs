mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(feature = "cuda")]
mod ffi;

use std::{
    borrow::Cow,
    fmt,
    io::{Cursor, Read},
    sync::{atomic::AtomicUsize, Arc},
};

#[cfg(feature = "cuda")]
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicBool, Ordering},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor},
    safetensors::MmapedSafetensors,
    DType, Device, Result, Tensor,
};
use candle_nn::Module;

use crate::{
    generate_isq, generate_isq_imatrix,
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    GgufRawTensor, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
    QuantizedSerdeType,
};

pub(crate) fn cpu_fused_moe_q4k_forward<F>(
    gate: &QMatMul,
    up: &QMatMul,
    down: &QMatMul,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    act: F,
) -> Result<Option<Tensor>>
where
    F: Fn(f32) -> f32 + Copy + Send + Sync,
{
    cpu::cpu_fused_moe_q4k_forward(gate, up, down, xs, topk_weights, topk_ids, act)
}

pub(crate) fn cpu_fused_moe_q4k_forward_raw<F>(
    gate: &GgufRawTensor<'_>,
    up: &GgufRawTensor<'_>,
    down: &GgufRawTensor<'_>,
    xs: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    act: F,
) -> Result<Option<Tensor>>
where
    F: Fn(f32) -> f32 + Copy + Send + Sync,
{
    cpu::cpu_fused_moe_q4k_forward_raw(gate, up, down, xs, topk_weights, topk_ids, act)
}

pub(crate) fn cpu_q4k_matmul(qmatmul: &QMatMul, xs: &Tensor) -> Result<Option<Tensor>> {
    cpu::cpu_q4k_matmul(qmatmul, xs)
}

#[cfg(feature = "cuda")]
static LOG_GGUF_CUDA_STAGED_MOE: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "cuda")]
type StagedCacheKey = (usize, candle_core::cuda::DeviceId, Vec<usize>);

#[cfg(feature = "cuda")]
struct StagedCacheEntry {
    tensor: Arc<QTensor>,
    bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
struct StagedCache {
    entries: HashMap<StagedCacheKey, StagedCacheEntry>,
    order: VecDeque<StagedCacheKey>,
    bytes: usize,
}

#[cfg(feature = "cuda")]
thread_local! {
    static GGUF_CUDA_STAGED_QTENSOR_CACHE: RefCell<StagedCache> = RefCell::new(StagedCache::default());
}

#[cfg(feature = "cuda")]
fn legacy_cuda_device(device: &Device) -> bool {
    let Device::Cuda(dev) = device else {
        return false;
    };

    use candle_core::cuda::cudarc::driver::{result, sys};
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(0);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);

    major * 100 + minor * 10 < 700
}

#[cfg(feature = "cuda")]
fn cuda_staged_dtype_supported(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

#[cfg(feature = "cuda")]
fn cuda_staged_cache_max_bytes() -> usize {
    std::env::var("MISTRALRS_GGUF_CUDA_MOE_STAGED_CACHE_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
        .saturating_mul(1024 * 1024)
}

#[cfg(feature = "cuda")]
fn touch_staged_cache_key(order: &mut VecDeque<StagedCacheKey>, key: &StagedCacheKey) {
    if let Some(pos) = order.iter().position(|existing| existing == key) {
        order.remove(pos);
    }
    order.push_back(key.clone());
}

#[cfg(feature = "cuda")]
fn staged_raw_layout(raw: &GgufRawTensor<'_>) -> Result<(usize, usize, usize, usize)> {
    let dims = raw.shape.as_ref();
    let &[num_experts, out_features, in_features] = dims else {
        candle_core::bail!(
            "GGUF staged CUDA MoE expects weights [experts, out, in], got {:?}",
            dims
        );
    };

    let block_size = raw.dtype.block_size();
    if in_features % block_size != 0 {
        candle_core::bail!(
            "GGUF staged CUDA MoE expects in_features {in_features} divisible by block size {block_size}"
        );
    }

    let expert_bytes = out_features
        .checked_mul(in_features)
        .and_then(|v| v.checked_div(block_size))
        .and_then(|v| v.checked_mul(raw.dtype.type_size()))
        .ok_or_else(|| candle_core::Error::msg("GGUF staged CUDA MoE expert byte size overflow"))?;
    let expected_bytes = expert_bytes
        .checked_mul(num_experts)
        .ok_or_else(|| candle_core::Error::msg("GGUF staged CUDA MoE tensor byte size overflow"))?;
    if expected_bytes != raw.data.len() {
        candle_core::bail!(
            "GGUF staged CUDA MoE raw byte length mismatch: expected {expected_bytes}, got {}",
            raw.data.len()
        );
    }
    Ok((num_experts, out_features, in_features, expert_bytes))
}

#[cfg(feature = "cuda")]
fn compact_raw_experts(
    raw: &GgufRawTensor<'_>,
    expert_ids: &[usize],
    device: &Device,
) -> Result<QTensor> {
    let (num_experts, out_features, in_features, expert_bytes) = staged_raw_layout(raw)?;
    let mut compact = Vec::with_capacity(expert_bytes * expert_ids.len());
    for &expert_id in expert_ids {
        if expert_id >= num_experts {
            candle_core::bail!(
                "GGUF staged CUDA MoE expert id {expert_id} out of range {num_experts}"
            );
        }
        let start = expert_id * expert_bytes;
        let end = start + expert_bytes;
        compact.extend_from_slice(&raw.data[start..end]);
    }

    qtensor_from_ggml(
        raw.dtype,
        &compact,
        vec![expert_ids.len(), out_features, in_features],
        device,
    )
}

#[cfg(feature = "cuda")]
fn cached_compact_raw_experts(
    raw: &GgufRawTensor<'_>,
    expert_ids: &[usize],
    device: &Device,
) -> Result<Arc<QTensor>> {
    let Device::Cuda(dev) = device else {
        candle_core::bail!("GGUF staged CUDA MoE cache requires CUDA device");
    };
    let (_, _, _, expert_bytes) = staged_raw_layout(raw)?;
    let bytes = expert_bytes
        .checked_mul(expert_ids.len())
        .ok_or_else(|| candle_core::Error::msg("GGUF staged CUDA MoE cache byte overflow"))?;
    let max_bytes = cuda_staged_cache_max_bytes();
    if max_bytes == 0 || bytes > max_bytes {
        return Ok(Arc::new(compact_raw_experts(raw, expert_ids, device)?));
    }

    let key = (raw.cache_key, dev.id(), expert_ids.to_vec());
    if let Some(hit) = GGUF_CUDA_STAGED_QTENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let tensor = cache.entries.get(&key).map(|entry| entry.tensor.clone());
        if tensor.is_some() {
            touch_staged_cache_key(&mut cache.order, &key);
        }
        tensor
    }) {
        return Ok(hit);
    }

    let tensor = Arc::new(compact_raw_experts(raw, expert_ids, device)?);
    GGUF_CUDA_STAGED_QTENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(old) = cache.entries.remove(&key) {
            cache.bytes = cache.bytes.saturating_sub(old.bytes);
        }
        touch_staged_cache_key(&mut cache.order, &key);
        cache.entries.insert(
            key.clone(),
            StagedCacheEntry {
                tensor: tensor.clone(),
                bytes,
            },
        );
        cache.bytes = cache.bytes.saturating_add(bytes);

        while cache.bytes > max_bytes {
            let Some(evict_key) = cache.order.pop_front() else {
                break;
            };
            if evict_key == key {
                cache.order.push_front(evict_key);
                break;
            }
            if let Some(evicted) = cache.entries.remove(&evict_key) {
                cache.bytes = cache.bytes.saturating_sub(evicted.bytes);
            }
        }
    });

    Ok(tensor)
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_staged_moe_forward_raw(
    gate: &GgufRawTensor<'_>,
    up: &GgufRawTensor<'_>,
    down: &GgufRawTensor<'_>,
    xs_flat: &Tensor,
    topk_weights: &Tensor,
    topk_ids: &Tensor,
    act_type: i32,
) -> Result<Option<Tensor>> {
    if !xs_flat.device().is_cuda()
        || !topk_ids.device().is_cuda()
        || !topk_weights.device().is_cuda()
    {
        return Ok(None);
    }
    if gate.dtype != up.dtype || gate.dtype != down.dtype {
        return Ok(None);
    }
    if !cuda_staged_dtype_supported(gate.dtype) {
        return Ok(None);
    }

    let (num_tokens, hidden_dim) = xs_flat.dims2()?;
    let (ids_tokens, topk) = topk_ids.dims2()?;
    let (weights_tokens, weights_topk) = topk_weights.dims2()?;
    if ids_tokens != num_tokens || weights_tokens != num_tokens || weights_topk != topk {
        candle_core::bail!(
            "GGUF staged CUDA MoE routing shape mismatch: xs={:?}, ids={:?}, weights={:?}",
            xs_flat.shape(),
            topk_ids.shape(),
            topk_weights.shape()
        );
    }

    let gate_dims = gate.shape.as_ref();
    let up_dims = up.shape.as_ref();
    let down_dims = down.shape.as_ref();
    let &[num_experts, intermediate_size, gate_hidden] = gate_dims else {
        candle_core::bail!(
            "GGUF staged CUDA MoE gate expects [experts, intermediate, hidden], got {:?}",
            gate_dims
        );
    };
    if up_dims != gate_dims {
        candle_core::bail!(
            "GGUF staged CUDA MoE gate/up shape mismatch {:?} vs {:?}",
            gate_dims,
            up_dims
        );
    }
    let &[down_experts, down_hidden, down_intermediate] = down_dims else {
        candle_core::bail!(
            "GGUF staged CUDA MoE down expects [experts, hidden, intermediate], got {:?}",
            down_dims
        );
    };
    if down_experts != num_experts
        || down_hidden != hidden_dim
        || down_intermediate != intermediate_size
        || gate_hidden != hidden_dim
    {
        candle_core::bail!(
            "GGUF staged CUDA MoE shape mismatch: xs_hidden={hidden_dim}, gate={:?}, down={:?}",
            gate_dims,
            down_dims
        );
    }

    let ids_cpu = topk_ids.to_device(&Device::Cpu)?;
    let ids_vec = ids_cpu.to_vec2::<u32>()?;
    let mut seen = vec![false; num_experts];
    let mut expert_ids = Vec::new();

    for token_ids in &ids_vec {
        for &expert_id in token_ids {
            let expert_id = expert_id as usize;
            if expert_id >= num_experts {
                candle_core::bail!(
                    "GGUF staged CUDA MoE expert id {expert_id} out of range {num_experts}"
                );
            }
            if !seen[expert_id] {
                seen[expert_id] = true;
                expert_ids.push(expert_id);
            }
        }
    }

    if expert_ids.is_empty() {
        return Ok(None);
    }
    expert_ids.sort_unstable();

    let mut remap = vec![usize::MAX; num_experts];
    for (compact_id, &expert_id) in expert_ids.iter().enumerate() {
        remap[expert_id] = compact_id;
    }

    let mut compact_ids = Vec::with_capacity(num_tokens * topk);
    for token_ids in &ids_vec {
        for &expert_id in token_ids {
            compact_ids.push(remap[expert_id as usize] as u32);
        }
    }

    if !LOG_GGUF_CUDA_STAGED_MOE.swap(true, Ordering::Relaxed) {
        tracing::info!(
            "Using staged GGUF CUDA MoE decode path (dtype={:?}, selected_experts={}, cache_mb={}, hidden={hidden_dim}, intermediate={intermediate_size})",
            gate.dtype,
            expert_ids.len(),
            cuda_staged_cache_max_bytes() / (1024 * 1024)
        );
    }

    let gate_qt = cached_compact_raw_experts(gate, &expert_ids, xs_flat.device())?;
    let up_qt = cached_compact_raw_experts(up, &expert_ids, xs_flat.device())?;
    let down_qt = cached_compact_raw_experts(down, &expert_ids, xs_flat.device())?;

    let compact_ids = Tensor::from_vec(compact_ids, (num_tokens, topk), topk_ids.device())?;
    let compact_ids_flat = compact_ids.flatten_all()?.contiguous()?;
    let (ids_storage, ids_layout) = compact_ids_flat.storage_and_layout();
    let candle_core::Storage::Cuda(ids_cuda) = &*ids_storage else {
        return Ok(None);
    };
    if ids_layout.start_offset() != 0 {
        return Ok(None);
    }
    let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;

    let topk_weights_f32 = topk_weights
        .flatten_all()?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let (weights_storage, weights_layout) = topk_weights_f32.storage_and_layout();
    let candle_core::Storage::Cuda(weights_cuda) = &*weights_storage else {
        return Ok(None);
    };
    let weights_slice = weights_cuda.as_cuda_slice::<f32>()?;
    let weights_ptr = {
        use candle_core::cuda::cudarc::driver::DevicePtr;
        weights_slice
            .slice(weights_layout.start_offset()..)
            .device_ptr(weights_slice.stream())
            .0 as *const f32
    };

    let dev = xs_flat.device().as_cuda_device()?;
    let out = unsafe {
        cuda::indexed_moe_fused_decode(
            &gate_qt,
            &up_qt,
            &down_qt,
            xs_flat,
            ids_slice,
            weights_ptr,
            num_tokens,
            topk,
            act_type,
            dev,
        )?
    };
    Ok(Some(out))
}

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
}

pub struct LazyGgufMatMul {
    artifacts: Arc<MmapedSafetensors>,
    name: String,
    data_offset: usize,
    data_len: usize,
    dtype: GgmlDType,
    dims: Vec<usize>,
    b: Option<Tensor>,
}

impl fmt::Debug for LazyGgufMatMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyGgufMatMul")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("dims", &self.dims)
            .field("data_len", &self.data_len)
            .field("has_bias", &self.b.is_some())
            .finish()
    }
}

fn ggml_dtype_from_uqff(dtype: u32) -> Result<GgmlDType> {
    match dtype {
        0 => Ok(GgmlDType::F32),
        1 => Ok(GgmlDType::F16),
        2 => Ok(GgmlDType::Q4_0),
        3 => Ok(GgmlDType::Q4_1),
        6 => Ok(GgmlDType::Q5_0),
        7 => Ok(GgmlDType::Q5_1),
        8 => Ok(GgmlDType::Q8_0),
        9 => Ok(GgmlDType::Q8_1),
        10 => Ok(GgmlDType::Q2K),
        11 => Ok(GgmlDType::Q3K),
        12 => Ok(GgmlDType::Q4K),
        13 => Ok(GgmlDType::Q5K),
        14 => Ok(GgmlDType::Q6K),
        15 => Ok(GgmlDType::Q8K),
        // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
        30 => Ok(GgmlDType::BF16),
        _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
    }
}

impl LazyGgufMatMul {
    pub fn from_uqff_mmap(
        artifacts: Arc<MmapedSafetensors>,
        name: String,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        let view = artifacts.get(&name)?;
        let data = view.data();
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            return Ok(None);
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;
        let has_bias = buffer.read_u8()? != 0;
        let dtype = ggml_dtype_from_uqff(buffer.read_u32::<LittleEndian>()?)?;

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let data_offset = buffer.position() as usize;
        let data_end = data_offset
            .checked_add(data_len)
            .ok_or_else(|| candle_core::Error::msg("UQFF GGUF tensor data length overflow"))?;
        if data_end > buffer.get_ref().len() {
            candle_core::bail!("UQFF GGUF tensor data for `{name}` extends past artifact length");
        }
        buffer.set_position(data_end as u64);

        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, &Device::Cpu)?)
        } else {
            None
        };

        Ok(Some(Arc::new(Self {
            artifacts,
            name,
            data_offset,
            data_len,
            dtype,
            dims,
            b,
        })))
    }

    fn with_raw_data<T>(&self, f: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
        let view = self.artifacts.get(&self.name)?;
        let data = view.data();
        let data_end = self.data_offset + self.data_len;
        f(&data[self.data_offset..data_end])
    }

    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        if let Some(ref b) = self.b {
            x.broadcast_add(&b.to_device(x.device())?)
        } else {
            Ok(x)
        }
    }

    fn cache_key(&self) -> usize {
        self as *const Self as usize
    }

    fn materialize_qmatmul(&self, device: &Device) -> Result<QMatMul> {
        self.with_raw_data(|raw| {
            let w = qtensor_from_ggml(self.dtype, raw, self.dims.clone(), device)?;
            QMatMul::from_qtensor(w)
        })
    }
}

impl GgufMatMul {
    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        if let Some(ref b) = self.b {
            x.broadcast_add(&b.to_device(x.device())?)
        } else {
            Ok(x)
        }
    }

    fn weight_device(&self) -> Device {
        match &self.w {
            QMatMul::QTensor(q) => q.device(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
        }
    }

    #[cfg(feature = "cuda")]
    fn uses_fast_mmvq(&self) -> bool {
        let QMatMul::QTensor(q) = &self.w else {
            return false;
        };
        if !q.device().is_cuda() || !fast_mmvq::supports(q.dtype()) {
            return false;
        }

        match std::env::var("MISTRALRS_GGUF_FAST_MMVQ") {
            Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
            Err(_) => !legacy_cuda_device(&q.device()),
        }
    }

    #[cfg(feature = "cuda")]
    fn try_fast_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if !self.uses_fast_mmvq()
            || !a.device().is_cuda()
            || !matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32)
        {
            return Ok(None);
        }

        let flat_batch = a.dims()[..a.dims().len().saturating_sub(1)]
            .iter()
            .product::<usize>();

        let QMatMul::QTensor(q) = &self.w else {
            unreachable!("uses_fast_mmvq() requires QTensor weights")
        };

        // Batch 1-8: use MMVQ (decode kernel)
        if (1..=fast_mmvq::MMVQ_MAX_BATCH).contains(&flat_batch) {
            return Ok(Some(fast_mmvq::plain(q, a)?));
        }

        // Batch > 8: use MMQ (prompt kernel)
        if flat_batch > fast_mmvq::MMVQ_MAX_BATCH {
            return Ok(Some(fast_mmq::plain(q, a)?));
        }

        Ok(None)
    }
}

impl QuantMethod for LazyGgufMatMul {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("LazyGgufMatMul must be constructed from a UQFF mmap artifact")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.materialize_qmatmul(&Device::Cpu)?
            .dequantize_f16()?
            .to_dtype(DType::F32)
    }

    fn forward_raw(&self, _a: &Tensor) -> Result<Tensor> {
        candle_core::bail!("Lazy GGUF UQFF layer only supports indexed MoE gather_forward")
    }

    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let target_device = x.device().clone();
        let x_cpu;
        let x = if target_device.is_cpu() {
            x
        } else {
            x_cpu = x.to_device(&Device::Cpu)?;
            &x_cpu
        };
        let indices_cpu;
        let indices = if indices.device().is_cpu() {
            indices
        } else {
            indices_cpu = indices.to_device(&Device::Cpu)?;
            &indices_cpu
        };

        let out = self.with_raw_data(|raw| {
            cpu::cpu_indexed_moe_forward_raw(
                self.dtype,
                &self.dims,
                raw,
                self.cache_key(),
                x,
                indices,
            )
        })?;
        self.add_bias(out)?.to_device(&target_device)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, Device::Cpu)
    }

    fn gguf_raw_tensor_and_bias(&self) -> Result<Option<(GgufRawTensor<'_>, Option<&Tensor>)>> {
        let view = self.artifacts.get(&self.name)?;
        let data = view.data();
        let data_end = self.data_offset + self.data_len;
        Ok(Some((
            GgufRawTensor {
                dtype: self.dtype,
                shape: Cow::Borrowed(&self.dims),
                data: Cow::Borrowed(&data[self.data_offset..data_end]),
                cache_key: self.cache_key(),
            },
            self.b.as_ref(),
        )))
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("Lazy GGUF UQFF layer does not support LoRA delta application")
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_some() {
            candle_core::bail!("Lazy GGUF UQFF layer cannot be re-quantized")
        }
        Ok(self)
    }
}

impl QuantizedSerde for LazyGgufMatMul {
    fn name(&self) -> &'static str {
        "gguf-lazy"
    }
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self {
                w: QMatMul::from_arc(q_weight)?,
                b,
            }),
            QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.w.dequantize_f16()?.to_dtype(DType::F32)
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = self.try_fast_forward(a)? {
                return self.add_bias(out);
            }

            let weight_device = self.weight_device();
            if weight_device.is_cuda() && !a.device().is_cuda() {
                let a_on_weight = a.to_device(&weight_device)?;
                if let Some(out) = self.try_fast_forward(&a_on_weight)? {
                    return self.add_bias(out)?.to_device(a.device());
                }

                let original_dtype = a_on_weight.dtype();
                let a_f32 = if original_dtype == DType::F32 {
                    a_on_weight
                } else {
                    a_on_weight.to_dtype(DType::F32)?
                };
                let x = self.w.forward(&a_f32)?;
                let x = if original_dtype == DType::F32 {
                    x
                } else {
                    x.to_dtype(original_dtype)?
                };
                return self.add_bias(x)?.to_device(a.device());
            }
        }

        if let Some(x) = cpu_q4k_matmul(&self.w, a)? {
            return self.add_bias(x);
        }

        // Fallback: Candle QMatMul requires F32
        let original_dtype = a.dtype();
        let a_f32 = if original_dtype == DType::F32 {
            a.clone()
        } else {
            a.to_dtype(DType::F32)?
        };
        let x = self.w.forward(&a_f32)?;
        let x = if original_dtype == DType::F32 {
            x
        } else {
            x.to_dtype(original_dtype)?
        };
        self.add_bias(x)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Use indexed_moe_forward for efficient indexed matmul
        // Expected shapes:
        // - x: (n_tokens, 1, hidden_dim) or (n_tokens, n_experts_per_tok, hidden_dim)
        // - indices: (n_tokens, n_experts_per_tok)
        // - weights (self): (n_experts, out_features, in_features)
        let weights_device = match &self.w {
            QMatMul::QTensor(q) => q.device(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
        };
        let res = if weights_device.is_cuda() && x.device().is_cuda() {
            #[cfg(feature = "cuda")]
            {
                cuda::qmatmul_indexed_moe_forward(&self.w, x, indices)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::bail!("GGUF indexed MoE CUDA path requires the `cuda` feature")
            }
        } else if weights_device.is_cpu() && !x.device().is_cpu() {
            let x_cpu = x.to_device(&Device::Cpu)?;
            let indices_cpu = indices.to_device(&Device::Cpu)?;
            return self
                .add_bias(cpu::cpu_indexed_moe_forward(&self.w, &x_cpu, &indices_cpu)?)?
                .to_device(x.device());
        } else {
            cpu::cpu_indexed_moe_forward(&self.w, x, indices)?
        };

        if let Some(ref b) = self.b {
            res.broadcast_add(b)
        } else {
            Ok(res)
        }
    }

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        match &self.w {
            candle_core::quantized::QMatMul::QTensor(qt) => Some(qt.clone()),
            _ => None,
        }
    }

    fn gguf_qmatmul_and_bias(&self) -> Option<(&QMatMul, Option<&Tensor>)> {
        Some((&self.w, self.b.as_ref()))
    }

    fn gguf_raw_tensor_and_bias(&self) -> Result<Option<(GgufRawTensor<'_>, Option<&Tensor>)>> {
        let QMatMul::QTensor(q) = &self.w else {
            return Ok(None);
        };
        if !q.device().is_cpu() {
            return Ok(None);
        }
        Ok(Some((
            GgufRawTensor {
                dtype: q.dtype(),
                shape: Cow::Borrowed(q.shape().dims()),
                data: q.data()?,
                cache_key: Arc::as_ptr(q) as usize,
            },
            self.b.as_ref(),
        )))
    }

    fn quantized_act_type(&self) -> Option<DType> {
        #[cfg(feature = "cuda")]
        {
            if self.uses_fast_mmvq() {
                return None;
            }
        }
        Some(DType::F32)
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self {
                w: QMatMul::Tensor(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::Tensor((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::TensorF16(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::TensorF16((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::QTensor(w),
                b,
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if let Some(dtype) = dtype {
            // F8Q8 is not a GgmlDType, so intercept before try_into()
            if dtype == IsqType::F8Q8 {
                let t = match &self.w {
                    QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                    QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
                };
                let t = t.to_device(&device)?;
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Arc::new(crate::F8Q8Linear::from_weight(
                    &t,
                    self.b.clone(),
                )?));
            }
            let t = match &self.w {
                QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
            };
            let dtype = dtype.try_into()?;
            let res = if let Some(imatrix_weight) = imatrix_weight {
                generate_isq_imatrix!(t, imatrix_weight, device, dtype, n_quantized, guard)
            } else {
                generate_isq!(t, device, dtype, n_quantized, guard)
            };
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: res,
                b: self.b.clone(),
            })?))
        } else {
            let w = match &self.w {
                QMatMul::QTensor(q) => QMatMul::QTensor(Arc::new(QTensor::quantize(
                    &q.dequantize(&device)?,
                    q.dtype(),
                )?)),
                QMatMul::Tensor(t) => QMatMul::Tensor(t.to_device(&device)?),
                QMatMul::TensorF16(t) => QMatMul::TensorF16(t.to_device(&device)?),
            };
            let b = if let Some(b) = &self.b {
                Some(b.to_device(&device)?)
            } else {
                None
            };
            Ok(Arc::new(GgufMatMul { w, b }))
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (0 for GGUF), u8, little endian
// -----------------------
// Tensor data length in bytes, u32, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Quantized dtype, u32, little endian
// -----------------------
// Num shape dims, u32, little endian
// -----------------------
// ...
// Array (in original order): quantized weight shape dims, u32, little endian
// ...
// -----------------------
// ...
// Array: quantized weight data, u8s
// ...
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for GgufMatMul {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "gguf"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.b.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = match &self.w {
            QMatMul::QTensor(qw) => {
                let w = qw.data()?.to_vec();
                let w_shape = qw.shape().dims();
                let dtype: u32 = match qw.dtype() {
                    GgmlDType::F32 => 0,
                    GgmlDType::F16 => 1,
                    GgmlDType::Q4_0 => 2,
                    GgmlDType::Q4_1 => 3,
                    GgmlDType::Q5_0 => 6,
                    GgmlDType::Q5_1 => 7,
                    GgmlDType::Q8_0 => 8,
                    GgmlDType::Q8_1 => 9,
                    GgmlDType::Q2K => 10,
                    GgmlDType::Q3K => 11,
                    GgmlDType::Q4K => 12,
                    GgmlDType::Q5K => 13,
                    GgmlDType::Q6K => 14,
                    GgmlDType::Q8K => 15,
                    // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                    GgmlDType::BF16 => 30,
                };

                let mut buffer = Vec::new();

                // Version is always first!
                buffer.extend(&UQFF_VERSION.to_le_bytes());

                // ISQ type for GGUF is 0
                buffer.push(QuantizedSerdeType::Gguf as u8);

                // Length
                buffer.extend(&(w.len() as u32).to_le_bytes());

                // Has bias
                buffer.push(bias.is_some() as u8);

                // Dtype (u32)
                buffer.extend(&dtype.to_le_bytes());

                // Shape
                buffer.extend((w_shape.len() as u32).to_le_bytes());
                for dim in w_shape {
                    buffer.extend((*dim as u32).to_le_bytes());
                }

                // Quantized W Vec<u8> (just append it)
                buffer.extend(&w);

                buffer
            }
            QMatMul::TensorF16(_) | QMatMul::Tensor(_) => {
                candle_core::bail!("Cannot serialize non-quantized")
            }
        };

        if let Some(b) = bias.as_ref() {
            serialize_tensor(&mut buffer, b)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        Ok(Arc::new(Self {
            w: QMatMul::QTensor(w.into()),
            b,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        Ok((
            Arc::new(Self {
                w: QMatMul::QTensor(w.into()),
                b: None,
            }),
            b,
        ))
    }
}

impl GgufMatMul {
    pub fn get_isq_type_from_uqff(data: Cow<[u8]>) -> Result<IsqType> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let _ = buffer.read_u32::<LittleEndian>()? as usize;

        let _ = buffer.read_u8()? != 0;

        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => candle_core::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        IsqType::try_from(dtype)
    }
}
