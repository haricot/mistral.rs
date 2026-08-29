use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, DeviceLocation, Result, Tensor};
use candle_nn::{Linear, Module};
#[cfg(test)]
use half::f16;
use safetensors::tensor::Dtype;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
#[cfg(feature = "cuda")]
use crate::ActivationQuantizationTransform;
use crate::{
    ActivationQuantizationScheme, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedActivation, QuantizedSerde, QuantizedSerdeType, Shard, UqffReader, UqffTensor,
};

#[cfg(feature = "cuda")]
mod cuda;

pub const HQZ4_SCHEMA_VERSION: u32 = 2;
const HQZ4_LEGACY_SCHEMA_VERSION: u32 = 1;
pub const HQZ4_BITS: usize = 4;
pub const HQZ4_DEFAULT_GROUP_SIZE: usize = 128;
pub const HQZ4_LAYOUT_ROW_MAJOR_NIBBLES: u8 = 0;
pub const HQZ4_TRANSFORM_SHARED_RHT: u8 = 0;

const HQZ4_MAX_LEVEL: i8 = 7;
const HQZ4_NIBBLE_MASK: u8 = 0x0f;
const SPLITMIX_INCREMENT: u64 = 0x9e37_79b9_7f4a_7c15;
const SPLITMIX_MULTIPLIER_1: u64 = 0xbf58_476d_1ce4_e5b9;
const SPLITMIX_MULTIPLIER_2: u64 = 0x94d0_49bb_1331_11eb;
const GROUP_MIX: u64 = 0xd1b5_4a32_d192_ed03;
const ELEMENT_MIX: u64 = 0x8cb9_2baa_7f3d_d15b;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hqz4Config {
    pub group_size: usize,
    pub seed: u64,
}

impl Default for Hqz4Config {
    fn default() -> Self {
        Self {
            group_size: HQZ4_DEFAULT_GROUP_SIZE,
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hqz4Tensor {
    rows: usize,
    cols: usize,
    group_size: usize,
    group_offset: usize,
    seed: u64,
    scales: Vec<f32>,
    codes: Vec<u8>,
}

impl Hqz4Tensor {
    pub fn encode(weights: &[f32], rows: usize, cols: usize, cfg: Hqz4Config) -> Result<Self> {
        validate_layout(rows, cols, cfg.group_size)?;
        let elements = checked_elements(rows, cols)?;
        if weights.len() != elements {
            candle_core::bail!(
                "HQZ4 weight length {} does not match shape [{rows}, {cols}].",
                weights.len()
            );
        }
        if weights.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("HQZ4 weights must be finite.");
        }

        let groups_per_row = cols / cfg.group_size;
        let scale_count = rows
            .checked_mul(groups_per_row)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale count overflow.".into()))?;
        let mut scales = Vec::with_capacity(scale_count);
        let mut codes = vec![0u8; elements / 2];
        let mut rotated = vec![0f32; cfg.group_size];

        for row in 0..rows {
            for group in 0..groups_per_row {
                let start = row * cols + group * cfg.group_size;
                rotated.copy_from_slice(&weights[start..start + cfg.group_size]);
                rotate_weight_group(&mut rotated, cfg.seed, group);

                let max_abs = rotated.iter().copied().map(f32::abs).fold(0f32, f32::max);
                let scale = if max_abs == 0.0 {
                    0.0
                } else {
                    let scale = max_abs / HQZ4_MAX_LEVEL as f32;
                    if !scale.is_finite() || scale == 0.0 {
                        candle_core::bail!(
                            "HQZ4 group scale is outside the finite F32 range at row {row}, group {group}."
                        );
                    }
                    scale
                };
                scales.push(scale);

                for (offset, value) in rotated.iter().copied().enumerate() {
                    let quantized = if scale == 0.0 {
                        0
                    } else {
                        (value / scale)
                            .round()
                            .clamp(-(HQZ4_MAX_LEVEL as f32), HQZ4_MAX_LEVEL as f32)
                            as i8
                    };
                    write_code(&mut codes, start + offset, quantized);
                }
            }
        }

        Self::from_parts(rows, cols, cfg, scales, codes)
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        cfg: Hqz4Config,
        scales: Vec<f32>,
        codes: Vec<u8>,
    ) -> Result<Self> {
        Self::from_parts_at_group_offset((rows, cols), cfg, 0, scales, codes)
    }

    fn from_parts_at_group_offset(
        shape: (usize, usize),
        cfg: Hqz4Config,
        group_offset: usize,
        scales: Vec<f32>,
        codes: Vec<u8>,
    ) -> Result<Self> {
        let (rows, cols) = shape;
        validate_layout(rows, cols, cfg.group_size)?;
        let elements = checked_elements(rows, cols)?;
        group_offset
            .checked_add(cols / cfg.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 group offset overflow.".into()))?;
        let expected_scales = rows
            .checked_mul(cols / cfg.group_size)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 scale count overflow.".into()))?;
        if scales.len() != expected_scales {
            candle_core::bail!(
                "HQZ4 scale count {} does not match expected count {expected_scales}.",
                scales.len()
            );
        }
        if scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            candle_core::bail!("HQZ4 scales must be finite and non-negative.");
        }
        let expected_codes = elements / 2;
        if codes.len() != expected_codes {
            candle_core::bail!(
                "HQZ4 code length {} does not match expected length {expected_codes}.",
                codes.len()
            );
        }
        if (0..elements).any(|index| read_code(&codes, index) < -HQZ4_MAX_LEVEL) {
            candle_core::bail!("HQZ4 code payload contains the reserved -8 level.");
        }
        for (group_index, scale) in scales.iter().enumerate() {
            if *scale != 0.0 {
                continue;
            }
            let row = group_index / (cols / cfg.group_size);
            let group = group_index % (cols / cfg.group_size);
            let start = row * cols + group * cfg.group_size;
            if (start..start + cfg.group_size).any(|index| read_code(&codes, index) != 0) {
                candle_core::bail!("HQZ4 zero-scale group contains non-zero codes.");
            }
        }

        Ok(Self {
            rows,
            cols,
            group_size: cfg.group_size,
            group_offset,
            seed: cfg.seed,
            scales,
            codes,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn group_offset(&self) -> usize {
        self.group_offset
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn decode_rotated_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            candle_core::bail!("HQZ4 row {row} is outside row count {}.", self.rows);
        }
        let groups_per_row = self.cols / self.group_size;
        let mut output = vec![0f32; self.cols];
        for group in 0..groups_per_row {
            let scale = self.scales[row * groups_per_row + group];
            let start = row * self.cols + group * self.group_size;
            let output_start = group * self.group_size;
            for offset in 0..self.group_size {
                output[output_start + offset] =
                    read_code(&self.codes, start + offset) as f32 * scale;
            }
        }
        Ok(output)
    }

    pub fn decode_row(&self, row: usize) -> Result<Vec<f32>> {
        let mut output = self.decode_rotated_row(row)?;
        for group in 0..self.cols / self.group_size {
            let start = group * self.group_size;
            inverse_rotate_weight_group(
                &mut output[start..start + self.group_size],
                self.seed,
                self.group_offset + group,
            );
        }
        Ok(output)
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(checked_elements(self.rows, self.cols)?);
        for row in 0..self.rows {
            output.extend(self.decode_row(row)?);
        }
        Ok(output)
    }

    pub fn transform_activation(&self, activation: &[f32]) -> Result<Vec<f32>> {
        if activation.len() != self.cols {
            candle_core::bail!(
                "HQZ4 activation length {} does not match input width {}.",
                activation.len(),
                self.cols
            );
        }
        if activation.iter().any(|value| !value.is_finite()) {
            candle_core::bail!("HQZ4 activations must be finite.");
        }

        let mut output = activation.to_vec();
        for group in 0..self.cols / self.group_size {
            let start = group * self.group_size;
            transform_activation_group(
                &mut output[start..start + self.group_size],
                self.seed,
                self.group_offset + group,
            );
        }
        Ok(output)
    }

    pub fn matvec(&self, activation: &[f32]) -> Result<Vec<f32>> {
        let activation = self.transform_activation(activation)?;
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let weights = self.decode_rotated_row(row)?;
            output.push(
                weights
                    .iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum(),
            );
        }
        Ok(output)
    }

    fn shard(&self, dim: usize, start: usize, len: usize) -> Result<Self> {
        let size = match dim {
            0 => self.rows,
            1 => self.cols,
            _ => candle_core::bail!("HQZ4 can only shard dimensions 0 or 1, got {dim}."),
        };
        let end = start
            .checked_add(len)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 shard range overflow.".into()))?;
        if end > size {
            candle_core::bail!(
                "HQZ4 shard range {start}..{end} exceeds dimension {dim} of size {size}."
            );
        }
        if start == 0 && len == size {
            return Ok(self.clone());
        }

        let cfg = Hqz4Config {
            group_size: self.group_size,
            seed: self.seed,
        };
        let groups_per_row = self.cols / self.group_size;
        if dim == 0 {
            let code_bytes_per_row = self.cols / 2;
            let codes = self.codes[start * code_bytes_per_row..end * code_bytes_per_row].to_vec();
            let scales = self.scales[start * groups_per_row..end * groups_per_row].to_vec();
            return Self::from_parts_at_group_offset(
                (len, self.cols),
                cfg,
                self.group_offset,
                scales,
                codes,
            );
        }

        if !start.is_multiple_of(self.group_size) || !len.is_multiple_of(self.group_size) {
            candle_core::bail!(
                "HQZ4 input shard {start}..{end} must align to group size {}.",
                self.group_size
            );
        }
        let shard_groups = len / self.group_size;
        let first_group = start / self.group_size;
        let mut codes = Vec::with_capacity(self.rows * len / 2);
        let mut scales = Vec::with_capacity(self.rows * shard_groups);
        for row in 0..self.rows {
            let code_start = (row * self.cols + start) / 2;
            codes.extend_from_slice(&self.codes[code_start..code_start + len / 2]);
            let scale_start = row * groups_per_row + first_group;
            scales.extend_from_slice(&self.scales[scale_start..scale_start + shard_groups]);
        }
        let group_offset = self
            .group_offset
            .checked_add(first_group)
            .ok_or_else(|| candle_core::Error::Msg("HQZ4 group offset overflow.".into()))?;
        Self::from_parts_at_group_offset((self.rows, len), cfg, group_offset, scales, codes)
    }
}

#[derive(Debug)]
pub struct HyperQuantLinear {
    weight: Hqz4Tensor,
    bias: Option<Tensor>,
    device: Device,
    #[cfg(feature = "cuda")]
    cuda_weight: Option<Hqz4CudaWeight>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
struct Hqz4CudaWeight {
    codes: Tensor,
    scales: Tensor,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct Hqz4CudaInner {
    codes: Tensor,
    scales: Tensor,
    rows: usize,
    cols: usize,
    group_size: usize,
}

#[cfg(feature = "cuda")]
pub(crate) fn try_fused_qkv_quantized(
    activation: &QuantizedActivation,
    q: &Hqz4CudaInner,
    key: &Hqz4CudaInner,
    value: &Hqz4CudaInner,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Ok((_, input_width)) = activation.quantized().dims2() else {
        return Ok(None);
    };
    if [q, key, value].iter().any(|weight| {
        weight.cols != input_width
            || weight.group_size != q.group_size
            || !activation.quantized().device().same_device(weight.codes.device())
            || !activation.quantized().device().same_device(weight.scales.device())
    }) {
        return Ok(None);
    }
    Ok(Some(cuda::qkv_matmul_quantized(
        activation.quantized(),
        activation.scales(),
        q,
        key,
        value,
        activation.source_shape(),
        activation.source_dtype(),
    )?))
}

#[cfg(feature = "cuda")]
pub(crate) fn try_fused_silu_gate_up_quantized(
    activation: &QuantizedActivation,
    gate: &Hqz4CudaInner,
    up: &Hqz4CudaInner,
) -> Result<Option<Tensor>> {
    let Ok((_, input_width)) = activation.quantized().dims2() else {
        return Ok(None);
    };
    if gate.rows != up.rows
        || gate.cols != input_width
        || up.cols != input_width
        || gate.group_size != up.group_size
        || !activation.quantized().device().same_device(gate.codes.device())
        || !activation.quantized().device().same_device(gate.scales.device())
        || !activation.quantized().device().same_device(up.codes.device())
        || !activation.quantized().device().same_device(up.scales.device())
    {
        return Ok(None);
    }
    Ok(Some(cuda::silu_gate_up_matmul_quantized(
        activation.quantized(),
        activation.scales(),
        gate,
        up,
        activation.source_shape(),
        activation.source_dtype(),
    )?))
}

impl HyperQuantLinear {
    fn supports_device_location(location: DeviceLocation) -> bool {
        match location {
            DeviceLocation::Cpu => true,
            #[cfg(feature = "cuda")]
            DeviceLocation::Cuda { .. } => cuda::HAVE_HQZ4_DP4A_KERNELS,
            #[cfg(not(feature = "cuda"))]
            DeviceLocation::Cuda { .. } => false,
            DeviceLocation::Metal { .. } => false,
        }
    }

    fn ensure_supported_device(device: &Device) -> Result<()> {
        if !Self::supports_device_location(device.location()) {
            candle_core::bail!(
                "HQZ4 supports CPU and CUDA DP4A builds targeting compute capability 6.1 or newer."
            );
        }
        Ok(())
    }

    pub fn from_weight(weight: &Tensor, bias: Option<Tensor>, config: Hqz4Config) -> Result<Self> {
        Self::ensure_supported_device(weight.device())?;
        let (rows, cols) = weight.dims2()?;
        let values = weight
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let encoded = Hqz4Tensor::encode(&values, rows, cols, config)?;
        Self::from_encoded_on_device(encoded, bias, weight.device())
    }

    pub fn from_encoded(weight: Hqz4Tensor, bias: Option<Tensor>) -> Result<Self> {
        let device = bias
            .as_ref()
            .map(|bias| bias.device().clone())
            .unwrap_or(Device::Cpu);
        Self::from_encoded_on_device(weight, bias, &device)
    }

    fn from_encoded_on_device(
        weight: Hqz4Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        Self::ensure_supported_device(device)?;
        if let Some(bias) = &bias {
            if bias.dims() != [weight.rows()] {
                candle_core::bail!(
                    "HQZ4 bias shape {:?} does not match output width {}.",
                    bias.dims(),
                    weight.rows()
                );
            }
        }
        let bias = bias.map(|bias| bias.to_device(device)).transpose()?;
        #[cfg(feature = "cuda")]
        let cuda_weight = if device.is_cuda() {
            Some(Hqz4CudaWeight {
                codes: Tensor::from_vec(
                    weight.codes().to_vec(),
                    (weight.rows(), weight.cols() / 2),
                    device,
                )?,
                scales: Tensor::from_vec(
                    weight.scales().to_vec(),
                    (weight.rows(), weight.cols() / weight.group_size()),
                    device,
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            cuda_weight,
        })
    }

    pub fn encoded(&self) -> &Hqz4Tensor {
        &self.weight
    }

    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &[
            "weight",
            "weight.bits",
            "weight.format",
            "weight.group_size",
            "weight.layout",
            "weight.scales",
            "weight.schema",
            "weight.seed_hi",
            "weight.seed_lo",
            "weight.shape",
            "weight.transform",
        ];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.tensor_dtype("weight", Dtype::U8)
            && layer.scalar("weight.bits", Dtype::U8)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.group_size", Dtype::U32)
            && layer.scalar("weight.layout", Dtype::U8)
            && (layer.tensor_dtype("weight.scales", Dtype::F16)
                || layer.tensor_dtype("weight.scales", Dtype::F32))
            && layer.scalar("weight.schema", Dtype::U32)
            && layer.scalar("weight.seed_hi", Dtype::U32)
            && layer.scalar("weight.seed_lo", Dtype::U32)
            && layer.u32_vector("weight.shape")
            && layer.scalar("weight.transform", Dtype::U8)
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::HyperQuant,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        tensors: &[UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        let bits = crate::uqff::u8_scalar_with_suffix(tensors, prefix, "weight.bits")?;
        if bits != HQZ4_BITS as u8 {
            candle_core::bail!("Unsupported HyperQuant bit width {bits}.");
        }
        Ok("hqz4".to_string())
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        Self::ensure_supported_device(device)?;
        let schema = reader.load_u32_scalar(&format!("{key}.weight.schema"))?;
        let layout = reader.load_u8_scalar(&format!("{key}.weight.layout"))?;
        let transform = reader.load_u8_scalar(&format!("{key}.weight.transform"))?;
        let bits = reader.load_u8_scalar(&format!("{key}.weight.bits"))? as usize;
        if !matches!(schema, HQZ4_LEGACY_SCHEMA_VERSION | HQZ4_SCHEMA_VERSION) {
            candle_core::bail!(
                "Unsupported HQZ4 schema {schema}; expected {HQZ4_LEGACY_SCHEMA_VERSION} or {HQZ4_SCHEMA_VERSION}."
            );
        }
        if layout != HQZ4_LAYOUT_ROW_MAJOR_NIBBLES {
            candle_core::bail!("Unsupported HQZ4 layout {layout}.");
        }
        if transform != HQZ4_TRANSFORM_SHARED_RHT {
            candle_core::bail!("Unsupported HQZ4 transform {transform}.");
        }
        if bits != HQZ4_BITS {
            candle_core::bail!("Unsupported HQZ4 bit width {bits}.");
        }

        let shape = reader.load_u32_vec(&format!("{key}.weight.shape"))?;
        let [rows, cols] = shape.as_slice() else {
            candle_core::bail!("HQZ4 weight shape must have rank 2, got {:?}.", shape);
        };
        let group_size = reader.load_u32_scalar(&format!("{key}.weight.group_size"))? as usize;
        validate_layout(*rows, *cols, group_size)?;
        let expected_weight_shape = [*rows, *cols / 2];
        let actual_weight_shape = reader.tensor_dims(&format!("{key}.weight"))?;
        if actual_weight_shape != expected_weight_shape {
            candle_core::bail!(
                "HQZ4 packed weight shape {:?} does not match expected {:?}.",
                actual_weight_shape,
                expected_weight_shape
            );
        }
        let expected_scale_shape = [*rows, *cols / group_size];
        let actual_scale_shape = reader.tensor_dims(&format!("{key}.weight.scales"))?;
        if actual_scale_shape != expected_scale_shape {
            candle_core::bail!(
                "HQZ4 scale shape {:?} does not match expected {:?}.",
                actual_scale_shape,
                expected_scale_shape
            );
        }

        let seed_lo = reader.load_u32_scalar(&format!("{key}.weight.seed_lo"))? as u64;
        let seed_hi = reader.load_u32_scalar(&format!("{key}.weight.seed_hi"))? as u64;
        let seed = seed_lo | seed_hi << 32;
        let codes = reader.load_raw_u8(&format!("{key}.weight"))?;
        let scales =
            reader.load_tensor(&format!("{key}.weight.scales"), &Device::Cpu)?;
        let expected_scale_dtype = if schema == HQZ4_LEGACY_SCHEMA_VERSION {
            DType::F16
        } else {
            DType::F32
        };
        if scales.dtype() != expected_scale_dtype {
            candle_core::bail!(
                "HQZ4 schema {schema} expects {expected_scale_dtype:?} scales, got {:?}.",
                scales.dtype()
            );
        }
        let scales = scales
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let config = Hqz4Config { group_size, seed };
        let mut weight = Hqz4Tensor::from_parts(*rows, *cols, config, scales, codes)?;
        let range = crate::uqff::shard_range(shard, &shape)?;
        if let Some((dim, start, len)) = range {
            weight = weight.shard(dim, start, len)?;
        }
        let bias = reader.load_bias(key, device, range, shape.len())?;
        Self::from_encoded_on_device(weight, bias, device)
    }

    fn scales_tensor(&self) -> Result<Tensor> {
        Tensor::from_vec(
            self.weight.scales().to_vec(),
            (
                self.weight.rows(),
                self.weight.cols() / self.weight.group_size(),
            ),
            &Device::Cpu,
        )
    }

    #[cfg(feature = "cuda")]
    fn embedding_forward_cuda(&self, ids: &Tensor, output_dtype: DType) -> Result<Tensor> {
        let mut output_shape = ids.dims().to_vec();
        output_shape.push(self.weight.cols());
        let ids = ids.to_dtype(DType::U32)?.flatten_all()?.contiguous()?;
        let cuda_weight = self
            .cuda_weight
            .as_ref()
            .expect("CUDA HQZ4 weights are initialized on CUDA devices");
        let kernel_dtype = if output_dtype == DType::F16 {
            DType::F16
        } else {
            DType::F32
        };
        crate::utils::log::once_log_info(
            "HQZ4 CUDA: using direct packed embedding lookup (no full dequantization).",
        );
        let output = cuda::embedding(
            &ids,
            &cuda_weight.codes,
            &cuda_weight.scales,
            &self.weight,
            kernel_dtype,
        )?
        .reshape(output_shape)?;
        if output.dtype() == output_dtype {
            Ok(output)
        } else {
            output.to_dtype(output_dtype)
        }
    }
}

impl QuantMethod for HyperQuantLinear {
    fn new(_method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        candle_core::bail!("HyperQuantLinear must be constructed from weights or HQZ4 parts.")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Tensor::from_vec(
            self.weight.decode()?,
            (self.weight.rows(), self.weight.cols()),
            &Device::Cpu,
        )
        .and_then(|weight| weight.to_device(&self.device))
    }

    fn embedding_forward(&self, ids: &Tensor, output_dtype: DType) -> Result<Tensor> {
        Self::ensure_supported_device(ids.device())?;
        if !ids.device().same_device(&self.device) {
            candle_core::bail!("HQZ4 embedding ids and weights must use the same device.");
        }
        if ids.device().is_cuda() {
            #[cfg(feature = "cuda")]
            return self.embedding_forward_cuda(ids, output_dtype);
            #[cfg(not(feature = "cuda"))]
            candle_core::bail!("HQZ4 CUDA support is not compiled.");
        }
        self.embedding_forward_raw(ids)?.to_dtype(output_dtype)
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        Self::ensure_supported_device(ids.device())?;
        if !ids.device().same_device(&self.device) {
            candle_core::bail!("HQZ4 embedding ids and weights must use the same device.");
        }
        if ids.device().is_cuda() {
            #[cfg(feature = "cuda")]
            return self.embedding_forward_cuda(ids, DType::F32);
            #[cfg(not(feature = "cuda"))]
            candle_core::bail!("HQZ4 CUDA support is not compiled.");
        }
        let mut output_shape = ids.dims().to_vec();
        output_shape.push(self.weight.cols());
        self.dequantize_w()?
            .index_select(&ids.to_dtype(DType::U32)?.flatten_all()?, 0)?
            .reshape(output_shape)
    }

    fn forward_raw(&self, activation: &Tensor) -> Result<Tensor> {
        Self::ensure_supported_device(activation.device())?;
        if !activation.device().same_device(&self.device) {
            candle_core::bail!("HQZ4 activations and weights must use the same device.");
        }
        if activation.device().is_cuda() {
            #[cfg(feature = "cuda")]
            {
                let original_dtype = activation.dtype();
                let kernel_activation = match original_dtype {
                    DType::F16 | DType::F32 => activation.clone(),
                    DType::BF16 => activation.to_dtype(DType::F16)?,
                    dtype => candle_core::bail!(
                        "HQZ4 CUDA supports F16, BF16, and F32 activations, got {dtype:?}."
                    ),
                };
                let kernel_activation = if kernel_activation.is_contiguous() {
                    kernel_activation
                } else {
                    kernel_activation.contiguous()?
                };
                let cuda_weight = self
                    .cuda_weight
                    .as_ref()
                    .expect("CUDA HQZ4 weights are initialized on CUDA devices");
                crate::utils::log::once_log_info(
                    "HQZ4 CUDA: using the A8/W4 DP4A backend (SM61+).",
                );
                let mut output = cuda::dp4a_matmul(
                    &kernel_activation,
                    &cuda_weight.codes,
                    &cuda_weight.scales,
                    &self.weight,
                )?;
                if let Some(bias) = &self.bias {
                    output = output.broadcast_add(&bias.to_dtype(output.dtype())?)?;
                }
                return if output.dtype() == original_dtype {
                    Ok(output)
                } else {
                    output.to_dtype(original_dtype)
                };
            }
            #[cfg(not(feature = "cuda"))]
            candle_core::bail!("HQZ4 CUDA support is not compiled.");
        }
        let weight = self.dequantize_w()?.to_dtype(activation.dtype())?;
        let bias = self
            .bias
            .as_ref()
            .map(|bias| bias.to_dtype(activation.dtype()))
            .transpose()?;
        Linear::new(weight, bias).forward(activation)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn activation_quantization_scheme(&self) -> Option<ActivationQuantizationScheme> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() && cuda::HAVE_HQZ4_DP4A_KERNELS {
            return Some(ActivationQuantizationScheme {
                // Candle stores the signed two's-complement A8 payload in U8
                // tensors; the DP4A kernel interprets those bytes as i8.
                dtype: DType::U8,
                block_shape: [1, self.weight.group_size()],
                transform: ActivationQuantizationTransform::Hqz4Rht {
                    seed: self.weight.seed(),
                    group_offset: self.weight.group_offset(),
                },
            });
        }
        None
    }

    #[cfg(feature = "cuda")]
    fn hqz4_cuda_inner(&self) -> Option<Hqz4CudaInner> {
        let cuda_weight = self.cuda_weight.as_ref()?;
        Some(Hqz4CudaInner {
            codes: cuda_weight.codes.clone(),
            scales: cuda_weight.scales.clone(),
            rows: self.weight.rows(),
            cols: self.weight.cols(),
            group_size: self.weight.group_size(),
        })
    }

    fn quantize_activation(&self, activation: &Tensor) -> Result<QuantizedActivation> {
        let scheme = self.activation_quantization_scheme().ok_or_else(|| {
            candle_core::Error::msg("HQZ4 shared activation quantization is unavailable")
        })?;
        if !activation.device().same_device(&self.device) {
            candle_core::bail!("HQZ4 activations and weights must use the same device.");
        }
        let source_shape = activation.dims().to_vec();
        let source_dtype = activation.dtype();
        let kernel_activation = match source_dtype {
            DType::F16 | DType::F32 => activation.clone(),
            DType::BF16 => activation.to_dtype(DType::F16)?,
            dtype => candle_core::bail!(
                "HQZ4 shared activation supports F16, BF16, and F32, got {dtype:?}."
            ),
        };
        #[cfg(feature = "cuda")]
        {
            let (quantized, scales) = cuda::quantize_activation(&kernel_activation, &self.weight)?;
            crate::utils::log::once_log_info(
                "HQZ4 CUDA: sharing one transformed A8 activation across compatible projections.",
            );
            QuantizedActivation::new(quantized, scales, source_shape, source_dtype, scheme)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (kernel_activation, source_shape, source_dtype, scheme);
            candle_core::bail!("HQZ4 CUDA support is not compiled.")
        }
    }

    fn forward_quantized(&self, activation: &QuantizedActivation) -> Result<Tensor> {
        let scheme = self.activation_quantization_scheme().ok_or_else(|| {
            candle_core::Error::msg("HQZ4 shared activation quantization is unavailable")
        })?;
        if activation.scheme() != scheme {
            candle_core::bail!(
                "HQZ4 activation scheme {:?} does not match layer scheme {:?}.",
                activation.scheme(),
                scheme
            );
        }
        if !activation.quantized().device().same_device(&self.device) {
            candle_core::bail!("HQZ4 quantized activations and weights must use the same device.");
        }
        #[cfg(feature = "cuda")]
        {
            let cuda_weight = self
                .cuda_weight
                .as_ref()
                .expect("CUDA HQZ4 weights are initialized on CUDA devices");
            let mut output = cuda::dp4a_matmul_quantized(
                activation.quantized(),
                activation.scales(),
                &cuda_weight.codes,
                &cuda_weight.scales,
                &self.weight,
                activation.source_shape(),
                activation.source_dtype(),
            )?;
            if let Some(bias) = &self.bias {
                output = output.broadcast_add(&bias.to_dtype(output.dtype())?)?;
            }
            Ok(output)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = activation;
            candle_core::bail!("HQZ4 CUDA support is not compiled.")
        }
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, self.device.clone())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            DType::F32,
            self.device.clone(),
            vec![self.weight.rows(), self.weight.cols()],
            request,
            true,
        ))
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Self::ensure_supported_device(delta.device())?;
        if !delta.device().same_device(&self.device) {
            candle_core::bail!("HQZ4 delta and weight must use the same device.");
        }
        let weight = (self.dequantize_w()?.to_dtype(delta.dtype())? + delta)?;
        Ok(Arc::new(Self::from_weight(
            &weight,
            self.bias.clone(),
            Hqz4Config {
                group_size: self.weight.group_size(),
                seed: self.weight.seed(),
            },
        )?))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_none() || (dtype == Some(IsqType::HQZ4) && imatrix_weight.is_none()) {
            Self::ensure_supported_device(&device)?;
            if self.device.same_device(&device) {
                return Ok(self);
            }
            let bias = self
                .bias
                .as_ref()
                .map(|bias| bias.to_device(&device))
                .transpose()?;
            return Ok(Arc::new(Self::from_encoded_on_device(
                self.weight.clone(),
                bias,
                &device,
            )?));
        }
        if dtype == Some(IsqType::HQZ4) {
            candle_core::bail!("HQZ4 does not support imatrix.");
        }
        let unquant = crate::UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.dequantize_w()?,
            self.bias.clone(),
        )))?;
        Arc::new(unquant).apply_isq(dtype, device, n_quantized, imatrix_weight, guard)
    }

    fn has_bias(&self) -> bool {
        self.bias.is_some()
    }
}

impl QuantizedSerde for HyperQuantLinear {
    fn name(&self) -> &'static str {
        "hyperquant-hqz4-linear"
    }

    fn isq_serde_supported(&self) -> bool {
        true
    }

    fn uqff_type(&self) -> Option<IsqType> {
        Some(IsqType::HQZ4)
    }

    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        if ty != IsqType::HQZ4 {
            candle_core::bail!("Cannot serialize HQZ4 layer as {ty}; actual type is hqz4.");
        }
        if self.weight.group_offset() != 0 {
            candle_core::bail!(
                "Cannot serialize an input-sharded HQZ4 layer as a full UQFF layer."
            );
        }

        let seed = self.weight.seed();
        let rows = u32::try_from(self.weight.rows())
            .map_err(|_| candle_core::Error::Msg("HQZ4 row count exceeds U32.".into()))?;
        let cols = u32::try_from(self.weight.cols())
            .map_err(|_| candle_core::Error::Msg("HQZ4 column count exceeds U32.".into()))?;
        let group_size = u32::try_from(self.weight.group_size())
            .map_err(|_| candle_core::Error::Msg("HQZ4 group size exceeds U32.".into()))?;
        let mut tensors = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::HyperQuant as u8,
            ),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.schema"), HQZ4_SCHEMA_VERSION),
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.layout"),
                HQZ4_LAYOUT_ROW_MAJOR_NIBBLES,
            ),
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.transform"),
                HQZ4_TRANSFORM_SHARED_RHT,
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), HQZ4_BITS as u8),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.group_size"), group_size),
            UqffTensor::from_u32_vec(format!("{prefix}.weight.shape"), vec![rows, cols], vec![2]),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.seed_lo"), seed as u32),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.seed_hi"), (seed >> 32) as u32),
            UqffTensor::from_raw_u8(
                format!("{prefix}.weight"),
                self.weight.codes().to_vec(),
                vec![self.weight.rows(), self.weight.cols() / 2],
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight.scales"), &self.scales_tensor()?)?,
        ];
        if let Some(bias) = &self.bias {
            tensors.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(tensors)
    }

    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }

    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType> {
        Ok(IsqType::HQZ4)
    }
}

fn validate_layout(rows: usize, cols: usize, group_size: usize) -> Result<()> {
    if rows == 0 || cols == 0 {
        candle_core::bail!("HQZ4 matrices must have non-zero dimensions.");
    }
    if group_size < 2 || !group_size.is_power_of_two() {
        candle_core::bail!("HQZ4 group size {group_size} must be a power of two of at least 2.");
    }
    if !cols.is_multiple_of(group_size) {
        candle_core::bail!("HQZ4 input width {cols} must be divisible by group size {group_size}.");
    }
    checked_elements(rows, cols)?;
    Ok(())
}

fn checked_elements(rows: usize, cols: usize) -> Result<usize> {
    rows.checked_mul(cols)
        .ok_or_else(|| candle_core::Error::Msg("HQZ4 element count overflow.".into()))
}

fn rotate_weight_group(values: &mut [f32], seed: u64, group: usize) {
    apply_signs(values, seed, group);
    normalized_hadamard(values);
}

fn inverse_rotate_weight_group(values: &mut [f32], seed: u64, group: usize) {
    normalized_hadamard(values);
    apply_signs(values, seed, group);
}

fn transform_activation_group(values: &mut [f32], seed: u64, group: usize) {
    apply_signs(values, seed, group);
    normalized_hadamard(values);
}

fn normalized_hadamard(values: &mut [f32]) {
    let mut half = 1;
    while half < values.len() {
        let span = half * 2;
        for start in (0..values.len()).step_by(span) {
            for offset in 0..half {
                let left = values[start + offset];
                let right = values[start + offset + half];
                values[start + offset] = left + right;
                values[start + offset + half] = left - right;
            }
        }
        half = span;
    }
    let normalization = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value *= normalization;
    }
}

fn apply_signs(values: &mut [f32], seed: u64, group: usize) {
    for (index, value) in values.iter_mut().enumerate() {
        if sign_is_negative(seed, group, index) {
            *value = -*value;
        }
    }
}

fn sign_is_negative(seed: u64, group: usize, index: usize) -> bool {
    let state =
        seed ^ (group as u64).wrapping_mul(GROUP_MIX) ^ (index as u64).wrapping_mul(ELEMENT_MIX);
    splitmix64(state) & 1 != 0
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(SPLITMIX_INCREMENT);
    state = (state ^ (state >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    state = (state ^ (state >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    state ^ (state >> 31)
}

fn write_code(codes: &mut [u8], index: usize, value: i8) {
    let nibble = value as u8 & HQZ4_NIBBLE_MASK;
    let byte = &mut codes[index / 2];
    if index.is_multiple_of(2) {
        *byte = (*byte & 0xf0) | nibble;
    } else {
        *byte = (*byte & HQZ4_NIBBLE_MASK) | (nibble << 4);
    }
}

fn read_code(codes: &[u8], index: usize) -> i8 {
    let byte = codes[index / 2];
    let nibble = if index.is_multiple_of(2) {
        byte & HQZ4_NIBBLE_MASK
    } else {
        byte >> 4
    };
    (nibble << 4) as i8 >> 4
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ROWS: usize = 5;
    const TEST_COLS: usize = 128;
    const TEST_GROUP_SIZE: usize = 32;
    const TEST_SEED: u64 = 0x5eed_61;
    const TEST_MAX_RELATIVE_L2: f32 = 0.12;
    const TEST_MIN_COSINE: f32 = 0.995;

    fn test_weights() -> Vec<f32> {
        (0..TEST_ROWS * TEST_COLS)
            .map(|index| {
                let index = index as f32;
                (index * 0.037).sin() * 0.35 + (index * 0.011).cos() * 0.08
            })
            .collect()
    }

    fn test_config() -> Hqz4Config {
        Hqz4Config {
            group_size: TEST_GROUP_SIZE,
            seed: TEST_SEED,
        }
    }

    fn a8_matvec_reference(encoded: &Hqz4Tensor, activation: &[f32]) -> Result<Vec<f32>> {
        let activation = encoded.transform_activation(activation)?;
        let groups_per_row = encoded.cols() / encoded.group_size();
        let mut quantized = vec![0i8; encoded.cols()];
        let mut activation_scales = vec![0f32; groups_per_row];
        for group in 0..groups_per_row {
            let start = group * encoded.group_size();
            let values = &activation[start..start + encoded.group_size()];
            let max_abs = values.iter().copied().map(f32::abs).fold(0f32, f32::max);
            let scale = if max_abs == 0.0 {
                0.0
            } else {
                max_abs / 127.0
            };
            activation_scales[group] = scale;
            for (offset, value) in values.iter().copied().enumerate() {
                quantized[start + offset] = if scale == 0.0 {
                    0
                } else {
                    (value / scale).round().clamp(-127.0, 127.0) as i8
                };
            }
        }

        let mut output = vec![0f32; encoded.rows()];
        for (row, output) in output.iter_mut().enumerate() {
            for group in 0..groups_per_row {
                let start = group * encoded.group_size();
                let dot = (0..encoded.group_size())
                    .map(|offset| {
                        i32::from(read_code(encoded.codes(), row * encoded.cols() + start + offset))
                            * i32::from(quantized[start + offset])
                    })
                    .sum::<i32>();
                *output += dot as f32
                    * encoded.scales()[row * groups_per_row + group]
                    * activation_scales[group];
            }
        }
        Ok(output)
    }

    #[test]
    fn rht_round_trip_restores_values() {
        let mut values = (0..TEST_GROUP_SIZE)
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let expected = values.clone();
        rotate_weight_group(&mut values, TEST_SEED, 3);
        inverse_rotate_weight_group(&mut values, TEST_SEED, 3);

        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn encoding_is_deterministic() -> Result<()> {
        let weights = test_weights();
        let first = Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?;
        let second = Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?;
        let different_seed = Hqz4Tensor::encode(
            &weights,
            TEST_ROWS,
            TEST_COLS,
            Hqz4Config {
                group_size: TEST_GROUP_SIZE,
                seed: TEST_SEED + 1,
            },
        )?;

        assert_eq!(first, second);
        assert_ne!(first.codes(), different_seed.codes());
        assert_eq!(first.codes().len(), weights.len() / 2);
        assert_eq!(
            first.scales().len(),
            TEST_ROWS * TEST_COLS / TEST_GROUP_SIZE
        );
        Ok(())
    }

    #[test]
    fn signed_nibbles_are_low_first() -> Result<()> {
        let encoded = Hqz4Tensor::from_parts(
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
            vec![1.0],
            vec![0x79],
        )?;

        assert_eq!(encoded.decode_rotated_row(0)?, vec![-7.0, 7.0]);
        Ok(())
    }

    #[test]
    fn row_decode_matches_full_decode() -> Result<()> {
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let decoded = encoded.decode()?;

        for row in 0..TEST_ROWS {
            assert_eq!(
                encoded.decode_row(row)?,
                decoded[row * TEST_COLS..(row + 1) * TEST_COLS]
            );
        }
        Ok(())
    }

    #[test]
    fn codec_preserves_weight_geometry() -> Result<()> {
        let weights = test_weights();
        let decoded =
            Hqz4Tensor::encode(&weights, TEST_ROWS, TEST_COLS, test_config())?.decode()?;
        let dot = weights
            .iter()
            .zip(&decoded)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let source_l2 = weights.iter().map(|value| value * value).sum::<f32>();
        let decoded_l2 = decoded.iter().map(|value| value * value).sum::<f32>();
        let error_l2 = weights
            .iter()
            .zip(&decoded)
            .map(|(left, right)| {
                let error = left - right;
                error * error
            })
            .sum::<f32>();
        let cosine = dot / (source_l2 * decoded_l2).sqrt();
        let relative_l2 = (error_l2 / source_l2).sqrt();

        assert!(cosine > TEST_MIN_COSINE, "cosine={cosine}");
        assert!(
            relative_l2 < TEST_MAX_RELATIVE_L2,
            "relative_l2={relative_l2}"
        );
        Ok(())
    }

    #[test]
    fn rotated_matvec_matches_decoded_dense_matvec() -> Result<()> {
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let activation = (0..TEST_COLS)
            .map(|index| (index as f32 * 0.023).cos() * 0.5)
            .collect::<Vec<_>>();
        let actual = encoded.matvec(&activation)?;
        let decoded = encoded.decode()?;
        let expected = decoded
            .chunks_exact(TEST_COLS)
            .map(|row| {
                row.iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[test]
    fn dynamic_a8_reference_tracks_float_activation_path() -> Result<()> {
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let activation = (0..TEST_COLS)
            .map(|index| (index as f32 * 0.023).cos() * 0.5)
            .collect::<Vec<_>>();
        let actual = a8_matvec_reference(&encoded, &activation)?;
        let expected = encoded.matvec(&activation)?;

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 0.02 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_hqz4_dp4a_kernels))]
    #[test]
    fn hyperquant_cuda_dp4a_matches_a8_reference() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let layer = HyperQuantLinear::from_encoded_on_device(encoded.clone(), None, &device)?;
        let activation = (0..3 * TEST_COLS)
            .map(|index| f16::from_f32((index as f32 * 0.017).sin() * 0.5))
            .collect::<Vec<_>>();
        let expected = activation
            .chunks_exact(TEST_COLS)
            .flat_map(|row| {
                let row = row.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
                a8_matvec_reference(&encoded, &row).expect("valid test activation")
            })
            .collect::<Vec<_>>();
        let activation = Tensor::from_vec(activation, (3, TEST_COLS), &device)?;
        let actual = layer
            .forward(&activation)?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 0.02 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_hqz4_dp4a_kernels))]
    #[test]
    fn hyperquant_cuda_embedding_decodes_only_selected_rows() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let expected_weights = encoded.decode()?;
        let layer = HyperQuantLinear::from_encoded_on_device(encoded, None, &device)?;
        let selected = [4u32, 1, 4, 0];
        let ids = Tensor::from_vec(selected.to_vec(), (2, 2), &device)?;
        let actual = layer
            .embedding_forward(&ids, DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = selected
            .iter()
            .flat_map(|row| {
                let start = *row as usize * TEST_COLS;
                expected_weights[start..start + TEST_COLS].iter().copied()
            })
            .collect::<Vec<_>>();

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-5 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[test]
    fn input_shard_matvec_uses_global_group_indices() -> Result<()> {
        let encoded = Hqz4Tensor::encode(&test_weights(), TEST_ROWS, TEST_COLS, test_config())?;
        let shard = encoded.shard(1, TEST_GROUP_SIZE, TEST_GROUP_SIZE * 2)?;
        assert_eq!(shard.group_offset(), 1);
        let activation = (0..TEST_GROUP_SIZE * 2)
            .map(|index| (index as f32 * 0.041).sin())
            .collect::<Vec<_>>();
        let actual = shard.matvec(&activation)?;
        let decoded = shard.decode()?;
        let expected = decoded
            .chunks_exact(TEST_GROUP_SIZE * 2)
            .map(|row| {
                row.iter()
                    .zip(&activation)
                    .map(|(weight, activation)| weight * activation)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();

        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[test]
    fn from_parts_rejects_reserved_code() {
        let error = Hqz4Tensor::from_parts(
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
            vec![1.0],
            vec![0x08],
        )
        .unwrap_err();
        assert!(error.to_string().contains("reserved -8"));
    }

    #[test]
    fn encode_rejects_invalid_inputs() {
        let invalid_group = Hqz4Tensor::encode(
            &[0.0; 12],
            2,
            6,
            Hqz4Config {
                group_size: 3,
                seed: 0,
            },
        )
        .unwrap_err();
        assert!(invalid_group.to_string().contains("power of two"));

        let non_finite = Hqz4Tensor::encode(
            &[0.0, f32::NAN],
            1,
            2,
            Hqz4Config {
                group_size: 2,
                seed: 0,
            },
        )
        .unwrap_err();
        assert!(non_finite.to_string().contains("finite"));
    }

    #[test]
    fn f32_scales_cover_bf16_weight_range() -> Result<()> {
        let mut weights = vec![0.0; HQZ4_DEFAULT_GROUP_SIZE];
        weights[0] = 10_000_000.0;
        let encoded = Hqz4Tensor::encode(
            &weights,
            1,
            HQZ4_DEFAULT_GROUP_SIZE,
            Hqz4Config::default(),
        )?;

        assert!(encoded.scales()[0] > f16::MAX.to_f32());
        assert!(encoded.scales()[0].is_finite());
        Ok(())
    }

    fn test_linear() -> Result<HyperQuantLinear> {
        let weight = Tensor::from_vec(test_weights(), (TEST_ROWS, TEST_COLS), &Device::Cpu)?;
        let bias = Tensor::from_vec(
            (0..TEST_ROWS)
                .map(|row| row as f32 * 0.125 - 0.25)
                .collect::<Vec<_>>(),
            TEST_ROWS,
            &Device::Cpu,
        )?;
        HyperQuantLinear::from_weight(&weight, Some(bias), test_config())
    }

    fn write_test_uqff(layer: &HyperQuantLinear, prefix: &str) -> Result<std::path::PathBuf> {
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(layer.serialize_uqff(prefix, IsqType::HQZ4)?);
        write_uqff_tensors(&tensors, "roundtrip")
    }

    fn write_uqff_tensors(tensors: &[UqffTensor], label: &str) -> Result<std::path::PathBuf> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mistralrs-hqz4-{label}-{}-{stamp}.uqff",
            std::process::id(),
        ));
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .map_err(candle_core::Error::wrap)?;
        Ok(path)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 1e-5 * expected.abs().max(1.0);
            assert!((actual - expected).abs() <= tolerance);
        }
        Ok(())
    }

    #[test]
    fn uqff_round_trip_preserves_hqz4_layer() -> Result<()> {
        const PREFIX: &str = "model.layers.0.mlp.down_proj";

        let layer = test_linear()?;
        let tensors = layer.serialize_uqff(PREFIX, IsqType::HQZ4)?;
        let (stored, shape) = crate::stored_type_from_tensors(&tensors, PREFIX)?;
        assert_eq!(stored, "hqz4");
        assert_eq!(shape, vec![TEST_ROWS, TEST_COLS]);

        let path = write_test_uqff(&layer, PREFIX)?;
        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        assert_eq!(reader.shard_alignment(PREFIX)?, TEST_GROUP_SIZE);
        assert_eq!(reader.pack_factor_for(PREFIX, DType::F16)?, Some(3));
        let loaded = reader
            .load_linear(PREFIX, &Device::Cpu, Shard::default())?
            .expect("HQZ4 layer must load");
        assert_eq!(loaded.uqff_type(), Some(IsqType::HQZ4));
        assert!(loaded.has_bias());
        assert_close(&loaded.dequantize_w()?, &layer.dequantize_w()?)?;

        let activation = Tensor::from_vec(
            (0..2 * TEST_COLS)
                .map(|index| (index as f32 * 0.019).sin())
                .collect::<Vec<_>>(),
            (2, TEST_COLS),
            &Device::Cpu,
        )?;
        assert_close(&loaded.forward(&activation)?, &layer.forward(&activation)?)?;
        drop(reader);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn uqff_schema_one_f16_scales_remain_readable() -> Result<()> {
        const PREFIX: &str = "model.layers.0.mlp.down_proj";

        let layer = test_linear()?;
        let schema_name = format!("{PREFIX}.weight.schema");
        let scales_name = format!("{PREFIX}.weight.scales");
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(
            layer
                .serialize_uqff(PREFIX, IsqType::HQZ4)?
                .into_iter()
                .filter(|tensor| {
                    tensor.name() != schema_name.as_str()
                        && tensor.name() != scales_name.as_str()
                }),
        );
        tensors.push(UqffTensor::from_u32_scalar(
            schema_name,
            HQZ4_LEGACY_SCHEMA_VERSION,
        ));
        let legacy_scales = Tensor::from_vec(
            layer
                .encoded()
                .scales()
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            (
                layer.encoded().rows(),
                layer.encoded().cols() / layer.encoded().group_size(),
            ),
            &Device::Cpu,
        )?;
        tensors.push(UqffTensor::from_tensor(scales_name, &legacy_scales)?);

        let path = write_uqff_tensors(&tensors, "schema-one")?;
        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let loaded = reader
            .load_linear(PREFIX, &Device::Cpu, Shard::default())?
            .expect("HQZ4 schema 1 layer must load");
        assert_close(&loaded.dequantize_w()?, &layer.dequantize_w()?)?;
        drop(reader);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn uqff_input_shard_preserves_global_rht_group_offset() -> Result<()> {
        const PREFIX: &str = "model.layers.0.self_attn.q_proj";

        let layer = test_linear()?;
        let path = write_test_uqff(&layer, PREFIX)?;
        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let loaded = reader
            .load_linear(
                PREFIX,
                &Device::Cpu,
                Shard::Offset {
                    dim: 1,
                    offset: TEST_GROUP_SIZE,
                    len: TEST_GROUP_SIZE * 2,
                },
            )?
            .expect("HQZ4 input shard must load");
        assert!(!loaded.has_bias());

        let expected_weight =
            layer
                .dequantize_w()?
                .narrow(1, TEST_GROUP_SIZE, TEST_GROUP_SIZE * 2)?;
        assert_close(&loaded.dequantize_w()?, &expected_weight)?;
        let activation = Tensor::from_vec(
            (0..TEST_GROUP_SIZE * 2)
                .map(|index| (index as f32 * 0.031).cos())
                .collect::<Vec<_>>(),
            (1, TEST_GROUP_SIZE * 2),
            &Device::Cpu,
        )?;
        let expected = Linear::new(expected_weight, None).forward(&activation)?;
        assert_close(&loaded.forward(&activation)?, &expected)?;

        let error = reader
            .load_linear(
                PREFIX,
                &Device::Cpu,
                Shard::Offset {
                    dim: 1,
                    offset: TEST_GROUP_SIZE / 2,
                    len: TEST_GROUP_SIZE * 2,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("align to group size"));
        drop(reader);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn uqff_output_shard_preserves_matching_bias() -> Result<()> {
        const PREFIX: &str = "model.layers.0.mlp.up_proj";

        let layer = test_linear()?;
        let path = write_test_uqff(&layer, PREFIX)?;
        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let loaded = reader
            .load_linear(
                PREFIX,
                &Device::Cpu,
                Shard::Offset {
                    dim: 0,
                    offset: 1,
                    len: 3,
                },
            )?
            .expect("HQZ4 output shard must load");
        assert!(loaded.has_bias());

        let activation = Tensor::from_vec(
            (0..TEST_COLS)
                .map(|index| (index as f32 * 0.029).sin())
                .collect::<Vec<_>>(),
            (1, TEST_COLS),
            &Device::Cpu,
        )?;
        let expected = layer.forward(&activation)?.narrow(1, 1, 3)?;
        assert_close(&loaded.forward(&activation)?, &expected)?;
        drop(reader);
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn uqff_reader_rejects_unknown_hqz4_schema() -> Result<()> {
        const PREFIX: &str = "model.layers.0.mlp.gate_proj";

        let layer = test_linear()?;
        let schema_name = format!("{PREFIX}.weight.schema");
        let mut tensors = crate::uqff_version_tensors();
        tensors.extend(layer.serialize_uqff(PREFIX, IsqType::HQZ4)?);
        tensors.retain(|tensor| tensor.name() != schema_name);
        tensors.push(UqffTensor::from_u32_scalar(schema_name, 2));
        let path = write_uqff_tensors(&tensors, "invalid-schema")?;
        let reader = UqffReader::open(std::slice::from_ref(&path))?;
        let error = reader
            .load_linear(PREFIX, &Device::Cpu, Shard::default())
            .unwrap_err();
        assert!(error.to_string().contains("Unsupported HQZ4 schema 2"));
        drop(reader);
        let _ = std::fs::remove_file(path);
        Ok(())
    }
}
