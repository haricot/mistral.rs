use super::Nvfp4LayerParts;
use crate::Nvfp4ActivationMode;
use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;

fn validate(parts: &Nvfp4LayerParts) -> Result<()> {
    if parts.weight.rank() != 2 || parts.activation != Nvfp4ActivationMode::None {
        candle_core::bail!("Pascal NVFP4 supports rank-2 W4A16 only; quantized activations and stacked experts are unsupported");
    }
    Ok(())
}

fn scale_bytes(scales: &Tensor) -> Result<Vec<u8>> {
    Ok(crate::utils::data_to_bytes::<F8E4M3>(
        scales
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<F8E4M3>()?,
    ))
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
pub(super) fn build(
    parts: &Nvfp4LayerParts,
) -> Result<candle_core::quantized::nvfp4_arch_e7::Nvfp4E7Linear> {
    validate(parts)?;
    let device = parts.weight.device().as_cuda_device()?;
    let (n, packed_k) = parts.weight.dims2()?;
    let packed = parts
        .weight
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    let scales = scale_bytes(&parts.scales)?;
    // E7 accepts one multiplier; per-row checkpoint multipliers are applied after its GEMV.
    candle_core::quantized::nvfp4_arch_e7::Nvfp4E7Linear::from_modelopt_bytes(
        device,
        n,
        packed_k * super::VALUES_PER_BYTE,
        &packed,
        &scales,
        1.0,
    )
}

pub(super) fn prepare_input(input: &Tensor, rows: usize, k: usize) -> Result<Tensor> {
    // contiguous() and copy() preserve contiguous views' offsets; E7 also requires exact storage length.
    input
        .to_dtype(DType::F16)?
        .force_contiguous()?
        .reshape((rows, k))
}

pub(super) fn finish(
    output: Tensor,
    parts: &Nvfp4LayerParts,
    input_shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let n = parts.weight.dim(0)?;
    let output = output
        .to_dtype(DType::F32)?
        .broadcast_mul(&parts.global_scales.reshape((1, n))?)?
        .to_dtype(dtype)?;
    let output = match &parts.bias {
        Some(bias) => output.broadcast_add(bias)?,
        None => output,
    };
    let mut shape = input_shape.to_vec();
    *shape.last_mut().unwrap() = n;
    output.reshape(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts() -> Result<Nvfp4LayerParts> {
        Ok(Nvfp4LayerParts {
            weight: Tensor::zeros((2, 8), DType::U8, &Device::Cpu)?,
            scales: Tensor::from_vec(vec![F8E4M3::from_bits(0x38); 2], (2, 1), &Device::Cpu)?,
            global_scales: Tensor::new(&[2f32, 3.0], &Device::Cpu)?,
            input_scale: None,
            activation: Nvfp4ActivationMode::None,
            bias: Some(Tensor::new(&[0.5f32, -0.5], &Device::Cpu)?),
            dtype: DType::F32,
        })
    }

    #[test]
    fn dense_only_and_no_quantized_activations() -> Result<()> {
        let mut parts = parts()?;
        validate(&parts)?;
        parts.activation = Nvfp4ActivationMode::DynamicBlock;
        assert!(validate(&parts).is_err());
        parts.activation = Nvfp4ActivationMode::None;
        parts.weight = parts.weight.unsqueeze(0)?;
        assert!(validate(&parts).is_err());
        Ok(())
    }

    #[test]
    fn scale_payload_preserves_every_e4m3_byte() -> Result<()> {
        let bytes = (0..=u8::MAX).collect::<Vec<_>>();
        let tensor = Tensor::from_vec(
            bytes
                .iter()
                .copied()
                .map(F8E4M3::from_bits)
                .collect::<Vec<_>>(),
            bytes.len(),
            &Device::Cpu,
        )?;
        assert_eq!(scale_bytes(&tensor)?, bytes);
        Ok(())
    }

    #[test]
    fn e7_input_compacts_views_and_preserves_logical_values() -> Result<()> {
        for dtype in [DType::F16, DType::BF16] {
            let base = Tensor::arange(0f32, 24f32, &Device::Cpu)?
                .reshape((3, 2, 4))?
                .to_dtype(dtype)?;
            let offset = base.narrow(0, 1, 1)?;
            assert!(offset.is_contiguous());
            assert_ne!(offset.layout().start_offset(), 0);
            let prefix = base.narrow(0, 0, 1)?;
            assert!(prefix.is_contiguous());
            assert_eq!(prefix.layout().start_offset(), 0);
            let strided = base.transpose(0, 1)?;
            assert!(!strided.is_contiguous());
            for input in [&base, &offset, &prefix, &strided] {
                let rows = input.elem_count() / 4;
                let expected = input
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let actual = prepare_input(input, rows, 4)?;
                assert_eq!(actual.dtype(), DType::F16);
                assert_eq!(actual.dims(), [rows, 4]);
                let (storage, layout) = actual.storage_and_layout();
                assert!(layout.is_contiguous());
                assert_eq!(layout.start_offset(), 0);
                match &*storage {
                    candle_core::Storage::Cpu(storage) => {
                        assert_eq!(storage.as_slice::<half::f16>()?.len(), rows * 4);
                    }
                    _ => unreachable!(),
                }
                assert_eq!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    expected
                );
            }
        }
        Ok(())
    }

    #[test]
    fn row_multipliers_precede_bias_and_restore_batch_shape() -> Result<()> {
        let parts = parts()?;
        let output = Tensor::new(&[[1f32, -2.0], [3.0, 4.0]], &Device::Cpu)?;
        let result = finish(output, &parts, &[1, 2, 16], DType::F32)?;
        assert_eq!(
            result.to_vec3::<f32>()?,
            vec![vec![vec![2.5, -6.5], vec![6.5, 11.5]]]
        );
        Ok(())
    }
}
