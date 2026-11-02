#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{
    blockwise_fp8_moe, BlockwiseFP8Linear, Fp8ActivationScheme, QuantMethod,
    QuantMethodConfig,
};

fn assert_rows(output: Tensor, expected: &[f32], dtype: DType) -> Result<()> {
    assert_eq!(output.dtype(), dtype);
    let actual = output.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(a.is_finite() && (*a - *b).abs() <= b.abs() * 0.02,
            "FP8 SM61 output[{i}]: actual={a} expected={b}");
    }
    Ok(())
}

#[test]
fn sm61_fp8_blockwise_packed_f16_w8a16() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let w = Tensor::ones((128, 128), DType::F32, &dev)?.to_dtype(DType::F8E4M3)?;
    let scales = Tensor::ones((1, 1), DType::F32, &dev)?;
    let layer = <BlockwiseFP8Linear as QuantMethod>::new(
        QuantMethodConfig::BlockwiseFP8 {
            weight: w,
            weight_scale_inv: scales,
            bias: None,
            dequant_dtype: DType::F16,
            weight_block_size: vec![128, 128],
            activation_scheme: None,
        }
    )?;
    let input = Tensor::ones((2, 128), DType::F16, &dev)?;
    assert_rows(layer.forward(&input)?, &vec![128.0; 2 * 128], DType::F16)
}

#[test]
fn sm61_fp8_blockwise_packed_bf16_preserves_range() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let w = Tensor::ones((128, 128), DType::F32, &dev)?.to_dtype(DType::F8E4M3)?;
    let scales = Tensor::ones((1, 1), DType::F32, &dev)?;
    let layer = <BlockwiseFP8Linear as QuantMethod>::new(
        QuantMethodConfig::BlockwiseFP8 {
            weight: w,
            weight_scale_inv: scales,
            bias: None,
            dequant_dtype: DType::BF16,
            weight_block_size: vec![128, 128],
            // Dynamic activation checkpoints are routed to W8ABF16, not
            // to FP8 W8A8 Tensor Core paths on unsupported Pascal hardware.
            activation_scheme: Some(Fp8ActivationScheme::Dynamic),
        }
    )?;
    let x = Tensor::from_vec(vec![1.0e8f32; 2 * 128], (2, 128), &dev)?
        .to_dtype(DType::BF16)?;
    let rounded = x.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?.to_vec2::<f32>()?[0][0];
    let expected = Tensor::new(128.0f32 * rounded, &Device::Cpu)?
        .to_dtype(DType::BF16)?.to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    assert!(expected > 65504.0);
    assert_rows(layer.forward(&x)?, &vec![expected; 2 * 128], DType::BF16)
}

#[test]
fn sm61_fp8_moe_groups_experts_without_dense_weights() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let w0 = Tensor::ones((1, 128, 128), DType::F32, &dev)?;
    let w1 = (&w0 * 2.0)?;
    let w = Tensor::cat(&[w0, w1], 0)?.to_dtype(DType::F8E4M3)?;
    let scales = Tensor::ones((2, 1, 1), DType::F32, &dev)?;
    let layer = blockwise_fp8_moe(
        w, scales, vec![128, 128], Some(Fp8ActivationScheme::Dynamic), DType::BF16,
    )?;
    let x = Tensor::ones((2, 128), DType::BF16, &dev)?;
    let indices = Tensor::from_vec(vec![0u32, 1, 1, 0], (2, 2), &dev)?;
    let result = layer.gather_forward(&x, &indices)?;
    assert_eq!(result.dims(), &[2, 2, 128]);
    let mut expected = Vec::with_capacity(4 * 128);
    for v in [128.0f32, 256.0, 256.0, 128.0] {
        expected.extend(std::iter::repeat_n(v, 128));
    }
    assert_rows(result, &expected, DType::BF16)
}

#[test]
fn sm61_fp8_moe_uses_per_slot_activations() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let weights = Tensor::ones((2, 128, 128), DType::F32, &dev)?
        .to_dtype(DType::F8E4M3)?;
    let scale = Tensor::ones((2, 1, 1), DType::F32, &dev)?;
    let layer = blockwise_fp8_moe(weights, scale, vec![128, 128], None, DType::F16)?;
    let x = Tensor::from_vec(
        [vec![1f32; 128], vec![3f32; 128]].concat(), (1, 2, 128), &dev,
    )?.to_dtype(DType::F16)?;
    let indices = Tensor::from_vec(vec![0u32, 1], (1, 2), &dev)?;
    let y = layer.gather_forward(&x, &indices)?;
    assert_eq!(y.dims(), &[1, 2, 128]);
    let expected = [vec![128.0f32; 128], vec![384.0; 128]].concat();
    assert_rows(y, &expected, DType::F16)
}
