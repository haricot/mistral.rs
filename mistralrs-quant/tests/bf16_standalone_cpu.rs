use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethod, QuantMethodConfig, UnquantLinear};

#[test]
fn cpu_bf16_linear_preserves_range_and_rounds_reference_once() -> Result<()> {
    let device = &Device::Cpu;
    let input = Tensor::from_vec(vec![100_000f32; 64], (2, 32), device)?.to_dtype(DType::BF16)?;
    let weight = Tensor::from_vec(vec![2f32; 128], (4, 32), device)?.to_dtype(DType::BF16)?;
    let expected = input
        .to_dtype(DType::F32)?
        .matmul(&weight.to_dtype(DType::F32)?.t()?)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weight, None)))?;
    let actual = layer.forward(&input)?;
    assert_eq!(actual.dtype(), DType::BF16);
    assert_eq!(actual.to_dtype(DType::F32)?.to_vec2::<f32>()?, expected);
    assert!(expected
        .iter()
        .flatten()
        .all(|x| x.is_finite() && *x > 65_504.0));
    Ok(())
}

#[test]
fn cpu_bf16_batched_linear_keeps_bias_and_output_shape() -> Result<()> {
    let device = &Device::Cpu;
    let input =
        Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 1, 2), device)?.to_dtype(DType::BF16)?;
    let weight = Tensor::new(&[[1f32, 1.], [2., 0.]], device)?.to_dtype(DType::BF16)?;
    let bias = Tensor::new(&[1f32, 2.], device)?.to_dtype(DType::BF16)?;
    let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
        weight,
        Some(bias),
    )))?;
    let output = layer.forward(&input)?;
    assert_eq!(output.dims(), &[2, 1, 2]);
    assert_eq!(
        output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?,
        vec![4., 4., 8., 8.]
    );
    Ok(())
}
