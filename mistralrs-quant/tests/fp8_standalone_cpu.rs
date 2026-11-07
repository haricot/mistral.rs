use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{BlockwiseFP8Linear, QuantMethod, QuantMethodConfig};

#[test]
fn cpu_fp8_blockwise_scales_and_bias_follow_the_checkpoint() -> Result<()> {
    let device = &Device::Cpu;
    let weight = Tensor::ones((2, 128), DType::F32, device)?.to_dtype(DType::F8E4M3)?;
    let scales = Tensor::new(&[[0.5f32], [2.]], device)?;
    let bias = Tensor::new(&[1f32, 3.], device)?;
    let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
        weight,
        weight_scale_inv: scales,
        bias: Some(bias),
        dequant_dtype: DType::F32,
        weight_block_size: vec![1, 128],
        activation_scheme: None,
    })?;
    let input = Tensor::ones((3, 128), DType::F32, device)?;
    let output = layer.forward(&input)?;
    assert_eq!(output.dims(), &[3, 2]);
    assert_eq!(output.to_vec2::<f32>()?, vec![vec![65., 259.]; 3]);
    Ok(())
}

#[test]
fn cpu_fp8_blockwise_keeps_batched_input_shape() -> Result<()> {
    let device = &Device::Cpu;
    let layer = BlockwiseFP8Linear::new(QuantMethodConfig::BlockwiseFP8 {
        weight: Tensor::ones((2, 128), DType::F32, device)?.to_dtype(DType::F8E4M3)?,
        weight_scale_inv: Tensor::ones((1, 1), DType::F32, device)?,
        bias: None,
        dequant_dtype: DType::F32,
        weight_block_size: vec![128, 128],
        activation_scheme: None,
    })?;
    let output = layer.forward(&Tensor::ones((2, 3, 128), DType::F32, device)?)?;
    assert_eq!(output.dims(), &[2, 3, 2]);
    assert_eq!(output.flatten_all()?.to_vec1::<f32>()?, vec![128.; 12]);
    Ok(())
}
