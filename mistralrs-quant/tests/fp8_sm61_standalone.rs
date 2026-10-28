#![cfg(all(feature = "cuda", feature = "cuda-legacy-fp8"))]
use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::{BlockwiseFP8Linear, QuantMethod, QuantMethodConfig};

#[test]
fn sm61_fp8_blockwise_packed_w8a16_smoke() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    // All E4M3FN weights equal +1, with a 1x1 block scale.
    // This verifies a BlockwiseFP8Linear forward through the MistralRS
    // quantization facade, not only Candle's standalone kernel.
    let w = Tensor::ones((128, 128), DType::F8E4M3, &dev)?;
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
    let out = layer.forward(&input)?;
    assert_eq!(out.dims(), &[2, 128]);
    assert_eq!(out.dtype(), DType::F16);
    let rows = out.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    for row in rows {
        for value in row {
            assert!((value - 128.0).abs() <= 2.0, "value={value}");
        }
    }
    Ok(())
}
