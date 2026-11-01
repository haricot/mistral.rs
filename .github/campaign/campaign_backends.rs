use candle_core::{Device, Result, Tensor};

fn check_backend(device: &Device) -> Result<()> {
    let input = Tensor::new(&[[1f32, 2.], [3., 4.]], device)?;
    let result = input.matmul(&input)?;
    device.synchronize()?;
    assert_eq!(
        result.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
        vec![vec![7., 10.], vec![15., 22.]]
    );
    Ok(())
}

#[test]
fn campaign_cpu_backend() -> Result<()> {
    check_backend(&Device::Cpu)
}

#[cfg(feature = "cuda")]
#[test]
fn campaign_cuda_backend() -> Result<()> {
    check_backend(&Device::new_cuda(0)?)
}

fn nvfp4_fixture(
    device: &Device,
    dtype: candle_core::DType,
) -> Result<mistralrs_quant::nvfp4::Nvfp4Layer> {
    use candle_core::DType;
    use mistralrs_quant::nvfp4::{Nvfp4Layer, Nvfp4LayerParts};
    use mistralrs_quant::Nvfp4ActivationMode;
    Nvfp4Layer::from_parts(Nvfp4LayerParts {
        weight: Tensor::from_vec(vec![0x22u8; 16], (2, 8), device)?,
        scales: Tensor::ones((2, 1), DType::F32, &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?
            .to_device(device)?,
        global_scales: Tensor::new(&[0.5f32, 2.0], device)?,
        input_scale: None,
        activation: Nvfp4ActivationMode::None,
        bias: None,
        dtype,
    })
}

#[test]
fn campaign_cpu_nvfp4_w4a16() -> Result<()> {
    use candle_core::DType;
    use mistralrs_quant::QuantMethod;
    let layer = nvfp4_fixture(&Device::Cpu, DType::F32)?;
    let input = Tensor::new(&[[1f32; 16], [2f32; 16]], &Device::Cpu)?;
    assert_eq!(
        layer.forward(&input)?.to_vec2::<f32>()?,
        vec![vec![8., 32.], vec![16., 64.]]
    );
    Ok(())
}

fn mxfp4_embedding(device: &Device, dtype: candle_core::DType) -> Result<()> {
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        DType,
    };
    use mistralrs_quant::EmbeddingTable;
    let dense = Tensor::from_vec(
        (0..64)
            .map(|i| (i as f32 - 31.5) * 0.125)
            .collect::<Vec<_>>(),
        (2, 32),
        &Device::Cpu,
    )?;
    let weights = QTensor::quantize(&dense, GgmlDType::Mxfp4)?;
    let ids = Tensor::new(&[1u32, 0, 1], &Device::Cpu)?;
    let reference = weights
        .dequantize(&Device::Cpu)?
        .index_select(&ids, 0)?
        .to_dtype(dtype)?
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?;
    #[cfg(feature = "cuda")]
    let weights = if device.is_cuda() {
        use candle_core::quantized::QStorage;
        QTensor::new(
            QStorage::from_data(weights.data()?, device, weights.dtype())?,
            weights.shape().clone(),
        )?
    } else {
        weights
    };
    let actual =
        EmbeddingTable::quantized(&weights)?.gather(&ids.to_device(device)?, dtype.into())?;
    device.synchronize()?;
    assert_eq!(
        actual
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?,
        reference
    );
    Ok(())
}

#[test]
fn campaign_cpu_mxfp4_embedding() -> Result<()> {
    mxfp4_embedding(&Device::Cpu, candle_core::DType::F32)
}

#[cfg(all(feature = "cuda", feature = "cuda-legacy-fp4"))]
#[test]
fn campaign_cuda_nvfp4_sm61_w4a16() -> Result<()> {
    use candle_core::DType;
    use mistralrs_quant::QuantMethod;
    let device = Device::new_cuda(0)?;
    let input = Tensor::new(&[[1f32; 16], [2f32; 16]], &device)?;
    for dtype in [DType::F16, DType::BF16] {
        let actual = nvfp4_fixture(&device, dtype)?.forward(&input.to_dtype(dtype)?)?;
        device.synchronize()?;
        assert_eq!(actual.dtype(), dtype);
        assert_eq!(
            actual
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec2::<f32>()?,
            vec![vec![8., 32.], vec![16., 64.]]
        );
    }
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        mxfp4_embedding(&device, dtype)?;
    }
    Ok(())
}
