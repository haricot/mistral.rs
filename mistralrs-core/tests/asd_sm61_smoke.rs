#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Result, Tensor};

#[test]
fn asd_sm61_smoke() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let data = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let shifted = (&data + &data)?;
    assert_eq!(shifted.dtype(), DType::F32);
    assert_eq!(shifted.to_device(&Device::Cpu)?.to_vec1::<f32>()?, vec![2.0, 4.0, 6.0, 8.0]);
    Ok(())
}
