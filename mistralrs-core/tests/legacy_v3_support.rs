use candle_core::{Device, Result, Tensor};

#[test]
fn cpu_backend_remains_available() -> Result<()> {
    let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
    assert_eq!(
        x.matmul(&x)?.to_vec2::<f32>()?,
        vec![vec![7., 10.], vec![15., 22.]]
    );
    Ok(())
}
