#![cfg(all(feature = "cuda", feature = "cuda-legacy-bf16"))]
use candle_core::{DType, Device, Result, Tensor};

#[test]
fn sm61_bf16_gemm_preserves_values_outside_f16_range() -> Result<()> {
    let gpu = Device::new_cuda(0)?;
    let cpu = Device::Cpu;
    let lhs = Tensor::from_vec(vec![100_000.0f32; 2 * 32], (2, 32), &cpu)?
        .to_dtype(DType::BF16)?;
    let rhs = Tensor::from_vec(vec![2.0f32; 32 * 4], (32, 4), &cpu)?
        .to_dtype(DType::BF16)?;
    let actual = lhs.to_device(&gpu)?.matmul(&rhs.to_device(&gpu)?)?;
    assert_eq!(actual.dtype(), DType::BF16);

    // Form the reference from the same rounded BF16 operands, accumulated
    // in F32 and rounded ONCE to BF16. Do not pass through FP16.
    let expected = lhs.to_dtype(DType::F32)?
        .matmul(&rhs.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)?;
    let got = actual.to_device(&cpu)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let want = expected.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    for (row_got, row_expected) in got.iter().zip(&want) {
        for (&g, &e) in row_got.iter().zip(row_expected) {
            assert!(g.is_finite() && g > 65_504.0,
                "BF16 GEMM crossed FP16 limit incorrectly: {g}");
            assert!((g - e).abs() <= e.abs() * 0.01,
                "BF16 GEMM mismatch: gpu={g} cpu={e}");
        }
    }
    Ok(())
}
