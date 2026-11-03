"""Keep scalar FP8 available without enabling SM80 grouped tensor-core kernels."""
from pathlib import Path


def replace_exact(text, old, new, count, path):
    if text.count(new) == count and old not in text:
        return text, False
    if text.count(old) != count or new in text:
        raise RuntimeError(f"unrecognized SM61 FP8 dispatch contract: {path}")
    return text.replace(old, new), True


def install(repo):
    repo = Path(repo)
    build = repo / "mistralrs-quant/build.rs"
    if not build.exists():
        return False
    text = build.read_text()
    old = '                "grouped_mm_*.cu",\n'
    new = old + '                "grouped_fp8.cu",\n'
    if text.count(new) == 1:
        changed = False
    elif text.count(old) == 1 and '"grouped_fp8.cu"' not in text:
        build.write_text(text.replace(old, new, 1))
        changed = True
    else:
        raise RuntimeError("unrecognized SM61 CUDA exclusion contract")
    for name, count, old, new in (
        ("moe/mod.rs", 1,
         '#[cfg(has_blockwise_fp8_kernels)]\npub(crate) mod grouped_fp8;',
         '#[cfg(has_fp8_mma_kernels)]\npub(crate) mod grouped_fp8;'),
        ("mxfp8/mod.rs", 1,
         '#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]',
         '#[cfg(all(feature = "cuda", has_fp8_mma_kernels))]'),
        ("pertensor_fp8/mod.rs", 4,
         '#[cfg(all(feature = "cuda", has_blockwise_fp8_kernels))]',
         '#[cfg(all(feature = "cuda", has_fp8_mma_kernels))]'),
    ):
        path = repo / "mistralrs-quant/src" / name
        text, patched = replace_exact(path.read_text(), old, new, count, name)
        if patched:
            path.write_text(text)
        changed = changed or patched
    path = repo / "mistralrs-quant/src/fp8/quantize.rs"
    text, patched = split_fp8_roundtrip(path.read_text())
    if patched:
        path.write_text(text)
    return changed or patched


def split_fp8_roundtrip(text):
    old = """    fn test_roundtrip_f8e4m3() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
"""
    new = """    fn test_roundtrip_f8e4m3() -> Result<()> {
        roundtrip_f8e4m3_on_device(Device::Cpu)
    }

    #[test]
    #[cfg(all(feature = "cuda", not(feature = "metal")))]
    fn cuda_sm61_roundtrip_f8e4m3() -> Result<()> {
        roundtrip_f8e4m3_on_device(Device::new_cuda(0)?)
    }

    #[cfg(not(feature = "metal"))]
    fn roundtrip_f8e4m3_on_device(dev: Device) -> Result<()> {
"""
    return replace_exact(text, old, new, 1, "FP8 CPU/CUDA roundtrip test")
