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
    return changed
