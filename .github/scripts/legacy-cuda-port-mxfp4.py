#!/usr/bin/env python3
"""Compatibility shim for Candle cuda_legacy GgmlDType::Mxfp4 (GGML code 39).

Rebased mistral.rs is from upstream before Candle's GGUF MXFP4 enum expansion.
Maintain correct serialization, 17-byte block alignment, and explicit rejection
of converting packed GGUF MXFP4 into the *distinct* runtime IsqType::MXFP4.
Fail closed if the maintained upstream source layout changes.
"""
from pathlib import Path


def replace_one(src: str, old: str, new: str, label: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"MXFP4 {label}: expected one upstream anchor; got {n}")
    return src.replace(old, new, 1)


archive_path = Path("mistralrs-quant/src/gguf/archive.rs")
archive = archive_path.read_text()
archive = replace_one(
    archive,
    "            BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1,",
    "            BlockMxfp4, BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1,",
    "Candle packed block import",
)
archive = replace_one(
    archive,
    "        GgmlDType::BF16 => align_of::<bf16>(),\n",
    "        GgmlDType::BF16 => align_of::<bf16>(),\n"
    "        GgmlDType::Mxfp4 => align_of::<BlockMxfp4>(),\n",
    "GGUF mmap alignment",
)
archive_path.write_text(archive)

gguf_path = Path("mistralrs-quant/src/gguf/mod.rs")
gguf = gguf_path.read_text()
gguf = replace_one(
    gguf,
    "        GgmlDType::BF16 => 30,\n",
    "        GgmlDType::BF16 => 30,\n"
    "        GgmlDType::Mxfp4 => 39,\n",
    "UQFF GGML type encoder",
)
gguf = replace_one(
    gguf,
    "        30 => Ok(GgmlDType::BF16),\n",
    "        30 => Ok(GgmlDType::BF16),\n"
    "        39 => Ok(GgmlDType::Mxfp4),\n",
    "UQFF GGML type decoder",
)
gguf = replace_one(
    gguf,
    '        30 => "bf16",\n',
    '        30 => "bf16",\n'
    '        39 => "mxfp4",\n',
    "GGUF dtype display",
)
gguf = replace_one(
    gguf,
    "#[cfg(test)]\nmod tests {\n    use super::*;\n",
    """#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxfp4_ggml_uqff_type_roundtrip() -> Result<()> {
        assert_eq!(ggml_dtype_to_uqff_code(GgmlDType::Mxfp4), 39);
        assert_eq!(ggml_dtype_from_uqff_code(39)?, GgmlDType::Mxfp4);
        assert_eq!(gguf_dtype_label(39), "mxfp4");
        Ok(())
    }

""",
    "type code roundtrip regression",
)
gguf_path.write_text(gguf)

lib_path = Path("mistralrs-quant/src/lib.rs")
lib = lib_path.read_text()
lib = replace_one(
    lib,
    "            GgmlDType::BF16 | GgmlDType::F32 | GgmlDType::F16 => {\n",
    """            // GGUF MXFP4 is a packed GGML tensor type; IsqType::MXFP4
            // is a distinct learned/activation-aware quantizer and must not
            // be silently conflated with this serialization representation.
            GgmlDType::Mxfp4
            | GgmlDType::BF16
            | GgmlDType::F32
            | GgmlDType::F16 => {
""",
    "ISQ conversion preserves distinct MXFP4 formats",
)
lib_path.write_text(lib)
print("PORT_MXFP4_R1_READY: GGML code=39; alignment=BlockMxfp4; ISQ identity preserved")
