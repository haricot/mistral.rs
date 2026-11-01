"""Preserve the pinned Candle MXFP4 encoding in the upstream embedding codec."""
from pathlib import Path


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("unrecognized MXFP4 embedding migration anchor")
    return text.replace(old, new, 1)


def install(repo):
    repo = Path(repo)
    path = repo / "mistralrs-quant/src/embedding/mod.rs"
    if not path.exists() or "GgmlDType::Mxfp4 =>" in path.read_text():
        return False
    text = path.read_text()
    for old, new in (
        ("    Iq4Nl,\n", "    Iq4Nl,\n    Mxfp4 = 18,\n"),
        ("            Self::Q8K => (256, 292),", "            Self::Q8K => (256, 292),\n            Self::Mxfp4 => (32, 17),"),
        ("            GgmlDType::Q8K => GgufRowType::Q8K,", "            GgmlDType::Q8K => GgufRowType::Q8K,\n            GgmlDType::Mxfp4 => GgufRowType::Mxfp4,"),
        ("const GGUF_IQ4_NL: u32 = 20;", "const GGUF_IQ4_NL: u32 = 20;\nconst GGUF_MXFP4: u32 = 39;"),
        ("            GGUF_IQ4_NL => Self::Gguf(GgufRowType::Iq4Nl),", "            GGUF_IQ4_NL => Self::Gguf(GgufRowType::Iq4Nl),\n            GGUF_MXFP4 => Self::Gguf(GgufRowType::Mxfp4),"),
        ("        GgufRowType::Q8K => dequantize_blocks::<BlockQ8K>(src, out),", "        GgufRowType::Q8K => dequantize_blocks::<BlockQ8K>(src, out),\n        GgufRowType::Mxfp4 => dequantize_blocks::<BlockMxfp4>(src, out),"),
    ):
        text = replace_once(text, old, new)
    path.write_text(text)
    path = repo / "mistralrs-quant/kernels/embedding/gather_rows.cu"
    text = path.read_text()
    text = replace_once(text, "  SYM_INT8 = 20,", "  SYM_INT8 = 20,\n  MXFP4 = 21,")
    text = replace_once(text, "  } else if constexpr (C == FP4) {", """  } else if constexpr (C == MXFP4) {
    const uint8_t *b = r.data + (col / 32) * 17;
    const uint32_t e = b[0];
    const float d = __uint_as_float(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      const int c = col % 32 + i;
      const int code = (b[1 + c % 16] >> (c < 16 ? 0 : 4)) & 15;
      v[i] = (2.0f * e2m1_to_float(code)) * d;
    }
  } else if constexpr (C == FP4) {""")
    text = replace_once(text, "    GATHER_CASE(IQ4_NL)", "    GATHER_CASE(IQ4_NL)\n    GATHER_CASE(MXFP4)")
    path.write_text(text)
    path = repo / "mistralrs-quant/src/metal_kernels/embedding.metal"
    text = path.read_text()
    text = replace_once(text, "SYM_INT8 = 20,", "SYM_INT8 = 20, MXFP4 = 21,")
    text = replace_once(text, "  } else if (C == FP4) {", """  } else if (C == MXFP4) {
    const device uchar *b = data + (col / 32) * 17;
    const uint e = b[0];
    const float d = as_type<float>(e < 2 ? (0x00200000u << e) : ((e - 1) << 23));
    const uint c = col % 32;
    const int code = (b[1 + c % 16] >> (c < 16 ? 0 : 4)) & 15;
    const float q = (code & 8) ? -fp4_values[code & 7] : fp4_values[code & 7];
    return (2.0f * q) * d;
  } else if (C == FP4) {""")
    text = replace_once(text, "EMBEDDING_CODEC(20)\n", "EMBEDDING_CODEC(20)\nEMBEDDING_CODEC(21)\n")
    path.write_text(text)
    return True
