#!/usr/bin/env python3
"""Voxtral4b B2-A: decode embedded Tekken bytes and audio assets only.

This never loads quantized model weights or claims that tokenizer IDs match
the 131072-row embedding matrix. Depends only on the Python standard library.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from voxtral4_gguf_probe import HeaderReader, inspect

ASSETS = {
    "tokenizer.tekken.vocab_tensor": (1335218,),
    "audio.mel_filters": (201, 128),
    "audio.mel_window": (400,),
}
F32_GGML_TYPE = 0
MAX_ASSET_BYTES = 8_000_000
MAX_SPECIALS = 2048


def _safe_product(dims):
    n = 1
    for d in dims:
        if not 1 <= d <= 2_000_000:
            raise ValueError("invalid Voxtral asset shape")
        n *= d
        if n * 4 > MAX_ASSET_BYTES:
            raise ValueError("Voxtral asset exceeds safe size limit")
    return n


def read_assets(path):
    reader = HeaderReader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise ValueError("B2 requires little-endian GGUF")
        reader.order = "<"
        version = reader.scalar("I")
        if version != 3:
            raise ValueError("B2 supports only GGUF version 3")
        n_tensors = reader.scalar("Q")
        n_metadata = reader.scalar("Q")
        if n_tensors > 50000 or n_metadata > 50000:
            raise ValueError("unbounded GGUF header")
        metadata = {}
        specials = None
        for _ in range(n_metadata):
            key = reader.text()
            typ = reader.scalar("I")
            if key == "tokenizer.tekken.specials":
                if typ != 9 or reader.scalar("I") != 8:
                    raise ValueError("Tekken specials must be an array of strings")
                n = reader.scalar("Q")
                if not 1 <= n <= MAX_SPECIALS:
                    raise ValueError("invalid Tekken special token count")
                specials = [reader.text() for _ in range(n)]
            else:
                value = reader.value(typ)
                if key.startswith("tokenizer.tekken.") or key in (
                    "general.architecture", "general.alignment",
                    "voxtral4b.llm.vocab_size",
                    "voxtral4b.audio_token_id",
                ):
                    metadata[key] = value
        tensors = {}
        for _ in range(n_tensors):
            name = reader.text()
            nd = reader.scalar("I")
            if not 1 <= nd <= 8:
                raise ValueError("invalid GGUF tensor dimensionality")
            dims = tuple(reversed([reader.scalar("Q") for _ in range(nd)]))
            dtype = reader.scalar("I")
            offset = reader.scalar("Q")
            if name in ASSETS:
                if name in tensors:
                    raise ValueError("duplicate Voxtral asset tensor")
                if dims != ASSETS[name] or dtype != F32_GGML_TYPE:
                    raise ValueError(
                        f"{name}: expected F32 {ASSETS[name]}, got type={dtype} {dims}"
                    )
                tensors[name] = (offset, dims)
        if set(tensors) != set(ASSETS):
            raise ValueError(f"missing GGUF asset tensors: {sorted(set(ASSETS)-set(tensors))}")
        align = metadata.get("general.alignment", 32)
        if type(align) is not int or not 1 <= align <= 4096 or align & (align - 1):
            raise ValueError("invalid GGUF alignment")
        start = (reader.file.tell() + align - 1) // align * align
        result = {}
        for name, (relative, dims) in tensors.items():
            nbytes = _safe_product(dims) * 4
            absolute = start + relative
            if absolute < start or absolute + nbytes > reader.end:
                raise ValueError(f"{name}: tensor outside GGUF data")
            reader.file.seek(absolute)
            result[name] = reader.read(nbytes)
        return result, metadata, specials
    finally:
        reader.file.close()


def decode_tekken(blob, n_vocab):
    """Converter format: 150000 repeated <u16 token_byte_length><token_bytes>."""
    if type(n_vocab) is not int or not 1 <= n_vocab <= 200000:
        raise ValueError("invalid Tekken BPE vocabulary count")
    # Converter stores each original serialized byte as one exact F32 integer.
    raw = bytearray()
    for (value,) in struct.iter_unpack("<f", blob):
        if not math.isfinite(value) or value < 0 or value > 255 or int(value) != value:
            raise ValueError("non-byte value in Tekken F32 vocabulary tensor")
        raw.append(int(value))
    offset = 0
    empty = 0
    max_token_bytes = 0
    tokens_digest = hashlib.sha256()
    for _ in range(n_vocab):
        if offset + 2 > len(raw):
            raise ValueError("truncated Tekken token length")
        length = raw[offset] | (raw[offset + 1] << 8)
        offset += 2
        if offset + length > len(raw):
            raise ValueError("truncated Tekken token data")
        token = raw[offset:offset + length]
        tokens_digest.update(struct.pack("<H", length))
        tokens_digest.update(token)
        offset += length
        empty += length == 0
        max_token_bytes = max(max_token_bytes, length)
    if offset != len(raw):
        raise ValueError(f"Tekken vocabulary has {len(raw)-offset} trailing bytes")
    return {
        "count": n_vocab,
        "packed_bytes": len(raw),
        "packed_sha256": hashlib.sha256(raw).hexdigest(),
        "parsed_sha256": tokens_digest.hexdigest(),
        "empty_tokens": empty,
        "max_token_bytes": max_token_bytes,
        "payload_byte_values_exact": True,
    }


def inspect_assets(path):
    # Reuse the full B1 header contract, not only a representative list.
    head = inspect(path)
    if (head["architecture"] != "voxtral4b"
            or head["b1_contract"]["status"] != "B1_HEADER_PASS_RUNTIME_NOT_QUALIFIED"):
        raise ValueError("GGUF failed the full B1 header contract")
    assets, metadata, specials = read_assets(path)
    if not specials or len(specials) != metadata.get("tokenizer.tekken.n_specials"):
        raise ValueError("Tekken special token names are missing or inconsistent")
    if len(set(specials)) != len(specials):
        raise ValueError("duplicate Tekken special token names")
    vocab = decode_tekken(assets["tokenizer.tekken.vocab_tensor"],
                          metadata.get("tokenizer.tekken.n_vocab"))
    count = metadata.get("voxtral4b.llm.vocab_size")
    if count != 131072:
        raise ValueError("unexpected embedding vocabulary size")
    filter_vals = [v[0] for v in struct.iter_unpack("<f", assets["audio.mel_filters"])]
    window = [v[0] for v in struct.iter_unpack("<f", assets["audio.mel_window"])]
    if not all(math.isfinite(v) and v >= 0 for v in filter_vals):
        raise ValueError("invalid Slaney filter coefficients")
    if not any(v > 0 for v in filter_vals):
        raise ValueError("empty Slaney filter matrix")
    max_hann_error = max(
        abs(value - (0.5 - 0.5 * math.cos(2.0 * math.pi * i / 400.0)))
        for i, value in enumerate(window)
    )
    if max_hann_error > 2e-6:
        raise ValueError(f"GGUF mel window diverges from periodic Hann: {max_hann_error}")
    return {
        "status": "B2_ASSETS_DECODED_TOKEN_ID_MAPPING_NOT_VALIDATED",
        "architecture": head["architecture"],
        "tensor_inventory_sha256": head["tensor_inventory_sha256"],
        "tekken": {
            **vocab,
            "n_specials": len(specials),
            "specials_sha256": hashlib.sha256(
                json.dumps(specials, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "specials_preview": specials[:8],
            "embedding_rows": count,
            "full_token_id_mapping": "NOT_VALIDATED",
            "vocab_exceeds_embedding_rows": vocab["count"] > count,
            "token_id_reconciliation": "REQUIRED_BEFORE_RUNTIME",
        },
        "audio": {
            "mel_filter_shape": list(ASSETS["audio.mel_filters"]),
            "mel_filter_sha256": hashlib.sha256(assets["audio.mel_filters"]).hexdigest(),
            "mel_filter_nonzero_count": sum(v > 0 for v in filter_vals),
            "mel_filter_range": [min(filter_vals), max(filter_vals)],
            "hann_length": len(window),
            "hann_sha256": hashlib.sha256(assets["audio.mel_window"]).hexdigest(),
            "hann_max_abs_error": max_hann_error,
            "sample_rate_hz": 16000,
            "streaming_mel_parity": "NOT_VALIDATED",
        },
        "quantized_forward": "NOT_VALIDATED",
        "audio_forward": "NOT_VALIDATED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = inspect_assets(args.gguf)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(encoded)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
