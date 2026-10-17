#!/usr/bin/env python3
"""B2-B: prove all reachable Tekken token IDs are within embedding rows.

CrispASR src/voxtral4b.cpp and src/voxtral_tekken_vocab.h bound
BPE rank to embedding_rows - special_count (issue #338).
This is an offline, GGUF-assets-only *domain* qualification, not yet
a tokenization/parity or neural forward qualification.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from voxtral4_gguf_assets import inspect_assets, read_assets


def audit_token_domain(encoded_float_blob, special_names, serialized_bpe_count,
                       embedding_rows, audio_token_id):
    if not isinstance(embedding_rows, int) or embedding_rows < 1:
        raise ValueError("invalid embedding rows")
    if not isinstance(serialized_bpe_count, int) or not 1 <= serialized_bpe_count <= 200000:
        raise ValueError("invalid serialized BPE rank count")
    if not isinstance(special_names, list) or not special_names:
        raise ValueError("missing special-token list")
    if len(set(special_names)) != len(special_names):
        raise ValueError("duplicate Tekken special tokens")
    n_specials = len(special_names)
    if n_specials >= embedding_rows:
        raise ValueError("special IDs would exceed the embedding matrix")
    if not isinstance(audio_token_id, int) or not 0 <= audio_token_id < n_specials:
        raise ValueError("invalid Voxtral audio token ID")

    active_limit = embedding_rows - n_specials
    active_count = min(serialized_bpe_count, active_limit)
    inactive_count = serialized_bpe_count - active_count
    if len(encoded_float_blob) % 4:
        raise ValueError("invalid F32-packed Tekken byte length")
    packed = bytearray()
    for (value,) in struct.iter_unpack("<f", encoded_float_blob):
        if not math.isfinite(value) or value < 0 or value > 255 or int(value) != value:
            raise ValueError("Tekken vocabulary contains non-byte F32 value")
        packed.append(int(value))

    offsets = (hashlib.sha256(), hashlib.sha256())
    token_lengths = (0, 0)
    active_bytes = set()
    active_unique = set()
    duplicates = 0
    at = 0
    for rank in range(serialized_bpe_count):
        if at + 2 > len(packed):
            raise ValueError("truncated Tekken BPE header")
        length = packed[at] | (packed[at + 1] << 8)
        at += 2
        if at + length > len(packed):
            raise ValueError("truncated Tekken BPE payload")
        token = bytes(packed[at:at + length])
        at += length
        is_active = rank < active_count
        offsets[0 if is_active else 1].update(struct.pack("<H", length) + token)
        if is_active:
            if token in active_unique:
                duplicates += 1
            active_unique.add(token)
            if len(token) == 1:
                active_bytes.add(token[0])
    if at != len(packed):
        raise ValueError("unexpected Tekken trailing bytes")
    max_active_id = n_specials + active_count - 1
    if max_active_id >= embedding_rows:
        raise ValueError("active Tekken rank would cause out-of-bounds embedding lookup")
    if len(active_bytes) != 256:
        raise ValueError(f"active vocabulary does not cover all 256 byte tokens: {len(active_bytes)}")

    return {
        "status": "B2B_TOKEN_DOMAIN_PASS_TOKENIZATION_PARITY_NOT_VALIDATED",
        "serialized_bpe": serialized_bpe_count,
        "special_token_count": n_specials,
        "embedding_rows": embedding_rows,
        "active_bpe": active_count,
        "inactive_bpe": inactive_count,
        "first_active_bpe_id": n_specials,
        "max_active_bpe_id": max_active_id,
        "first_inactive_bpe_id": n_specials + active_count if inactive_count else None,
        "active_token_ids_within_embedding_bounds": True,
        "active_single_byte_alphabet": len(active_bytes),
        "active_duplicate_piece_count": duplicates,
        "special_audio_token_id": audio_token_id,
        "special_audio_token_name": special_names[audio_token_id],
        "active_bpe_sha256": offsets[0].hexdigest(),
        "inactive_bpe_sha256": offsets[1].hexdigest(),
        "full_packed_vocab_sha256": hashlib.sha256(packed).hexdigest(),
        "mapping_basis": "CrispASR voxtral_tekken::active_bpe_count",
        "tekken_regex_parity": "NOT_VALIDATED",
        "text_to_token_ids_parity": "NOT_VALIDATED",
        "quantized_forward": "NOT_VALIDATED",
    }


def inspect_token_domain(path):
    # B2-A verifies F32 integrity / Slaney / Hann / the entire B1 inventory.
    b2a = inspect_assets(path)
    assets, metadata, specials = read_assets(path)
    audio_id = metadata.get("voxtral4b.audio_token_id")
    if audio_id is None:
        raise ValueError("Voxtral GGUF audio token ID metadata missing")
    result = audit_token_domain(
        assets["tokenizer.tekken.vocab_tensor"],
        specials,
        metadata.get("tokenizer.tekken.n_vocab"),
        metadata.get("voxtral4b.llm.vocab_size"),
        audio_id,
    )
    if result["full_packed_vocab_sha256"] != b2a["tekken"]["packed_sha256"]:
        raise ValueError("B2-A and B2-B disagree on packed Tekken vocab")
    result["b2a_status"] = b2a["status"]
    result["tensor_inventory_sha256"] = b2a["tensor_inventory_sha256"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = inspect_token_domain(args.gguf)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(encoded)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
