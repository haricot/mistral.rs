#!/usr/bin/env python3
"""Read only the GGUF header, to qualify the real Voxtral4 Q4_K inventory.

No additional Python dependencies, no mmap of tensor payload, no GPU.
This is diagnostic data and is NOT a proof of correct transcription.
"""
import argparse
import hashlib
import json
import os
import struct
from collections import Counter
from pathlib import Path

from voxtral4_gguf_contract import validate_header_contract

VALUE_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
MAX_STRING = 8_388_608
MAX_TENSORS = 50_000
MAX_METADATA = 50_000
MAX_ARRAY = 3_000_000


class HeaderReader:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.end = os.fstat(self.file.fileno()).st_size
        self.order = "<"

    def read(self, n):
        if n < 0 or n > MAX_STRING or self.file.tell() + n > self.end:
            raise ValueError("invalid or truncated GGUF header")
        data = self.file.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data

    def scalar(self, fmt):
        return struct.unpack(self.order + fmt, self.read(struct.calcsize(fmt)))[0]

    def text(self):
        n = self.scalar("Q")
        if n > MAX_STRING:
            raise ValueError("GGUF string exceeds safety bound")
        return self.read(n).decode("utf-8", errors="replace")

    def skip(self, n):
        if n < 0 or self.file.tell() + n > self.end:
            raise ValueError("invalid GGUF array length")
        self.file.seek(n, os.SEEK_CUR)

    def value(self, kind):
        if kind == 8:
            return self.text()
        if kind == 9:
            elem_type = self.scalar("I")
            count = self.scalar("Q")
            if count > MAX_ARRAY:
                raise ValueError("GGUF metadata array exceeds safety bound")
            if elem_type in VALUE_TYPES:
                self.skip(count * VALUE_TYPES[elem_type][1])
            elif elem_type == 8:
                for _ in range(count):
                    n = self.scalar("Q")
                    self.skip(n)
            else:
                raise ValueError(f"unsupported nested GGUF metadata array type {elem_type}")
            return {"array_type": elem_type, "count": count}
        if kind in VALUE_TYPES:
            return self.scalar(VALUE_TYPES[kind][0])
        raise ValueError(f"unsupported GGUF metadata value type {kind}")


def inspect(path):
    reader = HeaderReader(path)
    try:
        magic = reader.read(4)
        if magic == b"GGUF":
            reader.order = "<"
        elif magic == b"FUGG":
            reader.order = ">"
        else:
            raise ValueError("not a GGUF file")
        version = reader.scalar("I")
        if version not in (2, 3):
            raise ValueError(f"GGUF version {version} is unsupported by this probe")
        n_tensors = reader.scalar("Q")
        n_metadata = reader.scalar("Q")
        if n_tensors > MAX_TENSORS or n_metadata > MAX_METADATA:
            raise ValueError("GGUF header item count exceeds safety bound")
        metadata = {}
        for _ in range(n_metadata):
            key = reader.text()
            kind = reader.scalar("I")
            value = reader.value(kind)
            if key == "general.architecture" or key.startswith(
                ("voxtral4.", "voxtral4b.", "tokenizer.tekken.", "tokenizer.ggml.")
            ):
                metadata[key] = value
        tensor_names = set()
        tensor_info = {}
        counts = Counter()
        examples = []
        representative = {
            "token_embd.weight", "output_norm.weight",
            "audio.conv.1.weight", "audio.conv.2.weight",
            "audio.ln_post.weight", "proj1.weight", "proj2.weight",
            "audio.blk.0.attn_q.weight",
            "audio.blk.31.ffn_down.weight",
            "blk.0.ada_norm_down.weight",
            "blk.25.ada_norm_up.weight",
            "audio.mel_filters", "audio.mel_window",
            "tokenizer.tekken.vocab_tensor",
        }
        for _ in range(n_tensors):
            name = reader.text()
            nd = reader.scalar("I")
            if nd > 8:
                raise ValueError("GGUF tensor dimensionality exceeds safety bound")
            dims = [reader.scalar("Q") for _ in range(nd)]
            dtype = reader.scalar("I")
            reader.scalar("Q")  # relative tensor offset
            if name in tensor_names:
                raise ValueError(f"duplicate GGUF tensor {name}")
            tensor_names.add(name)
            tensor_info[name] = (tuple(reversed(dims)), dtype)
            counts[str(dtype)] += 1
            if name in representative:
                examples.append({"name": name, "shape": list(reversed(dims)), "ggml_type": dtype})
        contract = validate_header_contract(metadata, tensor_info)
        fingerprint = hashlib.sha256()
        for name, (shape, dtype) in sorted(tensor_info.items()):
            fingerprint.update(json.dumps([name, shape, dtype], separators=(",", ":")).encode("utf-8"))
            fingerprint.update(b"\n")
        return {
            "status": "GGUF_HEADER_ONLY_NOT_RUNTIME_QUALIFIED",
            "b1_contract": contract,
            "tensor_inventory_sha256": fingerprint.hexdigest(),
            "file": Path(path).name,
            "size_bytes": reader.end,
            "gguf_version": version,
            "architecture": metadata.get("general.architecture"),
            "metadata": metadata,
            "tensor_count": n_tensors,
            "tensor_type_counts": dict(counts),
            "representative_tensors": sorted(examples, key=lambda v: v["name"]),
            "missing_representative_tensors": sorted(representative.difference(tensor_names)),
            "audio_block_tensors": sum(n.startswith("audio.blk.") for n in tensor_names),
            "decoder_block_tensors": sum(n.startswith("blk.") for n in tensor_names),
            "header_bytes": reader.file.tell(),
        }
    finally:
        reader.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="Local GGUF file; only the header is read")
    parser.add_argument("--report", help="Write JSON report to this path")
    parser.add_argument("--require-b1", action="store_true", help="Exit nonzero unless full header contract passes")
    args = parser.parse_args()
    result = inspect(args.gguf)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    print(encoded)
    if args.report:
        Path(args.report).write_text(encoded + "\n", encoding="utf-8")
    if args.require_b1 and result["b1_contract"]["status"] != "B1_HEADER_PASS_RUNTIME_NOT_QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
