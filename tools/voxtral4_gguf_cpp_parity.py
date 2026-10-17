#!/usr/bin/env python3
"""B2-C2: reproducible real-GGUF Rust vs pinned CrispASR C++ Tekken token IDs.

Only GGUF's embedded Tekken resource (1.3MB) is read; the 2.52GB quantized
weights are never materialized. Source pin and git-blob SHA are mandatory.

This checks deployed CrispASR voxtral4b.cpp text behavior separately from
the richer Unicode pretokenizer in the CrispASR shared Tekken header.
Neither mode proves Mistral's official tokenizer parity or audio inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.request import urlopen

from voxtral4_gguf_assets import read_assets
from voxtral4_gguf_token_ids import inspect_token_domain

PINNED_CRISPASR_COMMIT = "369029c91d90a61d8346b8bbdafe426bfdde76d1"
PINNED_HEADER_BLOB_SHA1 = "66a3a97a74adf94b2b177e41552cd0130e90fd58"
PINNED_HEADER_URL = (
    "https://raw.githubusercontent.com/CrispStrobe/CrispASR/"
    + PINNED_CRISPASR_COMMIT + "/src/voxtral_tekken_vocab.h"
)


def git_blob_sha1(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def get_reference_header(path=None):
    if path:
        content = Path(path).read_bytes()
    else:
        with urlopen(PINNED_HEADER_URL, timeout=40) as response:
            content = response.read(100_000)
    digest = git_blob_sha1(content)
    if digest != PINNED_HEADER_BLOB_SHA1:
        raise ValueError(f"CrispASR header git blob SHA mismatch: {digest}")
    return content


def packed_tekken_bytes(f32_blob):
    if len(f32_blob) % 4:
        raise ValueError("bad GGUF Tekken F32 size")
    packed = bytearray()
    for (value,) in struct.iter_unpack("<f", f32_blob):
        if not 0 <= value <= 255 or int(value) != value:
            raise ValueError("non-byte-valued Tekken F32 tensor")
        packed.append(int(value))
    return bytes(packed)


def create_sidecar(path, metadata, specials, f32_blob):
    if metadata.get("voxtral4b.llm.vocab_size") != 131072:
        raise ValueError("wrong embedding table size")
    if (len(specials) != 1000
            or metadata.get("tokenizer.tekken.n_specials") != 1000):
        raise ValueError("unexpected Voxtral4b special-token count")
    if metadata.get("tokenizer.tekken.n_vocab") != 150000:
        raise ValueError("unexpected serialized BPE rank count")
    packed = packed_tekken_bytes(f32_blob)
    blob = bytearray()
    blob.extend(struct.pack("<II", 131072, len(specials)))
    for token in specials:
        encoded = token.encode("utf-8")
        if len(encoded) > 65535:
            raise ValueError("special token string too long")
        blob.extend(struct.pack("<H", len(encoded)))
        blob.extend(encoded)
    blob.extend(struct.pack("<II", 150000, len(packed)))
    blob.extend(packed)
    path.write_bytes(blob)
    return hashlib.sha256(packed).hexdigest()


def run_reference(executable, sidecar, mode, corpus):
    cmd = [str(executable), str(sidecar), mode]
    cmd.extend(text.encode("utf-8").hex() for text in corpus)
    output = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    rows = output.stdout.splitlines()
    if len(rows) != len(corpus):
        raise ValueError(f"CrispASR C++ reference returned {len(rows)} results for {len(corpus)} cases")
    parsed = [json.loads(row) for row in rows]
    for seq in parsed:
        if not isinstance(seq, list) or any(type(v) is not int or not 0 <= v < 131072 for v in seq):
            raise ValueError("CrispASR oracle returned a malformed/out-of-bounds ID")
    return parsed


def check(model, header, report_path, repo_root):
    compiler = shutil.which("g++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("C++ compiler unavailable: install gcc/g++")
    b2b = inspect_token_domain(model)
    if b2b["status"] != "B2B_TOKEN_DOMAIN_PASS_TOKENIZATION_PARITY_NOT_VALIDATED":
        raise ValueError(f"B2-B contract not qualified: {b2b['status']}")
    source = repo_root / "tools/voxtral4_gguf_cpp_oracle.cpp"
    corpus = json.loads((repo_root / "tools/voxtral4_gguf_french_corpus.json").read_text(encoding="utf-8"))
    if not isinstance(corpus, list) or len(corpus) < 20 or any(not isinstance(s, str) for s in corpus):
        raise ValueError("invalid fixed French test corpus")
    resources, metadata, specials = read_assets(model)
    if not all(k in resources for k in ("tokenizer.tekken.vocab_tensor",)):
        raise ValueError("Tekken GGUF resource missing")
    reference = get_reference_header(header)
    with tempfile.TemporaryDirectory(prefix="voxtral4-b2c2-") as temp:
        directory = Path(temp)
        (directory / "voxtral_tekken_vocab.h").write_bytes(reference)
        exe = directory / "voxtral4_cpp_oracle"
        sidecar = directory / "tekken_reference.bin"
        digest = create_sidecar(
            sidecar, metadata, specials, resources["tokenizer.tekken.vocab_tensor"]
        )
        if digest != b2b["full_packed_vocab_sha256"]:
            raise ValueError("B2-B and sidecar Tekken SHA256 disagree")
        subprocess.run(
            [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra",
             "-I", str(directory), str(source), "-o", str(exe)],
            check=True, timeout=120
        )
        deployed = run_reference(exe, sidecar, "runtime", corpus)
        unicode_tokens = run_reference(exe, sidecar, "unicode_header", corpus)
        divergent = [
            {"index": i, "text": corpus[i],
             "runtime_token_count": len(x), "unicode_token_count": len(y),
             "runtime_first_ids": x[:20], "unicode_first_ids": y[:20]}
            for i, (x, y) in enumerate(zip(deployed, unicode_tokens)) if x != y
        ]
        env = dict(os.environ)
        env.update(
            VOXTRAL4_GGUF_PATH=str(Path(model).resolve()),
            VOXTRAL4_CPP_ORACLE=str(exe),
            VOXTRAL4_CPP_SIDECAR=str(sidecar),
        )
        # A real model and the independently compiled C++ reference are passed
        # into the dedicated Rust unit test. A mismatched ID fails the command.
        test = subprocess.run(
            ["cargo", "test", "--locked", "-p", "mistralrs-core", "--lib",
             "gguf::voxtral4_tokenizer::tests::optional_crispasr_cpp_differential",
             "--", "--nocapture"],
            cwd=repo_root, env=env, capture_output=True, text=True, timeout=1800
        )
        result = {
            "status": ("B2C2_DEPLOYED_CRISPASR_PARITY_PASS_UNICODE_PARITY_NOT_ESTABLISHED"
                       if test.returncode == 0 else "B2C2_FAIL_RUST_CRISPASR_MISMATCH_OR_COMPILATION"),
            "checkpoint": Path(model).name,
            "tensor_inventory_sha256": b2b["tensor_inventory_sha256"],
            "packed_tekken_sha256": digest,
            "crispasr_commit": PINNED_CRISPASR_COMMIT,
            "crispasr_header_blob_sha1": PINNED_HEADER_BLOB_SHA1,
            "cpp_compiler": compiler,
            "french_corpus_cases": len(corpus),
            "rust_vs_deployed_cpp": "PASS" if test.returncode == 0 else "FAIL",
            "rust_test_exit_code": test.returncode,
            "rust_test_tail": (test.stdout + "\n" + test.stderr)[-2800:],
            "cpp_deployed_vs_unicode_header_mismatch_count": len(divergent),
            "cpp_deployed_vs_unicode_header_examples": divergent[:12],
            "original_mistral_tekken_parity": "NOT_ESTABLISHED",
            "audio_forward": "NOT_VALIDATED",
            "quantized_forward": "NOT_VALIDATED",
        }
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    print(encoded)
    if report_path:
        Path(report_path).write_text(encoded + "\n", encoding="utf-8")
    if result["rust_vs_deployed_cpp"] != "PASS":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="Full local Voxtral4b Q4_K GGUF")
    parser.add_argument("--reference-header", help="Offline CrispASR header path (must match pinned git blob SHA)")
    parser.add_argument("--report", default="voxtral4-b2c2-report.json")
    args = parser.parse_args()
    check(args.gguf, args.reference_header, args.report, Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
