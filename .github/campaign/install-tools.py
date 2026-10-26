#!/usr/bin/env python3
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
assets = [
    ("nextest", "cargo-nextest", f'https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-{config["nextest_version"]}/cargo-nextest-{config["nextest_version"]}-x86_64-unknown-linux-gnu.tar.gz'),
    ("sccache", "sccache", f'https://github.com/mozilla/sccache/releases/download/v{config["sccache_version"]}/sccache-v{config["sccache_version"]}-x86_64-unknown-linux-musl.tar.gz'),
]
for key, binary, url in assets:
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != config[f"{key}_sha256"]:
        raise RuntimeError(f"download digest mismatch: {binary}")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        matches = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == binary]
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous binary: {binary}")
        (destination / binary).write_bytes(archive.extractfile(matches[0]).read())
        (destination / binary).chmod(0o755)
