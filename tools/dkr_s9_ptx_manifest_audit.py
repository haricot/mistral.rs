#!/usr/bin/env python3
"""S9 boundary: verify REAL cudaforge PTX symbols before compiling CUDA crates.

This audit does not invoke Cargo, NVCC, or the GPU. Unlike the R1.21
manifest fixture preflight, it rejects PTX with no compiled .entry.
Run automatically after the one approved cudaforge package build,
immediately before cargo nextest's potentially costly Rust/static build.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from dkr_r121_manifest_preflight import PACKAGES as R121_PACKAGES

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPOSITORY / "kernel-packages" / "build" / "cuda" / "sm61"
BUILDER_SOURCE = (
    REPOSITORY / "mistralrs-kernel-runtime" / "src" / "bin"
    / "dkr_package_builder.rs"
)
PACKAGE_PATTERN = re.compile(r'Package\s*\{\s*name:\s*"([^"]+)"')
PTX_TARGET = re.compile(r'(?m)^\s*\.target\s+sm_(\d+)(?=\s|,|$)')
PTX_ENTRY = re.compile(
    r'(?m)^\s*(?:(?:\.(?:visible|weak|extern))\s+)*'
    r'\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\('
)


def check_ptx(content: bytes, entry: str, target_cc: int) -> None:
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AssertionError(f"PTX is not UTF-8: {error}") from error
    targets = {int(match) for match in PTX_TARGET.findall(decoded)}
    if target_cc not in targets:
        raise AssertionError(
            f"PTX architecture mismatch: expected sm_{target_cc}, got {sorted(targets)}"
        )
    entries = set(PTX_ENTRY.findall(decoded))
    if entry not in entries:
        raise AssertionError(
            f"missing compiled .entry {entry} (PTX declares {len(entries)} entries)"
        )


def builder_packages() -> set[str]:
    text = BUILDER_SOURCE.read_text(encoding="utf-8")
    names = PACKAGE_PATTERN.findall(text)
    if len(names) != len(set(names)) or not names:
        raise AssertionError("duplicate or empty cudaforge package list")
    return set(names)


def verify(root: Path, cc: int) -> tuple[int, int, int]:
    if not root.is_dir():
        raise AssertionError(
            f"package tree absent: {root}; compile external PTX first"
        )
    expected = builder_packages()
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise AssertionError(
            f"no compiled PTX manifests under {root}"
        )
    counts = Counter()
    entries_checked = 0
    # Use content hash to avoid rescanning identical PTX modules copied
    # into distinct typed profiles; still check every manifest entry.
    parsed = {}
    root = root.resolve()
    for manifest in manifests:
        path = manifest.resolve()
        if not path.is_relative_to(root):
            raise AssertionError(f"manifest path escapes package root: {manifest}")
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise AssertionError(f"manifest has no family/profile: {relative}")
        family = relative.parts[0]
        if family not in expected:
            raise AssertionError(f"unexpected PTX family: {family}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "DKR-KERNEL-MODULE-V1" or (
            data.get("abi") != "dkr.typed.v1"
        ):
            raise AssertionError(f"{relative}: invalid schema/ABI")
        if data.get("target_cc") != cc:
            raise AssertionError(f"{relative}: target_cc is not {cc}")
        if not (data.get("min_cc", 0) <= cc <= data.get("max_cc", 0)):
            raise AssertionError(f"{relative}: CC constraints reject sm_{cc}")
        artifact = data.get("artifact")
        if not isinstance(artifact, str) or (
            Path(artifact).name != artifact or not artifact.endswith(".ptx")
        ):
            raise AssertionError(f"{relative}: invalid PTX artifact name")
        binary = (path.parent / artifact).resolve()
        if binary.parent != path.parent or not binary.is_file():
            raise AssertionError(f"{relative}: missing/outside PTX artifact")
        payload = binary.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if sha != data.get("sha256"):
            raise AssertionError(f"{relative}: PTX digest mismatch")
        entry = data.get("entry")
        if not isinstance(entry, str) or not entry:
            raise AssertionError(f"{relative}: empty CUDA entry")
        cache_key = (sha, cc)
        if cache_key not in parsed:
            text = payload.decode("utf-8")
            parsed[cache_key] = (
                {int(m) for m in PTX_TARGET.findall(text)},
                set(PTX_ENTRY.findall(text)),
            )
        targets, symbols = parsed[cache_key]
        if cc not in targets:
            raise AssertionError(f"{relative}: PTX target mismatch {sorted(targets)}")
        if entry not in symbols:
            raise AssertionError(
                f"{relative}: compiled PTX does not export {entry}"
            )
        block = data.get("block")
        if not isinstance(block, list) or (
            len(block) != 3 or not all(isinstance(x, int) and x > 0 for x in block)
        ):
            raise AssertionError(f"{relative}: invalid CUDA block geometry")
        counts[family] += 1
        entries_checked += 1

    missing = expected - counts.keys()
    if missing:
        raise AssertionError(f"missing cudaforge families: {sorted(missing)}")
    for family, contract in R121_PACKAGES.items():
        if counts[family] != contract["profiles"]:
            raise AssertionError(
                f"{family}: expected {contract['profiles']} compiled PTX profiles, "
                f"got {counts[family]}"
            )
    total_r121 = sum(counts[f] for f in R121_PACKAGES)
    return len(counts), entries_checked, total_r121


def self_test() -> None:
    fixture = (
        b".version 7.0\n.target sm_61\n.address_size 64\n"
        b".visible .entry dkr_test_f16(\n.param .u64 p) {\n ret;\n}\n"
    )
    check_ptx(fixture, "dkr_test_f16", 61)
    failures = (
        (fixture, "does_not_exist", 61),
        (fixture, "dkr_test_f16", 80),
        (b".version 7.0\n.target sm_61\n", "dkr_test_f16", 61),
    )
    for code, entry, arch in failures:
        try:
            check_ptx(code, entry, arch)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"PTX validator accepted an invalid module: {entry=} {arch=}"
            )
    print("DKR_S9_PTX_AUDIT_SELF_TEST=PASS cases=4 no_cuda_compilation")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--cc", type=int, default=61)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    families, entries, r121 = verify(a.root, a.cc)
    print(
        f"DKR_S9_PTX_MANIFEST_AUDIT=PASS target=sm_{a.cc} "
        f"families={families} manifests={entries} "
        f"r121_profiles={r121} compiled_entries_verified={entries}"
    )


if __name__ == "__main__":
    try:
        main()
    except (
        AssertionError, OSError, UnicodeDecodeError,
        ValueError, KeyError, json.JSONDecodeError,
    ) as error:
        print(f"DKR_S9_PTX_MANIFEST_AUDIT=FAIL {error}", file=sys.stderr)
        sys.exit(1)
