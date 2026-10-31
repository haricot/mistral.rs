#!/usr/bin/env python3
"""Verify the one-and-only optimized Extended binary before reuse or publication."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = "haricot-extended-release-r1"
PORTABLE_RUSTFLAGS = "-C target-cpu=x86-64"
SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected(plan):
    if plan.get("target") != "extended" or not SHA.fullmatch(plan.get("candidate", "")):
        raise ValueError("Extended candidate must be pinned to a commit")
    matrix = [n for n in plan["matrix"] if n["component"] == "extended"]
    if len(matrix) != 1 or matrix[0]["candidate"] != plan["candidate"]:
        raise ValueError("Extended matrix candidate differs from the plan")
    image = plan["image"]
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("compiler image is not immutable")
    return {
        "schema": SCHEMA,
        "candidate": plan["candidate"],
        "recipe": plan["recipe"],
        "image": image,
        "target": "sm_61",
        "toolchain": plan["config"]["toolchain"],
        "profile": "release",
        "rustflags": PORTABLE_RUSTFLAGS,
        "features": sorted(set(plan["config"]["features"] + ["code-execution"])),
        "lock_sha256": matrix[0]["lock"],
    }


def inspect(plan_file, directory):
    plan = json.loads(Path(plan_file).read_text())
    directory = Path(directory)
    binary = directory / "mistralrs"
    manifest = directory / "release.json"
    if binary.is_symlink() or manifest.is_symlink():
        raise ValueError("release artifact must not contain symlinks")
    if not binary.is_file() or not manifest.is_file():
        raise ValueError("release binary or manifest is missing")
    if not os.access(binary, os.X_OK):
        raise ValueError("release binary is not executable")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("release executable is not ELF")
    details = json.loads(manifest.read_text())
    for key, value in expected(plan).items():
        if details.get(key) != value:
            raise ValueError(f"release identity mismatch: {key}")
    if not DIGEST.fullmatch(details.get("binary_sha256", "")):
        raise ValueError("invalid binary SHA-256")
    if hash_file(binary) != details["binary_sha256"]:
        raise ValueError("release binary SHA-256 mismatch")
    if details.get("binary_bytes") != binary.stat().st_size:
        raise ValueError("release binary size mismatch")
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("record", "verify"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        plan = json.loads(args.plan.read_text())
        binary = args.dir / "mistralrs"
        if not binary.is_file():
            raise ValueError("missing release binary")
        details = {**expected(plan), "binary_sha256": hash_file(binary),
                   "binary_bytes": binary.stat().st_size}
        (args.dir / "release.json").write_text(json.dumps(details, sort_keys=True, indent=2) + "\n")
    manifest = inspect(args.plan, args.dir)
    print(f"EXTENDED_RELEASE_R1=PASS candidate={manifest['candidate']} binary_sha256={manifest['binary_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        raise SystemExit("EXTENDED_RELEASE_R1_REJECT=" + str(error))
