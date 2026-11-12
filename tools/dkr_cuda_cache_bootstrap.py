#!/usr/bin/env python3
"""Bootstrap cudaforge v0.1.6 CUDA object caches without running nvcc.

Copies verified objects and rewrites the JSON cache keys from existing
Cargo OUT_DIRs to the stable MISTRALRS_CUDA_BUILD_ROOT layout already used
by mistralrs-quant, mistralrs-core and mistralrs-paged-attn.

Dry-run by default. Never changes CUDA_COMPUTE_CAP, Cargo.lock or target/
build outputs. Run while no Cargo/NVCC build is active.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PACKAGES = (
    ("mistralrs-quant", "quant", "kernels", "kernels"),
    ("mistralrs-core", "core", "", "kernels"),
    ("mistralrs-paged-attn", "paged-attn", "", "kernels"),
)
CACHE_FILE = ".cudaforge_cache.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_for_entry(key, entry, crate_dir):
    """Verify the entry without trusting a serialized source/object path."""
    obj_name = entry.get("object_path")
    if not isinstance(obj_name, str) or not key.endswith(":" + obj_name):
        return None
    source_name = key[: -len(obj_name) - 1]
    path = Path(source_name)
    if not path.is_absolute():
        path = crate_dir / path
    if not path.is_file() or path.suffix != ".cu":
        return None
    if sha256(path) != entry.get("content_hash"):
        return None
    return source_name


def valid_entries(cache_file, crate_dir, arch):
    try:
        cache = json.loads(cache_file.read_text())
    except (OSError, ValueError):
        return []
    if cache.get("version") != 1 or not isinstance(cache.get("entries"), dict):
        return []
    results = []
    for key, entry in cache["entries"].items():
        if not isinstance(entry, dict) or entry.get("gpu_arch") != arch:
            continue
        src_name = source_for_entry(key, entry, crate_dir)
        if src_name is None:
            continue
        obj = Path(entry["object_path"])
        # Only adopt compiler objects from this exact Cargo cache directory.
        if not obj.is_absolute() or obj.parent.resolve() != cache_file.parent.resolve():
            continue
        if not obj.is_file() or obj.suffix != ".o" or obj.stat().st_size == 0:
            continue
        if not all(isinstance(entry.get(key), str)
                   for key in ("watch_hash", "args_hash", "content_hash")):
            continue
        results.append((src_name, entry, obj))
    return results


def choose_candidate(target, package, subdir, crate_dir, arch):
    glob = f"{package}-*/out"
    found = []
    for out in (target / "debug" / "build").glob(glob):
        candidate = out / subdir / CACHE_FILE if subdir else out / CACHE_FILE
        if candidate.is_file():
            entries = valid_entries(candidate, crate_dir, arch)
            if entries:
                found.append((len(entries), candidate.stat().st_mtime, candidate, entries))
    return max(found, default=None, key=lambda item: (item[0], item[1]))


def atomic_copy(source, destination):
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as tmp:
        tmp_name = Path(tmp.name)
    try:
        shutil.copy2(source, tmp_name)
        os.replace(tmp_name, destination)
    finally:
        tmp_name.unlink(missing_ok=True)


def atomic_json(data, destination):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8",
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as tmp:
        tmp_name = Path(tmp.name)
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
    try:
        os.replace(tmp_name, destination)
    finally:
        tmp_name.unlink(missing_ok=True)


def run(repo, target, root, arch, apply):
    if target.resolve() == root.resolve():
        raise ValueError("cache root must not equal Cargo target directory")
    print(f"DKR_CACHE_BOOTSTRAP={'APPLY' if apply else 'DRY_RUN'} arch={arch}")
    print(f"DKR_CACHE_ROOT={root}")
    total = 0
    for package, section, old_subdir, new_subdir in PACKAGES:
        crate_dir = repo / package
        selected = choose_candidate(target, package, old_subdir, crate_dir, arch)
        if selected is None:
            print(f"{package}: NO_VALID_CARGO_OBJECT_CACHE (no action)")
            continue

        _, _, cache_file, records = selected
        dest_dir = root / section / new_subdir
        dest_cache = dest_dir / CACHE_FILE
        if dest_cache.exists():
            print(f"{package}: TARGET_CACHE_EXISTS ({dest_cache}); no overwrite")
            continue
        print(f"{package}: SEEDABLE={len(records)} from {cache_file}")
        total += len(records)
        if not apply:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        # The source list is unique for a given cache. Protect against
        # malformed duplicate object names and never follow an existing
        # destination symlink.
        new_entries = {}
        for src_name, entry, old_obj in records:
            new_obj = dest_dir / old_obj.name
            if new_obj.exists() or new_obj.is_symlink():
                if new_obj.is_symlink() or new_obj.stat().st_size != old_obj.stat().st_size:
                    raise ValueError(f"refusing conflicting destination object {new_obj}")
            else:
                atomic_copy(old_obj, new_obj)
            new_value = dict(entry)
            new_value["object_path"] = str(new_obj)
            key = f"{src_name}:{new_obj}"
            if key in new_entries:
                raise ValueError(f"duplicate cached entry {src_name}")
            new_entries[key] = new_value
        atomic_json({"entries": new_entries, "version": 1}, dest_cache)
        print(f"{package}: SEEDED={len(new_entries)}")

    print(f"DKR_CACHE_BOOTSTRAP_CANDIDATES={total}")
    if not apply:
        print("Dry-run only. Add --apply once to copy verified objects.")
    else:
        print("No CUDA kernels compiled; builder will still validate flags/headers/arch.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--target-dir", type=Path, default=Path("target"))
    parser.add_argument("--cache-root", type=Path,
                        default=Path("target/dkr-cuda-cache/sm61"))
    parser.add_argument("--arch", default="sm_61")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    target = (repo / args.target_dir).resolve()
    root = (repo / args.cache_root).resolve()
    if not (repo / "mistralrs-quant" / "build.rs").is_file():
        parser.error("run at mistral.rs workspace root or pass --repo")
    if not target.is_dir():
        parser.error(f"Cargo target directory does not exist: {target}")
    try:
        return run(repo, target, root, args.arch, args.apply)
    except (OSError, ValueError) as error:
        print(f"DKR_CACHE_BOOTSTRAP_ERROR={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
