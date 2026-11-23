#!/usr/bin/env python3
"""Read-only Cudaforge 0.1.6 cache audit, before another expensive SM61 build.

Replays BuildCache::needs_rebuild() comparisons possible without invoking nvcc:
object presence, source SHA256, watched-header SHA256 and cached architecture /
argument fingerprints. It cannot reconstruct *next* Cargo feature-derived args
without executing the build script. See source: guoqingbao/cudaforge src/hash.rs.
"""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


COMPONENTS = {
    "quant": ("mistralrs-quant", "kernels"),
    "paged-attn": ("mistralrs-paged-attn", "src/cuda"),
    "core": ("mistralrs-core", "src/cuda"),
}
HEADER_SUFFIXES = {".h", ".cuh", ".hpp"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def watched_hash(path, crate_root):
    """Match cudaforge::hash::hash_paths([PathBuf]) for a single directory.

    Builder supplies relative watch paths with crate cwd, so the serialized
    path in hash_paths is relative (not the absolute path used to read bytes).
    """
    h = hashlib.sha256()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(p for p in path.rglob("*")
                            if p.is_file() and p.suffix in HEADER_SUFFIXES)
    else:
        candidates = []
    for candidate in candidates:
        h.update(str(candidate.relative_to(crate_root)).encode())
        h.update(b":")
        h.update(digest(candidate).encode())
        h.update(b"\0")
    return h.hexdigest()


def resolve_path(path, crate_root):
    return path if path.is_absolute() else crate_root / path


def audit(root, source_checkout, component):
    crate, watched = COMPONENTS[component]
    cache_path = root / component / "kernels" / ".cudaforge_cache.json"
    source = source_checkout / crate
    result = {
        "component": component, "cache": str(cache_path),
        "source_root": str(source), "watch_path": watched,
        "watch_hash_current": None, "watched_header_count": 0,
        "entries": 0, "cache_exists": cache_path.is_file(),
        "reasons": {}, "architectures": [], "args_fingerprints": [],
        "items": {}, "examples": {},
    }
    if not cache_path.is_file():
        result["reasons"] = {"missing_cache": 1}
        return result
    try:
        raw = json.loads(cache_path.read_text())
        entries = raw["entries"]
        if not isinstance(entries, dict):
            raise ValueError("'entries' must be a JSON object")
    except (ValueError, OSError, KeyError) as error:
        result["reasons"] = {"unreadable_cache": 1}
        result["error"] = str(error)
        return result

    current_watch = watched_hash(source / watched, source) if (source / watched).exists() else None
    result["watch_hash_current"] = current_watch
    if (source / watched).is_dir():
        result["watched_header_count"] = sum(
            p.is_file() and p.suffix in HEADER_SUFFIXES
            for p in (source / watched).rglob("*")
        )
    reasons, arches, args = Counter(), set(), set()
    examples = {}
    for key, entry in sorted(entries.items()):
        obj_name = entry.get("object_path", "")
        suffix = ":" + obj_name
        src_name = key[:-len(suffix)] if obj_name and key.endswith(suffix) else key
        src_path = resolve_path(Path(src_name), source)
        obj_path = resolve_path(Path(obj_name), source) if obj_name else None
        mismatches = []
        if obj_path is None or not obj_path.is_file():
            mismatches.append("object_missing")
        if not src_path.is_file():
            mismatches.append("source_missing")
            present_hash = None
        else:
            present_hash = digest(src_path)
            if present_hash != entry.get("content_hash"):
                mismatches.append("source_changed")
        if current_watch is None:
            mismatches.append("watch_path_missing")
        elif entry.get("watch_hash") != current_watch:
            mismatches.append("watch_hash_changed")
        if not obj_name or not key.endswith(suffix):
            mismatches.append("invalid_composite_key")
        if not mismatches:
            reasons["ready_for_cached_arguments"] += 1
        for cause in mismatches:
            reasons[cause] += 1
            examples.setdefault(cause, [])
            if len(examples[cause]) < 3:
                examples[cause].append(src_name)
        arches.add(entry.get("gpu_arch", "unknown"))
        args.add(entry.get("args_hash", "unknown"))
        result["items"][src_name] = {
            "object": obj_name,
            "object_exists": obj_path is not None and obj_path.is_file(),
            "source_hash_cached": entry.get("content_hash"),
            "source_hash_current": present_hash,
            "watch_hash_cached": entry.get("watch_hash"),
            "args_hash_cached": entry.get("args_hash"),
            "gpu_arch": entry.get("gpu_arch"),
        }
    result["entries"] = len(entries)
    result["reasons"] = dict(sorted(reasons.items()))
    result["examples"] = examples
    result["architectures"] = sorted(arches)
    result["args_fingerprints"] = sorted(args)
    return result


def compare(a, b):
    changed = {}
    for name in COMPONENTS:
        left, right = a["components"].get(name, {}), b["components"].get(name, {})
        fields = ["watch_hash_current", "architectures", "args_fingerprints"]
        differences = {field: {"before": left.get(field), "after": right.get(field)}
                       for field in fields if left.get(field) != right.get(field)}
        li, ri = left.get("items", {}), right.get("items", {})
        moved = []
        for src in sorted(set(li) | set(ri)):
            x, y = li.get(src), ri.get(src)
            if x != y:
                moved.append(src)
        if moved:
            differences["changed_sources_or_objects"] = moved[:12]
            differences["changed_source_count"] = len(moved)
        if differences:
            changed[name] = differences
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=(os.environ.get("MISTRALRS_CUDA_BUILD_ROOT") or None))
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--snapshot", type=Path,
                        help="save immutable read-only evidence to a JSON file (no builds)")
    parser.add_argument("--compare", type=Path,
                        help="compare current audit against previous JSON snapshot")
    parser.add_argument("--require-clean", action="store_true",
                        help="block if cached objects, source or watched headers differ")
    args = parser.parse_args()
    if args.cache_root is None or str(args.cache_root) in ("", "."):
        parser.error("supply a NONEMPTY ABSOLUTE --cache-root or MISTRALRS_CUDA_BUILD_ROOT; do not pass an unset shell variable")
    if not args.cache_root.is_absolute():
        parser.error("--cache-root must be an absolute path to the persistent Cudaforge cache")
    root = args.cache_root.resolve()
    if not root.is_dir() or not any((root / name / "kernels" / ".cudaforge_cache.json").is_file() for name in COMPONENTS):
        parser.error(f"invalid Cudaforge cache root: {root} (no component cache indexes found)")
    checkout = args.source_root.resolve()
    data = {
        "schema": "dkr.cudaforge.cache.audit.v1",
        "cache_root": str(root), "source_checkout": str(checkout),
        "components": {name: audit(root, checkout, name) for name in COMPONENTS},
    }
    clean = True
    for name, result in data["components"].items():
        reasons = result["reasons"]
        ready = reasons.get("ready_for_cached_arguments", 0)
        other = {key: value for key, value in reasons.items()
                 if key != "ready_for_cached_arguments"}
        if other or not result["entries"]:
            clean = False
        print("DKR_CUDAFORGE_CACHE_AUDIT "
              f"component={name} entries={result['entries']} ready={ready} "
              f"headers={result['watched_header_count']} "
              f"arch={','.join(result['architectures']) or 'unknown'} "
              f"fingerprints={len(result['args_fingerprints'])} "
              f"reasons={json.dumps(other, sort_keys=True)}")
        for cause, examples in result["examples"].items():
            print(f"  {cause}: {', '.join(examples)}")
    if args.compare:
        old = json.loads(args.compare.read_text())
        diffs = compare(old, data)
        print("DKR_CUDAFORGE_CACHE_COMPARE=" +
              json.dumps(diffs, ensure_ascii=False, sort_keys=True))
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        # Do not silently overwrite an earlier snapshot.
        with args.snapshot.open("x") as out:
            json.dump(data, out, indent=2, sort_keys=True)
            out.write("\n")
        print(f"DKR_CUDAFORGE_CACHE_SNAPSHOT={args.snapshot}")
    print(f"DKR_CUDAFORGE_CACHE_STATUS={'CLEAN_CACHED_STATE' if clean else 'REVIEW'}")
    print("NOTE: cached arguments only; next Cargo feature set or nvcc/toolchain "
          "changes can still invalidate the cache, even on CLEAN_CACHED_STATE.")
    if args.require_clean and not clean:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
