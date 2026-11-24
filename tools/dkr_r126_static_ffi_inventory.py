#!/usr/bin/env python3
"""S9A: read-only producer / FFI candidate inventory.

A lexical source audit plus optional nm of existing cached objects. Not a
Rust cfg evaluator, macro expander, linker or proof of runtime reachability.
Never invoke Cargo, a device compiler, or modify any build/cache input.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess

COMPONENTS = ("mistralrs-core", "mistralrs-quant", "mistralrs-paged-attn")
CACHE = dict(zip(COMPONENTS, ("core", "quant", "paged-attn")))
PIN = "d40ac1b4d69a209d08f76c5b5bd4e267ec63b4f0"
EXTERN = re.compile(r'\bextern\s+"C(?:-unwind)?"\s*\{')
DECL = re.compile(r'\bfn\s+([A-Za-z_]\w*)\s*\(')
IDENT = re.compile(r'\b[A-Za-z_]\w*\b')
QUALIFIED = re.compile(r'\bffi\s*::\s*([A-Za-z_]\w*)')
PRODUCER = re.compile(r'\.source_(?:glob|files)\s*\(|\.exclude\s*\(|\.build_lib\s*\(|rustc-link-lib=')


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def loc(path, n):
    return f"{path}:{n}"


def extern_declarations(path, source):
    declarations = collections.defaultdict(list)
    macro_blocks = []
    for m in EXTERN.finditer(source):
        start = m.end()
        depth, end = 1, start
        while end < len(source) and depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        if depth:
            macro_blocks.append(loc(path, source.count("\n", 0, m.start()) + 1))
            continue
        block = source[start:end-1]
        if "$" in block:
            macro_blocks.append(loc(path, source.count("\n", 0, m.start()) + 1))
        for d in DECL.finditer(block):
            declarations[d.group(1)].append(loc(path, source.count("\n", 0, start+d.start()) + 1))
    return declarations, macro_blocks


def source_inventory(root, tracked, component):
    paths = [p for p in tracked if p.startswith(component + "/src/") and p.endswith(".rs")]
    cuda_root = component + ("/kernels/" if component == "mistralrs-quant" else "/src/cuda/")
    cuda = [p for p in tracked if p.startswith(cuda_root) and p.endswith(".cu")]
    data = {p: (root/p).read_text() for p in paths}
    declared = collections.defaultdict(list)
    macro_blocks = []
    for path, source in data.items():
        found, macros = extern_declarations(path, source)
        for name, sites in found.items():
            declared[name].extend(sites)
        macro_blocks.extend(macros)
    refs = collections.defaultdict(set)
    qualified = collections.defaultdict(set)
    # Record literal identifier references, including macro arguments. Do not
    # mistake a matching identifier for a proven live production call.
    for path, source in data.items():
        for n, line in enumerate(source.splitlines(), 1):
            for name in set(IDENT.findall(line)).intersection(declared):
                site = loc(path, n)
                if site not in declared[name]:
                    refs[name].add(site)
            for name in QUALIFIED.findall(line):
                qualified[name].add(loc(path, n))
    build = component + "/build.rs"
    build_sites = [dict(line=n, text=line.strip()) for n,line in enumerate(
        (root/build).read_text().splitlines(), 1) if PRODUCER.search(line)]
    symbols = {name: dict(declarations=sorted(sites),
                          lexical_references=sorted(refs[name]),
                          qualified_ffi_references=sorted(qualified[name]))
               for name, sites in sorted(declared.items())}
    return dict(cuda_translation_units=cuda, producer_sites=build_sites,
                explicit_ffi_symbols=symbols, macro_extern_blocks=sorted(set(macro_blocks)),
                qualified_ffi_names_without_explicit_declaration={
                    k:sorted(v) for k,v in sorted(qualified.items()) if k not in declared},
                build_rs_sha256=digest(root/build))


def cache_inventory(root, declared):
    objects = sorted(root.rglob("*.o"))
    errors, exports = [], collections.defaultdict(list)
    for path in objects:
        try:
            result = subprocess.run(["nm", "--defined-only", "--extern-only",
                                     "--format=posix", str(path)],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode:
                errors.append(dict(path=str(path), error=result.stderr.strip()[:400]))
                continue
            for line in result.stdout.splitlines():
                cols = line.split()
                if len(cols) >= 2 and cols[0] in declared and cols[1] in ("T", "W", "I"):
                    exports[cols[0]].append(str(path))
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(dict(path=str(path), error=str(exc)))
    return dict(root=str(root), objects=len(objects), errors=errors,
                declared_symbol_definitions=dict(sorted(exports.items())),
                cache_provenance="historical_objects_not_strict_link_evidence")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path.cwd())
    p.add_argument("--s8-evidence", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    root = a.source_root.resolve(strict=True)
    try:
        evidence = json.loads(a.s8_evidence.read_text())
        if (evidence.get("schema") != "dkr.r126.native.preflight.v1"
                or evidence.get("status") != "PASS"
                or evidence.get("target_cc") != 61 or evidence.get("candle_pin") != PIN
                or evidence.get("typed_profiles") != 216
                or evidence.get("native_families") != 25):
            raise ValueError("S8 native/typed preflight evidence required")
        head = git(root, "rev-parse", "HEAD").decode().strip()
        paths = [p for c in COMPONENTS for p in
                 (c+"/src", c+"/build.rs", c+"/Cargo.toml", c+"/kernels")]
        if git(root, "status", "--porcelain", "--untracked-files=no", "--", *paths).strip():
            raise ValueError("tracked production source changes; audit a clean revision")
        tracked = sorted(p.decode() for p in git(root, "ls-files", "-z", "--", *paths).split(b"\0") if p)
        report = {}
        for component in COMPONENTS:
            item = source_inventory(root, tracked, component)
            symbols = item["explicit_ffi_symbols"]
            direct = sum(bool(s["qualified_ffi_references"]) for s in symbols.values())
            if a.cache_root is not None:
                item["historical_cache"] = cache_inventory(
                    a.cache_root.resolve()/CACHE[component], symbols)
            report[component] = item
            print(f"DKR_R126_S9_COMPONENT component={component} "
                  f"cuda_sources={len(item['cuda_translation_units'])} "
                  f"explicit_ffi={len(symbols)} direct_qualified_refs={direct} "
                  f"macro_extern_blocks={len(item['macro_extern_blocks'])} "
                  f"producer_sites={len(item['producer_sites'])}", flush=True)
            if "historical_cache" in item:
                c = item["historical_cache"]
                print(f"DKR_R126_S9_CACHE component={component} objects={c['objects']} "
                      f"matched_definitions={len(c['declared_symbol_definitions'])} "
                      f"nm_errors={len(c['errors'])} strict_link_evidence=false", flush=True)
        if git(root, "rev-parse", "HEAD").decode().strip() != head or git(
                root, "status", "--porcelain", "--untracked-files=no", "--", *paths).strip():
            raise ValueError("source changed during inventory")
        result = dict(schema="dkr.r126.static-ffi.inventory.v1", status="COMPLETE",
                      source_head=head, s8_evidence_sha256=digest(a.s8_evidence),
                      components=report, analysis="lexical_candidates_not_cfg_or_macro_expanded",
                      compilation=False, static_producer_elimination_qualified=False,
                      static_fallback_elimination_qualified=False, full_model_qualified=False)
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"DKR_R126_S9_INVENTORY=COMPLETE components=3 compilation=false closure_qualified=false report={a.json_out}")
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"DKR_R126_S9_INVENTORY=BLOCKED reason={exc}") from exc


if __name__ == "__main__":
    main()
