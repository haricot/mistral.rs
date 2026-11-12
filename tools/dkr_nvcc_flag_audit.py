#!/usr/bin/env python3
"""Read-only DKR NVCC feature flag audit; never runs Cargo or nvcc.

In cudaforge v0.1.6 the compiler arguments hash is builder-wide, not
per source file. Every new builder-wide -DMISTRALRS_DKR_* flag can
invalidate all cached CUDA objects for that crate.
"""

import argparse
import re
import sys
from pathlib import Path

BUILDS = {
    "mistralrs-core": ("src/cuda/*.cu", "src/cuda", {
        "MISTRALRS_DKR_ADD_RMS_NORM",
        "MISTRALRS_DKR_INPUT_PAD",
        "MISTRALRS_DKR_RMS_VARIANTS",
        "MISTRALRS_DKR_CAUSAL_MASK",
    }),
    "mistralrs-quant": ("kernels/*/*.cu", "kernels", {
        "MISTRALRS_DKR_REDUCE_NORM",
    }),
    "mistralrs-paged-attn": ("src/cuda/*.cu", "src/cuda", set()),
}
DEFINE = re.compile(r'\.arg\(\s*"-D(MISTRALRS_DKR_[A-Z0-9_]+)(?:=[^"]*)?"\s*\)')
HEADER_SUFFIXES = {".h", ".cuh", ".hpp"}


def analyze(repo):
    total = 0
    unexpected = {}
    for crate, (glob, header_root, approved) in BUILDS.items():
        root = repo / crate
        content = (root / "build.rs").read_text()
        macros = set(DEFINE.findall(content))
        bad = macros - approved
        missing = approved - macros
        if bad or missing:
            unexpected[crate] = {"added": sorted(bad), "missing": sorted(missing)}
        sources = sorted(root.glob(glob))
        headers = sorted(p for p in (root / header_root).rglob("*")
                         if p.is_file() and p.suffix in HEADER_SUFFIXES)
        print(f"DKR_NVCC_BUILDER={crate} source_files={len(sources)}")
        print(f"DKR_NVCC_GLOBAL_DKR_DEFINES={','.join(sorted(macros)) or '<none>'}")
        for macro in sorted(macros):
            pattern = re.compile(r"\b" + re.escape(macro) + r"\b")
            users = [str(p.relative_to(root)) for p in sources
                     if pattern.search(p.read_text(errors="replace"))]
            hdr_users = [str(p.relative_to(root)) for p in headers
                         if pattern.search(p.read_text(errors="replace"))]
            print(f"  {macro}: source_uses={','.join(users) or '<none>'}"
                  f" header_uses={','.join(hdr_users) or '<none>'}")
            total += 1
        if "MISTRALRS_CUDA_BUILD_ROOT" not in content:
            unexpected.setdefault(crate, {})["cache_root"] = "missing"

    print(f"DKR_NVCC_GLOBAL_DEFINE_COUNT={total}")
    if unexpected:
        for crate, changed in unexpected.items():
            print(f"DKR_NVCC_FLAG_CONTRACT_DRIFT={crate} {changed}", file=sys.stderr)
        print("DKR_NVCC_FLAG_AUDIT=REVIEW", file=sys.stderr)
        return 1
    print("DKR_NVCC_FLAG_AUDIT=PASS no new builder-wide DKR defines")
    print("Future DKR defines must be scoped to affected CUDA translation units"
          " in separate compile domains; do not invalidate all static objects.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / "mistralrs-core" / "build.rs").is_file():
        parser.error("run from mistral.rs workspace root or set --repo")
    return analyze(repo)


if __name__ == "__main__":
    sys.exit(main())
