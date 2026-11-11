#!/usr/bin/env python3
"""Build every external CUDA package for one requested architecture."""

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("kernel-packages/build/cuda"))
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    builders = sorted(PACKAGE_ROOT.glob("*/build.py"))
    if not builders:
        parser.error("no external CUDA package builders found")
    for builder in builders:
        command = [
            sys.executable,
            str(builder),
            "--cc",
            str(args.cc),
            "--output",
            str(args.output),
            "--nvcc",
            args.nvcc,
        ]
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    print(f"DKR_CUDA_PACKAGES=PASS target_cc={args.cc} builders={len(builders)}")


if __name__ == "__main__":
    main()
