#!/usr/bin/env python3
import argparse
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

CUDA_ERROR_STUB_LIBRARY = 34


def run(stub, command):
    driver = ctypes.CDLL(str(stub.resolve()))
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    if driver.cuInit(0) != CUDA_ERROR_STUB_LIBRARY:
        raise RuntimeError("hosted CPU execution requires the CUDA link stub")
    with tempfile.TemporaryDirectory(prefix="mistral-cpu-driver-") as directory:
        (Path(directory) / "libcuda.so.1").symlink_to(stub.resolve())
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = directory + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        print("CPU_CUDA_LINK_STUB=ENABLED; GPU execution is unavailable", flush=True)
        return subprocess.run(command, env=env).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", type=Path, default=Path("/usr/local/cuda/lib64/stubs/libcuda.so"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a CPU test command is required")
    return run(args.stub, command)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        print(f"CPU_RUNTIME_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
