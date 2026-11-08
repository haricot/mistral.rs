#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDLE_REV = "9cba2e898af3d25a89257f1f2ec5acece0404dcb"
ABI = "mistral.linear.f32.v1"
BINARY = "legacy-provider-f32-poc"


def run(command, **kwargs):
    print("+ " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def source_scope():
    root = tomllib.loads((ROOT / "Cargo.toml").read_text())
    demo = tomllib.loads((HERE / "Cargo.toml").read_text())
    assert root["workspace"]["dependencies"]["candle-core"]["rev"] == CANDLE_REV
    assert demo["dependencies"]["candle-core"]["rev"] == CANDLE_REV
    assert demo["features"] == {"default": [], "cuda": ["candle-core/cuda", "mistralrs-kernel-runtime/cuda"]}
    assert not (HERE / "build.rs").exists()
    print("PROVIDER_F32_SOURCE_SCOPE=PASS", flush=True)


def cargo(*args):
    return ["cargo", *args, "--locked", "--manifest-path", HERE / "Cargo.toml"]


def packages(output, cc):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        candidate = Path(os.environ.get("CUDA_ROOT", "/opt/cuda")) / "bin/nvcc"
        if candidate.is_file():
            nvcc = str(candidate)
    if not nvcc:
        raise RuntimeError("nvcc missing: add your CUDA 12.9 bin directory to PATH")
    run([nvcc, "--version"])
    manifests = []
    for name, source, entry in [("a", "simple.cu", "linear_simple"),
                                ("b", "tiled.cu", "linear_tiled")]:
        directory = output / name
        directory.mkdir(parents=True)
        artifact = directory / "kernel.ptx"
        command = [nvcc, "--ptx", f"--gpu-architecture=compute_{cc}", "--fmad=false",
                   "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
        if os.environ.get("NVCC_CCBIN"):
            command += ["-ccbin", os.environ["NVCC_CCBIN"]]
        run(command + [HERE / "kernels" / source, "-o", artifact])
        ptx = artifact.read_text()
        assert re.search(rf"\.target\s+sm_{cc}\b", ptx), "wrong PTX target"
        assert re.search(rf"\.entry\s+{entry}\s*\(", ptx), "missing PTX entry"
        manifest = {"schema": 1, "abi": ABI, "implementation_id": "external-linear-swap",
                    "artifact": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "entry": entry, "target_cc": cc, "min_cc": cc, "max_cc": cc,
                    "dtype": "f32", "layout": "contiguous_zero_offset", "bias": False,
                    "block": [16, 16, 1], "max_shape": [4096, 4096, 4096]}
        path = directory / "module.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(path)
    assert json.loads(manifests[0].read_text())["sha256"] != json.loads(manifests[1].read_text())["sha256"]
    print(f"PROVIDER_F32_PTX_SM{cc}=PASS", flush=True)
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["cpu", "cuda-compile", "gpu"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_scope()
    if args.mode == "cpu":
        run(["cargo", "fmt", "--manifest-path", HERE / "Cargo.toml", "--", "--check"])
        run(cargo("check"))
        run(cargo("test"))
        run(["cargo", "test", "--locked", "--manifest-path", ROOT / "Cargo.toml", "-p", "mistralrs-kernel-runtime"])
        print("PROVIDER_F32_CPU=PASS", flush=True)
        return
    cc = os.environ.get("CUDA_COMPUTE_CAP", "61")
    if not cc.isdecimal() or not 50 <= int(cc) <= 999:
        raise RuntimeError("CUDA_COMPUTE_CAP must be numeric, for example 61")
    output = (args.output or Path(tempfile.mkdtemp(prefix="provider-f32-"))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(output / "target"))).resolve()
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    env["CUDA_COMPUTE_CAP"] = cc
    env.setdefault("CARGO_BUILD_JOBS", "2")
    env.setdefault("CARGO_PROFILE_DEV_OPT_LEVEL", "1")
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    run(cargo("build", "--features", "cuda"), env=env)
    binary = target / "debug" / BINARY
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    print(f"BINARY_SHA256_BEFORE={before}", flush=True)
    # Artifacts are built after the Rust executable and are never embedded in it.
    a, b = packages(output / "packages", int(cc))
    if args.mode == "gpu":
        for manifest in [a, b]:
            run([binary, "validate", manifest])
        evidence = output / "evidence.json"
        with evidence.open("w") as stream:
            run([binary, "swap", a, b], stdout=stream)
        report = json.loads(evidence.read_text())
        assert report["status"] == "PASS" and report["binary_sha256"] == before
        assert len(report["passes"]) == 3 and report["invalid_input_rejections"] == 6
        print(evidence.read_text(), flush=True)
        print(f"EVIDENCE={evidence}", flush=True)
    after = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert before == after, "executable changed while replacing external artifacts"
    (output / "build-evidence.json").write_text(json.dumps({
        "mode": args.mode, "binary_sha256": before, "candle_revision": CANDLE_REV,
        "target_cc": int(cc), "external_ptx_compiled_after_binary": True,
        "gpu_executed": args.mode == "gpu"}, indent=2) + "\n")
    print(f"BINARY_SHA256_AFTER={after}", flush=True)
    print(f"PROVIDER_F32_{args.mode.upper().replace('-', '_')}=PASS", flush=True)
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
