#!/usr/bin/env python3
"""Cheap, GPU-free guard of the Candle R1.24 -> Mistral typed DKR contract.

Does not compile Rust, resolve git dependencies, or attest GPU correctness.
"""
from __future__ import annotations

import pathlib
import re
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CANDLE = (
    "candle-core",
    "candle-nn",
    "candle-flash-attn-v3",
    "candle-metal-kernels",
    "dkr-kernel-registry",
)


def fail(reason: str) -> None:
    raise AssertionError(reason)


def main() -> None:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    deps = cargo["workspace"]["dependencies"]
    revs = {}
    for name in CANDLE:
        row = deps[name]
        if "github.com/haricot/candle.git" not in row["git"]:
            fail(f"{name} has a different Candle source")
        revs[name] = row["rev"]
    if len(set(revs.values())) != 1:
        fail(f"Candle packages are pinned to different revisions: {revs}")
    revision = next(iter(revs.values()))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        fail("Candle dependency must use an immutable 40-character commit")

    lock = (ROOT / "Cargo.lock").read_text()
    for name in CANDLE:
        match = re.search(
            r"(?ms)^\[\[package\]\]\nname = \"" + re.escape(name)
            + r"\"\n.*?(?=^\[\[package\]\]|\Z)",
            lock,
        )
        if match is None or (
            f"git+https://github.com/haricot/candle.git?rev={revision}#{revision}"
            not in match.group()
        ):
            fail(f"Cargo.lock does not match {name} at {revision}")
    if "95f9087c5210908c9ff7356bf35e914cddac178a" in lock:
        fail("stale pre-R1.24 Candle pin in Cargo.lock")

    runtime_toml = tomllib.loads(
        (ROOT / "mistralrs-kernel-runtime/Cargo.toml").read_text()
    )
    shared = runtime_toml["features"]["shared-registry"]
    if "candle-core/cuda-dkr-typed-image" not in shared:
        fail("typed-image transport is not enabled by shared-registry")
    if "candle-core/cuda-dkr-s31" in shared:
        fail("typed image transport must not force native S31 legacy compilation")
    core_toml = tomllib.loads((ROOT / "mistralrs-core/Cargo.toml").read_text())
    features = core_toml["features"]
    if "mistralrs-kernel-runtime/shared-registry" not in features["dkr-sm61-experimental"]:
        fail("SM61 DKR does not enable typed transport")
    if "candle-core/cuda-dkr-qualified-default" not in features["dkr-sm61-cubin-qualified"]:
        fail("qualified CUBIN feature does not enable Candle native default")
    pilot = features["dkr-r125-typed-real-pilot"]
    for name in ("dkr-moe", "dkr-indexed-copy", "dkr-paged-decode", "mistralrs-kernel-runtime/shared-registry"):
        if name not in pilot:
            fail(f"real typed pilot missing {name}")
    if "candle-core/cuda-legacy-bf16" not in pilot:
        fail("SM61 typed pilot requires Candle BF16 tensor fill/cast compatibility")
    if any(flag.startswith("cuda-legacy") or "cuda-dkr-s31" in flag
           or "mistralrs-quant/cuda-legacy" in flag
           or "mistralrs-paged-attn/cuda-legacy" in flag for flag in pilot):
        fail("typed pilot may enable only Candle's generic BF16 compatibility, not native S31 or Mistral static legacy families")

    quant_pilot = features["dkr-r125-quant-typed-pilot"]
    for name in ("dkr-quant-gemv", "dkr-quant-gemm",
                 "mistralrs-kernel-runtime/shared-registry",
                 "candle-core/cuda-legacy-bf16", "candle-core/cuda-legacy-fp8"):
        if name not in quant_pilot:
            fail(f"quant typed pilot missing {name}")
    if any(flag.startswith("cuda-legacy") or "cuda-dkr-s31" in flag
           or "mistralrs-quant/cuda-legacy" in flag
           or "mistralrs-paged-attn/cuda-legacy" in flag
           for flag in quant_pilot):
        fail("quant pilot must not force Mistral static legacy or native S31")
    if not (ROOT / "tools/dkr_r125_cudaforge_cache_audit.py").exists():
        fail("read-only Cudaforge cache audit missing")
    if not (ROOT / "tools/dkr_r125_quant_manifest_preflight.py").exists():
        fail("quant manifest preflight missing")

    loader = (ROOT / "mistralrs-kernel-runtime/src/universal_cuda.rs").read_text()
    if "get_or_load_dkr_typed_image_func" not in loader:
        fail("Mistral bypasses the Candle typed-image transport")
    if any(token in loader for token in ("CUBIN_MODULES", "load_cubin(", "ImageMode::")):
        fail("duplicate Mistral CUDA image policy/cache still present")
    if (ROOT / "mistralrs-kernel-runtime/src/image_policy.rs").exists():
        fail("Mistral independent CUBIN image policy still present")
    typed = (ROOT / "mistralrs-kernel-runtime/src/universal.rs").read_text()
    if "cubin_sha256" not in typed or "cubin_artifact" not in typed:
        fail("typed manifests do not attest CUBIN images")

    print(
        "DKR_R125_CANDLE_INTEGRATION_AUDIT=PASS "
        f"revision={revision} typed_transport=candle "
        "native_cubin=opt_in typed_real_pilot=3 asd_dispatch=unchanged gpu_qualified=false"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, KeyError, tomllib.TOMLDecodeError) as error:
        print(f"DKR_R125_CANDLE_INTEGRATION_AUDIT=FAIL reason={error}", file=sys.stderr)
        sys.exit(1)
