#!/usr/bin/env python3
"""Consolidate the four already qualified SM61 quant families without compilation.

Verify original recorded numerical evidence, exact image hashes and profile
contracts. Copy only manifests and their PTX/CUBIN payloads, never overwrite
source families. This is filesystem staging, NOT embedded-runtime qualification.
"""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from dkr_r125_quant_evidence import verify as verify_quant
from dkr_r125_fp8_forward_evidence import verify_forward
from dkr_r126_nvfp4_combined import IMAGES as NV_IMAGES, verify as verify_nv
from dkr_r126_nvfp4_preflight import digest

IMAGES = {
    "quant-gemv": (
        "f3a89fad0567164865b46b6fb971df288ce47e4b24804de6af4951d95be44288",
        "1fb3e5e76048c22c9c04a6662258278714b8a7a3901f303242edf8b984abb9b1",
    ),
    "quant-gemm": (
        "3017abe46a75675321e5ef5beb8092de3c582999cbc26ebb491025fdfee3affb",
        "49dad02c21b2d5df48b28f0df5e0d83a713d1da15c9c8ec50f5375bd90e77497",
    ),
    **NV_IMAGES,
}


def profiles(family):
    op = family.split("-")[1]
    geometry = "b128" if op == "gemv" else "t16"
    block = [128, 1, 1] if op == "gemv" else [16, 16, 1]
    for dtype in ("f16", "bf16"):
        formats = ("nvfp4",) if family.startswith("nvfp4") else ("fp8", "mxfp4")
        for fmt in formats:
            for mode in (("plain",) if fmt == "fp8" else ("plain", "bias")):
                directory = Path(family)
                if fmt != "nvfp4":
                    directory /= fmt
                directory = directory / dtype / mode
                operation = (f"quant.fp8.blockwise.{op}.{geometry}.v1" if fmt == "fp8"
                             else f"quant.{fmt}.{op}.{mode}.{geometry}.v1")
                entry = (f"dkr_fp8_block_{op}_{dtype}" if fmt == "fp8"
                         else f"dkr_{fmt}_{op}_{dtype}_{mode}")
                yield directory, operation, entry, block, dtype, fmt, mode


def inspect(root, family):
    cubin_hash, ptx_hash = IMAGES[family]
    files = {}
    expected_paths = set()
    for directory, operation, entry, block, dtype, fmt, mode in profiles(family):
        rel = directory / "manifest.json"
        path = root / rel
        expected_paths.add(rel)
        data = json.loads(path.read_text())
        required = dict(schema="DKR-KERNEL-MODULE-V1", abi="dkr.typed.v1",
                        operation=operation, entry=entry, target_cc=61,
                        block=block, sha256=ptx_hash, cubin_sha256=cubin_hash)
        for name, expected in required.items():
            if data.get(name) != expected:
                raise ValueError(f"{path}: unexpected {name}: {data.get(name)!r}")
        args = data.get("arguments")
        if not isinstance(args, list) or not args or args[0].get("dtype") != dtype:
            raise ValueError(f"{path}: activation dtype does not match profile")
        files[str(rel)] = digest(path)
        for field, expected in (("artifact", ptx_hash), ("cubin_artifact", cubin_hash)):
            name = data.get(field)
            if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
                raise ValueError(f"{path}: unsafe artifact path")
            artifact = directory / name
            source = root / artifact
            if digest(source) != expected:
                raise ValueError(f"{source}: image digest mismatch")
            if field == "artifact":
                code = source.read_text()
                if not re.search(r"\.target\s+sm_61(?:\s|,|$)", code):
                    raise ValueError(f"{source}: wrong PTX target")
                if not re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\(", code):
                    raise ValueError(f"{source}: expected entry missing")
            else:
                with source.open("rb") as f:
                    header = f.read(64)
                if (len(header) != 64 or header[:6] != b"\x7fELF\x02\x01"
                        or int.from_bytes(header[18:20], "little") != 190
                        or int.from_bytes(header[48:52], "little") & 255 != 61):
                    raise ValueError(f"{source}: invalid SM61 CUDA ELF64 image")
            files[str(artifact)] = expected
    actual = {p.relative_to(root) for p in (root / family).rglob("manifest.json")}
    if actual != expected_paths:
        raise ValueError(f"{family}: missing or additional manifest profiles")
    return files


def collect(quant, nv):
    return {
        family: inspect(nv if family.startswith("nvfp4") else quant, family)
        for family in IMAGES
    }


def stage(root, quant, nv, expected):
    if root.exists():
        if collect(root, root) != expected:
            raise ValueError("existing consolidated catalogue differs; refusing overwrite")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".quant-catalog-", dir=root.parent) as temp:
            staging = Path(temp) / "sm61"
            staging.mkdir()
            for family, files in expected.items():
                source = nv if family.startswith("nvfp4") else quant
                for name in files:
                    dest = staging / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source / name, dest)
            if collect(staging, staging) != expected:
                raise ValueError("staged manifest/image mismatch")
            staging.rename(root)
    if len(list(root.rglob("manifest.json"))) != 20:
        raise ValueError("consolidated catalogue must contain exactly 20 manifests")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quant-root", type=Path, required=True)
    p.add_argument("--nvfp4-root", type=Path, required=True)
    p.add_argument("--quant-gemv-log", type=Path, required=True)
    p.add_argument("--quant-gemm-log", type=Path, required=True)
    p.add_argument("--fp8-forward-log", type=Path, required=True)
    p.add_argument("--nvfp4-combined-log", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    try:
        out = a.output.resolve()
        for src in (a.quant_root.resolve(), a.nvfp4_root.resolve()):
            if out == src or out.is_relative_to(src) or src.is_relative_to(out):
                raise ValueError("output must be separate from source catalogues")
        if a.json_out.exists():
            raise ValueError("report already exists; use a new report path")
        if a.json_out.resolve().is_relative_to(out):
            raise ValueError("report must be outside the embedded-input directory")
        expected = collect(a.quant_root, a.nvfp4_root)
        verify_quant(a.quant_root, a.quant_gemv_log, "gemv")
        verify_quant(a.quant_root, a.quant_gemm_log, "gemm")
        verify_forward(a.quant_root, a.fp8_forward_log)
        verify_nv(a.nvfp4_root, a.nvfp4_combined_log)
        stage(out, a.quant_root, a.nvfp4_root, expected)
        logs = {name: digest(path) for name, path in (
            ("quant_gemv", a.quant_gemv_log), ("quant_gemm", a.quant_gemm_log),
            ("fp8_forward", a.fp8_forward_log), ("nvfp4_combined", a.nvfp4_combined_log)
        )}
        report = dict(schema="dkr.r126.quant.catalog.evidence.v1", status="PASS",
                      target_cc=61, profiles=20, cubin_images=4,
                      catalogue=str(out), image_pairs=IMAGES, files=expected,
                      source_log_sha256=logs, prior_numeric_evidence_reverified=True,
                      combined_four_family_gpu_run=False, embedded_runtime_qualified=False,
                      full_model_qualified=False, performance_qualified=False)
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"DKR_R126_QUANT_CATALOG=FAIL reason={exc}")
        raise SystemExit(1)
    print("DKR_R126_QUANT_CATALOG=PASS profiles=20 cubin_images=4 "
          "prior_evidence=true compilation=false embedded_qualified=false full_model=false")
    print(f"DKR_R126_QUANT_CATALOG_ROOT={out}")
    print(f"DKR_R126_QUANT_CATALOG_REPORT={a.json_out}")


if __name__ == "__main__":
    main()
