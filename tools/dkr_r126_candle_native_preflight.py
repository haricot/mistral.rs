#!/usr/bin/env python3
"""S8 read-only native + typed catalogue preflight, no Cargo build or GPU.

Use the verifier from the exact pinned Candle checkout identified by Cargo
metadata. Discover old qualification receipts, never regenerate their inputs.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from dkr_r126_full_catalog_inventory import inspect, package_dir
from dkr_r126_nvfp4_preflight import digest

PIN = "d40ac1b4d69a209d08f76c5b5bd4e267ec63b4f0"
SCHEMAS = {"CANDLE-DKR-R125-P1-RECEIPT-V1", "CANDLE-DKR-R125-P2-RECEIPT-V1"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def candle_source(metadata):
    obj = json.loads(metadata.read_text())
    candidates = [p for p in obj["packages"] if p["name"] == "candle-core"
                  and p.get("source", "").endswith("#" + PIN)]
    require(len(candidates) == 1, "exact pinned Candle source not resolved")
    root = Path(candidates[0]["manifest_path"]).resolve(strict=True).parent.parent
    script = root / "tools/dkr-r124-s55c-qualify.py"
    spec = importlib.util.spec_from_file_location("candle_native_verifier", script)
    mod = importlib.util.module_from_spec(spec)
    require(spec.loader is not None, "Candle verifier unavailable")
    spec.loader.exec_module(mod)
    return root, mod


def typed_gate(report):
    obj = json.loads(report.read_text())
    require(obj["schema"] == "dkr.r126.full-catalog.staged.v1"
            and obj["status"] == "PASS" and obj["target_cc"] == 61
            and obj["families"] == 26 and obj["profiles"] == 216
            and obj["cubin_images"] == 27, "wrong S7 staged report")
    root = Path(obj["catalogue"]).resolve(strict=True)
    require(len(obj["family_artifacts"]) == 26, "incomplete family set")
    for family, expected in obj["family_artifacts"].items():
        observed = inspect(root, family)
        require(json.loads(json.dumps(observed)) == expected, f"{family}: artifact drift")
        require(observed["profiles"] == observed["paired_profiles"], f"{family}: unpaired")
    actual = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    require(actual == obj["files"], "S7 file inventory changed")
    require(len(list(root.rglob("manifest.json"))) == 216, "typed profile count changed")
    return root


def candidates(roots, receipts):
    paths = {p.resolve() for p in receipts}
    for root in roots:
        if not root.is_dir():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in (".git", "incremental", "deps", ".fingerprint",
                         "node_modules", ".venv", "__pycache__")]
            if "receipt.json" in files:
                paths.add(Path(directory) / "receipt.json")
    return sorted(paths)


def native_gate(path, verifier):
    obj = json.loads(path.read_text())
    require(obj.get("schema") in SCHEMAS and obj.get("target_cc") == 61, "not a native R1.25 receipt")
    seal, cubins, bundle = (Path(obj[k]).resolve(strict=True)
                           for k in ("seal", "cubin_dir", "catalogue_dir"))
    for file, key in ((seal, "seal_sha256"), (cubins/"manifest.json", "cubin_manifest_sha256"),
                      (bundle/"catalogue.json", "catalogue_sha256")):
        require(digest(file) == obj[key], f"receipt digest changed: {file}")
    verified = verifier.verify(bundle, seal, cubins)
    require(verified["catalogue_sha256"] == obj["catalogue_sha256"], "native bundle identity changed")
    env = dict(DKR_EMBED_NATIVE_DIR=str(Path(verified["seal"]["catalogue"]).resolve(strict=True)),
               DKR_EMBED_CUBIN_DIR=str(cubins),
               DKR_EMBED_S55_MMQ_DIR=str(verified["mmq_dir"].resolve(strict=True)),
               DKR_EMBED_S55B_RECIPES_DIR=str(verified["recipes_dir"].resolve(strict=True)),
               DKR_EMBED_S55C_CUDAFORGE_DIR=str(bundle))
    return obj, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--typed-evidence", type=Path, required=True)
    p.add_argument("--search-root", type=Path, action="append", default=[])
    p.add_argument("--receipt", type=Path, action="append", default=[])
    p.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    try:
        sys.dont_write_bytecode = True
        root, verifier = candle_source(a.metadata)
        typed = typed_gate(a.typed_evidence)
        print("DKR_R126_S8_TYPED_INPUT=PASS families=26 profiles=216 cubin_images=27", flush=True)
        valid = []
        for path in candidates(a.search_root, a.receipt):
            try:
                obj = json.loads(path.read_text())
                if obj.get("schema") not in SCHEMAS:
                    continue
                obj, env = native_gate(path, verifier)
                valid.append((path, obj, env))
                print(f"DKR_R126_S8_NATIVE_RECEIPT=PASS path={path}", flush=True)
            except (OSError, ValueError, TypeError, KeyError, SystemExit) as exc:
                print(f"DKR_R126_S8_NATIVE_RECEIPT=REJECT path={path} reason={exc}", flush=True)
        require(valid, "no valid native receipt; locate existing R1.25 P1/P2 receipt, do not rebuild")
        identities = {tuple(sorted(env.items())) for _,_,env in valid}
        require(len(identities) == 1, "multiple native inputs; select one with --receipt only")
        path, obj, env = valid[0]
        env["DKR_EMBED_TYPED_DIR"] = str(typed)
        result = dict(schema="dkr.r126.native.preflight.v1", status="PASS", target_cc=61,
                      candle_pin=PIN, candle_checkout=str(root),
                      native_receipt=str(path), native_receipt_sha256=digest(path),
                      native_catalogue_sha256=obj["catalogue_sha256"],
                      typed_evidence_sha256=digest(a.typed_evidence),
                      build_environment=env, native_families=25, native_recipes=4, mmq_codecs=10,
                      typed_families=26, typed_profiles=216, typed_cubin_images=27,
                      compilation=False, integration_gpu_qualified=False,
                      static_fallback_elimination_qualified=False)
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
        print("DKR_R126_S8_NATIVE_INPUT=PASS families=25 recipes=4 mmq_codecs=10")
        for key, value in env.items():
            print(f"{key}={value}")
        print(f"DKR_R126_S8_PREFLIGHT=PASS compilation=false gpu_qualified=false report={a.json_out}")
    except (ValueError, OSError, KeyError, TypeError, ImportError, SystemExit) as exc:
        print(f"DKR_R126_S8_PREFLIGHT=BLOCKED reason={exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
