#!/usr/bin/env python3
"""Read-only inventory of registered DKR families, local packages and evidence.

Counts package coverage, not static fallback elimination or model readiness.
Never calls Cargo, CUDA, nvcc, ptxas or writes to existing package roots.
"""
import argparse
import json
import re
from pathlib import Path

from dkr_r126_quant_embedded import validate as validate_quant
from dkr_r126_nvfp4_preflight import digest

QUANT = {"quant-gemv", "quant-gemm", "nvfp4-gemv", "nvfp4-gemm"}
REAL = {"moe": "dkr_moe_gpu", "indexed-copy": "dkr_indexed_copy_gpu",
        "paged-decode": "dkr_paged_decode_gpu"}


PACKAGE_DIRS = {"logits": "sparse-logits"}


def package_dir(family):
    return PACKAGE_DIRS.get(family, family)


def inspect(root, family):
    paths = sorted((root / package_dir(family)).rglob("manifest.json"))
    ptxs, cubins, pairs, keys = set(), set(), set(), set()
    paired = 0
    for path in paths:
        m = json.loads(path.read_text())
        if m.get("schema") != "DKR-KERNEL-MODULE-V1" or m.get("abi") != "dkr.typed.v1" or m.get("target_cc") != 61:
            raise ValueError(f"invalid typed SM61 contract: {path}")
        key = (m["operation"], m["implementation_id"], m["target_cc"])
        if key in keys:
            raise ValueError(f"duplicate catalogue identity: {path}")
        keys.add(key)
        def artifact(field, hash_field):
            name, sha = m[field], m[hash_field]
            if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
                raise ValueError(f"unsafe artifact: {path}")
            p = path.parent / name
            if p.is_symlink() or not p.is_file() or digest(p) != sha:
                raise ValueError(f"artifact missing or changed: {p}")
            return p
        p = artifact("artifact", "sha256")
        code = p.read_text()
        if not re.search(r"\.target\s+sm_61(?:\s|,|$)", code) or not re.search(
            r"\.entry\s+" + re.escape(m["entry"]) + r"\s*\(", code):
            raise ValueError(f"PTX target/entry mismatch: {p}")
        ptxs.add(m["sha256"])
        if (m.get("cubin_artifact") is None) != (m.get("cubin_sha256") is None):
            raise ValueError(f"incomplete CUBIN pair: {path}")
        if m.get("cubin_artifact") is not None:
            p = artifact("cubin_artifact", "cubin_sha256")
            with p.open("rb") as f:
                header = f.read(64)
            if (len(header) != 64 or header[:6] != b"\x7fELF\x02\x01"
                or int.from_bytes(header[18:20], "little") != 190
                or (int.from_bytes(header[48:52], "little") & 255) != 61):
                raise ValueError(f"invalid CUDA SM61 ELF: {p}")
            paired += 1
            cubins.add(m["cubin_sha256"])
            pairs.add((m["cubin_sha256"], m["sha256"]))
    return dict(profiles=len(paths), paired_profiles=paired,
                ptx_images=sorted(ptxs), cubin_images=sorted(cubins),
                image_pairs=sorted(pairs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--quant", type=Path, required=True)
    p.add_argument("--quant-evidence", type=Path, required=True)
    p.add_argument("--embedded-evidence", type=Path, required=True)
    p.add_argument("--real", type=Path, required=True)
    p.add_argument("--real-log", type=Path, required=True)
    p.add_argument("--json-out", type=Path, required=True)
    a = p.parse_args()
    try:
        builder = Path(__file__).resolve().parents[1] / "mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs"
        families = re.findall(r'Package\s*\{\s*name:\s*"([^"]+)"', builder.read_text())
        if not families or len(families) != len(set(families)) or not QUANT.union(REAL) <= set(families):
            raise ValueError("unexpected package builder inventory")
        validate_quant(a.quant, a.quant_evidence)
        embedded = json.loads(a.embedded_evidence.read_text())
        source = json.loads(a.quant_evidence.read_text())
        if (embedded.get("schema") != "dkr.r126.quant.embedded.evidence.v1"
            or embedded.get("status") != "PASS"
            or embedded.get("embedded_runtime_qualified") is not True
            or embedded.get("strict_cubin") is not True
            or embedded.get("source_catalogue_evidence_sha256") != digest(a.quant_evidence)
            or embedded.get("image_pairs") != source.get("image_pairs")):
            raise ValueError("S5 report is not bound to S4 catalogue")
        log = a.real_log.read_text() if a.real_log.is_file() else ""
        real_pass = bool(log) and not (
            "CANDLE_DKR_TYPED_IMAGE_FALLBACK" in log
            or re.search(r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm\d+\s+format=ptx", log)
            or re.search(r"(?m)^\s*(?:FAIL|TIMEOUT|ABORT|LEAK)\s+\[", log)
        )
        loaded = set(re.findall(
            r"CANDLE_DKR_TYPED_IMAGE_LOADED\s+target=sm61\s+format=cubin\s+"
            r"cubin_sha256=([0-9a-f]{64})\s+ptx_sha256=([0-9a-f]{64})", log
        ))
        rows = {}
        for family in families:
            root = a.quant if family in QUANT else a.real if family in REAL and (a.real / family).is_dir() else a.source
            try:
                info = inspect(root, family)
                status = ("MISSING" if not info["profiles"] else
                          "PAIRED" if info["profiles"] == info["paired_profiles"] else
                          "PTX_ONLY" if not info["paired_profiles"] else "PARTIAL_PAIR")
                evidence = "NONE"
                if family in QUANT:
                    evidence = "S5_EMBEDDED_TARGETED_PASS"
                elif family in REAL and real_pass and re.search(
                    r"(?m)^\s*PASS\s+\[[^\]]+\]\s+.*\b" + re.escape(REAL[family]) + r"\s*$", log
                ) and set(info["image_pairs"]) & loaded:
                    evidence = "R125_FILESYSTEM_TARGETED_PASS"
                rows[family] = dict(root=str(root), package_dir=package_dir(family), status=status, evidence=evidence, **info)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                rows[family] = dict(root=str(root), status="INVALID", evidence="NONE", reason=str(exc))
            r = rows[family]
            print(f"DKR_R126_FAMILY family={family} status={r['status']} "
                  f"profiles={r.get('profiles',0)} paired={r.get('paired_profiles',0)} evidence={r['evidence']}")
        counts = {status: sum(r["status"] == status for r in rows.values())
                  for status in ("MISSING", "INVALID", "PTX_ONLY", "PARTIAL_PAIR", "PAIRED")}
        report = dict(schema="dkr.r126.full-catalog.inventory.v1", target_cc=61,
                      registered_families=len(families), counts=counts, families=rows,
                      s4_evidence_sha256=digest(a.quant_evidence),
                      s5_evidence_sha256=digest(a.embedded_evidence),
                      real_log_sha256=digest(a.real_log) if log else None,
                      full_catalogue_qualified=False, full_model_qualified=False,
                      static_fallback_elimination_qualified=False)
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"DKR_R126_FULL_INVENTORY=FAIL reason={exc}")
        raise SystemExit(1)
    print(f"DKR_R126_FULL_INVENTORY=COMPLETE families={len(families)} "
          + " ".join(f"{k.lower()}={v}" for k,v in counts.items())
          + " full_catalogue_qualified=false compilation=false")
    print(f"DKR_R126_FULL_INVENTORY_REPORT={a.json_out}")


if __name__ == "__main__":
    main()
