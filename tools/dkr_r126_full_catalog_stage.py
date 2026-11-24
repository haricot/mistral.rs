#!/usr/bin/env python3
"""Stage only missing CUBIN pairs; consolidate without modifying original packages.

The caller runs the existing Cudaforge builder --pair-existing on the isolated
assembly root between 'stage' and 'finalize'. No compiler is called here.
"""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

from dkr_r126_full_catalog_inventory import inspect, package_dir
from dkr_r126_nvfp4_preflight import digest


def require(ok, why):
    if not ok:
        raise ValueError(why)


def snapshot(root, family):
    directory = root / package_dir(family)
    result = {}
    for p in sorted(directory.rglob("manifest.json")):
        rel = p.relative_to(root)
        m = json.loads(p.read_text())
        result[str(rel)] = (m, digest(p))
    return result


def copy_profiles(src, dest, rows):
    for rel, (m, _) in rows.items():
        path = Path(rel)
        folder = dest / path.parent
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("artifact", "cubin_artifact"):
            if m.get(name) is not None:
                shutil.copyfile(src / path.parent / m[name], folder / m[name])
        shutil.copyfile(src / path, dest / path)


def atomic_catalog(destination, writer, validator):
    if destination.exists():
        validator(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dkr-stage-", dir=destination.parent) as tmp:
        staged = Path(tmp) / "sm61"
        staged.mkdir()
        writer(staged)
        validator(staged)
        staged.rename(destination)


def load_plan(inventory, logits_root):
    data = json.loads(inventory.read_text())
    require(data.get("schema") == "dkr.r126.full-catalog.inventory.v1"
            and data.get("target_cc") == 61, "wrong inventory schema/target")
    builder = Path(__file__).resolve().parents[1] / "mistralrs-kernel-runtime/src/bin/dkr_package_builder.rs"
    registered = set(re.findall(r'Package\s*\{\s*name:\s*"([^"]+)"', builder.read_text()))
    require(set(data["families"]) == registered and len(registered) == 26,
            "inventory does not cover current 26 registered families")
    plan = {}
    for family, row in data["families"].items():
        root = Path(row["root"]).resolve()
        if row["status"] == "MISSING" and family == "logits" and logits_root:
            root = logits_root.resolve()
            actual = inspect(root, family)
            require(actual["profiles"] == 3, "isolated logits package needs three profiles")
        else:
            require(row["status"] in ("PAIRED", "PTX_ONLY"),
                    f"{family} {row['status']}; correct inventory before assembly")
            actual = inspect(root, family)
            for field in ("profiles", "paired_profiles", "ptx_images", "cubin_images"):
                require(actual[field] == row[field], f"{family}: inventory drift in {field}")
            require(actual["image_pairs"] == [tuple(p) for p in row["image_pairs"]],
                    f"{family}: inventory image-pair drift")
        require(actual["profiles"] > 0 and actual["paired_profiles"] in (0, actual["profiles"]),
                f"{family}: missing or partial original package")
        plan[family] = (root, actual, snapshot(root, family))
    return plan


def validate_copy(root, plan, final):
    expected_paths, keys = set(), set()
    for family, (_, original, old) in plan.items():
        current = inspect(root, family)
        require(current["profiles"] == original["profiles"]
                and current["ptx_images"] == original["ptx_images"],
                f"{family}: changed PTX/profile set")
        now = snapshot(root, family)
        require(set(now) == set(old), f"{family}: changed manifest paths")
        expected_paths.update(now)
        for path, (m, sha) in now.items():
            prior, prior_sha = old[path]
            key = (m["operation"], m["implementation_id"], m["target_cc"])
            require(key not in keys, f"duplicate catalogue identity: {path}")
            keys.add(key)
            if original["paired_profiles"]:
                require(sha == prior_sha, f"qualified manifest changed: {path}")
            else:
                stripped = {k:v for k,v in m.items() if k not in ("cubin_artifact", "cubin_sha256")}
                expected = {k:v for k,v in prior.items() if k not in ("cubin_artifact", "cubin_sha256")}
                require(stripped == expected, f"non-pairing manifest metadata changed: {path}")
        if original["paired_profiles"]:
            require(current["image_pairs"] == original["image_pairs"], f"{family}: existing CUBIN changed")
        if final:
            require(current["profiles"] == current["paired_profiles"], f"{family}: CUBIN pairing not complete")
    require({str(p.relative_to(root)) for p in root.rglob("manifest.json")} == expected_paths,
            "unexpected manifest outside planned families")
    return {family: inspect(root, family) for family in plan}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("stage", "finalize"))
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--assembly", type=Path, required=True, help="isolated /sm61 directory for missing pairs only")
    ap.add_argument("--output", type=Path, required=True, help="final /sm61 directory")
    ap.add_argument("--logits-root", type=Path, help="optional isolated replacement for genuinely missing logits")
    ap.add_argument("--json-out", type=Path, required=True)
    a = ap.parse_args()
    try:
        plan = load_plan(a.inventory, a.logits_root)
        assembly, output = a.assembly.resolve(), a.output.resolve()
        roots = {root for root, _, _ in plan.values()}
        for dest in (assembly, output):
            require(all(dest != src and not dest.is_relative_to(src) and not src.is_relative_to(dest)
                        for src in roots), "output overlaps original package root")
        require(assembly != output and not assembly.is_relative_to(output)
                and not output.is_relative_to(assembly), "assembly/output overlap")
        pending = {f: v for f,v in plan.items() if not v[1]["paired_profiles"]}
        preserved = {f: v for f,v in plan.items() if v[1]["paired_profiles"]}
        require(pending, "no missing CUBIN pairs to stage")
        if a.phase == "stage":
            atomic_catalog(assembly,
                           lambda dest: [copy_profiles(src, dest, rows) for src,_,rows in pending.values()],
                           lambda dest: validate_copy(dest, pending, False))
            unique = {h for _, info, _ in pending.values() for h in info["ptx_images"]}
            print(f"DKR_R126_FULL_STAGE=PASS preserved_families={len(preserved)} pending_families={len(pending)} unique_pending_ptx={len(unique)}")
            return
        require(not a.json_out.exists(), "report already exists")
        require(not a.json_out.resolve().is_relative_to(output), "report must be outside catalogue")
        validate_copy(assembly, pending, True)
        def writer(dest):
            for family,(root,_,rows) in preserved.items():
                copy_profiles(root, dest, rows)
            for family in pending:
                copy_profiles(assembly, dest, snapshot(assembly, family))
        atomic_catalog(output, writer, lambda dest: validate_copy(dest, plan, True))
        results = validate_copy(output, plan, True)
        files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
        images = {h for r in results.values() for h in r["cubin_images"]}
        report = dict(schema="dkr.r126.full-catalog.staged.v1", status="PASS", target_cc=61,
                      families=len(results), profiles=sum(r["profiles"] for r in results.values()),
                      cubin_images=len(images), preserved_families=sorted(preserved),
                      newly_paired_families=sorted(pending), inventory_sha256=digest(a.inventory),
                      catalogue=str(output), files=files, family_artifacts=results,
                      full_catalogue_qualified=False, embedded_full_qualified=False,
                      static_fallback_elimination_qualified=False, full_model_qualified=False)
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        with a.json_out.open("x") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"DKR_R126_FULL_CATALOG_STAGE=PASS families={report['families']} profiles={report['profiles']} cubin_images={len(images)} full_catalogue_qualified=false")
        print(f"DKR_R126_FULL_CATALOG_REPORT={a.json_out}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"DKR_R126_FULL_CATALOG_STAGE=BLOCKED reason={exc}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
