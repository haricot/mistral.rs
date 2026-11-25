#!/usr/bin/env python3
"""Plan and stamp bounded FlashInfer decode images. Never invoke CUDA."""
import argparse
import hashlib
import itertools
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from dkr_r126_flashinfer_decode_plan import report

DTYPES = ("f16", "bf16", "f32")
GROUPS = (1, 2, 3, 4, 5, 6, 7, 8, 16)
HEADS = (64, 128, 256, 512)
ARTIFACT = "flashinfer_decode.ptx"
SHARED_LIMIT = 48 * 1024
MAX_IMAGE = 32 * 1024 * 1024
MAX_MANIFEST = 256 * 1024

def tensor(name, dtype, rank, access, representation=None):
    result = dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)
    if representation:
        result["representation"] = representation
    return result

def scalar(name, dtype="u32"):
    return dict(kind="scalar", name=name, dtype=dtype)

def arguments(dt, cache, partition):
    cd = dt if cache == "plain" else "fp8_e4m3"
    args = [tensor("q_span", dt, 1, "read", "flashinfer.q.strided_span.v1")]
    args += [tensor(n, cd, 4, "read", "flashinfer.hnd.v1") for n in ("key_cache", "value_cache")]
    args += [tensor(n, "i32", 1, "read") for n in
             ("indptr", "indices", "last_page_len", "request_indices", "tile_indices", "o_indptr", "chunk_size")]
    args += [tensor("block_valid_mask", "u8", 1, "read")]
    if partition:
        args += [tensor("partial", dt, 3, "write", "flashinfer.splitkv.partial.v1"),
                 tensor("lse", "f32", 2, "write", "flashinfer.splitkv.log2_lse.v1")]
    else:
        args += [tensor("output", dt, 3, "write")]
    args += [scalar(n) for n in ("batch", "padded_batch", "heads", "kv_heads", "page", "magic", "shift", "add")]
    args += [scalar("q_stride_n", "i32"), scalar("q_stride_h", "i32"), scalar("use_mask"),
             scalar("window_left", "i32")]
    args += [scalar(n, "f32") for n in ("softcap", "scale", "k_scale", "v_scale")]
    return args

def safe_geometry(cc, width, head, group, historical):
    """Preserve old safe geometry; otherwise full warps and disjoint scratch."""
    stages = historical["stages"]
    x = historical["block"][0]
    old_z = historical["block"][2]
    old_tile = historical["tile"]
    intended = historical["intended_threads"]
    old_ok = (not historical["partial_warp"] and
              not historical["exceeds_requested_shared_limit"] and
              (old_z == 1 or 2*stages*old_tile*width >= 4))
    if old_ok:
        return dict(historical, recovered=False)
    candidates = []
    for tile in range(old_tile, 0, -1):
        for z in range(1, 1024//(x*group)+1):
            threads = x*group*z
            if threads % 32:
                continue
            stage_bytes = 2*stages*tile*group*z*head*width
            state_bytes = 4*group*z*head if z > 1 else 0
            # Offset table is live only during the K/V pipeline. m/d are
            # live only during reduction; the F32 vector and m/d must not overlap.
            shared = max(stage_bytes+tile*threads*8,
                         max(stage_bytes,state_bytes)+2*group*z*4)
            if shared <= SHARED_LIMIT:
                candidates.append(((-tile,abs(threads-intended),threads),tile,z,shared))
    if not candidates:
        return None
    _,tile,z,shared = min(candidates)
    return dict(historical, block=[x,group,z], actual_threads=x*group*z,
                tile=tile, dynamic_shared_bytes=shared, partial_warp=False,
                exceeds_requested_shared_limit=False, recovered=True)

def plan(cc, base=411):
    audit = report(cc, base, SHARED_LIMIT)
    eligible, rejected = [], []
    for row in audit["geometries"]:
        width = 1 if row["cache"] == "fp8" else (4 if row["dtype"] == "f32" else 2)
        candidate = safe_geometry(cc, width, row["head"], row["group"], row)
        if candidate is None:
            rejected.append(dict(row, reasons=["no_full_warp_geometry_within_shared_budget"]))
        else:
            eligible.append(candidate)
    images = []
    # The full typed signatures are larger than the synthetic S9B4d2 rows.
    # Six groups (48 entries) leave headroom in the 256 KiB manifest,
    # including the later pretty-printed CUBIN pairing fields.
    max_groups = min(6, audit["profiles_per_shard_limit"] // 8)
    if not 1 <= max_groups <= 6:
        raise ValueError("unsupported image grouping policy")
    for dt, cache, hd in itertools.product(DTYPES, ("plain", "fp8"), HEADS):
        rows = [r for r in eligible if (r["dtype"], r["cache"], r["head"]) == (dt, cache, hd)]
        for start in range(0, len(rows), max_groups):
            chunk = rows[start:start+max_groups]
            mask = sum(1 << GROUPS.index(r["group"]) for r in chunk)
            image = DTYPES.index(dt) | ((cache == "fp8") << 2) | (HEADS.index(hd) << 3) | (mask << 5)
            profiles = []
            for r in chunk:
                g = r["group"]
                for w, s, mode in itertools.product((0, 1), (0, 1), ("direct", "partition")):
                    entry = f"dkr_flashinfer_decode_g{g}_w{w}_s{s}_{mode}"
                    suffix = f"d{hd}.g{g}.w{w}.s{s}.{mode}.v1"
                    profiles.append(dict(schema="DKR-KERNEL-MODULE-V1", abi="dkr.typed.v1",
                        operation=f"attention.flashinfer.decode.{suffix}",
                        implementation_id=f"cuda.flashinfer_decode.{dt}.{cache}.safe2.{suffix}",
                        artifact=ARTIFACT, entry=entry, target_cc=cc, min_cc=cc, max_cc=cc,
                        block=r["block"], shared_mem_bytes=r["dynamic_shared_bytes"],
                        arguments=arguments(dt, cache, mode == "partition")))
            images.append(dict(image=image, dtype=dt, cache=cache, head=hd,
                               groups=[r["group"] for r in chunk], geometries=chunk, profiles=profiles))
    logical = sum(len(i["profiles"]) for i in images)
    # Conservative bound: every existing logical profile occupies a separate manifest.
    physical_upper = base + audit["merge_profiles"] + len(images)
    total = base + audit["merge_profiles"] + logical
    if physical_upper > audit["catalogue_limit"] or total > audit["logical_profile_limit"]:
        raise ValueError("planned catalogue exceeds resource bounds")
    return dict(schema="dkr.r126.flashinfer.decode.shards.v1", cc=cc,
                eligible_geometries=len(eligible), recovered_geometries=sum(r["recovered"] for r in eligible), quarantined_geometries=rejected,
                decode_profiles=logical, images=images, physical_manifest_upper=physical_upper,
                prospective_logical_profiles=total, compilation=False, gpu_qualified=False,
                closure_qualified=False)

def widths(profile):
    return [64 if a["kind"] == "tensor" else 32 for a in profile["arguments"]]

def verify_ptx(data, image, cc):
    if len(data) > MAX_IMAGE:
        raise ValueError("PTX image budget exceeded")
    source = data.decode()
    if not re.search(r"\.target\s+sm_" + str(cc) + r"(?:[\s,]|$)", source) or not re.search(r"\.address_size\s+64\b", source):
        raise ValueError("PTX target/address mismatch")
    entries = {}
    for match in re.finditer(r"\.entry\s+(\w+)\s*\(([^)]*)\)", source, re.S):
        name, params = match.groups()
        if name in entries:
            raise ValueError("duplicate PTX entry")
        fields = [] if not params.strip() else params.split(",")
        parsed = []
        for field in fields:
            width = re.search(r"\.(?:u|s|b|f)(8|16|32|64)\b", field)
            if not width:
                raise ValueError("unsupported PTX parameter: " + field)
            parsed.append(int(width.group(1)))
        entries[name] = parsed
    expected = {p["entry"]: widths(p) for p in image["profiles"]}
    if entries != expected:
        raise ValueError("PTX entry set or typed parameter widths differ from plan")

def package(output, cc, image, data):
    verify_ptx(data, image, cc)
    sha = hashlib.sha256(data).hexdigest()
    folder = output / f"sm{cc}" / "flashinfer-decode" / f"safe2-image-{image['image']:05d}"
    profiles = [dict(p, sha256=sha) for p in image["profiles"]]
    shard = dict(schema="DKR-KERNEL-SHARD-V1", profiles=profiles)
    encoded = (json.dumps(shard, indent=2) + "\n").encode()
    # Reserve space for CUBIN path/hash and pretty-print overhead.
    if len(encoded) + 256 * len(profiles) > MAX_MANIFEST:
        raise ValueError("manifest pairing budget exceeded")
    # Mixing two generations would create ambiguous typed profiles.
    family = folder.parent
    if any(family.glob("image-*/manifest.json")):
        raise ValueError("old decode generation present; stage safe2 in a fresh root")
    folder.mkdir(parents=True, exist_ok=True)
    manifest, artifact = folder / "manifest.json", folder / ARTIFACT
    if manifest.is_file() and artifact.is_file():
        old = json.loads(manifest.read_text())
        stripped = dict(old)
        stripped["profiles"] = [{k:v for k,v in p.items() if k not in ("cubin_artifact", "cubin_sha256")}
                                for p in old.get("profiles", [])]
        if stripped == shard and artifact.read_bytes() == data:
            return
        # Changed artifacts must use a fresh output root, not overwrite a qualified image.
        raise ValueError("existing FlashInfer shard differs; use a fresh output root")
    if manifest.exists() or artifact.exists():
        raise ValueError("incomplete existing shard; inspect the staging directory")
    artifact.write_bytes(data)
    manifest.write_bytes(encoded)

def self_test(cc):
    result = plan(cc)
    images = result["images"]
    assert len({i["image"] for i in images}) == len(images)
    ids = [p["implementation_id"] for i in images for p in i["profiles"]]
    assert len(set(ids)) == len(ids) == result["decode_profiles"]
    assert all(0 < len(i["profiles"]) <= 48 for i in images)
    if cc == 61:
        assert result["eligible_geometries"] == 215 and len(result["quarantined_geometries"]) == 1
        assert result["recovered_geometries"] == 76
        assert len(images) == 48 and len(ids) == 1720
        assert result["physical_manifest_upper"] == 471
    with tempfile.TemporaryDirectory() as tmp:
        for i in images:
            src = f".version 8.0\n.target sm_{cc}\n.address_size 64\n"
            for p in i["profiles"]:
                params = ",".join(f".param .b{w} arg{j}" for j,w in enumerate(widths(p)))
                src += f".visible .entry {p['entry']}({params}) {{ ret; }}\n"
            data = src.encode()
            package(Path(tmp), cc, i, data)
            manifest = Path(tmp) / f"sm{cc}" / "flashinfer-decode" / f"safe2-image-{i['image']:05d}" / "manifest.json"
            paired = json.loads(manifest.read_text())
            for row in paired["profiles"]:
                row["cubin_artifact"] = "flashinfer_decode.cubin"
                row["cubin_sha256"] = "0" * 64
            encoded = (json.dumps(paired, indent=2) + "\n").encode()
            assert len(encoded) <= MAX_MANIFEST
            manifest.write_bytes(encoded)
            package(Path(tmp), cc, i, data)
            assert manifest.read_bytes() == encoded
            try:
                verify_ptx(data.replace(b".param .b64", b".param .b32", 1), i, cc)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid ABI was accepted")
        assert len(list(Path(tmp).rglob("manifest.json"))) == len(images)
    print(f"DKR_R126_S9B4D5_SELF_TEST=PASS cc={cc} images={len(images)} profiles={len(ids)} synthetic_ptx=true")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cc", type=int, required=True)
    p.add_argument("--list-images", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--json-out", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--image", type=int)
    p.add_argument("--ptx-file", type=Path)
    a = p.parse_args()
    if not 50 <= a.cc <= 120:
        p.error("invalid cc")
    r = plan(a.cc)
    if a.list_images:
        for i in r["images"]:
            print(i["image"])
        return
    if a.self_test:
        self_test(a.cc)
        return
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(r, indent=2) + "\n")
        print(f"DKR_R126_S9B4D5_PLAN=COMPLETE cc={a.cc} eligible={r['eligible_geometries']} recovered={r['recovered_geometries']} "
              f"quarantined={len(r['quarantined_geometries'])} profiles={r['decode_profiles']} "
              f"images={len(r['images'])} physical_upper={r['physical_manifest_upper']} "
              f"logical_total={r['prospective_logical_profiles']} compilation=false closure_qualified=false")
        return
    if a.output is None or a.image is None or a.ptx_file is None:
        p.error("specify --output --image --ptx-file or a planning mode")
    image = next((i for i in r["images"] if i["image"] == a.image), None)
    if image is None:
        p.error("unknown image")
    package(a.output, a.cc, image, a.ptx_file.read_bytes())
    print(f"DKR_FLASHINFER_DECODE_SHARD=STAGED cc={a.cc} image={a.image} profiles={len(image['profiles'])}")

if __name__ == "__main__":
    main()
