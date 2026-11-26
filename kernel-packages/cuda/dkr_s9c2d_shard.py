"""Read-only PTX verification and single-image shard staging for S9C.2D.

The compiler is owned by dkr_package_builder (cudaforge), never this helper.
"""
import hashlib
import json
import re
import tempfile
from pathlib import Path

MAX_MANIFEST = 256 * 1024
MAX_IMAGE = 32 * 1024 * 1024

def tensor(name, dtype, rank, access):
    return dict(kind="tensor", name=name, dtype=dtype, rank=rank, access=access)

def scalar(name, dtype):
    return dict(kind="scalar", name=name, dtype=dtype)

def profile(cc, operation, impl, entry, block, args, artifact, max_cc=None):
    return dict(schema="DKR-KERNEL-MODULE-V1", abi="dkr.typed.v1",
                operation=operation, implementation_id=impl, entry=entry,
                artifact=artifact, target_cc=cc, min_cc=cc,
                max_cc=cc if max_cc is None else max_cc,
                block=block, shared_mem_bytes=0, arguments=args)

def widths(row):
    return [64 if a["kind"]=="tensor" or a["dtype"] in ("i64","u64") else 32
            for a in row["arguments"]]

def verify(data, cc, rows):
    if not 1 <= len(rows) <= 64 or len(data) > MAX_IMAGE:
        raise ValueError("shard or image limit exceeded")
    if any(len(r["arguments"]) > 32 for r in rows):
        raise ValueError("argument limit exceeded")
    if len({r["implementation_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate implementation")
    if len({r["artifact"] for r in rows}) != 1:
        raise ValueError("multiple artifacts in one image")
    src = data.decode()
    if not re.search(r"\.target\s+sm_"+str(cc)+r"(?:[\s,]|$)",src) or not re.search(r"\.address_size\s+64\b",src):
        raise ValueError("PTX target/address mismatch")
    entries = {}
    for name,params in re.findall(r"\.entry\s+(\w+)\s*\(([^)]*)\)",src,re.S):
        if name in entries:
            raise ValueError("duplicate entry")
        entries[name] = []
        for arg in params.split(",") if params.strip() else []:
            m = re.search(r"\.(?:u|s|b|f)(8|16|32|64)\b",arg)
            if not m:
                raise ValueError("unrecognized parameter")
            entries[name].append(int(m.group(1)))
    expected = {r["entry"]: widths(r) for r in rows}
    if len(expected) != len(rows) or entries != expected:
        raise ValueError("PTX entry/ABI mismatch")

def stage(folder, cc, rows, data):
    verify(data, cc, rows)
    digest = hashlib.sha256(data).hexdigest()
    shard = dict(schema="DKR-KERNEL-SHARD-V1",
                 profiles=[dict(r,sha256=digest) for r in rows])
    paired = dict(shard,profiles=[dict(r,cubin_artifact=Path(r["artifact"]).with_suffix(".cubin").name,
                                      cubin_sha256="0"*64) for r in shard["profiles"]])
    if len((json.dumps(paired,indent=2)+"\n").encode()) > MAX_MANIFEST:
        raise ValueError("paired manifest exceeds budget")
    folder.mkdir(parents=True,exist_ok=True)
    manifest,artifact = folder/"manifest.json",folder/rows[0]["artifact"]
    if manifest.exists() or artifact.exists():
        if not manifest.is_file() or not artifact.is_file():
            raise ValueError("partial image; use a fresh staging root")
        previous=json.loads(manifest.read_text())
        stripped=dict(previous,profiles=[{k:v for k,v in r.items()
                       if k not in ("cubin_artifact","cubin_sha256")}
                       for r in previous.get("profiles",[])])
        if stripped != shard or artifact.read_bytes() != data:
            raise ValueError("existing image differs; use a fresh staging root")
        return
    artifact.write_bytes(data)
    manifest.write_text(json.dumps(shard,indent=2)+"\n")

def self_test(cc, rows):
    src=f".version 8.0\n.target sm_{cc}\n.address_size 64\n"
    for row in rows:
        params=",".join(f".param .b{w} a{i}" for i,w in enumerate(widths(row)))
        src+=f".visible .entry {row['entry']}({params}) {{ret;}}\n"
    data=src.encode()
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        stage(folder,cc,rows,data)
        manifest=folder/"manifest.json"
        paired=json.loads(manifest.read_text())
        for row in paired["profiles"]:
            row.update(cubin_artifact=Path(row["artifact"]).with_suffix(".cubin").name,
                       cubin_sha256="0"*64)
        original=json.dumps(paired,indent=2)+"\n"
        manifest.write_text(original)
        stage(folder,cc,rows,data)
        assert manifest.read_text()==original
        for invalid in (data.replace(b".param .b64",b".param .b32",1),
                        data.replace(b".address_size 64",b".address_size 32")):
            try:
                verify(invalid,cc,rows)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid PTX accepted")
