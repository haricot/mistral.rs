import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CRATES = ["mistralrs", "mistralrs-cli", "mistralrs-core", "mistralrs-quant",
          "mistralrs-paged-attn", "mistralrs-server-core", "mistralrs-pyo3"]
RESOLVED = {f"{name}/Cargo.toml" for name in CRATES} | {
    "mistralrs-core/build.rs", "mistralrs-core/src/cuda/moe.rs",
    "mistralrs-quant/build.rs", "mistralrs-paged-attn/build.rs",
}


def git(repo, *args, check=True):
    return subprocess.run(["git", "-c", "user.name=Legacy V3 preparation", "-c", "user.email=ci@localhost", "-C", str(repo), *args], check=check,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def show(repo, ref, path):
    return git(repo, "show", f"{ref}:{path}").stdout


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"resolution preimage changed: {before!r}")
    return text.replace(before, after, 1)


def verify_source(repo, meta, source):
    sha = source["sha"]
    git(repo, "merge-base", "--is-ancestor", meta["base_master_sha"], sha)
    inherited = git(repo, "merge-base", "--is-ancestor", meta["source_legacy_sha"], sha, check=False)
    if inherited.returncode != 1:
        raise ValueError("standalone must not inherit old Legacy")
    contract = json.loads(show(repo, sha, source["manifest"]))
    if contract["component"] != source["component"] or contract["base_master_sha"] != meta["base_master_sha"]:
        raise ValueError("component/base mismatch")
    if contract["requires_mistral_branches"] or contract["candle"]["revision"] != meta["candle_sha"]:
        raise ValueError("unexpected branch/backend dependency")
    paths = git(repo, "diff", "--name-only", meta["base_master_sha"], sha).stdout.decode().splitlines()
    if not set(paths) <= set(contract["allowed_paths"]):
        raise ValueError("standalone scope changed")
    for path, expected in contract["payload_sha256"].items():
        if hashlib.sha256(show(repo, sha, path)).hexdigest() != expected:
            raise ValueError(f"standalone payload changed: {path}")
    return paths


def resolved_files(repo, meta):
    sources = {source["component"]: source["sha"] for source in meta["sources"]}
    result = {}
    for name in CRATES:
        path = f"{name}/Cargo.toml"
        content = show(repo, meta["base_master_sha"], path).decode()
        features = {}
        for component, sha in sources.items():
            data = tomllib.loads(show(repo, sha, path).decode())
            key = f"cuda-legacy-{component}"
            features[key] = data["features"][key]
        features["cuda-legacy"] = [f"cuda-legacy-{component}" for component in sources]
        additions = "".join(f"{key} = {json.dumps(value)}\n" for key, value in features.items())
        result[path] = replace_once(content, "[features]\n", "[features]\n" + additions)
    result["mistralrs-core/build.rs"] = show(repo, sources["bf16"], "mistralrs-core/build.rs").decode()
    path = "mistralrs-quant/build.rs"
    result[path] = replace_once(show(repo, sources["fp8"], path).decode(),
                                "let legacy_bf16 = false;",
                                'let legacy_bf16 = cfg!(feature = "cuda-legacy-bf16") && compute_cap == 61;')
    path = "mistralrs-paged-attn/build.rs"
    before = '    let allow_legacy_fp8 = cfg!(feature = "cuda-legacy-fp8") && compute_cap == 61;'
    result[path] = replace_once(show(repo, sources["fp8"], path).decode(), before,
        '    if cfg!(feature = "cuda-legacy-bf16") && compute_cap == 61 {\n'
        '        builder = builder.arg("-DALLOW_LEGACY_BF16");\n    }\n' + before)
    path = "mistralrs-core/src/cuda/moe.rs"
    text = show(repo, sources["bf16"], path).decode()
    text = replace_once(text, '#[cfg(all(test, feature = "cuda", feature = "cuda-legacy-bf16"))]',
        '#[cfg(all(test, feature = "cuda", any(feature = "cuda-legacy-bf16", feature = "cuda-legacy-fp8", feature = "cuda-legacy-fp4")))]')
    result[path] = replace_once(text, '.chain(Some(DType::BF16))',
                               '.chain(cfg!(feature = "cuda-legacy-bf16").then_some(DType::BF16))')
    return result


def verify_payload_union(repo, output, meta, source_paths):
    expected = {}
    for source in meta["sources"]:
        for path in source_paths[source["component"]]:
            if path not in RESOLVED:
                data = show(repo, source["sha"], path)
                if path in expected and expected[path] != data:
                    raise ValueError(f"unreviewed shared payload: {path}")
                expected[path] = data
    for path, data in expected.items():
        if (output / path).read_bytes() != data:
            raise ValueError(f"source payload lost during composition: {path}")


def compose(repo, output, meta, support_ref):
    if output.exists():
        raise ValueError("output must not exist")
    source_paths = {s["component"]: verify_source(repo, meta, s) for s in meta["sources"]}
    resolutions = resolved_files(repo, meta)
    git(repo, "worktree", "add", "--detach", str(output), meta["base_master_sha"])
    try:
        for source in meta["sources"]:
            merged = git(output, "merge", "--no-commit", "--no-ff", source["sha"], check=False)
            conflicts = set(git(output, "diff", "--name-only", "--diff-filter=U").stdout.decode().splitlines())
            if conflicts - RESOLVED or (merged.returncode and not conflicts):
                raise ValueError(f"unreviewed merge failure: {merged.stderr.decode()} {sorted(conflicts)}")
            for path, text in resolutions.items():
                (output / path).write_text(text)
            git(output, "add", ".")
            git(output, "-c", "user.name=Legacy V3 preparation", "-c", "user.email=ci@localhost",
                "commit", "-m", f"Prepare locked {source['component']} input")
        git(output, "merge", "--no-ff", "--no-commit", support_ref)
        rust = [str(output / path) for path in sorted(RESOLVED) if path.endswith(".rs")]
        subprocess.run(["rustfmt", "--edition", "2021", "--config", "skip_children=true", *rust], check=True)
        verify_payload_union(repo, output, meta, source_paths)
        allowed = set().union(*map(set, source_paths.values())) | RESOLVED
        allowed |= set(git(repo, "diff", "--name-only", meta["base_master_sha"], support_ref).stdout.decode().splitlines())
        changed = set(git(output, "diff", "--name-only", meta["base_master_sha"]).stdout.decode().splitlines())
        if changed - allowed:
            raise ValueError(f"unreviewed output paths: {sorted(changed - allowed)}")
        git(output, "add", ".")
        git(output, "-c", "user.name=Legacy V3 preparation", "-c", "user.email=ci@localhost",
            "commit", "-m", "Add residual support and cross-format contracts")
        print(json.dumps({"candidate_sha": git(output, "rev-parse", "HEAD").stdout.decode().strip(),
                          "sources": meta["sources"], "support_sha": support_ref,
                          "gpu_qualified": False}, indent=2))
    except Exception:
        print(f"Composition failed; worktree retained for inspection: {output}")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    meta = json.loads((ROOT / "legacy-v3-support/sources.json").read_text())
    support = git(ROOT, "rev-parse", "HEAD").stdout.decode().strip()
    for source in meta["sources"]:
        git(ROOT, "cat-file", "-e", source["sha"] + "^{commit}")
    compose(ROOT, args.output.resolve(), meta, support)


if __name__ == "__main__":
    main()
