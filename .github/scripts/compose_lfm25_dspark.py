#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

CANDLE_CRATES = (
    "candle-core", "candle-nn", "candle-flash-attn-v3", "candle-metal-kernels"
)
RETAIN_SEED_PATHS = (
    "Cargo.toml", "Cargo.lock",
    "mistral-integration/README.md", "mistral-integration/sources.json",
)
SOURCE_BRANCH = "lfm25_dspark_standalone"
MANIFEST_PATH = "mistral-integration/sources.json"


def git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], text=True, capture_output=True
    )
    if check and result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result


def normalized_candle_manifest(text):
    for crate in CANDLE_CRATES:
        pattern = rf"(?m)^{re.escape(crate)}\s*=\s*\{{[^\n]+\}}$"

        def normalize(match):
            line = re.sub(r'git\s*=\s*"[^"]+"', 'git = "<candle>"', match[0])
            return re.sub(r'rev\s*=\s*"[0-9a-f]{40}"', 'rev = "<pin>"', line)

        text, count = re.subn(pattern, normalize, text)
        if count != 1:
            raise RuntimeError(f"expected one Candle manifest entry for {crate}")
    return text


def compose(repo, source_sha, candle_sha):
    repo = Path(repo).resolve()
    if not all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (source_sha, candle_sha)):
        raise RuntimeError("source and Candle pins must be full commit SHAs")
    if git(repo, "status", "--porcelain").stdout:
        raise RuntimeError("composition requires a clean seed checkout")
    source_text = git(repo, "show", f"{source_sha}:{MANIFEST_PATH}").stdout
    source = json.loads(source_text)
    if source.get("schema_version") != 1 or source.get("kind") != "lfm25-dspark-standalone":
        raise RuntimeError("source is not an LFM2.5 DSpark standalone contract")
    if source.get("candle_sha") != candle_sha:
        raise RuntimeError("standalone source must use the validated Candle Extended pin")
    base = source["base_master_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise RuntimeError("invalid standalone master baseline")
    git(repo, "merge-base", "--is-ancestor", base, "HEAD")
    git(repo, "merge-base", "--is-ancestor", base, source_sha)
    source_cargo = git(repo, "show", f"{source_sha}:Cargo.toml").stdout
    seed_cargo = (repo / "Cargo.toml").read_text()
    if normalized_candle_manifest(source_cargo) != normalized_candle_manifest(seed_cargo):
        raise RuntimeError("standalone dependencies differ beyond the Candle source pin")

    result = git(
        repo, "-c", "user.name=github-actions[bot]",
        "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "merge", "--no-commit", "--no-ff", source_sha, check=False,
    )
    conflicts = set(git(repo, "diff", "--name-only", "--diff-filter=U").stdout.splitlines())
    unexpected = conflicts - set(RETAIN_SEED_PATHS)
    if unexpected:
        raise RuntimeError(f"functional merge conflicts: {', '.join(sorted(unexpected))}")
    if result.returncode and not conflicts:
        raise RuntimeError(f"standalone merge failed: {result.stderr.strip()}")
    # Cargo.lock is resolved after the workflow pins the validated Candle source.
    git(repo, "checkout", "HEAD", "--", *RETAIN_SEED_PATHS)
    if git(repo, "diff", "--name-only", "--diff-filter=U").stdout:
        raise RuntimeError("unresolved standalone composition conflicts")

    components = repo / "mistral-integration/components"
    components.mkdir(parents=True, exist_ok=True)
    (components / "lfm25-dspark.json").write_text(source_text)
    (components / "lfm25-dspark.md").write_text(
        git(repo, "show", f"{source_sha}:mistral-integration/README.md").stdout
    )
    path = repo / MANIFEST_PATH
    target = json.loads(path.read_text())
    component = {
        "branch": SOURCE_BRANCH,
        "source_sha": source_sha,
        "base_master_sha": base,
        "source_manifest_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
    }
    target.setdefault("components", {})["lfm25_dspark"] = component
    path.write_text(json.dumps(target, indent=2) + "\n")
    git(repo, "add", "--", MANIFEST_PATH, "mistral-integration/components")
    git(repo, "diff", "--cached", "--check")
    print(f"LFM25_DSPARK_COMPOSED={source_sha}")
    return component


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--candle-sha", required=True)
    args = parser.parse_args()
    compose(args.repo, args.source_sha, args.candle_sha)


if __name__ == "__main__":
    main()
