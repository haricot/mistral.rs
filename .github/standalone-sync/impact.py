#!/usr/bin/env python3
"""Read-only master/standalone merge impact inspection (no builds or branch writes)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def git(repo, *args, check=True):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[:600]}")
    return result


def sha(repo, ref):
    value = git(repo, "rev-parse", "--verify", ref + "^{commit}").stdout.strip()
    if len(value) != 40 or any(x not in "0123456789abcdef" for x in value):
        raise RuntimeError("invalid Git source identity")
    return value


def contains(repo, older, newer):
    return git(repo, "merge-base", "--is-ancestor", older, newer, check=False).returncode == 0


def changed_paths(repo, left, right):
    result = git(repo, "diff", "--name-only", left, right)
    return set(filter(None, result.stdout.splitlines()))


def inspect_branch(repo, branch, master_sha, prefixes):
    branch_sha = sha(repo, f"refs/remotes/origin/{branch}")
    merge_base = git(repo, "merge-base", branch_sha, master_sha).stdout.strip()
    # An unrelated history is not rebased automatically, even if merge-tree
    # happens to report a clean tree in future git versions.
    if not merge_base:
        raise RuntimeError(f"{branch}: missing common ancestor with master")
    upstream_paths = changed_paths(repo, merge_base, master_sha)
    own_paths = changed_paths(repo, merge_base, branch_sha)
    overlap = sorted(upstream_paths & own_paths)
    high_impact = sorted(path for path in upstream_paths if any(
        path == prefix or path.startswith(prefix) for prefix in prefixes
    ))
    current = contains(repo, master_sha, branch_sha)
    # git merge-tree computes conflicts without changing the index/worktree,
    # the original ref, or publishing any candidate.
    merge_result = None if current else git(
        repo, "merge-tree", "--write-tree", branch_sha, master_sha, check=False
    )
    clean = True if current else merge_result.returncode == 0
    raw_meta = git(repo, "show", f"{branch_sha}:mistral-integration/sources.json",
                   check=False)
    try:
        metadata = json.loads(raw_meta.stdout) if raw_meta.returncode == 0 else {}
    except json.JSONDecodeError:
        metadata = {}
    return {
        "branch": branch,
        "branch_sha": branch_sha,
        "master_in_branch": current,
        "recorded_master_sha": metadata.get("base_master_sha"),
        "merge_base_sha": merge_base,
        "textual_merge_clean": clean,
        "needs_recomposition": not current,
        "upstream_changed_path_count": len(upstream_paths),
        "high_impact_paths": high_impact[:30],
        "high_impact_path_count": len(high_impact),
        "overlapping_paths": overlap[:30],
        "overlap_count": len(overlap),
        "status": "UP_TO_DATE" if current else (
            "MERGE_REVIEW" if not clean or overlap else
            "MERGE_CLEAN_UNTESTED"
        ),
        "qualification": "NOT_EXECUTED",
    }


def inspect(repo, config):
    master_sha = sha(repo, f"refs/remotes/origin/{config['master_branch']}")
    mirror_sha = sha(repo, f"refs/remotes/origin/{config['orchestrator_branch']}")
    entries = [inspect_branch(repo, branch, master_sha, config["high_impact_prefixes"])
               for branch in config["tracked_standalones"]]
    return {
        "schema_version": 1,
        "mode": "READ_ONLY",
        "source_master_sha": master_sha,
        "orchestrator_sha": mirror_sha,
        "mirror_contains_master": contains(repo, master_sha, mirror_sha),
        "standalones": entries,
        "gpu_tests_executed": False,
        "source_branches_modified": False,
        "promotions": [],
    }


def render_summary(report):
    lines = [
        "### Mistral master → standalone compatibility (read only)",
        f"Master: \`{report['source_master_sha']}\`",
        f"Mirror contains master: **{report['mirror_contains_master']}**",
        "",
        "| Branch | State | Textual merge | Overlap paths | High-impact upstream paths |",
        "|---|---|---|---:|---:|",
    ]
    for node in report["standalones"]:
        lines.append(
            f"| \`{node['branch']}\` | {node['status']} | "
            f"{'clean' if node['textual_merge_clean'] else 'conflicts'} | "
            f"{node['overlap_count']} | {node['high_impact_path_count']} |"
        )
    lines.extend([
        "",
        "**No standalone was updated.** A clean textual merge is NOT a "
        "compiled or GPU-qualified binary. Full-model LFM3/SM61 tests remain "
        "required before promotion.",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--require-mirror-current", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema_version") != 1:
        raise RuntimeError("unsupported standalone watch configuration schema")
    branches = config["tracked_standalones"]
    if len(branches) != len(set(branches)) or not branches:
        raise RuntimeError("missing/duplicate standalone watch entries")
    report = inspect(args.repo.resolve(), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("a") as out:
            out.write(render_summary(report))
    print(f"STANDALONE_UPSTREAM_IMPACT=OK master={report['source_master_sha']}"
          f" mirror_current={report['mirror_contains_master']}")
    for entry in report["standalones"]:
        print(f"{entry['branch']}: {entry['status']} "
              f"overlaps={entry['overlap_count']}")
    if args.require_mirror_current and not report["mirror_contains_master"]:
        raise RuntimeError("mirror_orch does not yet contain master: run master sync first")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"STANDALONE_IMPACT_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
