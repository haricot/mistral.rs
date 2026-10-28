#!/usr/bin/env python3
"""Publish immutable, merge-clean standalone candidates. NEVER promote production branches."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{40}$")
BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

def run(repo, *args, check=True, env=None):
    p = subprocess.run(["git", "-C", str(repo), *args], text=True,
                       capture_output=True, env=env, check=False)
    if check and p.returncode:
        raise RuntimeError(f"git {args[0]} failed: {p.stderr[-800:]}")
    return p

def oid(repo, ref):
    result = run(repo, "rev-parse", "--verify", ref + "^{commit}").stdout.strip()
    if not SHA.fullmatch(result):
        raise RuntimeError(f"invalid pinned commit: {ref}")
    return result

def remote(repo, branch):
    result = run(repo, "ls-remote", "--heads", "origin", f"refs/heads/{branch}").stdout.strip()
    if not result:
        return None
    value = result.split()[0]
    if not SHA.fullmatch(value):
        raise RuntimeError(f"invalid remote head for {branch}")
    return value

def staged_ref(master, source, branch):
    if not SHA.fullmatch(master) or not SHA.fullmatch(source) or not BRANCH.fullmatch(branch):
        raise RuntimeError("invalid immutable staging identity")
    return f"integration/staged-master/{master[:12]}/{source[:12]}/{branch}"

def commit_tree(repo, tree, source, master, branch):
    for item in (tree, source, master):
        if not SHA.fullmatch(item):
            raise RuntimeError("invalid staging git object")
    timestamp = max(int(run(repo, "show", "-s", "--format=%ct", parent).stdout)
                    for parent in (source, master)) + 1
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Haricot Mistral Staging",
        "GIT_AUTHOR_EMAIL": "344493+haricot@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "Haricot Mistral Staging",
        "GIT_COMMITTER_EMAIL": "344493+haricot@users.noreply.github.com",
        "GIT_AUTHOR_DATE": f"{timestamp} +0000",
        "GIT_COMMITTER_DATE": f"{timestamp} +0000",
    }
    message = (f"build: stage {branch} against master {master}\n\n"
               f"Standalone-Source-SHA: {source}\n"
               f"Upstream-Master-SHA: {master}\n"
               "Qualification: NOT_EXECUTED\nPromotion: PROHIBITED\n")
    return run(repo, "commit-tree", tree, "-p", source, "-p", master,
               "-m", message, env=env).stdout.strip()

def stage(repo, report, dry_run=False):
    master = report["source_master_sha"]
    if not SHA.fullmatch(master) or report.get("mirror_contains_master") is not True:
        raise RuntimeError("master is not synchronized into mirror_orch")
    if report.get("promotions") != [] or report.get("gpu_tests_executed") is not False:
        raise RuntimeError("unexpected qualifications or promotions in read-only source evidence")
    if remote(repo, "master") != master:
        raise RuntimeError("master head moved since impact inspection")
    current_mirror = remote(repo, "mirror_orch")
    if current_mirror is None or run(
        repo, "merge-base", "--is-ancestor", master, current_mirror, check=False
    ).returncode != 0:
        raise RuntimeError("remote mirror does not contain inspected master")
    entries = report["standalones"]
    if len({e["branch"] for e in entries}) != len(entries):
        raise RuntimeError("duplicate branches in impact report")
    candidates = []
    for node in entries:
        branch = node["branch"]
        source = node["branch_sha"]
        if not BRANCH.fullmatch(branch) or not SHA.fullmatch(source):
            raise RuntimeError("invalid source identity")
        if remote(repo, branch) != source:
            raise RuntimeError(f"source branch moved since inspection: {branch}")
        if node["status"] == "UP_TO_DATE":
            continue
        if node["status"] not in ("OVERLAP_REVIEW_UNTESTED", "MERGE_CLEAN_UNTESTED"):
            continue
        if node["textual_merge_clean"] is not True:
            raise RuntimeError(f"unclean candidate marked merge-clean: {branch}")
        result = run(repo, "merge-tree", "--write-tree", source, master, check=False)
        if result.returncode:
            raise RuntimeError(f"merge-tree status changed for {branch}")
        tree = result.stdout.splitlines()[0].strip()
        if not SHA.fullmatch(tree):
            raise RuntimeError(f"merge-tree produced no tree for {branch}")
        ref = staged_ref(master, source, branch)
        candidates.append({"branch": branch, "source_sha": source,
                           "master_sha": master, "tree_sha": tree, "ref": ref})
    published = []
    for candidate in candidates:
        sha = commit_tree(repo, candidate["tree_sha"], candidate["source_sha"],
                          master, candidate["branch"])
        if not SHA.fullmatch(sha):
            raise RuntimeError("invalid new staged commit")
        prior = remote(repo, candidate["ref"])
        if prior is not None and prior != sha:
            raise RuntimeError("immutable candidate ref collision")
        if prior is None and not dry_run:
            run(repo, "push", "origin", f"{sha}:refs/heads/{candidate['ref']}")
            if remote(repo, candidate["ref"]) != sha:
                raise RuntimeError("staged candidate publication not verified")
        published.append({**candidate, "candidate_sha": sha,
                          "status": "DRY_RUN" if dry_run else
                              ("ALREADY_STAGED" if prior else "STAGED_UNTESTED"),
                          "qualification": "NOT_EXECUTED",
                          "promotion": False})
    return {
        "schema_version": 1, "master_sha": master, "mirror_sha": current_mirror,
        "scope": "IMMUTABLE_CANDIDATES_ONLY", "dry_run": dry_run,
        "candidates": published,
        "skipped": [{"branch": n["branch"], "status": n["status"]}
                    for n in entries if n["branch"] not in
                    {c["branch"] for c in published}],
        "source_branches_modified": False,
        "gpu_tests_executed": False,
        "promotions": [],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    result = stage(args.repo.resolve(), json.loads(args.report.read_text()),
                   dry_run=args.dry_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if args.summary:
        with args.summary.open("a") as f:
            f.write("\n### Immutable standalone staging (NOT QUALIFIED)\n")
            for item in result["candidates"]:
                f.write(f"- \`{item['branch']}\`: \`{item['ref']}\`"
                        f" — {item['status']}\n")
            for item in result["skipped"]:
                f.write(f"- \`{item['branch']}\`: not staged ({item['status']})\n")
            f.write("\nNo production standalone was modified; no GPU tests ran.\n")
    print("STANDALONE_CANDIDATES_STAGED=" + str(len(result["candidates"])))
    for item in result["candidates"]:
        print(f"{item['branch']}: {item['status']} {item['candidate_sha']}")

if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyError, ValueError, IndexError) as e:
        print(f"STANDALONE_STAGING_ERROR: {e}", file=sys.stderr)
        sys.exit(1)
