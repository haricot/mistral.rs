"""Rename the qualified branch without changing any source commit or artifact."""
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = "haricot/mistral.rs"
OLD = "allow_old_card"
NEW = "legacy"
EXPECTED = "e4f7349cf973b5603ccbfe344e90c4d019a62754"
TREE = "dc68441d5a78e4daf7fd06ace4087c567bf1f01f"
CAMPAIGNS = {".github/workflows/mistral-cuda-legacy-sm61.yml",
             ".github/workflows/mistral-extended-sm61.yml",
             ".github/workflows/lfm3-model-sm61-qualification.yml",
             ".github/workflows/lfm3-model-sm61-promote.yml"}


def api(method, path, data=None, missing=False):
    command = ["gh", "api", "-X", method, "/repos/" + REPOSITORY + path]
    if data is not None:
        command += ["--input", "-"]
    result = subprocess.run(command, input=json.dumps(data) if data is not None else None,
                            text=True, capture_output=True)
    if result.returncode:
        if missing and "HTTP 404" in result.stderr:
            return None
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def identity(branch):
    if (branch["commit"]["sha"] != EXPECTED or
            branch["commit"]["commit"]["tree"]["sha"] != TREE):
        raise ValueError("qualified Legacy branch changed; refuse stale rename")


def migrate(client=api):
    for status in ("queued", "in_progress", "waiting", "pending"):
        runs = client("GET", "/actions/runs?status=" + status + "&per_page=100")
        if runs["total_count"] != len(runs["workflow_runs"]):
            raise ValueError("incomplete active workflow listing")
        if any(run["path"] in CAMPAIGNS for run in runs["workflow_runs"]):
            raise ValueError("a source campaign is active; do not rename or cancel it")
    old_ref = client("GET", "/git/ref/heads/" + OLD, missing=True)
    old = client("GET", "/branches/" + OLD) if old_ref is not None else None
    new = client("GET", "/branches/" + NEW, missing=True)
    if new is not None:
        identity(new)
        if old is not None:
            raise ValueError("both branch names exist; refuse an ambiguous migration")
    else:
        if old is None:
            raise ValueError("qualified source branch missing")
        identity(old)
        client("POST", "/branches/" + OLD + "/rename", {"new_name": NEW})
        identity(client("GET", "/branches/" + NEW))
        if client("GET", "/git/ref/heads/" + OLD, missing=True) is not None:
            raise ValueError("old Git ref still exists after rename")
    return {"status": "PASS", "old_name": OLD, "new_name": NEW,
            "commit": EXPECTED, "tree": TREE, "source_changed": False,
            "binaries_rebuilt": False, "gpu_started": False,
            "historical_evidence_rewritten": False}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("output path required")
    report = migrate()
    Path(sys.argv[1]).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("LEGACY_BRANCH_RENAME=PASS candidate=" + EXPECTED)
