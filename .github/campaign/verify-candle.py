#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import sys
import urllib.request


def get(path):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "haricot-sm61-campaign"}
    if os.environ.get("GH_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
    request = urllib.request.Request("https://api.github.com/repos/haricot/candle" + path, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def verify(plan):
    expected_branches = {"cuda_legacy"} if plan["target"] == "legacy" else {"cuda_legacy", "extended"}
    if set(plan["candle"]) != expected_branches:
        raise RuntimeError("missing pinned Candle qualification")
    for branch, entry in plan["candle"].items():
        kind = "legacy" if branch == "cuda_legacy" else "extended"
        refname = entry["validation_ref"]
        match = re.fullmatch(r"refs/(?:heads|tags)/archive/" + kind + r"/sm61-validated-(\d+)(?:-\d+)?", refname)
        if not match or not re.fullmatch("[a-f0-9]{40}", entry["sha"]):
            raise RuntimeError("invalid Candle physical proof identity")
        ref = get("/git/ref/" + refname.removeprefix("refs/"))
        if ref.get("ref") != refname:
            raise RuntimeError("Candle physical proof ref mismatch")
        obj = ref.get("object", {})
        for _ in range(4):
            if obj.get("type") != "tag":
                break
            if not re.fullmatch("[a-f0-9]{40}", str(obj.get("sha", ""))):
                raise RuntimeError("invalid annotated validation tag")
            obj = get("/git/tags/" + obj["sha"]).get("object", {})
        if obj.get("type") != "commit" or obj.get("sha") != entry["sha"]:
            raise RuntimeError("Candle physical proof is for another commit")
        run = get("/actions/runs/" + match.group(1))
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            raise RuntimeError("Candle physical campaign is not successful")
    print("PINNED_CANDLE_PHYSICAL_PROOFS=PASS")


if __name__ == "__main__":
    verify(json.loads(Path(sys.argv[1]).read_text()))
