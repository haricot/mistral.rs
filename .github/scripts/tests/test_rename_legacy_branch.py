import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("rename", Path(__file__).parents[1] / "rename-legacy-branch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RenameTests(unittest.TestCase):
    def setUp(self):
        self.branch = {"commit":{"sha":m.EXPECTED,"commit":{"tree":{"sha":m.TREE}}}}
        self.old = copy.deepcopy(self.branch)
        self.new = None
        self.posts = []
        self.active = []

    def client(self, method, path, data=None, missing=False):
        if path.startswith("/actions/runs?"):
            return {"total_count":len(self.active),"workflow_runs":self.active}
        if method == "POST":
            self.assertEqual(path,"/branches/allow_old_card/rename")
            self.assertEqual(data,{"new_name":"legacy"})
            self.posts.append(data);self.new=self.old;self.old=None
            return self.new
        if path in ("/branches/allow_old_card","/git/ref/heads/allow_old_card"):
            return self.old
        if path == "/branches/legacy":return self.new
        raise AssertionError(path)

    def test_exact_commit_rename_and_idempotent_retry(self):
        report=m.migrate(self.client)
        self.assertEqual(report["commit"],m.EXPECTED)
        self.assertFalse(report["source_changed"])
        self.assertEqual(m.migrate(self.client),report)
        self.assertEqual(len(self.posts),1)

    def test_changed_source_or_conflicting_destination_is_rejected(self):
        self.old["commit"]["sha"]="a"*40
        with self.assertRaises(ValueError):m.migrate(self.client)
        self.old=copy.deepcopy(self.branch);self.new=copy.deepcopy(self.branch)
        with self.assertRaises(ValueError):m.migrate(self.client)
        self.assertFalse(self.posts)

    def test_dedicated_gpu_campaign_blocks_rename_without_cancellation(self):
        self.active=[{"path":".github/workflows/mistral-extended-sm61.yml"}]
        with self.assertRaisesRegex(ValueError,"source campaign is active"):m.migrate(self.client)
        self.assertFalse(self.posts)

    def test_unrelated_upstream_workflow_is_left_alone(self):
        self.active=[{"path":".github/workflows/ci_cuda.yaml"}]
        self.assertEqual(m.migrate(self.client)["status"],"PASS")
