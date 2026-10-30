import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import call, patch


spec = importlib.util.spec_from_file_location("candle_proof", Path(__file__).resolve().parents[1] / "verify-candle.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandleProofTests(unittest.TestCase):
    def setUp(self):
        self.ref = "refs/heads/archive/legacy/sm61-validated-123-1"
        self.sha = "a" * 40
        self.plan = {"target": "legacy", "candle": {"cuda_legacy": {"sha": self.sha, "validation_ref": self.ref}}}
        self.responses = [{"ref": self.ref, "object": {"type": "commit", "sha": self.sha}},
                          {"status": "completed", "conclusion": "success"}]

    def test_exact_physical_commit_is_required(self):
        with patch.object(module, "get", side_effect=self.responses):
            module.verify(self.plan)
        self.responses[0]["object"]["sha"] = "b" * 40
        with patch.object(module, "get", side_effect=self.responses):
            with self.assertRaisesRegex(RuntimeError, "another commit"):
                module.verify(self.plan)

    def test_failed_physical_run_is_rejected(self):
        self.responses[1]["conclusion"] = "failure"
        with patch.object(module, "get", side_effect=self.responses):
            with self.assertRaisesRegex(RuntimeError, "not successful"):
                module.verify(self.plan)

    def test_extended_requires_both_candle_variants(self):
        self.plan["target"] = "extended"
        with self.assertRaisesRegex(RuntimeError, "missing pinned"):
            module.verify(self.plan)

    def test_legacy_cannot_use_an_extended_proof_name(self):
        self.plan["candle"]["cuda_legacy"]["validation_ref"] = self.ref.replace("/legacy/", "/extended/")
        with self.assertRaisesRegex(RuntimeError, "invalid Candle"):
            module.verify(self.plan)

    def archived_responses(self):
        ref = "refs/tags/archive/candle-pruned-20261016/archive/legacy/sm61-validated-38898658386-1"
        self.plan["candle"]["cuda_legacy"]["validation_ref"] = ref
        return [
            {"ref": ref, "object": {"type": "tag", "sha": "b" * 40}},
            {"object": {"type": "commit", "sha": self.sha}},
            {"status": "completed", "conclusion": "success"},
        ]

    def test_pruned_annotated_tag_keeps_exact_commit_and_physical_run_checks(self):
        responses = self.archived_responses()
        with patch.object(module, "get", side_effect=responses) as get:
            module.verify(self.plan)
        self.assertEqual(get.call_args_list, [
            call("/git/ref/tags/archive/candle-pruned-20261016/archive/legacy/sm61-validated-38898658386-1"),
            call("/git/tags/" + "b" * 40),
            call("/actions/runs/38898658386"),
        ])

    def test_pruned_tag_for_another_commit_is_rejected(self):
        responses = self.archived_responses()
        responses[1]["object"]["sha"] = "c" * 40
        with patch.object(module, "get", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "another commit"):
                module.verify(self.plan)

    def test_pruned_tag_does_not_bypass_failed_physical_workflow(self):
        responses = self.archived_responses()
        responses[2]["conclusion"] = "failure"
        with patch.object(module, "get", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "not successful"):
                module.verify(self.plan)

    def test_unknown_archive_namespace_is_rejected_before_any_api_request(self):
        for prefix in ("refs/tags/archive/unknown/archive/", "refs/heads/archive/candle-pruned-20261016/archive/"):
            with self.subTest(prefix=prefix):
                self.plan["candle"]["cuda_legacy"]["validation_ref"] = prefix + "legacy/sm61-validated-38898658386-1"
                with patch.object(module, "get") as get:
                    with self.assertRaisesRegex(RuntimeError, "invalid Candle"):
                        module.verify(self.plan)
                    get.assert_not_called()

    def test_archived_legacy_proof_cannot_satisfy_extended_identity(self):
        self.archived_responses()
        entry = self.plan["candle"]["cuda_legacy"]
        self.plan = {"target": "extended", "candle": {"cuda_legacy": entry, "extended": entry}}
        with patch.object(module, "get", side_effect=self.archived_responses()):
            with self.assertRaisesRegex(RuntimeError, "invalid Candle"):
                module.verify(self.plan)

    def test_hosted_preparation_stops_before_docker_when_candle_proof_fails(self):
        script = Path(__file__).resolve().parents[1] / "prepare-hosted.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "commands.jsonl"
            shim = root / "python3"
            shim.write_text("#!/usr/bin/env bash\n"
                            'printf "%s\\n" "$*" >> "$COMMAND_LOG"\n'
                            '[[ "$1" != .github/campaign/verify-candle.py ]] || exit 9\n')
            shim.chmod(0o755)
            docker = root / "docker"
            docker.write_text('#!/usr/bin/env bash\nprintf "DOCKER\\n" >> "$COMMAND_LOG"\n')
            docker.chmod(0o755)
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"],
                       COMMAND_LOG=str(log), RUNNER_TEMP=str(root), GITHUB_WORKSPACE=str(root),
                       CAMPAIGN_TARGET="legacy", CI_CUDA_IMAGE="image@sha256:" + "0" * 64)
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
            commands = log.read_text().splitlines()
            self.assertEqual(len(commands), 2)
            self.assertIn("campaign.py snapshot", commands[0])
            self.assertIn("verify-candle.py", commands[1])

    def test_both_workflows_authenticate_the_hosted_candle_proof_check(self):
        root = Path(__file__).resolve().parents[2]
        expected = """      - name: Prepare independent sources and lock composition
        id: prepare
        env:
          GH_TOKEN: ${{ github.token }}
        run: bash .github/campaign/prepare-hosted.sh"""
        for name in ("mistral-cuda-legacy-sm61.yml", "mistral-extended-sm61.yml"):
            self.assertIn(expected, (root / "workflows" / name).read_text())
