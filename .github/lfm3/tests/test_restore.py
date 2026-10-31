import importlib.util
from pathlib import Path
import shutil
import unittest
from unittest import mock

import test_qualification as fixtures

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("restore", ROOT / "restore.py")
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


class RestoreTests(unittest.TestCase):
    setUp = fixtures.QualificationTests.setUp
    evidence = fixtures.QualificationTests.evidence
    report = fixtures.QualificationTests.report
    def test_matching_model_proof_is_restored_without_model_execution(self):
        context, report = self.report()
        report['run_id'] = '987654321'
        restore.q.write(self.root / 'report.json', report)
        plan = self.root / 'plan.json'
        restore.q.write(plan, {'candidate': 'a' * 40, 'target': 'legacy'})
        artifacts = [{'name': 'lfm3-model-proof-' + report['fingerprint'], 'expired': False,
                      'workflow_run': {'id': 987654321}, 'digest': 'sha256:' + 'd' * 64}]
        def api(path):
            if path.startswith('/actions/artifacts?'): return {'artifacts': artifacts}
            return {'path': restore.q.WORKFLOW, 'event': 'workflow_dispatch', 'head_branch': 'mirror_orch', 'conclusion': 'success'}
        def download(_artifact, dest):
            dest.mkdir(parents=True)
            for p in self.root.iterdir():
                if p.is_file() and p.name != 'plan.json': shutil.copy2(p, dest / p.name)
        proofs = self.root / 'proofs'
        with mock.patch.object(restore.q, 'api', side_effect=api), mock.patch.object(restore.q, 'download_artifact', side_effect=download), \
             mock.patch.object(restore.q, 'source_fingerprint', return_value=context['runtimes']['legacy']['fingerprint']), \
             mock.patch.object(restore.q, 'attach') as attach:
            restore.restore(self.root, plan, proofs)
            attach.assert_called_once()
        self.assertTrue((proofs / 'model-evidence/model-proof/report.json').is_file())

    def test_changed_runtime_never_reuses_stale_model_proof(self):
        _, report = self.report()
        plan = self.root / 'plan.json'
        restore.q.write(plan, {'candidate': 'a' * 40, 'target': 'legacy'})
        artifacts = [{'name': 'lfm3-model-proof-' + report['fingerprint'], 'expired': False, 'workflow_run': {'id': 987654321}}]
        def api(path):
            if path.startswith('/actions/artifacts?'): return {'artifacts': artifacts}
            return {'path': restore.q.WORKFLOW, 'event': 'workflow_dispatch', 'head_branch': 'mirror_orch', 'conclusion': 'success'}
        def download(_artifact, dest):
            dest.mkdir(parents=True)
            for p in self.root.iterdir():
                if p.is_file() and p.name != 'plan.json': shutil.copy2(p, dest / p.name)
        with mock.patch.object(restore.q, 'api', side_effect=api), mock.patch.object(restore.q, 'download_artifact', side_effect=download), \
             mock.patch.object(restore.q, 'source_fingerprint', return_value='f' * 64), self.assertRaises(RuntimeError):
            restore.restore(self.root, plan, self.root / 'proofs')

    def test_untrusted_or_failed_workflow_is_not_downloaded(self):
        plan = self.root / 'plan.json'
        restore.q.write(plan, {'candidate': 'a' * 40, 'target': 'legacy'})
        def api(path):
            if path.startswith('/actions/artifacts?'):
                return {'artifacts': [{'name': 'lfm3-model-proof-test', 'expired': False, 'workflow_run': {'id': 123456789}}]}
            return {'path': restore.q.WORKFLOW, 'event': 'pull_request', 'head_branch': 'mirror_orch', 'conclusion': 'success'}
        with mock.patch.object(restore.q, 'api', side_effect=api), mock.patch.object(restore.q, 'download_artifact') as download, \
             mock.patch.object(restore.q, 'source_fingerprint', return_value='f' * 64), self.assertRaises(RuntimeError):
            restore.restore(self.root, plan, self.root / 'proofs')
        download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
