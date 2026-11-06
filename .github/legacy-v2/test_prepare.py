import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('legacy_v2_prepare', ROOT / 'prepare.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PreparationContracts(unittest.TestCase):
    def test_output_scope_is_separate_from_published_targets(self):
        manifest = p.read(ROOT / 'sources.json')
        self.assertEqual(set(p.OUTPUTS), set(manifest['outputs']))
        for output in ('legacy', 'extended', 'asd_runner_standalone', 'lfm25_dspark_standalone', 'nemotron_bolt_dflash_standalone'):
            self.assertNotIn(output, manifest['outputs'])
        for output in p.OUTPUTS[1:]:
            self.assertEqual(manifest['outputs'][output]['requires'], ['legacy_v2'])

    def test_all_inputs_are_frozen_and_support_has_no_workflow_or_component_metadata(self):
        m = p.read(ROOT / 'sources.json')
        for sha in m['sources'].values():
            self.assertRegex(sha, r'^[a-f0-9]{40}$')
        self.assertEqual(m['sources']['master'], m['master'])
        for path in m['legacy_support']['paths']:
            self.assertFalse(path.startswith('.github/'))
            self.assertNotEqual(path, p.META)
        for layer in ('legacy_support', 'fp8_packed'):
            self.assertEqual(p.hash_bytes((ROOT / m[layer]['patch']).read_bytes()), m[layer]['sha256'])

    def test_fp4_provenance_and_no_independent_sm61_prerequisite(self):
        m = p.read(ROOT / 'sources.json')
        self.assertTrue(m['fp4']['exclude_bolt_dflash_from_legacy'])
        self.assertEqual(m['fp4']['generic_source_blob'], '629cc55973a299825f8f84e91c980d08dd8958cf')
        self.assertEqual(m['outputs']['legacy_v2']['requires'], [])

    def test_identity_ignores_run_but_changes_with_recipe_or_cpu_flags(self):
        identity = {'sources': {'master': 'a' * 40}, 'recipe': 'x', 'cpu_flags': '-C target-cpu=x86-64'}
        self.assertEqual(p.key(identity), p.key(copy.deepcopy(identity)))
        for field in ('recipe', 'cpu_flags'):
            other = {**identity, field: 'changed'}
            self.assertNotEqual(p.key(identity), p.key(other))

    def fixture(self, path):
        plan = {'input_key': 'key', 'run_id': '123', 'run_attempt': '1', 'workflow_sha': 'f' * 40,
                'nodes': {name: {'sha': str(i) * 40, 'tree': 'a' * 40, 'cargo_lock_sha256': 'c' * 64}
                          for i, name in enumerate(p.OUTPUTS, 1)}}
        for name, node in plan['nodes'].items():
            report = {'target': name, 'candidate': node['sha'], 'tree': node['tree'],
                      'input_key': plan['input_key'], 'cargo_lock_sha256': node['cargo_lock_sha256'],
                      'run_id': plan['run_id'], 'run_attempt': plan['run_attempt'],
                      'workflow_sha': plan['workflow_sha'], 'command': p.CPU_COMMAND, 'status': 'PASS'}
            p.write(path / (name.replace('/', '-') + '.json'), report)
        return plan

    def test_accepts_only_evidence_for_all_exact_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            reports = Path(d)
            p.validate_attestations(self.fixture(reports), reports)

    def test_rejects_old_attempt_or_other_candidate_or_unlocked_command(self):
        for field, value in [('run_attempt', '2'), ('candidate', '0' * 40), ('command', ['cargo', 'check']), ('status', 'FAIL')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                reports = Path(d)
                plan = self.fixture(reports)
                target = reports / 'legacy_v2.json'
                report = p.read(target)
                report[field] = value
                p.write(target, report)
                with self.assertRaisesRegex(RuntimeError, 'mismatched CPU'):
                    p.validate_attestations(plan, reports)

    def test_real_atomic_publication_to_bare_remote(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            remote, repo, output, reports = [root / name for name in ('remote.git', 'repo', 'output', 'reports')]
            p.run(['git', 'init', '--bare', remote])
            p.run(['git', 'init', repo])
            p.git(repo, 'remote', 'add', 'origin', str(remote))
            (repo / 'Cargo.lock').write_text('test lock\n')
            p.git(repo, 'add', 'Cargo.lock')
            env = {**os.environ, **p.IDENTITY}
            import subprocess
            subprocess.run(['git', '-C', str(repo), 'commit', '-m', 'fixture'], env=env, check=True, capture_output=True)
            candidate = p.git(repo, 'rev-parse', 'HEAD').decode().strip()
            tree = p.git(repo, 'rev-parse', 'HEAD^{tree}').decode().strip()
            output.mkdir()
            identity = {'sources': {}, 'format_sources': True}
            nodes = {}
            for name in p.OUTPUTS:
                ref = 'refs/heads/integration/legacy-v2/test/' + name.replace('/', '-')
                p.git(repo, 'update-ref', ref, candidate)
                nodes[name] = {'sha': candidate, 'tree': tree, 'immutable_ref': ref,
                               'cargo_lock_sha256': p.hash_bytes((repo / 'Cargo.lock').read_bytes())}
            bundle = output / 'candidates.bundle'
            p.git(repo, 'bundle', 'create', str(bundle), *(node['immutable_ref'] for node in nodes.values()))
            plan = {'identity': identity, 'input_key': p.key(identity), 'nodes': nodes,
                    'leases': {name: None for name in p.OUTPUTS},
                    'bundle_sha256': p.hash_bytes(bundle.read_bytes()),
                    'run_id': '123', 'run_attempt': '1', 'workflow_sha': candidate}
            p.write(output / 'plan.json', plan)
            for name, node in nodes.items():
                p.write(reports / (name.replace('/', '-') + '.json'),
                        {'target': name, 'candidate': candidate, 'tree': tree, 'input_key': plan['input_key'],
                         'cargo_lock_sha256': node['cargo_lock_sha256'], 'run_id': '123', 'run_attempt': '1',
                         'workflow_sha': candidate, 'command': p.CPU_COMMAND, 'status': 'PASS'})
            with patch.object(p, 'verify_sources'), patch.dict(os.environ, {'GITHUB_RUN_ID': '123',
                    'GITHUB_RUN_ATTEMPT': '1', 'GITHUB_SHA': candidate}):
                p.publish(repo, output, reports)
            refs = p.git(repo, 'ls-remote', '--heads', 'origin').decode().splitlines()
            self.assertEqual(len(refs), 6)
            for name in p.OUTPUTS:
                self.assertEqual(p.head(repo, name), candidate)
            self.assertIsNone(p.head(repo, 'legacy'))
            self.assertIsNone(p.head(repo, 'extended'))

    def test_rejects_bundle_tampering_before_git_import(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            identity = {'sources': 'test'}
            p.write(folder / 'plan.json', {'nodes': {n: {} for n in p.OUTPUTS},
                    'leases': {n: None for n in p.OUTPUTS}, 'identity': identity,
                    'input_key': p.key(identity), 'bundle_sha256': '0' * 64})
            (folder / 'candidates.bundle').write_bytes(b'tampered')
            with self.assertRaisesRegex(RuntimeError, 'bundle checksum'):
                p.verify_bundle(folder, folder)

    def test_rejects_moved_inputs_before_fetch(self):
        with patch.object(p, 'head', return_value='b' * 40), patch.object(p, 'git') as git:
            with self.assertRaisesRegex(RuntimeError, 'source moved'):
                p.verify_sources(Path('.'), {'sources': {'master': 'a' * 40}})
            git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
