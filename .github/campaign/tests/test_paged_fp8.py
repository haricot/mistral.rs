import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compat_paged_fp8 as patch
import fp8_kv
import campaign
import release_r1


class PagedFp8Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self):
        files = {
            "mistralrs-core/src/paged_attention/cache_engine.rs": (ROOT.parents[1] / "mistralrs-core/src/paged_attention/cache_engine.rs").read_text(),
            "mistralrs-paged-attn/src/cuda/mod.rs": 'pub const USE_FP8: bool = cfg!(has_fp8);\n',
            "mistralrs-paged-attn/build.rs": '\n'.join(
                f'fn main{index}() {{\n    println!("cargo::rustc-check-cfg=cfg(has_fp8)");\n' +
                ('    if using_fp8 {\n        println!("cargo:rustc-cfg=has_fp8");\n    }\n' if index == 0 else '') + '}\n'
                for index in range(3)),
        }
        for name, content in files.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def test_patch_is_idempotent_and_binds_runtime_to_explicit_sm61_build(self):
        self.fixture()
        self.assertTrue(patch.install(self.root))
        build = (self.root / "mistralrs-paged-attn/build.rs").read_text()
        self.assertIn('using_fp8 && allow_legacy_fp8 && compute_cap == 61', build)
        self.assertEqual(build.count('rustc-check-cfg=cfg(has_fp8_legacy_sm61)'), 3)
        engine = (self.root / "mistralrs-core/src/paged_attention/cache_engine.rs").read_text()
        self.assertIn('CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR', engine)
        self.assertIn('mistralrs_paged_attn::USE_FP8_LEGACY_SM61', engine)
        before = {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertFalse(patch.install(self.root))
        self.assertEqual(before, {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        if shutil.which('rustfmt'):
            subprocess.run(['rustfmt', '--edition', '2021', str(self.root / 'mistralrs-core/src/paged_attention/cache_engine.rs')], check=True)
            self.assertFalse(patch.install(self.root))

    def test_changed_upstream_gate_requires_review(self):
        self.fixture()
        path = self.root / "mistralrs-core/src/paged_attention/cache_engine.rs"
        path.write_text(path.read_text().replace('major >= 8', 'major >= 9'))
        with self.assertRaisesRegex(RuntimeError, 'paged FP8 contract'):
            patch.install(self.root)

    @unittest.skipUnless(shutil.which('rustc'), 'Rust compiler required')
    def test_actual_arch_selector_on_cpu_including_missing_kernels_and_legacy_opt_in(self):
        source = self.root / 'matrix.rs'
        source.write_text('mod fp8_support {\n' + (ROOT / 'paged_fp8_support.rs').read_text() + '\n}\n')
        binary = self.root / 'matrix'
        subprocess.run(['rustc', '--edition=2021', '--test', source, '-o', binary], check=True, capture_output=True)
        result = subprocess.run([binary], check=True, capture_output=True, text=True)
        self.assertIn('1 passed', result.stdout)

    def report(self):
        case = {'query_heads':32, 'kv_heads':8, 'head_dim':64, 'exact_scaled_cache_roundtrip':True,
                'fp16_reference_max_abs_error':0.001, 'fp8_reference_max_abs_error':0.001,
                'fp8_fp16_max_abs_error':0, 'decode_speed_ratio_fp16_over_fp8':0.5}
        for name in fp8_kv.METRICS:
            value = 2.0 if name.startswith('fp8') else 1.0
            case[name+'_us'] = [value] * 9
            case[name+'_median_us'] = value
        cases = []
        for context in fp8_kv.CONTEXTS:
            item = copy.deepcopy(case)
            item.update(context=context, fp16_cache_bytes=context*8*64*4, fp8_cache_bytes=context*8*64*2)
            cases.append(item)
        return {'schema':1, 'status':'PASS', 'scope':'synthetic-paged-kv-cache', 'full_model_benchmark':False,
                'candidate':'a'*40, 'component':'legacy', 'run_id':'123', 'run_attempt':'1',
                'samples':9, 'iterations_per_sample':8, 'warmup':4, 'performance_threshold':None, 'cases':cases}

    def verify(self, report):
        path = self.root / 'legacy-fp8-kv.json'
        path.write_text(json.dumps(report))
        return fp8_kv.verify(path, {'candidate':'a'*40, 'component':'legacy'}, '123', '1')

    def test_slower_fp8_is_recorded_without_inventing_a_performance_gate(self):
        self.assertEqual(len(self.verify(self.report())), 64)

    def test_tampered_measurements_and_other_attempts_are_rejected(self):
        for change in ('attempt', 'candidate', 'memory', 'nan', 'median', 'parity', 'scope'):
            report = self.report()
            if change == 'attempt': report['run_attempt'] = '2'
            if change == 'candidate': report['candidate'] = 'b'*40
            if change == 'memory': report['cases'][0]['fp8_cache_bytes'] *= 2
            if change == 'nan': report['cases'][0]['fp8_decode_us'][0] = float('nan')
            if change == 'median': report['cases'][0]['fp8_decode_median_us'] = 1.0
            if change == 'parity': report['cases'][0]['fp8_reference_max_abs_error'] = 0.02
            if change == 'scope': report['full_model_benchmark'] = True
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.verify(report)

    def test_master_candle_comments_migration_keeps_one_pinned_dependency_per_crate(self):
        actual = ''.join(f'{name} = {{ version = "0.11.1" }}\n' for name in campaign.CANDLE_CRATES)
        old = ''.join(f'{name} = {{ version = "0.11.0", git = "old", rev = "a" }}\n' for name in campaign.CANDLE_CRATES)
        comments = ''.join(f'# {name} = {{ version = "0.11.0", git = "upstream", rev = "b" }}\n' for name in campaign.CANDLE_CRATES)
        text = '[workspace.dependencies]\nzip = { version = "9.1.3", default-features = false, features = ["deflate"] }\n' + actual + '<<<<<<< HEAD\n' + old + '=======\n' + comments + '>>>>>>> master\n'
        resolved = campaign.resolve_candle_conflicts(text)
        self.assertNotIn('<<<<<<<', resolved)
        path = self.root / 'Cargo.toml'
        path.write_text(resolved)
        campaign.pin_candle(self.root, 'c'*40, '0.11.0')
        pinned = path.read_text()
        campaign.pin_candle(self.root, 'c'*40, '0.11.0')
        self.assertEqual(path.read_text(), pinned)
        import tomllib
        dependencies = tomllib.loads(path.read_text())['workspace']['dependencies']
        self.assertEqual(dependencies['zip']['version'], '9.1.3')
        self.assertEqual(dependencies['zip']['rev'], 'af2ed2a3e12af1fdd733b1886b3e9067bf510be5')
        for name in campaign.CANDLE_CRATES:
            self.assertEqual(dependencies[name]['version'], '0.11.0')
            self.assertEqual(dependencies[name]['rev'], 'c'*40)
        self.assertIn('<<<<<<<', campaign.resolve_candle_conflicts(text.replace('0.11.1', '0.12.0')))

    def inference_fixture(self, cache):
        q = release_r1.model_module()
        (self.root / 'config.json').write_text(json.dumps({'num_hidden_layers':24}))
        process = mock.Mock()
        process.poll.return_value = None
        def http(_port, endpoint, _payload=None):
            if endpoint == '/v1/models': return {'data':[{'id':'text-fixture'}]}
            return {'usage':{'prompt_tokens':12, 'completion_tokens':32}, 'choices':[{'message':{'content':'fixture'}}]}
        with mock.patch.object(q.subprocess, 'Popen', return_value=process) as popen, mock.patch.object(q, 'http', side_effect=http) as requests:
            result = q.infer(self.root / 'mistralrs', 'text', 'cuda', {'text':self.root},
                             {'dtype':'f16','max_model_len':2048,'max_tokens':32,'top_logprobs':5},
                             self.root / 'server.log', paged_cache_type=cache)
        process.terminate.assert_called_once()
        return q, result, popen.call_args.args[0], requests

    def test_real_model_perf_invocation_is_offline_bounded_and_has_warmup(self):
        q, result, command, requests = self.inference_fixture('f8e4m3')
        self.assertEqual(requests.call_count, 1 + q.KV_BENCH_WARMUP + q.KV_BENCH_REPEATS)
        summary = release_r1.paged_summary(result, 'f8e4m3', q)
        self.assertEqual(summary['completion_tokens'], [32,32,32])
        self.assertIn('--pa-cache-type', command)
        result['paged_kv']['command'][command.index('--paged-attn')+1] = 'off'
        with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
            release_r1.paged_summary(result, 'f8e4m3', q)

    def test_existing_numeric_reference_still_disables_paged_attention(self):
        _, result, command, requests = self.inference_fixture(None)
        self.assertEqual(command[command.index('--paged-attn')+1], 'off')
        self.assertEqual(requests.call_count, 2)
        self.assertNotIn('paged_kv', result)

    def test_distinct_rerere_variants_are_part_of_the_immutable_resolution_identity(self):
        directory = self.root / 'rr-cache/common/conflict'
        directory.mkdir(parents=True)
        (directory / 'preimage').write_text('original')
        (directory / 'postimage').write_text('resolved')
        with mock.patch.object(campaign, 'ROOT', self.root):
            before = campaign.resolution_digest('legacy')
            (directory / 'postimage.1').write_text('second reviewed context')
            self.assertNotEqual(before, campaign.resolution_digest('legacy'))
            before = campaign.resolution_digest('legacy')
            (directory / 'thisimage').write_text('temporary merge state')
            self.assertEqual(before, campaign.resolution_digest('legacy'))


if __name__ == '__main__':
    unittest.main()
