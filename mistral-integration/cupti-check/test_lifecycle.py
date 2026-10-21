"""Exercise the actual Rust collector against synthetic FFI, without a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile


here = Path(__file__).resolve().parent
metadata = json.loads(subprocess.check_output([
    'cargo', 'metadata', '--no-deps', '--format-version', '1',
    '--manifest-path', str(here / 'Cargo.toml')], text=True))
binary = Path(metadata['target_directory']) / 'debug/d7-native-cupti-check'
subprocess.run(['cargo', 'build', '--locked', '--manifest-path', str(here / 'Cargo.toml')], check=True)
with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    library = root / 'libd7cuptimock.so'
    subprocess.run(['cc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                    '-Wl,-soname,libd7cuptimock.so', str(here / 'fake-cupti.c'), '-o', str(library)], check=True)
    for name in ('libcupti.so', 'libcupti.so.12', 'libcuda.so', 'libcuda.so.1'):
        (root / name).symlink_to(library.name)
    for case, option, expected_error in (
        ('finish', None, None),
        ('drop', 'D7_CUPTI_CHECK_DROP', None),
        ('write_error', None, 'write Flow Tuner CUPTI trace'),
        ('flush_error', 'D7_FAKE_FLUSH_ERROR', 'CUPTI activity flush failed'),
        ('sync_error', 'D7_FAKE_SYNC_ERROR', 'CUDA primary context synchronization failed'),
        ('finalize_error', 'D7_FAKE_FINALIZE_ERROR', 'CUPTI finalize failed with result 1'),
        ('missing_callback', 'D7_FAKE_SKIP_CALLBACK', 'CUPTI finalize callback was not invoked'),
    ):
        output, log = root / (case + '.json'), root / (case + '.log')
        if case == 'write_error':
            output.mkdir()
        env = dict(os.environ, LD_LIBRARY_PATH=str(root), FLOW_TUNER_CUPTI_OUT=str(output),
                   FLOW_TUNER_CUPTI_MODE='runtime-driver-diagnostic', D7_FAKE_LOG=str(log))
        if option:
            env[option] = '1'
        result = subprocess.run([str(binary)], env=env, capture_output=True, text=True)
        assert result.returncode == (1 if expected_error else 0), (case, result.returncode, result.stderr)
        if expected_error:
            assert expected_error in result.stderr, (case, result.stderr)
        calls = log.read_text().splitlines()
        assert calls.count('subscribe') == calls.count('flush') == calls.count('cuInit') == 1, (case, calls)
        assert calls.count('disable') == 5, (case, calls)
        contexts = 1 if case == 'sync_error' else 2
        assert all(calls.count(name) == contexts for name in ('retain', 'synchronize', 'restore', 'release')), (case, calls)
        assert calls.index('disable') < calls.index('synchronize') < calls.index('flush'), (case, calls)
        assert calls.count('finalize_at_exit') == (0 if case == 'missing_callback' else 1), (case, calls)
        assert 'unsubscribe' not in calls, (case, calls)
        if case == 'finish':
            trace = json.loads(output.read_text())
            assert trace['flush_complete'] and trace['finalize_complete'] and trace['cuda_synchronized']
            assert not trace['activities'] and not trace['dropped_records']
        elif case != 'write_error':
            assert not output.exists(), case
        print(case + ': PASS (synthetic FFI, no GPU qualification)')

    subprocess.run(['cargo', 'build', '--locked', '--features', 'd7-trace', '--manifest-path', str(here / 'Cargo.toml')], check=True)
    for case, option, error in (
        ('correlated', None, None),
        ('push_error', 'D7_FAKE_PUSH_ERROR', 'external correlation incomplete'),
        ('wrong_pop', 'D7_FAKE_WRONG_POP', 'external correlation incomplete'),
        ('held_scope', 'D7_FAKE_HELD_SCOPE', 'external correlation incomplete'),
    ):
        output, log, semantic = root / (case + '.json'), root / (case + '.log'), root / (case + '-trace')
        env = dict(os.environ, LD_LIBRARY_PATH=str(root), FLOW_TUNER_CUPTI_OUT=str(output),
                   FLOW_TUNER_CUPTI_MODE='cpu-gpu-correlated', D7_FAKE_LOG=str(log), D7_FAKE_CORRELATED='1',
                   D7_TRACE_RUN_ID=case, D7_TRACE_NONCE='nonce-' + case, D7_TRACE_DIR=str(semantic))
        if option:
            env[option] = '1'
        result = subprocess.run([str(binary)], env=env, capture_output=True, text=True, timeout=10)
        assert result.returncode == (1 if error else 0), (case, result.stderr)
        calls = log.read_text().splitlines()
        assert calls.count('finalize_at_exit') == 1 and calls.count('disable') == 6, (case, calls)
        assert 'pop' not in calls[calls.index('finalize_at_exit') + 1:], (case, calls)
        if error:
            assert error in result.stderr and not output.exists(), (case, result.stderr)
        else:
            trace = json.loads(output.read_text())
            assert trace['correlation']['complete'] and trace['correlation']['entered_scopes'] == trace['correlation']['exited_scopes'] == 2
            assert len(trace['external_correlations']) == len(trace['activities']) == 2
            events = [json.loads(line) for line in (semantic / 'events.ndjson').read_text().splitlines()]
            begins = {e['scope_id']: e for e in events if e['event'] == 'op.begin'}
            threads = set()
            for mapping, api in zip(trace['external_correlations'], trace['activities']):
                assert mapping['correlation_id'] == api['correlation_id']
                assert mapping['external_kind'] == trace['correlation']['external_kind_id']
                assert api['thread_id'] == begins[mapping['scope_id']]['cpu']['os_thread_id']
                assert api['process_id'] == trace['process_id']
                threads.add(api['thread_id'])
            assert len(threads) == 2
            assert calls.index('pop') < calls.index('flush') < calls.index('finalize_at_exit')
        assert json.loads((semantic / 'trace-manifest.json').read_text())['complete']
        print(case + ': PASS (synthetic FFI, no GPU qualification)')
