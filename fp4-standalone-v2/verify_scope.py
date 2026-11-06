import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
META = Path(__file__).resolve().parent / 'sources.json'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', action='store_true')
    args = parser.parse_args()
    meta = json.loads(META.read_text())
    base = meta['base_master_sha']
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', base, 'HEAD'], check=True)
    assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                           meta['source_legacy_sha'], 'HEAD'], capture_output=True).returncode == 1
    paths = set(git('diff', '--name-only', base).splitlines())
    paths.update(filter(None, git('ls-files', '--others', '--exclude-standard').splitlines()))
    assert paths <= set(meta['allowed_paths']), sorted(paths - set(meta['allowed_paths']))
    for path, expected in meta['payload_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    expected_feature = 'cuda-legacy-' + meta['component']
    for name in ['mistralrs', 'mistralrs-cli', 'mistralrs-core', 'mistralrs-quant',
                 'mistralrs-paged-attn', 'mistralrs-server-core', 'mistralrs-pyo3']:
        data = tomllib.loads((ROOT / name / 'Cargo.toml').read_text())
        actual = {key for key in data['features'] if key.startswith('cuda-legacy-')}
        assert actual == {expected_feature}, (name, actual)
    cargo = tomllib.loads((ROOT / 'Cargo.toml').read_text())
    for name in ['candle-core', 'candle-nn', 'candle-metal-kernels', 'candle-flash-attn-v3']:
        assert cargo['workspace']['dependencies'][name]['rev'] == meta['candle']['revision']
    locked = tomllib.loads((ROOT / 'Cargo.lock').read_text())
    for package in locked['package']:
        if package['name'] in ['candle-core', 'candle-nn', 'candle-kernels', 'candle-metal-kernels', 'candle-flash-attn-v3']:
            assert package['source'].endswith('#' + meta['candle']['revision'])
    for path in ['mistralrs-quant/src/unquantized/mod.rs', 'mistralrs-quant/src/blockwise_fp8/mod.rs', 'mistralrs-core/src/paged_attention/config.rs', 'mistralrs-core/src/speculative/dflash.rs']:
        assert git('hash-object', path) == git('rev-parse', base + ':' + path), path
    if args.format:
        rust = [str(ROOT / path) for path in sorted(paths) if path.endswith('.rs')]
        subprocess.run(['rustfmt', '--check', '--edition', '2021', '--config', 'skip_children=true', *rust], check=True)
    print('STANDALONE_' + meta['component'].upper() + '_SOURCE_SCOPE=PASS')


if __name__ == '__main__':
    main()
