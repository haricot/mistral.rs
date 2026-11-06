#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
META = 'mistral-integration/sources.json'
SHA = re.compile(r'[0-9a-f]{40}\Z')
OUTPUTS = ('legacy_v2', 'contrib/lfm25_dspark', 'contrib/nemotron_bolt_dflash')
IDENTITY = {'GIT_AUTHOR_NAME': 'Haricot Preparation', 'GIT_COMMITTER_NAME': 'Haricot Preparation',
            'GIT_AUTHOR_EMAIL': '344493+haricot@users.noreply.github.com',
            'GIT_COMMITTER_EMAIL': '344493+haricot@users.noreply.github.com'}
CPU_COMMAND = ['cargo', 'check', '--locked', '-p', 'mistralrs-cli', '--no-default-features', '--features', 'code-execution']


def run(args, cwd=None, data=None):
    result = subprocess.run([str(a) for a in args], cwd=cwd, input=data, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args}\n{result.stdout.decode(errors="replace")}\n{result.stderr.decode(errors="replace")}')
    return result.stdout


def git(repo, *args, data=None):
    return run(['git', '-C', repo, *args], data=data)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def key(value):
    return hash_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def head(repo, branch):
    lines = git(repo, 'ls-remote', '--heads', 'origin', 'refs/heads/' + branch).decode().splitlines()
    if len(lines) > 1:
        raise RuntimeError('ambiguous branch ' + branch)
    return lines[0].split()[0] if lines else None


def verify_sources(repo, manifest):
    for branch, expected in manifest['sources'].items():
        if not SHA.fullmatch(expected) or head(repo, branch) != expected:
            raise RuntimeError('source moved: ' + branch + '; review the pinned manifest before preparing')
        git(repo, 'fetch', '--no-tags', 'origin', expected)
        if git(repo, 'rev-parse', expected).decode().strip() != expected:
            raise RuntimeError('invalid source object')
    fp4 = manifest['fp4']
    for ref, expected in ((fp4['source'], fp4['generic_source_blob']),
                          (manifest['sources']['legacy'], fp4['migrated_legacy_blob'])):
        if git(repo, 'rev-parse', f"{ref}:{fp4['path']}").decode().strip() != expected:
            raise RuntimeError('NVFP4 provenance mismatch')
    for name in ('legacy_support', 'fp8_packed'):
        entry = manifest[name]
        if hash_bytes((ROOT / entry['patch']).read_bytes()) != entry['sha256']:
            raise RuntimeError('patch checksum mismatch: ' + name)
    fp8 = manifest['fp8_packed']
    actual_fp8 = git(repo, 'diff', '--full-index', fp8['base'], fp8['source'], '--', fp8['upstream_path'])
    if hash_bytes(actual_fp8) != fp8['upstream_delta_sha256']:
        raise RuntimeError('packed FP8 source provenance mismatch')
    support = manifest['legacy_support']
    actual = git(repo, 'diff', '--full-index', '--binary', support['base'], support['source'], '--', *support['paths'])
    if hash_bytes(actual) != support['sha256']:
        raise RuntimeError('reviewed Legacy support patch no longer matches its source')


def recipe_hash():
    return key({str(p.relative_to(ROOT)): hash_bytes(p.read_bytes()) for p in sorted(ROOT.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.once'})


def make_commit(repo, parent, message):
    git(repo, 'add', '-A')
    tree = git(repo, 'write-tree').decode().strip()
    timestamp = int(git(repo, 'show', '-s', '--format=%ct', parent)) + 1
    env = {**os.environ, **IDENTITY, 'GIT_AUTHOR_DATE': f'{timestamp} +0000', 'GIT_COMMITTER_DATE': f'{timestamp} +0000'}
    result = subprocess.run(['git', '-C', str(repo), 'commit-tree', tree, '-p', parent, '-m', message],
                            env=env, capture_output=True, check=True)
    sha = result.stdout.decode().strip()
    git(repo, 'reset', '--hard', sha)
    return sha, tree


def clone(repo, path, ref):
    run(['git', 'clone', '--shared', '--no-checkout', repo, path])
    git(path, 'checkout', '--detach', ref)


def apply(repo, data):
    git(repo, 'apply', '--index', '--whitespace=nowarn', data=data)


def format_changed(repo, before, toolchain):
    paths = git(repo, 'diff', '--name-only', before).decode().splitlines()
    rust = [p for p in paths if p.endswith('.rs') and (repo / p).exists()]
    if rust:
        run(['rustfmt', '+' + toolchain, '--edition', '2021', '--config', 'skip_children=true', *rust], cwd=repo)


def compose(repo, output, manifest, format_sources=True):
    output.mkdir(parents=True, exist_ok=False)
    leases = {name: head(repo, name) for name in OUTPUTS}
    identity = {'sources': manifest, 'recipe': recipe_hash(), 'cpu_flags': '-C target-cpu=x86-64',
                'cpu_command': CPU_COMMAND, 'format_sources': format_sources}
    fingerprint = key(identity)
    nodes = {}
    for branch in OUTPUTS:
        spec = manifest['outputs'][branch]
        directory = output / branch.replace('/', '-')
        parent = manifest['master'] if branch == 'legacy_v2' else nodes['legacy_v2']['sha']
        clone(repo, directory, parent)
        if branch == 'legacy_v2':
            for name in ('legacy_support', 'fp8_packed'):
                apply(directory, (ROOT / manifest[name]['patch']).read_bytes())
            for entry in manifest['test_imports']:
                content = git(repo, 'show', f"{manifest['sources'][entry['branch']]}:{entry['path']}")
                if hash_bytes(content) != entry['sha256']:
                    raise RuntimeError('standalone test provenance mismatch')
                target = directory / entry['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        else:
            source = manifest['sources'][spec['source']]
            paths = git(repo, 'diff', '--name-only', manifest['sources']['legacy'], source).decode().splitlines()
            paths = [p for p in paths if p not in (META, 'mistral-integration/README.md') and not p.startswith('.github/')]
            if any(p.startswith('.github/') or p == META for p in paths):
                raise RuntimeError('unexpected contribution scope')
            apply(directory, git(repo, 'diff', '--full-index', '--binary', manifest['sources']['legacy'], source, '--', *paths))
        contract = {'schema_version': 1, 'kind': 'mistral-source-preparation', 'target': branch,
                    'master': manifest['master'], 'source_key': fingerprint, 'sources': manifest['sources'],
                    'requires': {n: nodes[n]['sha'] for n in spec['requires']},
                    'features': manifest['features'], 'qualification': 'cpu-preparation-only',
                    'gpu_validated': False, 'full_models_executed': False}
        write(directory / META, contract)
        if format_sources:
            format_changed(directory, parent, manifest['toolchain'])
        sha, tree = make_commit(directory, parent, 'build: prepare ' + branch + ' from ' + fingerprint)
        # Transfer the deterministic commit into the source object store for bundling.
        git(repo, 'fetch', '--no-tags', str(directory), sha)
        immutable = 'refs/heads/integration/legacy-v2/' + fingerprint + '/' + branch.replace('/', '-')
        git(repo, 'update-ref', immutable, sha)
        nodes[branch] = {'sha': sha, 'tree': tree, 'parent': parent, 'immutable_ref': immutable,
                         'cargo_lock_sha256': hash_bytes((directory / 'Cargo.lock').read_bytes())}
    bundle = output / 'candidates.bundle'
    git(repo, 'bundle', 'create', str(bundle), *(n['immutable_ref'] for n in nodes.values()))
    git(repo, 'bundle', 'verify', str(bundle))
    plan = {'schema_version': 1, 'input_key': fingerprint, 'identity': identity, 'nodes': nodes,
            'leases': leases, 'bundle_sha256': hash_bytes(bundle.read_bytes()),
            'run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
            'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1'),
            'workflow_sha': os.environ.get('GITHUB_SHA', git(repo, 'rev-parse', 'HEAD').decode().strip())}
    write(output / 'plan.json', plan)
    return plan


def verify_bundle(repo, folder):
    plan = read(folder / 'plan.json')
    if set(plan['nodes']) != set(OUTPUTS) or set(plan['leases']) != set(OUTPUTS):
        raise RuntimeError('unexpected publication targets')
    if plan['input_key'] != key(plan['identity']):
        raise RuntimeError('input identity mismatch')
    if hash_bytes((folder / 'candidates.bundle').read_bytes()) != plan['bundle_sha256']:
        raise RuntimeError('bundle checksum mismatch')
    git(repo, 'bundle', 'verify', str(folder / 'candidates.bundle'))
    for node in plan['nodes'].values():
        git(repo, 'fetch', '--no-tags', str(folder / 'candidates.bundle'), node['immutable_ref'])
        if git(repo, 'rev-parse', node['sha'] + '^{tree}').decode().strip() != node['tree']:
            raise RuntimeError('candidate tree mismatch')
    return plan


def validate_attestations(plan, reports):
    for branch, node in plan['nodes'].items():
        attestation = read(reports / (branch.replace('/', '-') + '.json'))
        expected = {'target': branch, 'candidate': node['sha'], 'tree': node['tree'],
                    'input_key': plan['input_key'], 'cargo_lock_sha256': node['cargo_lock_sha256'],
                    'run_id': plan['run_id'], 'run_attempt': plan['run_attempt'],
                    'workflow_sha': plan['workflow_sha'], 'command': CPU_COMMAND, 'status': 'PASS'}
        if attestation != expected:
            raise RuntimeError('missing or mismatched CPU evidence: ' + branch)


def publish(repo, folder, reports):
    plan = verify_bundle(repo, folder)
    if not plan['identity']['format_sources']:
        raise RuntimeError('unformatted probe is not publishable')
    if any(str(plan[name]) != os.environ.get(env) for name, env in
           [('run_id', 'GITHUB_RUN_ID'), ('run_attempt', 'GITHUB_RUN_ATTEMPT'), ('workflow_sha', 'GITHUB_SHA')]):
        raise RuntimeError('run/attempt/workflow provenance mismatch')
    validate_attestations(plan, reports)
    verify_sources(repo, plan['identity']['sources'])
    args = ['push', '--atomic', 'origin']
    for branch, node in plan['nodes'].items():
        previous = plan['leases'][branch]
        current = head(repo, branch)
        if current == node['sha']:
            continue
        if current != previous:
            raise RuntimeError('destination lease changed: ' + branch)
        args.append('--force-with-lease=refs/heads/' + branch + ':' + (previous or ''))
        args.append(node['sha'] + ':refs/heads/' + branch)
        if previous:
            backup = 'refs/heads/archive/legacy-v2/' + branch.replace('/', '-') + '/' + previous
            existing = head(repo, backup.removeprefix('refs/heads/'))
            if existing not in (None, previous):
                raise RuntimeError('backup collision')
            args.append(previous + ':' + backup)
        ref = node['immutable_ref']
        existing = head(repo, ref.removeprefix('refs/heads/'))
        if existing not in (None, node['sha']):
            raise RuntimeError('immutable candidate ref collision')
        args.append(node['sha'] + ':' + ref)
    if len(args) > 3:
        git(repo, *args)
    write(folder / 'published.json', {'run_id': plan['run_id'], 'input_key': plan['input_key'],
                                     'branches': {name: node['sha'] for name, node in plan['nodes'].items()},
                                     'qualification': 'cpu-preparation-only'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['compose', 'checkout', 'attest', 'publish'])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=OUTPUTS)
    parser.add_argument('--reports', type=Path)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'sources.json')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    if args.command == 'compose':
        manifest = read(args.manifest)
        verify_sources(repo, manifest)
        plan = compose(repo, output, manifest)
        print(json.dumps({name: node['sha'] for name, node in plan['nodes'].items()}))
    elif args.command == 'publish':
        publish(repo, output, args.reports.resolve())
    else:
        plan = verify_bundle(repo, output)
        node = plan['nodes'][args.target]
        if args.command == 'checkout':
            git(repo, 'checkout', '--detach', node['sha'])
        else:
            if git(repo, 'rev-parse', 'HEAD').decode().strip() != node['sha'] or git(repo, 'status', '--porcelain').strip():
                raise RuntimeError('CPU candidate modified during validation')
            report = {'target': args.target, 'candidate': node['sha'], 'tree': node['tree'],
                      'input_key': plan['input_key'], 'cargo_lock_sha256': node['cargo_lock_sha256'],
                      'run_id': plan['run_id'], 'run_attempt': plan['run_attempt'],
                      'workflow_sha': plan['workflow_sha'], 'command': CPU_COMMAND, 'status': 'PASS'}
            write(args.reports / (args.target.replace('/', '-') + '.json'), report)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, KeyError) as error:
        sys.exit('LEGACY_V2_PREPARATION_REJECT=' + str(error))
