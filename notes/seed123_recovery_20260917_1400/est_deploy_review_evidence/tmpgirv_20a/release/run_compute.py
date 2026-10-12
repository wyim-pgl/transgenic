#!/usr/bin/env python3
"""Verify a controller-pinned small release; run one diagnostic CLI, never retry."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import stat
import subprocess
import sys
import time

MODULES = ('est_real_sample', 'est_hsp_contract', 'est_hsp_expanded', 'est_phase_accept', 'est_phase_project')
FILES = {'code/evidence/' + name + '.py' for name in MODULES} | {
    'code/evidence/__init__.py', 'manifest.json', 'run_compute.py', 'sample.sbatch'}
PYTHON = '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11'
PYTHON_SHA = '74d2337199fe0880e01a38cd136c03f28de4383b833393657ce8420ce3c57415'
READER_SHA = 'a579aed31a13841cc0950a487b59729355d0b9fbcc39293e0b90b960133217bc'
RUNTIME = {'python': PYTHON, 'python_version': '3.11.14', 'python_sha256': PYTHON_SHA,
           'reader_sha256': READER_SHA, 'reader_samtools': '1.21', 'reader_htslib': '1.23'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def no_links(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'symlink path: ' + str(part))
    return path


def small_bytes(path, limit=8 * 1024**2):
    path = no_links(path)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'not an unlinked regular file: ' + str(path))
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'small-file limit: ' + str(path))
    return data


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_release(release, expected):
    release = no_links(release)
    raw = small_bytes(release / 'ledger.json')
    require(digest(raw) == expected, 'external ledger SHA mismatch')
    ledger = json.loads(raw)
    require(ledger['schema'] == 1 and ledger['runtime'] == RUNTIME, 'ledger schema/runtime mismatch')
    require(set(ledger['files']) == FILES, 'ledger file set mismatch')
    found = set()
    dirs = set()
    for root, children, files in os.walk(release, followlinks=False):
        for name in children + files:
            path = no_links(Path(root) / name)
            rel = path.relative_to(release).as_posix()
            if path.is_dir():
                dirs.add(rel)
            else:
                found.add(rel)
                small_bytes(path)
    require(dirs == {'code', 'code/evidence'}, 'unexpected directory inventory')
    require(found == FILES | {'ledger.json'}, 'unexpected file inventory')
    for name, sha in ledger['files'].items():
        require(digest(small_bytes(release / name)) == sha, 'file SHA mismatch: ' + name)
    require(small_bytes(release / 'code/evidence/__init__.py') == b'', 'initializer must be empty')
    return ledger


def runtime_identity(manifest):
    require(sys.flags.isolated == 1 and sys.flags.no_user_site == 1, 'Python -I required')
    require(os.path.abspath(sys.executable) == PYTHON, 'wrong interpreter path')
    require(platform.python_version() == RUNTIME['python_version'], 'wrong Python version')
    require(file_sha('/proc/self/exe') == PYTHON_SHA and file_sha(PYTHON) == PYTHON_SHA,
            'interpreter executable SHA mismatch')
    reader = manifest['files']['samtools']
    require(reader['path'] == str(Path(PYTHON).with_name('samtools')) and reader['sha256'] == READER_SHA,
            'reader manifest pin mismatch')
    require(file_sha(reader['path']) == READER_SHA, 'reader executable SHA mismatch')
    result = subprocess.run([reader['path'], '--version'], capture_output=True, timeout=10, check=True)
    text = result.stdout.decode('utf-8', errors='strict')
    require('samtools 1.21' in text.splitlines() and 'Using htslib 1.23' in text.splitlines(), 'reader version mismatch')
    return dict(RUNTIME, executable=sys.executable, proc_exe=os.readlink('/proc/self/exe'),
                sys_version=sys.version, reader_version_output=text,
                libraries='Not hermetic: shared libraries/stdlib are not individually pinned')


def closure(release, ledger, original=None):
    require(ledger is not None, 'closure ledger not established')
    names = {'evidence'} | {'evidence.' + name for name in MODULES}
    require({n for n in sys.modules if n == 'evidence' or n.startswith('evidence.')} == names,
            'loaded evidence module inventory mismatch')
    modules = {}
    for name in sorted(names):
        module = sys.modules[name]
        relative = 'code/evidence/' + ('__init__' if name == 'evidence' else name.split('.')[1]) + '.py'
        target = release / relative
        require(Path(module.__file__) == target and Path(module.__spec__.origin) == target,
                'module origin mismatch: ' + name)
        require(digest(small_bytes(target)) == ledger['files'][relative], 'loaded module SHA mismatch: ' + name)
        if original is not None:
            require(module is original[name], 'loaded module replaced: ' + name)
        modules[name] = module
    require(list(sys.modules['evidence'].__path__) == [str(release / 'code/evidence')], 'package path mismatch')
    return modules


def report_binding(sample, manifest_sha):
    report_bytes = small_bytes(sample / 'report.json', 1024**2)
    report = json.loads(report_bytes)
    status = report['status']
    require(status in {'COMPLETE', 'FAILED', 'INCOMPLETE'}, 'unknown worker status')
    markers = {name for name in ('COMPLETE', 'FAILED', 'INCOMPLETE') if (sample / name).exists()}
    require(markers == {status}, 'missing/ambiguous worker marker')
    report_sha = digest(report_bytes)
    require(small_bytes(sample / status) == (report_sha + '\n').encode(), 'report marker SHA mismatch')
    # Diagnostics are bounded by the worker's 256 MiB cap; no large input hashing here.
    diag_sha = digest(small_bytes(sample / 'diagnostics.jsonl', 256 * 1024**2))
    require(report['diagnostics_sha256'] == diag_sha, 'diagnostics SHA mismatch')
    require(report['manifest_file']['sha256'] == manifest_sha, 'report manifest binding mismatch')
    for key in ('training_eligible', 'full_c0', 'route_a_complete', 'molecule_counts_complete', 'raster'):
        require(report.get(key) is False, 'unexpected biological approval: ' + key)
    require(report.get('a11_status') == 'unassessable', 'unexpected A11 assertion')
    return {'worker_status': status, 'report_sha256': report_sha, 'diagnostics_sha256': diag_sha}


def run(release, expected, out):
    release, out = no_links(release), no_links(out)
    require(release != out and release not in out.parents, 'output must be outside release')
    out.mkdir(exist_ok=False)
    control = out / 'control'
    control.mkdir()
    start = time.monotonic()
    result = dict(status='INCOMPLETE', ledger_sha256=expected, argv=list(sys.argv),
                  worker_returncode=None, postcheck_attempted=False, errors=[],
                  slurm={k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
                  environment={k: os.environ[k] for k in ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
                               'LD_PRELOAD', 'PYTHONHOME', 'PYTHONPATH', 'BASH_ENV') if k in os.environ},
                  runtime_libraries_hermetic=False)
    loaded = None
    ledger = None
    previous_path = list(sys.path)
    sys.dont_write_bytecode = True
    try:
        ledger = verify_release(release, expected)
        manifest = json.loads(small_bytes(release / 'manifest.json'))
        result['runtime_before'] = runtime_identity(manifest)
        require(not any(n == 'evidence' or n.startswith('evidence.') for n in sys.modules),
                'preloaded/shadow evidence package')
        sys.path.insert(0, str(release / 'code'))
        for name in MODULES:
            importlib.import_module('evidence.' + name)
        loaded = closure(release, ledger)
        result['loaded_modules'] = {name: {'origin': module.__file__,
                                    'sha256': file_sha(module.__file__)} for name, module in loaded.items()}
        argv = ['--manifest', str(release / 'manifest.json'), '--sha', ledger['files']['manifest.json'],
                '--out', str(out / 'sample')]
        result['worker_argv'] = argv
        try:
            rc = loaded['evidence.est_real_sample'].main(argv)
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else 1
        require(isinstance(rc, int), 'worker CLI must return integer status')
        result['worker_returncode'] = rc
    except BaseException as exc:
        result['errors'].append(type(exc).__name__ + ': ' + str(exc))
    finally:
        # Never exec away the verifier; independently try all after-checks on failure too.
        result['postcheck_attempted'] = True
        result['postchecks'] = {}
        for label, check in (
            ('ledger', lambda: verify_release(release, expected)),
            ('imports', lambda: closure(release, ledger, loaded)),
            ('runtime', lambda: runtime_identity(json.loads(small_bytes(release / 'manifest.json'))))):
            try:
                value = check()
                result['postchecks'][label] = 'PASS'
                if label == 'runtime':
                    result['runtime_after'] = value
            except BaseException as exc:
                result['postchecks'][label] = 'FAIL'
                result['errors'].append('postcheck ' + label + ': ' + str(exc))
        sys.path[:] = previous_path
        try:
            result['worker_binding'] = report_binding(out / 'sample', ledger['files']['manifest.json'])
        except BaseException as exc:
            result['errors'].append('worker binding: ' + str(exc))
        result['elapsed_seconds'] = time.monotonic() - start
        result['rusage_self'] = list(resource.getrusage(resource.RUSAGE_SELF))
        result['rusage_children'] = list(resource.getrusage(resource.RUSAGE_CHILDREN))
        success = (not result['errors'] and result['worker_returncode'] == 0
                   and result['worker_binding']['worker_status'] == 'COMPLETE')
        if success:
            result['status'] = 'VERIFIED_DIAGNOSTIC_COMPLETE'
        write_json(control / 'result.json', result)
        if success:
            write_json(control / 'verification.json', dict(status=result['status'], biological_acceptance=False,
                       ledger_sha256=expected, result_sha256=file_sha(control / 'result.json'),
                       **result['worker_binding']))
    return 0 if success else (result['worker_returncode'] or 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='small package only; no worker/runtime/input access')
    parser.add_argument('release', type=Path)
    parser.add_argument('expected_ledger_sha')
    parser.add_argument('out', type=Path, nargs='?')
    args = parser.parse_args()
    if args.verify_only:
        verify_release(args.release, args.expected_ledger_sha)
        print('Package inventory/SHA verified; no compute or biological acceptance')
        return 0
    require(args.out is not None and os.environ.get('SLURM_JOB_ID'), 'fresh output and Slurm allocation required')
    # This executing launcher must be the same bytes included in the pinned package.
    ledger = verify_release(args.release, args.expected_ledger_sha)
    require(digest(small_bytes(Path(__file__))) == ledger['files']['run_compute.py'], 'launcher SHA mismatch')
    return run(args.release, args.expected_ledger_sha, args.out)


if __name__ == '__main__':
    raise SystemExit(main())
