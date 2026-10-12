#!/usr/bin/env python3
"""Local-only deployment review. All generated files stay in this evidence directory."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
PREP = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_execution_20260917/est_deployment')
ROOT = Path('/tmp/est-real-source-binding-20260917')
NAMES = ('est_real_sample', 'est_hsp_contract', 'est_hsp_expanded', 'est_phase_accept', 'est_phase_project')
MANIFEST = ROOT / 'tasknotes/real_sample/sitalica.manifest.json'
FILES = [PREP / n for n in ('build_release.py', 'run_compute.py', 'sample.sbatch', 'test_deployment.py')]
FILES += [ROOT / ('revision/scripts/evidence/' + n + '.py') for n in NAMES]
FILES += [MANIFEST, ROOT / 'tests/test_est_real_sample.py']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory():
    return {str(p): sha(p) for p in FILES}

def save(name, value):
    with (BASE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')

before = inventory()
save('before.json', before)
commands = []
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(BASE))

def command(argv, name, cwd=BASE, timeout=40):
    p = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout)
    (BASE / (name + '.stdout')).write_bytes(p.stdout)
    (BASE / (name + '.stderr')).write_bytes(p.stderr)
    commands.append(dict(argv=argv, cwd=str(cwd), timeout=timeout, returncode=p.returncode,
                         stdout=name+'.stdout', stderr=name+'.stderr'))
    return p

p = command([sys.executable, '-B', str(PREP / 'test_deployment.py'), '-v'], 'synthetic')
assert p.returncode == 0, p.stderr.decode()
assert command(['/bin/bash', '-n', str(PREP / 'sample.sbatch')], 'bash_syntax').returncode == 0
pins = {'revision/scripts/evidence/' + n + '.py': sha(ROOT / ('revision/scripts/evidence/' + n + '.py')) for n in NAMES}
save('source-pins.json', pins)
assert pins['revision/scripts/evidence/est_real_sample.py'] == '93ea442d0dc60c1d8df2ee0b479361d89a188f9b43bba4327740f896efec27b2'
assert sha(MANIFEST) == '7d1af21faf7cf99c46120c07781d0af0ad1318a2c2aa12875fcfdd74b4e5f88c'
assert sha(ROOT / 'tests/test_est_real_sample.py') == '5920616a012ae1cdb2dd941dbafc5752b9643eaef564230597ab0ad9488044dc'
for n in NAMES[1:]:
    path = 'revision/scripts/evidence/' + n + '.py'
    p = command(['git', 'show', 'fca911e58fde44124b9359f9adc03a9d20059599:' + path], 'base_' + n, ROOT)
    assert p.returncode == 0 and hashlib.sha256(p.stdout).hexdigest() == pins[path]
release = BASE / 'actual-source-release'
p = command([sys.executable, '-I', '-B', str(PREP / 'build_release.py'), '--root', str(ROOT), '--out', str(release), '--source-hashes', str(BASE / 'source-pins.json'), '--manifest', str(MANIFEST), '--manifest-sha', sha(MANIFEST)], 'build')
assert p.returncode == 0, p.stderr.decode()
ledger_sha = p.stdout.decode().strip()
assert command([sys.executable, '-I', '-B', str(release / 'run_compute.py'), '--verify-only', str(release), ledger_sha], 'verify_only').returncode == 0
# The exact packaged closure imports under isolated Python, without invoking the worker.
code = "import sys,importlib,json;sys.path.insert(0,sys.argv[1]);[importlib.import_module('evidence.'+n) for n in sys.argv[2:]];print(json.dumps({n:m.__file__ for n,m in sys.modules.items() if n=='evidence' or n.startswith('evidence.')},sort_keys=True))"
p = command([sys.executable, '-I', '-B', '-c', code, str(release / 'code'), *NAMES], 'actual_imports')
assert p.returncode == 0, p.stderr.decode()
origins = json.loads(p.stdout)
assert set(origins) == {'evidence'} | {'evidence.' + n for n in NAMES}
# Runtime must fail closed locally, and direct login-style entry must not start.
p = command([sys.executable, '-I', '-B', str(release / 'run_compute.py'), str(release), ledger_sha, str(BASE / 'must-not-create')], 'no_allocation')
assert p.returncode != 0 and not (BASE / 'must-not-create').exists()
# Execute a tiny noisy fixture to check whether release output itself retains its streams.
spec = importlib.util.spec_from_file_location('deployment_test', PREP / 'test_deployment.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
case = t.Deployment('test_exception_still_postchecks'); case.setUp()
try:
    case.build("print('REVIEW_STDOUT'); __import__('sys').stderr.write('REVIEW_STDERR\\n'); raise RuntimeError('review failure')")
    with patch.object(case.runner, 'runtime_identity', return_value={'fixture': True}):
        rc = case.runner.run(case.release, case.sha, BASE / 'noisy-failure-output')
    assert rc != 0
    result = json.loads((BASE / 'noisy-failure-output/control/result.json').read_text())
    assert result['postcheck_attempted'] and 'RuntimeError: review failure' in result['errors']
    saved = b''.join(p.read_bytes() for p in (BASE / 'noisy-failure-output').rglob('*') if p.is_file())
    save('stream_probe.json', dict(returncode=rc, stdout_retained_in_output=b'REVIEW_STDOUT' in saved,
                                  stderr_retained_in_output=b'REVIEW_STDERR' in saved,
                                  result_preserved=True))
finally:
    case.doCleanups()
after = inventory()
save('after.json', after)
assert before == after
save('commands.json', commands)
save('summary.json', dict(accepted_source_hashes_match=True, tracked_dependencies_match_base=True,
                          deployment_tests=16, actual_package_verified=True,
                          exact_five_module_import_closure=True, production_files_unchanged=True,
                          ledger_sha256=ledger_sha, local_python=sys.version, source_pins=pins))
print('LOCAL REVIEW CHECKS PASS; stdout/stderr output-retention probe recorded separately')
