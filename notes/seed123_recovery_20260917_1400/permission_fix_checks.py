import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

BASE = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local')
SOURCE = BASE / 'notes/seed123_execution_20260917/est_deployment'
E = Path(tempfile.mkdtemp(prefix='permission_fix_evidence_', dir=str(BASE / 'notes/seed123_recovery_20260917_1400')))
def hashes():
    return {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in ('prepare_controller.py', 'test_deployment.py')}
before = hashes()
env = dict(os.environ, TMPDIR='/tmp', PYTHONDONTWRITEBYTECODE='1')
tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_deployment.Deployment.test_controller_rejects_ineffective_private_permissions', 'test_deployment.Deployment.test_controller_preparation_contract'], cwd=str(SOURCE), env=env, capture_output=True, text=True, timeout=40)
(E / 'tests.stdout').write_text(tests.stdout)
(E / 'tests.stderr').write_text(tests.stderr)
assert tests.returncode == 0, tests.stderr
spec = importlib.util.spec_from_file_location('test_deployment', SOURCE / 'test_deployment.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
tempfile.tempdir = '/tmp'
fixture = m.Deployment()
fixture.setUp()
results = {'hashes_before': before, 'test_returncode': tests.returncode, 'evidence': str(E)}
try:
    fixture.build()
    pins = json.loads((fixture.release / 'ledger.json').read_text())['files']
    def invoke(namespace, label):
        command = [sys.executable, '-I', '-B', str(SOURCE / 'prepare_controller.py'), '--release', str(fixture.release), '--ledger-sha', fixture.sha, '--launcher-sha', pins['run_compute.py'], '--batch-sha', pins['sample.sbatch'], '--namespace', str(namespace), '--stdout', str(namespace / 'stdout'), '--stderr', str(namespace / 'stderr')]
        p = subprocess.run(command, capture_output=True, text=True, timeout=15)
        (E / (label + '.stdout')).write_text(p.stdout)
        (E / (label + '.stderr')).write_text(p.stderr)
        return p, command
    ns = fixture.root / 'positive-reservation'
    p, command = invoke(ns, 'positive')
    assert p.returncode == 0, p.stderr
    receipt = json.loads((ns / 'preparation.json').read_text())
    assert receipt['status'] == 'PREPARED_NOT_SUBMITTED'
    modes = {}
    for path, expected in ((ns, 0o700), (ns / 'stdout', 0o600), (ns / 'stderr', 0o600), (ns / 'preparation.json', 0o600)):
        info = path.stat()
        modes[path.name] = {'mode': oct(stat.S_IMODE(info.st_mode)), 'uid': info.st_uid}
        assert info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == expected
    assert (ns / 'stdout').read_bytes() == (ns / 'stderr').read_bytes() == b''
    assert not (ns / 'worker').exists()
    results['positive'] = dict(command=command, returncode=p.returncode, observed=modes)
    ns = E / 'failed-reservation'
    p, command = invoke(ns, 'unsupported')
    assert p.returncode != 0 and p.stdout == ''
    assert 'private owner/mode not established' in p.stderr
    assert ns.is_dir() and stat.S_IMODE(ns.stat().st_mode) != 0o700
    assert not (ns / 'preparation.json').exists()
    assert not (ns / 'worker').exists()
    retained = ns.stat().st_ino
    again, _ = invoke(ns, 'refused-reuse')
    assert again.returncode != 0 and again.stdout == '' and 'FileExistsError' in again.stderr
    assert ns.stat().st_ino == retained
    results['unsupported'] = dict(command=command, returncode=p.returncode, stdout_empty=p.stdout == '', mode=oct(stat.S_IMODE(ns.stat().st_mode)), uid=ns.stat().st_uid, receipt_exists=False, reservation_retained=True, entries=[p.name for p in ns.iterdir()], reuse_returncode=again.returncode)
finally:
    fixture.doCleanups()
results['hashes_after'] = hashes()
assert results['hashes_after'] == before
results['verdict'] = 'PASS: scoped D2.1'
(E / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
