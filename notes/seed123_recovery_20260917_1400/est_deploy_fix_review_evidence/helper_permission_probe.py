"""Retained synthetic preparation CLI reproduction. Never submits."""
import importlib.util, json, os, platform, shutil, subprocess, sys
from pathlib import Path
E = Path(__file__).resolve().parent
D = E.parent.parent/'seed123_execution_20260917/est_deployment'
spec = importlib.util.spec_from_file_location('fixtures', D/'test_deployment.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
c = f.Deployment(); c.setUp()
try:
    c.build(); release = E/'permission-release'; shutil.copytree(c.release, release)
    pins = json.loads((release/'ledger.json').read_text())['files']
    ns = E/'permission-reservation'
    argv = [sys.executable, '-I', '-B', str(D/'prepare_controller.py'), '--release', str(release),
            '--ledger-sha', c.sha, '--launcher-sha', pins['run_compute.py'], '--batch-sha', pins['sample.sbatch'],
            '--namespace', str(ns), '--stdout', str(ns/'stdout'), '--stderr', str(ns/'stderr')]
    commands = []
    for label in ('prepare', 'refuse-reuse'):
        with (E/(label+'.stdout')).open('wb') as out, (E/(label+'.stderr')).open('wb') as err:
            p = subprocess.run(argv, stdout=out, stderr=err, timeout=10)
        commands.append(dict(label=label, argv=argv, timeout=10, returncode=p.returncode))
    modes = {str(p.relative_to(E)): oct(p.stat().st_mode & 0o777) for p in (ns,ns/'stdout',ns/'stderr',ns/'preparation.json')}
    record = dict(commands=commands, modes=modes, python=platform.python_version(), worker_absent=not (ns/'worker').exists(),
                  receipt=json.loads((ns/'preparation.json').read_text()))
    (E/'helper-permission-result.json').write_text(json.dumps(record,indent=2)+'\n')
    assert commands[0]['returncode'] == 0 and commands[1]['returncode'] != 0
    assert record['worker_absent']
    print(json.dumps(modes,indent=2))
finally: c.doCleanups()
