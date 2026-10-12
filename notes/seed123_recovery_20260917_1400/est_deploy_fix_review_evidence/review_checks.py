"""Independent bounded scoped checks. Synthetic fixtures only."""
import difflib, hashlib, json, os, pathlib, subprocess, sys
E = pathlib.Path(__file__).resolve().parent
D = E.parent.parent / 'seed123_execution_20260917/est_deployment'
B = E.parent / 'est_deploy_fix_backup'
NAMES = ('build_release.py', 'run_compute.py', 'sample.sbatch', 'test_deployment.py', 'prepare_controller.py')
def hashes():
    return {n: hashlib.sha256((D/n).read_bytes()).hexdigest() for n in NAMES}
before = hashes()
(E/'before.json').write_text(json.dumps(before, indent=2)+'\n')
for n in NAMES:
    old = (B/n).read_text() if (B/n).exists() else ''
    (E/(n+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True), (D/n).read_text().splitlines(True), fromfile='backup/'+n, tofile='current/'+n)))
commands = []
for label, argv, timeout in [
    ('suite', [sys.executable, '-B', str(D/'test_deployment.py'), '-v'], 90),
    ('shell-syntax', ['/bin/bash', '-n', str(D/'sample.sbatch')], 5),
    ('adversarial', [sys.executable, '-B', str(E/'adversarial.py'), '-v'], 45),
]:
    with (E/(label+'.stdout')).open('wb') as out, (E/(label+'.stderr')).open('wb') as err:
        p = subprocess.run(argv, cwd=str(D), stdout=out, stderr=err, timeout=timeout,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(E)))
    commands.append(dict(label=label, argv=argv, timeout=timeout, returncode=p.returncode))
(E/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
after = hashes()
(E/'after.json').write_text(json.dumps(after, indent=2)+'\n')
assert before == after
print(json.dumps(commands, indent=2))
assert all(c['returncode'] == 0 for c in commands)
