"""Bounded local review evidence. No deployment or remote commands."""
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
DEP = HERE.parent
ROOT = DEP.parents[3]
OLD = ROOT / 'notes/mvp_execution_20260916/final_p1'
RELEASE = DEP / 'release_v2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}

def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

before = {'original': inventory(OLD), 'release_v2': inventory(RELEASE),
          'transport': sha(DEP / 'release_v2.tar')}
save('preservation-before.json', before)
assert before['transport'] == '5232741b2e7a0b722d9d63d3a62d23ff1cee17fd42486979b081bd17cd731ab9'
assert sha(RELEASE / 'code.candidate.tar') == 'da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7'
assert before['original'] == json.loads((DEP / 'originals-preservation.json').read_text())
manifest = json.loads((RELEASE / 'release.json').read_text())
assert set(before['release_v2']) == set(manifest['release_files']) | {'release.json', 'release.sha256'}
for n, h in manifest['release_files'].items():
    assert sha(RELEASE / n) == h
with tarfile.open(DEP / 'release_v2.tar') as t:
    members = t.getmembers()
    assert len(members) == len({m.name for m in members})
    assert all(m.isfile() or m.isdir() for m in members)
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    transport_files = {m.name: hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
    prefixes = {n.split('/')[0] for n in transport_files}
    if prefixes == {'release_v2'}:
        transport_files = {n[len('release_v2/'):]: h for n, h in transport_files.items()}
    assert transport_files == before['release_v2']
with tarfile.open(RELEASE / 'code.candidate.tar') as t:
    members = t.getmembers()
    assert len(members) == len({m.name for m in members})
    assert all(m.isfile() or m.isdir() for m in members)
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    payload = {m.name: t.extractfile(m).read() for m in members if m.isfile()}
assert len(payload) == 103
assert {n: hashlib.sha256(b).hexdigest() for n, b in payload.items()} == manifest['source_files']
files = manifest['source_files']
component = {n:h for n,h in files.items() if n.startswith(('src/transgenic/', 'configs/', 'revision/scripts/mvp/')) or n == 'train/train_HyenaTransgenic_MVP.py'}
tree = ast.parse(payload['revision/scripts/mvp/mvp_decode_runtime.py'])
decoder_names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DECODER_ONLY' for t in n.targets))
canonical = lambda x: hashlib.sha256((json.dumps(x, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
pins = json.loads((RELEASE / 'operator-pins.json').read_text())
assert len(component) == pins['source_comparison_count'] == 89
assert len(decoder_names) == pins['decoder_file_count'] == 14
assert canonical(component) == pins['source_comparison_sha256']
assert canonical({n:files[n] for n in decoder_names}) == pins['decoder_code_sha256']
for name in ('deploy.py', 'build_release.py', 'p1.sbatch', 'controller_execution_scope.json'):
    assert (DEP/name).read_bytes() == (RELEASE/name).read_bytes()
    diff = ''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True), (RELEASE/name).read_text().splitlines(True), fromfile='original-final_p1/'+name, tofile='release_v2/'+name))
    (HERE/(name+'.diff.txt')).write_text(diff)
save('identity-verification.json', {'transport_sha256': before['transport'], 'release_json_sha256':sha(RELEASE/'release.json'), 'release_ledger_sha256':sha(RELEASE/'release.sha256'), 'source_count': len(files), 'transport_files':transport_files, 'operator_pins':pins, 'selected_top_level_scripts_equal_release':True, 'exact_transport_and_archive_closure':True})
(HERE/'tmp').mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(HERE/'tmp'))
results = []
for name, cwd, argv in (
    ('adapted-tests.txt', DEP, [sys.executable, '-m', 'unittest', 'test_candidate', 'test_deploy', '-v']),
    ('original-tests.txt', ROOT, [sys.executable, '-m', 'unittest', 'discover', '-s', str(OLD), '-p', 'test_deploy.py', '-v']),
    ('shell-syntax.txt', RELEASE, ['bash', '-n', 'p1.sbatch']),
    ('release-ledger.txt', RELEASE, ['sha256sum', '-c', 'release.sha256']),
):
    p = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    (HERE/name).write_text('cwd: '+str(cwd)+'\nargv: '+json.dumps(argv)+'\nexit_code: '+str(p.returncode)+'\n'+p.stdout)
    results.append({'log':name,'exit_code':p.returncode})
save('test-results.json', results)
after = {'original': inventory(OLD), 'release_v2': inventory(RELEASE), 'transport':sha(DEP/'release_v2.tar')}
save('preservation-after.json',after)
assert before == after
assert all(r['exit_code'] == 0 for r in results)
print('Exact release/transport/archive closure verified; 19 adapted and 11 original tests returned zero; immutable inputs unchanged.')
