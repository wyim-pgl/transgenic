#!/usr/bin/env python3
"""Local artifact verification only. Not an independent code review or runtime admission."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
TOP = HERE.parent
RELEASE = TOP/'release_v3'
ORIGINAL = TOP.parents[2]/'mvp_execution_20260916/final_p1'
PRIOR = TOP.parent/'deployment_v1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def inventory(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}

before = {'original':inventory(ORIGINAL), 'prior':inventory(PRIOR), 'release':inventory(RELEASE)}
(HERE/'preservation-before.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
(HERE/'tmp').mkdir(exist_ok=True)
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(HERE/'tmp'))
results = []
for label, cwd, argv in [
    ('adapted-and-gate-tests', TOP, [sys.executable,'-m','unittest','test_deploy','test_candidate','test_queued_gate','test_admission_cases']),
    ('original-tests', ORIGINAL, [sys.executable,'-m','unittest','test_deploy']),
    ('shell-syntax', TOP, ['/bin/bash','-n','p1.sbatch']),
    ('release-ledger', RELEASE, ['/usr/bin/sha256sum','-c','release.sha256'])]:
    start=time.monotonic()
    r=subprocess.run(argv,cwd=cwd,env=env,capture_output=True,text=True,timeout=180)
    (HERE/(label+'.txt')).write_text(r.stdout+r.stderr)
    results.append(dict(label=label,command=argv,cwd=str(cwd),exit_code=r.returncode,seconds=time.monotonic()-start))
    assert r.returncode==0, label
manifest=json.loads((RELEASE/'release.json').read_text())
with tarfile.open(RELEASE/'code.candidate.tar') as t:
    members=t.getmembers()
    assert len({m.name for m in members})==len(members)
    assert all(m.isfile() or m.isdir() for m in members)
    closure={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
assert len(closure)==103 and closure==manifest['source_files']
assert sha(RELEASE/'code.candidate.tar')=='da161088fb401dbe87382685c1b5faadc17c7255d4a89f705fe1126ecc5bcec7'
assert (RELEASE/'operator-pins.json').read_bytes()==(PRIOR/'release_v2/operator-pins.json').read_bytes()
expected=set(manifest['release_files'])|{'release.json','release.sha256'}
assert set(inventory(RELEASE))==expected
for name in ('deploy.py','admission.py','p1.sbatch','build_release.py','controller_execution_scope.json','authorization.example.json'):
    assert (RELEASE/name).read_bytes()==(TOP/name).read_bytes()
transport=TOP/'release_v3.tar'
with tarfile.open(transport, mode='x') as t:
    t.add(RELEASE,arcname='release_v3')
with tarfile.open(transport) as t:
    members=t.getmembers()
    assert len({m.name for m in members})==len(members)
    assert all(m.isfile() or m.isdir() for m in members)
    transported={m.name.removeprefix('release_v3/') if hasattr(str,'removeprefix') else m.name[len('release_v3/'):]:
                 hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members if m.isfile()}
assert transported==inventory(RELEASE)
after={'original':inventory(ORIGINAL),'prior':inventory(PRIOR),'release':inventory(RELEASE)}
assert before==after
assert before['original']==json.loads((TOP/'originals-preservation.json').read_text())
assert sha(PRIOR/'release_v2.tar')=='5232741b2e7a0b722d9d63d3a62d23ff1cee17fd42486979b081bd17cd731ab9'
(HERE/'preservation-after.json').write_text(json.dumps(after,sort_keys=True,indent=2)+'\n')
(HERE/'test-results.json').write_text(json.dumps(results,indent=2)+'\n')
identity=dict(transport_sha256=sha(transport),source_archive_sha256=sha(RELEASE/'code.candidate.tar'),
              release_json_sha256=sha(RELEASE/'release.json'),ledger_sha256=sha(RELEASE/'release.sha256'),
              source_count=len(closure),operator_pins=json.loads((RELEASE/'operator-pins.json').read_text()),
              status='LOCAL_VERIFICATION_ONLY_PENDING_INDEPENDENT_CODE_REVIEW')
(HERE/'identity-verification.json').write_text(json.dumps(identity,indent=2,sort_keys=True)+'\n')
print(json.dumps(identity,indent=2,sort_keys=True))
