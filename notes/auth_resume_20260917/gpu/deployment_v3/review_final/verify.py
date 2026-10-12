"""Independent local-only verification. Writes exclusively below review_final."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
GPU = HERE.parent
ROOT = GPU.parents[2]
RELEASE = HERE / 'release_v5'
TMP = OUT / 'tmp'
TMP.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n, v): (OUT/n).write_text(json.dumps(v, indent=2, sort_keys=True)+'\n')
def snapshot():
    roots = [GPU/'deployment_v1', GPU/'deployment_v2', GPU/'candidate', RELEASE,
             ROOT/'notes/mvp_execution_20260916/final_p1']
    return {str(p):sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
before=snapshot(); save('preservation-before.json',before)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(TMP))
commands=[('adapted-tests',[sys.executable,'-B','-m','unittest','test_deploy','test_candidate','test_queued_gate','test_admission_cases','test_recovery','test_capture'],HERE),
          ('original-tests',[sys.executable,'-B','-m','unittest','test_deploy'],ROOT/'notes/mvp_execution_20260916/final_p1'),
          ('shell-syntax',['/bin/bash','-n',str(RELEASE/'p1.sbatch')],OUT),
          ('release-ledger',['/usr/bin/sha256sum','-c','release.sha256'],RELEASE)]
results=[]
for label,argv,cwd in commands:
    start=time.monotonic()
    p=subprocess.run(argv,cwd=str(cwd),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=180)
    (OUT/(label+'.txt')).write_text(p.stdout+p.stderr)
    results.append(dict(label=label,argv=argv,cwd=str(cwd),exit_code=p.returncode,seconds=time.monotonic()-start))
    save('test-results.json',results)
    assert p.returncode==0, label
manifest=json.loads((RELEASE/'release.json').read_text())
expected=set(manifest['release_files'])|{'release.json','release.sha256'}
assert expected=={str(p.relative_to(RELEASE)) for p in RELEASE.rglob('*') if p.is_file()}
ledger={n:h for h,n in [line.split('  ',1) for line in (RELEASE/'release.sha256').read_text().splitlines()]}
assert set(ledger)==expected-{'release.sha256'}
assert all(sha(RELEASE/n)==h for n,h in ledger.items())
with tarfile.open(HERE/'release_v5.tar') as t:
    members=t.getmembers()
    assert len(members)==len(expected) and {m.name for m in members}=={'release_v5/'+n for n in expected}
    for m in members:
        assert m.isfile() and t.extractfile(m).read()==(HERE/m.name).read_bytes()
with tarfile.open(RELEASE/'code.candidate.tar') as t:
    members=t.getmembers(); assert len({m.name for m in members})==len(members)
    assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in members)
    payload={m.name:t.extractfile(m).read() for m in members if m.isfile()}
assert {n:hashlib.sha256(v).hexdigest() for n,v in payload.items()}==manifest['source_files']
assert len(payload)==103
for n,v in payload.items():
    p=TMP/'source'/n; p.parent.mkdir(exist_ok=True,parents=True); p.write_bytes(v)
# No source-worktree files are opened or changed: comparison is against the frozen v2 transport.
with tarfile.open(GPU/'deployment_v2/release_v3.tar') as t:
    old={m.name.split('/',1)[-1]:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
for n in ('admission.py','deploy.py','build_release.py','p1.sbatch','controller_execution_scope.json'):
    (OUT/(n+'.v2.diff.txt')).write_text(''.join(difflib.unified_diff(old[n].decode().splitlines(True),(RELEASE/n).read_text().splitlines(True),fromfile='v2/'+n,tofile='v5/'+n)))
assert old['code.candidate.tar']==(RELEASE/'code.candidate.tar').read_bytes()
for n in ('operator-pins.json','source-manifest.json'):
    assert old[n]==(RELEASE/n).read_bytes()
for n in ('deploy.py','admission.py','build_release.py','p1.sbatch','controller_execution_scope.json','authorization.example.json'):
    assert (HERE/n).read_bytes()==(RELEASE/n).read_bytes()
compat=json.loads((HERE/'review/host-compat-streaming-result.json').read_text())
assert compat['exit_code']==0
assert all(sha(RELEASE/n)==h for n,h in compat['sources_sha256'].items())
identities={n:sha(p) for n,p in [('transport',HERE/'release_v5.tar'),('release_json',RELEASE/'release.json'),('ledger',RELEASE/'release.sha256'),('source_archive',RELEASE/'code.candidate.tar')]}
assert identities['transport']=='b3dafe316605a8ccaae61e94854fe527bd8591cb4600ae152afae017d2807458'
assert identities['release_json']=='e0d2262a7aa94b49cdf08de764d6b807c75779dc32a5a7e83ff12810084c5389'
identities.update(source_files=len(payload),transport_files=len(expected),host_compat_source_pins_match=True,prior_v2_source_and_operator_pins_unchanged=True)
save('identities.json',identities)
after=snapshot(); save('preservation-after.json',after); assert before==after
print(json.dumps(dict(identities=identities,tests=results,preserved_files=len(before)),indent=2))
