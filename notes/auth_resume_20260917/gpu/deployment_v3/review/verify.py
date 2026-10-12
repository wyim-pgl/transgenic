"""Local verification only; never authorizes execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent.parent
GPU=HERE.parent
sys.path.insert(0,str(HERE))
import deploy as d

def preservation():
    expected=json.loads((HERE/'review/preservation-before.json').read_text())
    actual={str(p.relative_to(GPU)):d.sha(p) for name in ('deployment_v1','deployment_v2','candidate')
            for p in (GPU/name).rglob('*') if p.is_file()}
    assert actual==expected, 'prior release or source candidate changed'

preservation()
commands=[([sys.executable,'-m','unittest','test_deploy','test_candidate','test_queued_gate','test_admission_cases','test_recovery','test_capture'],HERE),
          ([sys.executable,'-m','unittest','test_deploy'],GPU.parents[1]/'mvp_execution_20260916/final_p1'),
          (['/bin/bash','-n','p1.sbatch'],HERE),
          (['/usr/bin/sha256sum','-c','release.sha256'],HERE/'release_v5')]
results=[]
for argv,cwd in commands:
    start=time.time()
    r=subprocess.run(argv,cwd=str(cwd),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=120)
    results.append(dict(argv=argv,cwd=str(cwd),seconds=time.time()-start,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    assert r.returncode==0, results[-1]
preservation()
release=HERE/'release_v5'
manifest=json.loads((release/'release.json').read_text())
expected=set(manifest['release_files'])|{'release.json','release.sha256'}
assert expected=={str(p.relative_to(release)) for p in release.rglob('*') if p.is_file()}
for name,digest in manifest['release_files'].items():
    assert d.sha(release/name)==digest
    if (HERE/name).is_file(): assert (release/name).read_bytes()==(HERE/name).read_bytes()
transport=HERE/'release_v5.tar'
with tarfile.open(str(transport),'x') as tar:
    for name in sorted(expected): tar.add(str(release/name),arcname='release_v5/'+name,recursive=False)
with tarfile.open(str(transport)) as tar:
    assert {m.name for m in tar.getmembers()}=={'release_v5/'+n for n in expected}
    for m in tar.getmembers():
        assert m.isfile()
        assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==d.sha(HERE/m.name)
transport.chmod(0o444)
d.save(HERE/'review/test-results.json',results)
d.save(HERE/'review/identity.json',dict(transport_sha256=d.sha(transport),release_json_sha256=d.sha(release/'release.json'),
    ledger_sha256=d.sha(release/'release.sha256'),preserved_prior_files=len(json.loads((HERE/'review/preservation-before.json').read_text())),
    status='LOCAL_VERIFIED_ONLY_PENDING_INDEPENDENT_REVIEW'))
print(json.dumps(json.loads((HERE/'review/identity.json').read_text()),indent=2))
