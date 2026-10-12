"""Install a pinned tiny tool bundle and validate; never activates production cleanup."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

version, server = sys.argv[1:3]
if server not in ('deltaai', 'pronghorn'):
    raise SystemExit('server must be deltaai or pronghorn')
repo=Path('/tmp/b6-checkpoint-archive')
base=Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/checkpoint_archive_20261009')
commit=subprocess.check_output(['git','rev-parse',version+'^{commit}'],cwd=str(repo),universal_newlines=True).strip()
version=commit[:7]
files=['revision/scripts/b6/checkpoint_archive.py','revision/scripts/b6/checkpoint_archive_cli.py','revision/scripts/b6/storage_preflight.py','tests/test_checkpoint_archive.py','tests/test_checkpoint_archive_cli.py','deploy/deltaai/checkpoint_archive.example.json','docs/checkpoint_archive.md']
blob=io.BytesIO();hashes={}
with tarfile.open(fileobj=blob,mode='w:gz') as package:
    def add(name,data):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=int(time.time())
        package.addfile(info,io.BytesIO(data))
    for name in files:
        data=subprocess.check_output(['git','show',commit+':'+name],cwd=str(repo));hashes[name]=hashlib.sha256(data).hexdigest();add(name,data)
    add('SOURCE_SHA256.json',json.dumps(hashes,sort_keys=True).encode());add('SOURCE_COMMIT',(commit+'\n').encode())
    add('pronghorn_e2e_probe.py',(base/'pronghorn_e2e_probe.py').read_bytes())
parent='/work/nvme/bilv/wcyim/transgenic' if server=='deltaai' else '/data/gpfs/assoc/pgl/data/Transgenic'
root=parent+'/ops/checkpoint_archive/'+version
python='/usr/bin/python3' if server=='deltaai' else '/data/gpfs/assoc/pgl/data/Transgenic/.venv-ssh/bin/python'
verify="import hashlib,json,pathlib; rows=json.load(open('SOURCE_SHA256.json')); assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in rows.items()); print('INSTALLED_SHA256_MATCH',len(rows))"
runner="import os,sys,subprocess; env=dict(os.environ,PATH=os.path.dirname(sys.executable)+os.pathsep+os.environ.get('PATH','')); raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s','tests','-p','test_checkpoint_archive*.py','-v'],env=env))"
remote='mkdir -p '+shlex.quote(root)+' && tar xzf - -C '+shlex.quote(root)+' && cd '+shlex.quote(root)+' && '+shlex.join([python,'-c',verify])+' && '+shlex.join([python,'-c',runner])
if server=='pronghorn': remote+=' && '+shlex.join([python,'pronghorn_e2e_probe.py',version])
if server=='deltaai': remote=shlex.join(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','dtai',remote])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pronghorn',remote],input=blob.getvalue(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=480)
data=(r.stdout+r.stderr).decode(errors='replace')
log=base/(server+'_final_'+version+'.log');log.write_text(data)
print(data if r.returncode else '\n'.join(line for line in data.splitlines() if not line.startswith('test_')))
print('VALIDATION_EXIT',r.returncode,'LOG',str(log))
raise SystemExit(r.returncode)
