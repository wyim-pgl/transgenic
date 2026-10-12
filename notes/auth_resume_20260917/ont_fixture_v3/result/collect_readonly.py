"""Run locally: python3 -B collect_readonly.py. Remote operations are reads only."""
import subprocess, pathlib, json, base64, hashlib
HERE=pathlib.Path(__file__).resolve().parent
REMOTE=r'''
import os,json,stat,hashlib,base64,datetime
from pathlib import Path
root=Path('/data/gpfs/home/wyim/ont_f4_20260917_1653')
limit=4*1024*1024
budget=22*1024*1024
used=0
result={'time':datetime.datetime.now().astimezone().isoformat(),'inventory':[],'files':{},'runtime':[]}
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def meta(p):
 s=os.stat(p)
 return dict(mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,size=s.st_size,resolved=os.path.realpath(p))
# Walk only the approved job namespace, never runtime/environment trees.
for folder,dirs,files in os.walk(str(root),followlinks=False):
 for name in sorted(dirs+files):
  p=Path(folder)/name; s=p.lstat(); rel=str(p.relative_to(root))
  r=dict(path=rel,logical=s.st_size,allocated=s.st_blocks*512,mode=oct(stat.S_IMODE(s.st_mode)),uid=s.st_uid,gid=s.st_gid,type='file' if stat.S_ISREG(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'other')
  if stat.S_ISREG(s.st_mode):
   r['sha256']=digest(p)
   # Preserve every output/log and small deployed evidence, excluding transport archive.
   wanted=rel.startswith('out/') or rel in ('stdout','stderr','controller_approval.json','release/MANIFEST.sha256','release/runtime_evidence/expected_closure.json')
   if wanted:
    if s.st_size<=limit and used+s.st_size<=budget:
     b=p.read_bytes();used+=len(b);result['files'][rel]=base64.b64encode(b).decode()
    else:r['retrieval_refused']='per-file or total bound'
  result['inventory'].append(r)
closure=json.loads((root/'release/runtime_evidence/expected_closure.json').read_text())
records={}; parents={}
def walk(x):
 if isinstance(x,dict):
  if 'sha256' in x and 'path' in x:
   records[x['path']]=x
   for p in x.get('parents',[]):parents[p['path']]=p
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(closure)
for p,e in sorted(records.items()):
 r={'path':p,'expected_sha256':e['sha256']}
 try:
  a=meta(p);a['sha256']=digest(p)
  if 'symlink' in e:a['symlink']=os.readlink(p) if os.path.islink(p) else None
  r['actual']=a;r['differences']={k:[e[k],a.get(k)] for k in a if k in e and e[k]!=a[k]}
 except Exception as ex:r['error']=repr(ex)
 result['runtime'].append(r)
result['runtime_parents']=[]
for p,e in sorted(parents.items()):
 r={'path':p}
 try:
  a=meta(p);r['actual']={k:a[k] for k in ('mode','uid','gid')};r['differences']={k:[e[k],a[k]] for k in ('mode','uid','gid') if e[k]!=a[k]}
 except Exception as ex:r['error']=repr(ex)
 result['runtime_parents'].append(r)
result['retrieved_file_bytes']=used
print(json.dumps(result,sort_keys=True))
'''
p=subprocess.Popen(['ssh','-o','ConnectTimeout=15','pronghorn','/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python3.11 -I -B -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 out,err=p.communicate(REMOTE.encode(),timeout=120)
except subprocess.TimeoutExpired:
 p.kill();out,err=p.communicate();(HERE/'collection_partial.stdout').write_bytes(out);(HERE/'collection_partial.stderr').write_bytes(err);raise
(HERE/'collection.stderr').write_bytes(err)
(HERE/'collection.json').write_bytes(out)
if len(out)+len(err)>32*1024*1024:raise RuntimeError('retrieval bound exceeded')
if p.returncode:raise RuntimeError(p.returncode)
d=json.loads(out)
for rel,b64 in d['files'].items():
 b=base64.b64decode(b64);dest=HERE/'raw'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 inv=next(i for i in d['inventory'] if i['path']==rel)
 assert hashlib.sha256(b).hexdigest()==inv['sha256']
print(json.dumps({'wire_bytes':len(out)+len(err),'file_bytes':d['retrieved_file_bytes'],'runtime_files':len(d['runtime']),'runtime_mismatches':[r for r in d['runtime'] if r.get('differences') or r.get('error')],'parent_mismatches':[r for r in d['runtime_parents'] if r.get('differences') or r.get('error')]},indent=2))
