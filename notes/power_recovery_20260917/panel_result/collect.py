import pathlib, subprocess, json, base64
D=pathlib.Path(__file__).resolve().parent
remote=r'''
import os,json,base64,stat
root='/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916'
rows=[]; small={}
for sub in ['outputs/job-6165628','logs']:
 for parent,dirs,files in os.walk(root+'/'+sub):
  if sub=='logs': files=[n for n in files if n in ['panel-6165628.out','panel-6165628.err']]
  for n in sorted(files):
   p=parent+'/'+n; s=os.lstat(p); rel=os.path.relpath(p,root)
   rows.append(dict(path=rel,size=s.st_size,mtime=s.st_mtime,mode=s.st_mode,inode=s.st_ino))
   if stat.S_ISREG(s.st_mode) and s.st_size<2000000 and (n.endswith(('.json','.out','.err')) or n=='COMPLETE'):
    small[rel]=base64.b64encode(open(p,'rb').read()).decode('ascii')
for rel in ['transport.json','source/source_pins.json']:
 p=root+'/'+rel
 small[rel]=base64.b64encode(open(p,'rb').read()).decode('ascii')
print(json.dumps(dict(files=rows,small=small)))
'''
p=subprocess.run(['ssh','pronghorn','/usr/bin/python -'],input=remote,text=True,capture_output=True,timeout=90)
(D/'collection.stderr.txt').write_text(p.stderr); p.check_returncode()
result=json.loads(p.stdout); (D/'remote_file_stats.json').write_text(json.dumps(result['files'],indent=2))
for name,encoded in result['small'].items():
 dest=D/'copied'/name; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(base64.b64decode(encoded))
print(json.dumps(result['files'],indent=2))
