import subprocess, pathlib, json, hashlib, datetime, shlex, base64
here=pathlib.Path(__file__).resolve().parent
remote=r'''
import os,json,base64,stat
root='/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_input_audit_20260916_bindfix1'
job=root+'/outputs/job-6165559'
paths=[root+'/'+x for x in ('transport.json','slurm-6165559.out','slurm-6165559.err')]
listing=[]
for folder in (job,job+'/panel'):
 if os.path.isdir(folder):
  for name in sorted(os.listdir(folder)):
   p=folder+'/'+name; s=os.lstat(p)
   listing.append({'path':p,'bytes':s.st_size,'regular':stat.S_ISREG(s.st_mode)})
   if stat.S_ISREG(s.st_mode) and (name.endswith('.json') or name in ('COMPLETE','FAILED','DONE')): paths.append(p)
files=[]; total=0
for p in paths:
 if not os.path.exists(p): continue
 s=os.lstat(p)
 if not stat.S_ISREG(s.st_mode): raise ValueError('not regular '+p)
 total+=s.st_size
 if total>10000000: raise ValueError('aggregate metadata limit')
 with open(p,'rb') as f: data=f.read(s.st_size+1)
 if len(data)!=s.st_size: raise ValueError('changed size '+p)
 files.append({'path':p,'bytes':len(data),'base64':base64.b64encode(data).decode('ascii')})
print(json.dumps({'root':root,'listing':listing,'files':files,'bytes':total}))
'''
def ssh(cmd):
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','pronghorn',cmd],capture_output=True,timeout=90)
 (here/'ssh_stderr.txt').open('ab').write(r.stderr)
 r.check_returncode(); return r.stdout
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
status=ssh('date -Is; sacct -j 6165559 --parsable2 --format=JobID,State,ExitCode,Elapsed,ElapsedRaw,Timelimit,Start,End,NodeList,MaxRSS; scontrol show job 6165559')
(here/'terminal_status.txt').write_bytes(stamp.encode()+b'\n'+status)
result=json.loads(ssh('/usr/bin/python -c '+shlex.quote(remote)))
records=[]
for item in result.pop('files'):
 rel=pathlib.PurePosixPath(item['path']).relative_to(result['root'])
 dest=here/'metadata'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
 data=base64.b64decode(item['base64']); dest.write_bytes(data)
 records.append({'remote_path':item['path'],'local_path':str(dest.relative_to(here)),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
result.update(collected_at=stamp,files=records,hash_location='local fetched bytes; no remote hashing')
(here/'collection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'bytes':result['bytes'],'files':len(records),'collection':str(here/'collection.json')}))
