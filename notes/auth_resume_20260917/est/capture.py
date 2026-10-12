"""Read-only fixed-path metadata capture. No data-file reads or remote writes."""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REMOTE = r'''
import os,sys,json,hashlib,stat,datetime
request=json.load(sys.stdin)
result={'utc':datetime.datetime.utcnow().isoformat()+'Z','records':[],'read_bytes':0}
for action,path in request:
    r={'action':action,'path':path}
    try:
        st=os.stat(path)
        r.update(size=st.st_size,mtime_ns=st.st_mtime_ns)
        if action=='list':
            r['entries']=[]
            for entry in sorted(os.scandir(path),key=lambda e:e.name):
                s=entry.stat(follow_symlinks=False)
                r['entries'].append({'name':entry.name,'size':s.st_size,'directory':stat.S_ISDIR(s.st_mode)})
                if len(r['entries'])>=300: raise ValueError('listing cap reached')
        elif action=='read':
            if not stat.S_ISREG(st.st_mode) or st.st_size>1048576: raise ValueError('per-file cap/nonregular')
            if result['read_bytes']+st.st_size>1048576*16: raise ValueError('total cap')
            with open(path,'rb') as f: data=f.read(st.st_size)
            result['read_bytes']+=len(data)
            after=os.stat(path)
            if (after.st_size,after.st_mtime_ns)!=(st.st_size,st.st_mtime_ns): raise ValueError('file changed')
            r.update(sha256=hashlib.sha256(data).hexdigest(),text=data.decode('utf8'))
        elif action!='stat': raise ValueError('unknown action')
    except Exception as exc: r['error']=str(exc)
    result['records'].append(r)
print(json.dumps(result,indent=2))
'''
import shlex
name=sys.argv[1]
actions=json.loads((HERE/(name+'.request.json')).read_text())
previous=sum(json.loads(p.read_text()).get('read_bytes',0) for p in HERE.glob('*.capture.json'))
if previous+sum(1048576 for a,p in actions if a=='read')>16*1048576:
    raise SystemExit('conservative cumulative reservation exceeds cap')
cmd='/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python -B -c '+shlex.quote(REMOTE)
r=subprocess.run(['ssh','pronghorn',cmd],input=json.dumps(actions),text=True,capture_output=True,timeout=100)
(HERE/(name+'.stderr.txt')).write_text(r.stderr)
if r.returncode: raise SystemExit(r.stderr)
data=json.loads(r.stdout)
assert previous+data['read_bytes']<=16*1048576
with (HERE/(name+'.capture.json')).open('x') as f: f.write(r.stdout)
print(json.dumps(data,indent=2))
