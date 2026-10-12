"""Read-only bounded SSH metadata collection. Never imports panel code or opens DBs."""
import datetime, hashlib, json, pathlib, shlex, subprocess
HERE = pathlib.Path(__file__).resolve().parent
NOTES = HERE.parents[1]
manifest_path = NOTES/'power_recovery_20260917/panel_arrays/copied/panel/manifest.json'
b = manifest_path.read_bytes()
assert hashlib.sha256(b).hexdigest() == '7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e'
m = json.loads(b)
paths = [r['path'] for r in m['inputs']]
paths += ['/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916/outputs/job-6165628/panel/manifest.json', '/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916/outputs/job-6165628/panel/COMPLETE', '/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif', '/data/gpfs/assoc/pgl/singularity/transgenic-ngc-26.08.sif']
code = '''import os,json,platform,shutil
from pathlib import Path
paths = %r
rows=[]
for name in paths:
 p=Path(name)
 try:
  s=p.stat(); rows.append(dict(path=name,resolved=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,wal=os.path.lexists(name+'.wal')))
 except OSError as e: rows.append(dict(path=name,error=str(e)))
print(json.dumps(dict(machine=platform.machine(),host=platform.node(),python=platform.python_version(),singularity=shutil.which('singularity'),apptainer=shutil.which('apptainer'),files=rows),indent=2))
''' % paths
cmd = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pronghorn','/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11 -B -c '+shlex.quote(code)]
r = subprocess.run(cmd, text=True, capture_output=True, timeout=40)
record = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,scope='stat-only; no payload hashes, arrays, DB opens or jobs')
with (HERE/'remote_metadata_explicit_python.json').open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print(r.stdout); print(r.stderr)
