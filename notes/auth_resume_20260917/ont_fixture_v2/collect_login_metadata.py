"""Local driver: read-only SSH metadata command, writes evidence ONLY locally."""
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
runtime = json.loads((HERE/'runtime.json').read_text())
source = (HERE/'runtime_closure.py').read_text()
payload = source+'\nRUNTIME = '+repr(runtime)+'''
import datetime, platform, socket
operational = dict(scontrol='/cm/shared/apps/slurm/current/bin/scontrol',
                   srun='/cm/shared/apps/slurm/current/bin/srun',
                   sbatch='/cm/shared/apps/slurm/current/bin/sbatch', timeout='/usr/bin/timeout')
RUNTIME['operations'] = {k:dict(path=str(Path(v).resolve(strict=True)), sha256=identity(v)['sha256']) for k,v in operational.items()}
env = dict(PATH=str(Path(RUNTIME['tools']['minimap2']['path']).parent), HOME='/tmp', TMPDIR='/tmp',
           LANG='C', LC_ALL='C', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', GOMAXPROCS='4')
config = subprocess.run([operational['scontrol'], 'show', 'config'], capture_output=True, text=True, timeout=15, env=env)
keys = ('ProctrackType', 'TaskPlugin', 'JobAcctGatherType', 'JobAcctGatherParams', 'KillWait', 'UnkillableStepTimeout', 'SlurmctldVersion', 'SlurmdParameters')
selected = [line.strip() for line in config.stdout.splitlines() if line.strip().split(' ',1)[0] in keys]
site = {}
for name in ('/etc/slurm/cgroup.conf', '/etc/slurm-llnl/cgroup.conf', '/cm/shared/apps/slurm/current/etc/cgroup.conf'):
    p=Path(name)
    if p.is_file():
        site[name] = dict(identity=identity(p), settings=[line.strip() for line in p.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')])
print(json.dumps(dict(closure=collect(RUNTIME, env), runtime=RUNTIME,
    observation=dict(host=socket.gethostname(), platform=platform.platform(), architecture=platform.machine(),
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     effective_environment=env, launcher_mode='env -i Python -I -B',
                     scheduler_paths={k:identity(v) for k,v in operational.items()},
                     scheduler_config_returncode=config.returncode, scheduler_config_selected=selected,
                     site_cgroup_configuration=site)), sort_keys=True))
'''
command = '/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 '+shlex.quote(runtime['tools']['python']['path'])+' -I -B -c '+shlex.quote(payload)
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'pronghorn', command], capture_output=True, text=True, timeout=180)
evidence = HERE/'runtime_evidence'; evidence.mkdir(exist_ok=True)
(evidence/'ssh_metadata.stderr.txt').write_text(result.stderr)
if result.returncode:
    (evidence/'metadata_failure.txt').write_text(result.stdout+'\nexit='+str(result.returncode)+'\n')
    raise SystemExit('metadata collection failed; see runtime_evidence')
data=json.loads(result.stdout)
for name,value in [('expected_closure.json', data['closure']), ('login_observation.json', data['observation']), ('runtime_with_operations.json',data['runtime'])]:
    (evidence/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
print(str(evidence))
