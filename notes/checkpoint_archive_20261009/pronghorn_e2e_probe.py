"""Run on pronghorn; exercises only explicitly marked tiny generated fixtures."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import uuid

revision = sys.argv[1]
pg_install = Path('/data/gpfs/assoc/pgl/data/Transgenic/ops/checkpoint_archive') / revision
remote_install = '/work/nvme/bilv/wcyim/transgenic/ops/checkpoint_archive/' + revision
cli = pg_install / 'revision/scripts/b6/checkpoint_archive_cli.py'
remote_cli = remote_install + '/revision/scripts/b6/checkpoint_archive_cli.py'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'dtai']

def delta(code, *args):
    result = subprocess.run(ssh + [shlex.join(['python3', '-c', code] + list(args))], capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout

create = r'''
from pathlib import Path
import hashlib,json,os,sys
nonce=sys.argv[1]
source=Path('/work/nvme/bilv/wcyim/transgenic/runs')/('.archive_e2e_'+nonce)
hdd=Path('/work/hdd/bilv/wcyim/transgenic')/('.archive_e2e_'+nonce)
for p in (source,hdd):
 p.mkdir(); (p/'.probe-owned').write_text(nonce)
run=source/'run space $value';run.mkdir()
for number in range(1,5):
 e=run/('epoch_%02d'%number); a=e/'accelerate_state';a.mkdir(parents=True)
 (e/'meta.json').write_text(json.dumps({'epoch':number,'global_step':number*10,'seed':123,'pending_micro':0}))
 (e/'eval.json').write_text(json.dumps({'epoch':number,'global_step':number*10}))
 for rel in ('model.safetensors','accelerate_state/model.safetensors','accelerate_state/optimizer.bin','accelerate_state/scheduler.bin','accelerate_state/random_states_0.pkl','extra $file [value].bin'):
  (e/rel).write_bytes((('fixture-%d-%s|'%(number,rel)).encode()*20000)[:262144])
(run/'best').symlink_to('epoch_01'); (run/'latest_state').mkdir(); (run/'latest_state'/'KEEP').write_text('resume-state')
expected={str(p.relative_to(run/'epoch_02')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'epoch_02').rglob('*') if p.is_file()}
print(json.dumps({'source':str(source),'run':str(run),'hdd':str(hdd),'expected':expected}))
'''
inspect = r'''
from pathlib import Path
import json,sys
run=Path(sys.argv[1]);print(json.dumps({'epochs':{str(n):(run/('epoch_%02d'%n)).exists() for n in range(1,5)},'latest':(run/'latest_state'/'KEEP').read_text(),'best':str((run/'best').readlink()) if hasattr(Path,'readlink') else __import__('os').readlink(str(run/'best'))}))
'''
cleanup = r'''
from pathlib import Path
import sys,shutil
nonce=sys.argv[1]
for base in ('/work/nvme/bilv/wcyim/transgenic/runs','/work/hdd/bilv/wcyim/transgenic'):
 p=Path(base)/('.archive_e2e_'+nonce)
 assert p.name=='.archive_e2e_'+nonce and (p/'.probe-owned').read_text()==nonce
 shutil.rmtree(str(p))
print('DELTA_GENERATED_FIXTURES_REMOVED')
'''

def invoke(config, flags, want=0):
    result = subprocess.run([sys.executable, str(cli), '--config', str(config)] + flags,
                            capture_output=True, text=True, timeout=150)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if (want == 0 and result.returncode != 0) or (want != 0 and result.returncode == 0):
        raise AssertionError((result.returncode, result.stdout, result.stderr))
    return records

for index, route in enumerate(('direct-timeout-retry', 'direct', 'hdd-forward'), 1):
    nonce = uuid.uuid4().hex
    source = json.loads(delta(create, nonce))
    try:
        with tempfile.TemporaryDirectory(prefix='.archive_e2e_', dir='/data/gpfs/assoc/pgl/data/Transgenic/archives') as local:
            destination = Path(local) / 'archive space $target'
            target = {'name':'fixture','transport':'ssh-pull','source_kind':'run','source_root':source['run'],
                      'destination_root':str(destination),'storage':{'kind':'local'},'min_age_seconds':0,
                      'min_free_bytes':1024**3,'max_bytes':8*1024**2,'max_checkpoints':1,
                      'remote':{'host':'dtai','python':'/usr/bin/python3','helper':remote_cli,'connect_timeout':10,'timeout':120},
                      'rsync_timeout':60}
            if route == 'hdd-forward':
                hdd_target=dict(target)
                hdd_target.update(transport='local',source_kind='run',source_root=source['run'],destination_root=source['hdd']+'/archive',
                                  storage={'kind':'lustre','required_pool':'ddn_hdd','required_project':22169,'mount':'/work/hdd/bilv'})
                hdd_target.pop('remote');hdd_target.pop('rsync_timeout')
                setup="from pathlib import Path; import sys,json; p=Path(sys.argv[1])/'hdd-config.json'; p.write_text(sys.argv[2]); print(str(p))"
                hdd_config=delta(setup,source['source'],json.dumps({'version':1,'targets':[hdd_target]})).strip()
                output=subprocess.check_output(ssh+[shlex.join(['/usr/bin/python3',remote_cli,'--config',hdd_config,'--apply','--remove-source'])],text=True,timeout=150)
                hdd_receipt=[json.loads(line)['receipt'] for line in output.splitlines() if json.loads(line)['status']=='verified'][0]
                target.update(source_kind='archive',source_root=hdd_receipt['archive_path'])
            config=Path(local)/'config.json'
            config.write_text(json.dumps({'version':1,'targets':[target]}))
            planned=invoke(config,['--once'])
            assert not destination.exists(), 'dry-run created destination'
            assert planned[0]['status']=='planned',planned
            if route == 'direct-timeout-retry':
                target['rate_limit_bytes']=32768;target['remote']['timeout']=2;target['rsync_timeout']=1
                config.write_text(json.dumps({'version':1,'targets':[target]}))
                failed=invoke(config,['--apply','--remove-source'],want=1)
                assert any(row['status']=='error' for row in failed),failed
                state=json.loads(delta(inspect,source['run']))
                assert state['epochs']['2'],state
                target.pop('rate_limit_bytes');target['remote']['timeout']=120;target['rsync_timeout']=60
                config.write_text(json.dumps({'version':1,'targets':[target]}))
            copied=invoke(config,['--apply'])
            receipt=[row['receipt'] for row in copied if row['status']=='verified'][0]
            archive=Path(receipt['archive_path'])
            for relative,digest in source['expected'].items():
                assert hashlib.sha256((archive/relative).read_bytes()).hexdigest()==digest,relative
            assert json.loads(delta('from pathlib import Path; import sys,json; print(json.dumps(Path(sys.argv[1]).exists()))',target['source_root']))
            removed=invoke(config,['--apply','--remove-source','--watch','--cycles','2','--interval','0.01'])
            assert any(row['status']=='removed' for row in removed),removed
            state=json.loads(delta(inspect,source['run']))
            assert state['epochs']=={'1':True,'2':False,'3':True,'4':True},state
            assert state['latest']=='resume-state' and state['best']=='epoch_01',state
            if route == 'hdd-forward':
                assert not json.loads(delta('from pathlib import Path; import sys,json; print(json.dumps(Path(sys.argv[1]).exists()))',target['source_root']))
            for relative,digest in source['expected'].items():
                assert hashlib.sha256((archive/relative).read_bytes()).hexdigest()==digest
            print(json.dumps({'case':route,'iteration':index,'result':'PASS','files':len(source['expected']),
                              'manifest_id':receipt['manifest_id'],'protected_epochs':[1,3,4],'latest_preserved':True}),flush=True)
    finally:
        print(delta(cleanup,nonce).strip(),flush=True)
print('ALL_PRONGHORN_GENERATED_FIXTURES_REMOVED',flush=True)
