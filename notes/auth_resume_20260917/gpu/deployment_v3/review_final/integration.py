"""Tiny local artifacts only. Real archived checkpoint main/copies/hashes.
Heavy runtime probes are not run. Publication uses a DrvFS-only adapter below;
this is output-inventory verification, not target-filesystem atomicity testing.
"""
import ast
import copy
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import types
from unittest.mock import patch
OUT=Path(__file__).resolve().parent
R=OUT.parent/'release_v5'
SOURCE=OUT/'tmp/source'
sys.path.insert(0,str(R))
import admission as a
import deploy as d
sys.path.insert(0,str(SOURCE/'revision/scripts/mvp'))
import stage_checkpoint as stage
import make_launch_readiness as ready
text=(SOURCE/'src/transgenic/training/mvp_finalize.py').read_text()
node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_rename_noreplace')
helper=types.ModuleType('transgenic.training.mvp_finalize')
helper.__dict__.update(dict(Path=Path,os=os,sys=sys,ctypes=ctypes,errno=errno))
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE/'src/transgenic/training/mvp_finalize.py'),'exec'),helper.__dict__)
# DrvFS returns EINVAL for renameat2(RENAME_NOREPLACE). Only this local
# filesystem adapter is substituted, not checkpoint copying/inventory/hashing.
def local_publish(source,destination):
    if os.path.lexists(destination): raise FileExistsError(str(destination))
    os.rename(source,destination)
helper._rename_noreplace=local_publish
sys.modules[helper.__name__]=helper

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,obj): p.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
results={}
for mutation in ('clean','source_unknown','source_regular_changed','destination_unknown','prepare_wrong_seed'):
    with tempfile.TemporaryDirectory(dir=OUT/'tmp') as tmp:
        root=Path(tmp); run=root/'seed123'; epoch=run/'epoch_05'; epoch.mkdir(parents=True)
        for name,raw in [('model.safetensors',b'tiny synthetic weights'),('meta.json',b'{}'+b' '*130),('eval.json',b'{}')]: (epoch/name).write_bytes(raw)
        for name in ('accelerate_state','pending_grads'):
            (epoch/name).mkdir(); (epoch/name/'untouched').write_bytes(b'synthetic resume state')
        (run/'run_config.json').write_text('{"seed":123}')
        (run/'runtime_identity.txt').write_text('synthetic runtime')
        inputs=root/'inputs'; inputs.mkdir()
        control=root/'control';control.mkdir()
        scratch=root/'scratch';scratch.mkdir()
        for n in ('inductor','triton'): (scratch/n).mkdir()
        record(control/'quota-checkpoint.json',dict(path=str(inputs),available_bytes=10**12,measured_at_unix=time.time()))
        pinfiles=dict(weights=epoch/'model.safetensors',meta=epoch/'meta.json',eval=epoch/'eval.json',run_config=run/'run_config.json',runtime_identity=run/'runtime_identity.txt')
        before={str(p.relative_to(run)):(p.read_bytes(),p.stat().st_ino,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in run.rglob('*') if p.is_file()}
        commands=[]
        def mapped(value):
            value=str(value)
            for old,new in [('/work/repo',SOURCE),('/source/seed123',run),('/mvp/inputs',inputs),('/mvp/probe/p1_control',control),('/scratch',scratch)]:
                if value==old or value.startswith(old+'/'): return Path(str(new)+value[len(old):])
            return Path(value)
        def execute(argv):
            argv=[str(x) for x in argv]; commands.append(argv)
            if argv[1].endswith('/stage_checkpoint.py'):
                assert stage.main(argv[2:])==0
                if mutation=='source_unknown': (epoch/'unknown').write_text('reject')
                if mutation=='source_regular_changed': (epoch/'model.safetensors').write_bytes(b'changed source')
                if mutation=='destination_unknown': (inputs/'b5_seed123_epoch05/checkpoint/unknown').write_text('reject')
            elif argv[1].endswith('/make_launch_readiness.py'):
                assert argv[2]=='prepare' and argv[argv.index('--profile')+1]=='development-single' and argv[argv.index('--seed')+1]=='123'
                parsed=ready.build_parser().parse_args(argv[2:])
                assert parsed.command=='prepare' and parsed.profile=='development-single' and type(parsed.seed) is int and parsed.seed==123
                bindings=dict(profile='development-single',seed=1 if mutation=='prepare_wrong_seed' else 123,run_role='mvp_finetune')
                record(control/'prepared.json',dict(bindings=bindings,bindings_sha256=d.binding_digest(bindings)))
                record(control/'consumed-code.json',{'fixture':True})
            elif argv[1].endswith('/run_bounded.py'):
                assert argv[argv.index('--timeout-seconds')+1]=='900'
            else: raise AssertionError(argv)
        def tiles(c):
            record(c/'tiles.json',{'fixture':True}); record(c/'tile-availability.json',dict(status='available_and_actual_measured_batches_checked'))
        probe=types.ModuleType('real_probes');probe.tree_identity=lambda p: {'fixture':True}
        with patch.object(d,'RESERVE',0),patch.object(d,'Path',side_effect=mapped),patch.dict(d.PINS,{k:digest(v) for k,v in pinfiles.items()}),patch.object(d,'run',side_effect=execute),patch.object(d,'select_tiles',side_effect=tiles),patch.dict(sys.modules,real_probes=probe),patch.dict(os.environ,MVP_DATA_MOVER_ALLOCATION_ID='LOCAL_TINY_FIXTURE_NOT_A_REAL_ALLOCATION'):
            error=None
            try: d.container_stage()
            except ValueError as e: error=str(e)
            if mutation=='clean':
                assert error is None,error
                checkpoint=inputs/'b5_seed123_epoch05/checkpoint'
                assert {p.name for p in checkpoint.iterdir()}=={'model.safetensors','meta.json','eval.json'}
                assert d.checkpoint_inventory(checkpoint)==[]
                assert ready.optional_sidecars(checkpoint)=={'config.json':None,'runtime.json':None,'meta.json':{'sha256':digest(epoch/'meta.json'),'bytes':132}}
                after={str(p.relative_to(run)):(p.read_bytes(),p.stat().st_ino,p.stat().st_mtime_ns,p.stat().st_ctime_ns) for p in run.rglob('*') if p.is_file()}
                assert before==after
                d.container_p1(); assert len(commands)==3
                bad=json.loads((control/'prepared.json').read_text());bad['bindings']['seed']=1;bad['bindings_sha256']=d.binding_digest(bad['bindings'])
                (control/'prepared.json').chmod(0o600)
                (control/'s0-inputs.json').chmod(0o600)
                record(control/'prepared.json',bad)
                hashes=json.loads((control/'s0-inputs.json').read_text());hashes['prepared.json']=digest(control/'prepared.json');record(control/'s0-inputs.json',hashes)
                try: d.container_p1()
                except ValueError as e: assert 'profile' in str(e)
                else: raise AssertionError('P1 admitted wrong seed')
                assert len(commands)==3
                results[mutation]=dict(stage_inventory=sorted(p.name for p in checkpoint.iterdir()),run_root_inventory=sorted(p.name for p in checkpoint.parent.iterdir()),source_unchanged=True,prepare_profile='development-single',prepare_seed=123,p1_wrong_seed_rejected=True)
            else:
                assert error is not None,mutation
                assert len(commands)==(2 if mutation=='prepare_wrong_seed' else 1)
                results[mutation]=dict(rejected=error,commands_before_rejection=len(commands))
# Exercise both complete live snapshots, through the real runner with harmless child outputs.
sys.path.insert(1,str(OUT.parent))
from test_admission_cases import LivePathTests
case=LivePathTests()
def advancing(argv,out,commands):
    if argv[0]=='/usr/bin/scontrol' and 'job' in argv and sum(c==argv for c in commands)==2:
        out=out.strip().replace('RunTime=00:01:00','RunTime=00:01:17')+' LastSchedEval=2026-09-18T00:01:17\n'
    if argv[0]=='/usr/bin/scontrol' and 'node' in argv and sum(c==argv for c in commands)==2:
        out=out.strip()+' CPULoad=9.99 FreeMem=123\n'
    return out,'',0
case.run_live(advancing)
results['two_snapshots_advancing_elapsed_and_volatile_node_fields']='admitted'
def changed_capacity(argv,out,commands):
    if argv[0]=='/usr/bin/scontrol' and 'node' in argv and sum(c==argv for c in commands)==2: out=out.replace('288','128')
    return out,'',0
try: case.run_live(changed_capacity)
except ValueError as e: results['changed_final_capacity']=str(e)
else: raise AssertionError('capacity change admitted')
def changed_request(argv,out,commands):
    if argv[0]=='/usr/bin/scontrol' and 'job' in argv and sum(c==argv for c in commands)==2: out=out.replace('CPUsPerTask=64','CPUsPerTask=288')
    return out,'',0
try: case.run_live(changed_request)
except ValueError as e: results['changed_final_request']=str(e)
else: raise AssertionError('request change admitted')
# Supplement capture suite: invalid encoding, failed spawn, timeout with descendant holding pipes.
import subprocess
capture={}
for label,argv,timeout in [('decode',[sys.executable,'-c',"import os;os.write(1,b'\\xff')"],2),('spawn',['/nonexistent/local-review-command'],1),('descendant-timeout',[sys.executable,'-c',"import os,time; p=os.fork(); time.sleep(2)"],0.05)]:
    with tempfile.TemporaryDirectory(dir=OUT/'tmp') as tmp:
        try: a.query(Path(tmp),label,argv,timeout)
        except ValueError: pass
        else: raise AssertionError('bad query admitted')
        r=json.loads((Path(tmp)/(label+'.json')).read_text())
        assert 'error' in r and r['captured_bytes']<=16777216
        capture[label]={k:r.get(k) for k in ('error','timed_out','exit_code','captured_bytes','kill_error','reap_error')}
results['capture_extra']=capture
results['limitations']='Local tiny fixtures. Actual archived stage main, copy, fsync and hashes executed. Local DrvFS rejects renameat2 with EINVAL, so publication used a fixture-only no-existing-path os.rename adapter, not production atomicity verification. Tiny-fixture reserve set to zero because local disk has less than 40GiB free. Prepare/GPU/tile work replaced with explicit fixture outputs. No real allocation, data hashing, or remote actions.'
record(OUT/'integration-results.json',results)
print(json.dumps(results,indent=2))
