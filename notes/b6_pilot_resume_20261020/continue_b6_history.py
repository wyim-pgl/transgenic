#!/usr/bin/env python3
"""B6-only receipt-bound continuation. Host Python3.6/stdlib; no tensor reads."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
C2_DIR = HERE/'c2' if (HERE/'c2').is_dir() else HERE.parent/'checkpoint_archive_20261009/c2_failure_3345919/resume_determinism'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
infra=load('b6_unchanged_host_infrastructure',C2_DIR/'continue_c2_history.py')
contract=load('b6_pure_receipt_contract',HERE/'b6_receipt_contract.py')
# Explicit infrastructure imports; no C2 Controller inheritance or global mutation.
for _name in ('Blocked','Wait','require','integer','digest_ok','safe_path','regular','read_bytes','sha',
              'encoded','canonical_sha','read_json','fsync_dir','atomic_json','exclusive_json','Lock',
              'checked_pin','source_member_sha','manifest_check','Slurm','terminal_ok','submission_token','submission_command'):
    globals()[_name]=getattr(infra,_name)
SOURCE='198ce14ce53266f53b7bda5d0c4cdfd94deb43b4'
PROFILE=dict(arm='B6',target=4099,seed=123,aug_seed=20261037,world_size=4,batch_size=1,
             mixed_precision='bf16',deterministic=True,checkpointing='nonreentrant')
FINAL_MARKERS=infra.FINAL_MARKERS
PROOF_MEMBERS={'b6_history_proof.py','b6_history_proof.sbatch','test_b6_history_proof.py','b6_receipt_contract.py','test_b6_receipt_contract.py','c2/history_proof.py','c2/probe.py','c2/production_link.py'}
PRODUCTION_MEMBERS={'b6_production_link.py','b6_production_link.sbatch','continue_b6_history.py','b6_receipt_contract.py','c2/continue_c2_history.py','c2/probe.py'}
SOURCE_MEMBERS={'train/train_HyenaTransgenic_B6.py','src/transgenic/training/b6_runtime.py','src/transgenic/training/b6_compile_history.py',
                'src/transgenic/training/b6_compile_replay.py','src/transgenic/training/b6_replay_state.py','revision/scripts/b6/storage_preflight.py'}
PHASES={'waiting_validation','validation_accepted','retire_intent','retire_waiting','both_old_jobs_retired','activation_intent',
        'candidate_activated','production_submit_intent','submission_uncertain','waiting_production','link_accepted','final_verified','complete'}


def validate_identity(observed,pin):
    require(type(observed) is dict,'missing job observation')
    for key in ('job_id','owner','uid','created','token'):
        require(type(observed.get(key)) is type(pin[key]) and observed[key]==pin[key],'job identity mismatch: '+key)
    require(submission_token(observed.get('token'),observed.get('argv',[]))==pin['token'],'job token mismatch')
    actual=list(observed.get('argv',[]));expected=list(pin['argv'])
    command=submission_command(actual,observed.get('work_dir'))
    require(command==pin['command']==submission_command(expected,pin['work_dir']) and observed.get('command')==command,'job full script mismatch')
    actual[-1]=command;expected[-1]=pin['command']
    require(actual==expected,'job full submission argv mismatch')


def validate_config(c):
    fields=('version operation_id owner uid work sif image_sha256 source_commit source_snapshot_git source_dir source_manifest_sha256 '
            'phases proof_root proof_job_id proof_identity obsolete_jobs bindings bindings_sha256 config_relative config_sha256 '
            'expected_runtime_fingerprint storage_python storage_script resources').split()
    require(type(c) is dict and set(c)==set(fields),'config fields mismatch')
    require(type(c['version']) is int and c['version']==1 and c['owner']=='wcyim' and type(c['uid']) is int and c['uid']==97434,'wrong config owner/version')
    require(re.fullmatch('[A-Za-z0-9_-]{1,64}',c['operation_id']) is not None,'invalid operation ID')
    require(Path(c['work']).is_absolute() and Path(c['work']).is_dir(),'WORK must exist')
    require(c['source_commit']==SOURCE and digest_ok(c['source_snapshot_git'],40),'wrong immutable source')
    require(c['sif']=='/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif','wrong NGC image')
    for key in ('image_sha256','source_manifest_sha256','bindings_sha256','config_sha256'): require(digest_ok(c[key]),'invalid '+key)
    for key in ('source_dir','proof_root','bindings','storage_script'):safe_path(c['work'],c[key])
    safe_path(safe_path(c['work'],c['source_dir']),c['config_relative'])
    require(set(c['phases'])=={'proof','production'},'phase set mismatch')
    for p in c['phases'].values():
        require(set(p)=={'helper_dir','script','helper_manifest_sha256'},'phase fields mismatch')
        safe_path(c['work'],p['helper_dir']);safe_path(c['work'],p['script']);require(digest_ok(p['helper_manifest_sha256']),'helper pin')
    require(re.fullmatch('[0-9]+',c['proof_job_id']) is not None and c['proof_identity']['job_id']==c['proof_job_id'],'proof identity mismatch')
    require([p['job_id'] for p in c['obsolete_jobs']]==['3345918','3345917'],'retirement order differs')
    for p in [c['proof_identity']]+c['obsolete_jobs']:
        require(p['owner']==c['owner'] and p['uid']==c['uid'],'pinned job owner mismatch')
        validate_identity(p,p)
    require(c['proof_identity']['command']==str(safe_path(c['work'],c['phases']['proof']['script'])),'proof script mismatch')
    resources=c['resources'];require(set(resources)=={'account','partition','time','cpus_per_task','exclude','output'},'resource fields')
    require(resources['partition']=='ghx4' and resources['account']=='bilv-dtai-gh' and integer(resources['cpus_per_task'],1),'production resources')
    match=re.fullmatch(r'(\d{1,2}):([0-5]\d):([0-5]\d)',resources['time'])
    require(match is not None and 0<sum(int(x)*y for x,y in zip(match.groups(),(3600,60,1)))<=48*3600,'walltime must be <=48h')
    safe_path(c['work'],resources['output'])
    require(type(resources['exclude']) is str and resources['exclude'],'explicit node exclusion required')
    return c


class Controller:
    @classmethod
    def initialize(cls,config,state_dir):
        c=validate_config(config);state=Path(state_dir)
        require(state.is_absolute(),'state must be absolute');safe_path(c['work'],str(state.relative_to(c['work'])))
        protected=[c['proof_root'],c['source_dir']]+[p['helper_dir'] for p in c['phases'].values()]
        for relative in protected:
            path=safe_path(c['work'],relative)
            require(state!=path and path not in state.parents and state not in path.parents,'state overlaps protected artifacts')
        lock=Path(c['work'])/'.b6_history.lock';claim=Path(c['work'])/'.b6_history.owner.json'
        with Lock(lock):
            require(not os.path.lexists(str(state)) and not os.path.lexists(str(claim)),'B6 operation already exists; inspect/reconcile')
            state.mkdir();fsync_dir(state.parent)
            exclusive_json(state/'config.json',c)
            journal=dict(version=1,operation_id=c['operation_id'],config_sha256=sha(state/'config.json'),phase='waiting_validation',
                revision=0,intent=None,receipts={},retirements=[],links=[],epoch_events=[],last_error=None,last_wait=None)
            atomic_json(state/'journal.json',journal)
            exclusive_json(claim,dict(operation_id=c['operation_id'],state_dir=str(state),config_sha256=journal['config_sha256']))
    def __init__(self,state_dir,scheduler=None):
        self.state=Path(state_dir);self.c=validate_config(read_json(self.state/'config.json'))
        safe_path(self.c['work'],str(self.state.relative_to(self.c['work'])))
        self.lock_path=Path(self.c['work'])/'.b6_history.lock'
        self.scheduler=scheduler or Slurm();self._fresh_intent=None;self.j=self.status()
    def path(self,relative):return safe_path(self.c['work'],relative)
    def status(self):
        j=read_json(self.state/'journal.json')
        require(j.get('version')==1 and j.get('operation_id')==self.c['operation_id'] and j.get('phase') in PHASES and
                j.get('config_sha256')==sha(self.state/'config.json'),'journal/config drift')
        return j
    @contextlib.contextmanager
    def locked(self):
        with Lock(self.lock_path):
            claim=read_json(Path(self.c['work'])/'.b6_history.owner.json')
            require(claim==dict(operation_id=self.c['operation_id'],state_dir=str(self.state),config_sha256=sha(self.state/'config.json')),'logical B6 ownership drift')
            self.j=self.status();yield
    def save(self,phase,**values):
        values.setdefault('last_error',None);values.setdefault('last_wait',None)
        self.j.update(values);self.j.update(phase=phase,revision=self.j['revision']+1,updated=time.time())
        atomic_json(self.state/'journal.json',self.j);return self.j
    def pins(self):
        source=self.path(self.c['source_dir'])
        require(read_bytes(source/'SOURCE_COMMIT').decode().strip()==SOURCE,'source commit drift')
        result=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
        require(result.returncode==0 and result.stdout.strip()==self.c['source_snapshot_git'],'source snapshot drift')
        members=manifest_check(source,'validation_sources.sha256',self.c['source_manifest_sha256'])
        require(SOURCE_MEMBERS|{self.c['config_relative']}<=members,'source manifest incomplete')
        for kind,phase in self.c['phases'].items():
            helper=self.path(phase['helper_dir']);members=manifest_check(helper,'manifest.sha256',phase['helper_manifest_sha256'])
            require((PROOF_MEMBERS if kind=='proof' else PRODUCTION_MEMBERS)<=members,'helper manifest incomplete')
            require(str(self.path(phase['script']).relative_to(helper)) in members,'script not manifested')
        require(sha(self.path(self.c['bindings']))==self.c['bindings_sha256'],'bindings drift')
        require(sha(safe_path(source,self.c['config_relative']))==self.c['config_sha256'],'config drift')
    def active_run(self):return self.c['proof_root']+'/'+self.c['proof_job_id']+'/resumed'
    def storage(self):
        try:value=self.scheduler.storage(self.c,self.path(self.active_run()))
        except Blocked as exc:
            if 'RuntimeError: insufficient CP4 storage:' in str(exc):raise Wait(str(exc))
            raise
        require(value.get('project_id')==22169 and value.get('pool')=='ddn_ssd','storage project/pool mismatch')
        for key in ('quota_remaining_bytes','filesystem_available_bytes'):
            require(integer(value.get(key)),'invalid storage observation')
            if value[key]<100*1024**3:raise Wait('below 100GiB storage reserve: '+key)
        return value
    def held_old(self,observation,pin):
        validate_identity(observation,pin)
        require(observation.get('active') is True and observation.get('state')=='PENDING' and
                observation.get('reason')=='JobHeldUser','obsolete job no longer same held PENDING allocation')
    def no_writers(self):
        retired={p['job_id'] for p in self.j['retirements']}
        old={p['job_id']:p for p in self.c['obsolete_jobs']}
        allowed={self.c['proof_job_id']}|{p['job_id'] for p in self.j['links']}
        for job in self.scheduler.active(self.c['owner']):
            if job['job_id'] in old:
                require(job['job_id'] not in retired,'retired job active again')
                self.held_old(self.scheduler.observe(job['job_id'],self.c['owner']),old[job['job_id']]);continue
            if job['job_id'] in allowed:
                require((job['job_id']==self.c['proof_job_id'] and self.j['phase']=='waiting_validation') or
                        (self.j['phase']=='waiting_production' and job['job_id']==self.j['links'][-1]['job_id']),
                        'accepted allocation became active again')
                continue
            command=job.get('command','');token=job.get('token','')
            require('b6' not in command.lower() and not token.startswith('b6-') and
                    self.active_run() not in command,'competing B6 writer: '+job['job_id'])
    def checkpoint_small(self,run,relative,position,payload=True):
        directory=safe_path(run,relative);meta=read_json(directory/'meta.json');identity=read_json(directory/'identity.json')
        contract.identity(identity)
        require(all(integer(meta.get(k)) and meta[k]==v for k,v in position.items()),'checkpoint position drift')
        require(meta.get('arm')=='B6' and type(meta.get('seed')) is int and meta['seed']==123,'checkpoint profile drift')
        history=read_json(directory/'compile_history.json');history_sha=sha(directory/'compile_history.json')
        require(meta.get('compile_history_sha256')==history_sha and meta.get('identity_sha256')==canonical_sha(identity),'checkpoint embedded digest mismatch')
        require(type(history.get('version')) is int and history['version']==1 and type(history.get('world_size')) is int and history['world_size']==4 and
                encoded(history.get('checkpoint'))==encoded(position) and history.get('identity')==identity and
                history.get('fingerprint')==self.c['expected_runtime_fingerprint'],'history identity/position/runtime drift')
        require(history.get('bootstrap','missing') is None and type(history.get('entries')) is list and history['entries'] and
                all(e.get('origin')=='observed' for e in history['entries']),'history requires null bootstrap and observed entries')
        for entry in history['entries']:
            require([r.get('rank') for r in entry.get('ranks',[])]==[0,1,2,3],'history rank set')
        if payload:
            for name in ['model.safetensors','optimizer.bin','scheduler.bin']+['random_states_%d.pkl'%r for r in range(4)]:
                require(regular(directory/'accelerate_state'/name).stat().st_size>0,'empty checkpoint payload')
            if position['pending_micro']:
                for rank in range(4):regular(directory/('pending_grads_rank%d.pt'%rank))
        return dict(meta_sha256=sha(directory/'meta.json'),identity_sha256=sha(directory/'identity.json'),history_sha256=history_sha)
    def proof_files(self,live_candidate=True):
        root=self.path(self.c['proof_root'])/self.c['proof_job_id']
        report=read_json(root/'comparison.json');contract.validate_report(report,SOURCE,self.c['proof_job_id'])
        verified=read_json(root/'verified_record.json');fresh=read_json(root/'fresh_start.json')
        provenance=dict(image_sha256=self.c['image_sha256'],source_manifest_sha256=self.c['source_manifest_sha256'],helper_manifest_sha256=self.c['phases']['proof']['helper_manifest_sha256'])
        require(verified.get('version')==1 and verified.get('kind')=='b6_fresh_allocation_verification' and
                verified.get('source_commit')==SOURCE and verified.get('job_id')==self.c['proof_job_id'] and
                verified.get('bindings_sha256')==self.c['bindings_sha256'] and verified.get('recipe_sha256')==self.c['config_sha256'] and verified.get('provenance')==provenance,'proof verification provenance mismatch')
        require(sha(root/'verified_record.json')==report['verified_record_sha256'] and sha(root/'fresh_start.json')==report['fresh_start_sha256'] and
                fresh.get('verified_record_sha256')==report['verified_record_sha256'] and fresh.get('job_id')==self.c['proof_job_id'],'fresh verification binding')
        require(report['runtime_fingerprint_sha256']==canonical_sha(self.c['expected_runtime_fingerprint']),'proof runtime changed')
        for phase in ('continuous','resumed'):
            for rank in range(4):
                key=phase+'/rank%d'%rank;path=root/phase/('prepared_schema_rank%d.json'%rank)
                require(sha(path)==report['prepared_schema_file_sha256'][key] and canonical_sha(read_json(path))==report['prepared_schema_sha256'] and
                        read_json(path)==report['prepared_schema'],'prepared schema mismatch')
                trace=regular(root/phase/('trace_rank%d.jsonl'%rank))
                with trace.open('rb') as stream:
                    first=stream.readline(16*1024*1024+1)
                require(len(first)<=16*1024*1024,'prepared trace record too large');row=json.loads(first.decode())
                require(row.get('event')=='prepared' and type(row.get('rank')) is int and row['rank']==rank and
                        row.get('micro')==(0 if phase=='continuous' else 9600) and row.get('prepared_schema_sha256')==sha(path),'prepared trace binding mismatch')
        for phase,position in [('continuous',contract.POSITION),('at400',contract.SNAPSHOT)]+([('resumed',contract.POSITION)] if live_candidate else []):
            bindings=self.checkpoint_small(root/phase,'latest_state',position)
            expected=report['checkpoint_bindings'][phase]
            require(all(bindings[k]==expected[k] for k in bindings),'proof checkpoint small binding changed')
            identity=read_json(root/phase/'latest_state/identity.json')
            require(identity==expected['identity'] and all(identity.get(k)==v for k,v in verified['identity_from_bindings'].items()),'proof input identity differs')
            require(identity['recipe_sha256']==verified['recipe_sha256'] and verified['recipe']==self.c['expected_runtime_fingerprint']['recipe'],'verified recipe mismatch')
        candidate=dict(relative_run='resumed',relative_checkpoint='latest_state',**contract.POSITION)
        candidate.update({k:report['checkpoint_bindings']['resumed'][k] for k in ('meta_sha256','identity_sha256','history_sha256')})
        candidate['canonical_checkpoint_sha256']=report['checkpoint_digests']['resumed']
        expected=dict(version=1,kind='b6_history_validation_ready',source_commit=SOURCE,validation_job_id=self.c['proof_job_id'],
            comparison_sha256=sha(root/'comparison.json'),verified_record_sha256=sha(root/'verified_record.json'),provenance=provenance,
            runtime_fingerprint_sha256=canonical_sha(self.c['expected_runtime_fingerprint']),candidate=candidate)
        ready=read_json(root/'validation_receipt.json');contract.validate_ready(ready,report,expected)
        return dict(ready_relative_path=str((root/'validation_receipt.json').relative_to(self.c['work'])),ready_sha256=sha(root/'validation_receipt.json'),
                    comparison_sha256=sha(root/'comparison.json'),candidate=candidate)
    def recheck_receipts(self,live_candidate=None):
        if live_candidate is None:
            # A persisted sbatch intent may already be running after a lost reply.
            live_candidate=not self.j['links'] and not (self.j['intent'] and self.j['intent']['kind']=='production')
        pin=self.j['receipts']['validation']
        self.observe_success(self.c['proof_job_id'],self.c['proof_identity'])
        current=self.proof_files(live_candidate=live_candidate)
        require(current==pin,'accepted proof drift')
        if 'activation' in self.j['receipts']:
            checked_pin(self.c['work'],self.j['receipts']['activation'])
            payload=read_json(self.state/'activation.json')
            require(payload['validation']==pin and payload['active_run']==self.active_run() and payload['retirements']==self.j['retirements'],'activation receipt drift')
            root=self.path(self.c['proof_root'])/self.c['proof_job_id']
            require(payload['comparison']==read_json(root/'comparison.json') and payload['ready']==read_json(root/'validation_receipt.json'),'embedded accepted proof drift')
            for name,key in [('meta.json','meta_sha256'),('identity.json','identity_sha256'),('compile_history.json','history_sha256')]:
                item=payload['initial'][name]
                require(hashlib.sha256(item['text'].encode()).hexdigest()==item['sha256']==pin['candidate'][key],'embedded initial410 binding drift')
    def observe_success(self,job_id,identity):
        observation=self.scheduler.observe(job_id,self.c['owner'])
        if observation is None:raise Wait('accounting unavailable')
        validate_identity(observation,identity)
        if observation.get('active'):raise Wait('job pending/running')
        terminal_ok(observation,job_id);return observation
    def accept_validation(self):
        observation=self.observe_success(self.c['proof_job_id'],self.c['proof_identity'])
        self.no_writers();pin=self.proof_files()
        return self.save('validation_accepted',receipts=dict(validation=pin),validation_observation=observation,last_wait=None)
    def retire_prepare(self):
        self.recheck_receipts();self.no_writers()
        index=len(self.j['retirements'])
        if index==2:return self.save('both_old_jobs_retired',intent=None)
        pin=self.c['obsolete_jobs'][index];self.held_old(self.scheduler.observe(pin['job_id'],self.c['owner']),pin)
        intent=dict(kind='retire',job_id=pin['job_id'],pin=pin,created=time.time(),
                    argv=['scancel','--ctld','--state=PENDING','--user='+self.c['owner'],pin['job_id']])
        self.save('retire_intent',intent=intent)
        # Only this stack frame issues scancel. Recovered intent observes accounting.
        self.held_old(self.scheduler.observe(pin['job_id'],self.c['owner']),pin)
        self.save('retire_waiting')
        try:result=self.scheduler.cancel(intent['argv'])
        except (Wait,OSError,TimeoutError) as exc:return self.save('retire_waiting',last_wait=str(exc))
        return self.save('retire_waiting',cancel_output=result)
    def retire_observe(self):
        intent=self.j['intent'];observation=self.scheduler.observe(intent['job_id'],self.c['owner'])
        if observation is None:raise Wait('retirement accounting unavailable')
        validate_identity(observation,intent['pin'])
        if observation.get('active'):
            self.held_old(observation,intent['pin']);raise Wait('cancellation not terminal; never reissued')
        rows=observation.get('rows',{})
        if intent['job_id'] not in rows:raise Wait('retirement parent accounting unavailable')
        require(rows[intent['job_id']][0]=='CANCELLED','obsolete allocation not CANCELLED')
        return self.save('validation_accepted',retirements=self.j['retirements']+[observation],intent=None,last_wait=None)
    def activation(self):
        self.recheck_receipts();self.no_writers();self.storage()
        require(len(self.j['retirements'])==2,'both retirements required')
        for prior in self.j['retirements']:
            current=self.scheduler.observe(prior['job_id'],self.c['owner']);validate_identity(current,prior)
            require(not current['active'] and current['rows'][prior['job_id']][0]=='CANCELLED','retired allocation changed')
        initial={}
        for name in ('meta.json','identity.json','compile_history.json'):
            path=self.path(self.active_run())/'latest_state'/name
            initial[name]=dict(text=read_bytes(path).decode(),sha256=sha(path))
        payload=dict(version=1,kind='b6_in_place_activation',operation_id=self.c['operation_id'],active_run=self.active_run(),
                     validation=self.j['receipts']['validation'],retirements=self.j['retirements'],initial=initial,
                     comparison=read_json(self.path(self.c['proof_root'])/self.c['proof_job_id']/'comparison.json'),
                     ready=read_json(self.path(self.c['proof_root'])/self.c['proof_job_id']/'validation_receipt.json'),
                     source_commit=SOURCE,provenance=dict(image_sha256=self.c['image_sha256'],source_manifest_sha256=self.c['source_manifest_sha256'],helper_manifest_sha256=self.c['phases']['proof']['helper_manifest_sha256']))
        self.save('activation_intent');exclusive_json(self.state/'activation.json',payload)
        receipts=dict(self.j['receipts']);receipts['activation']=dict(relative_path=str((self.state/'activation.json').relative_to(self.c['work'])),sha256=sha(self.state/'activation.json'))
        return self.save('candidate_activated',receipts=receipts)
    def build_request(self):
        require(self.j['phase'] in ('candidate_activated','link_accepted') and self.j['intent'] is None,'request phase conflict')
        self.pins();self.recheck_receipts();self.storage();self.no_writers()
        index=len(self.j['links'])+1
        if index==1:
            pin=self.j['receipts']['validation'];start={k:v for k,v in pin['candidate'].items() if k not in ('relative_run','relative_checkpoint')};start['relative_dir']='latest_state'
            predecessor=dict(relative_path=pin['ready_relative_path'],sha256=pin['ready_sha256'])
        else:
            prior=self.j['links'][-1];checked_pin(self.c['work'],prior['receipt']);receipt=read_json(self.path(prior['receipt']['relative_path']))
            require(not self.validate_link_receipt(prior,receipt),'cannot continue final')
            cp=receipt['checkpoint'];start=dict(cp['position']);start.update({k:cp[k] for k in ('relative_dir','meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256')});predecessor=prior['receipt']
        helper=self.c['phases']['production'];relative=str(self.state.relative_to(self.c['work']))+'/links/%04d'%index
        request=dict(version=1,operation_id=self.c['operation_id'],token='b6-%s-production-%04d'%(self.c['operation_id'],index),link_index=index,
            source_commit=SOURCE,source_snapshot_git=self.c['source_snapshot_git'],source_dir=self.c['source_dir'],source_manifest_sha256=self.c['source_manifest_sha256'],
            helper_dir=helper['helper_dir'],helper_manifest_sha256=helper['helper_manifest_sha256'],run_dir=self.active_run(),bindings=self.c['bindings'],bindings_sha256=self.c['bindings_sha256'],
            config_relative=self.c['config_relative'],config_sha256=self.c['config_sha256'],receipt_dir=relative,profile=PROFILE,start_checkpoint=start,predecessor=predecessor,
            expected_runtime_fingerprint=self.c['expected_runtime_fingerprint'],completed_epoch_events=self.j['epoch_events'],image_sha256=self.c['image_sha256'],
            activation=self.j['receipts']['activation'],
            mmap_dir=read_json(self.path(self.c['proof_root'])/self.c['proof_job_id']/'verified_record.json')['mmap_dir'])
        directory=self.path(relative);directory.mkdir(parents=True,exist_ok=True);exclusive_json(directory/'request.json',request)
        return dict(relative_path=relative+'/request.json',sha256=sha(directory/'request.json'))
    def prepare_submission(self,pin):
        require(self.j['intent'] is None and self.j['phase'] in ('candidate_activated','link_accepted'),'pending submission')
        self.pins();self.recheck_receipts();self.storage();self.no_writers();checked_pin(self.c['work'],pin)
        request=read_json(self.path(pin['relative_path']));r=self.c['resources'];helper=self.c['phases']['production']
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG') if k in os.environ}
        env.update(WORK=self.c['work'],SIF=self.c['sif'],REPO_DIR=str(self.path(self.c['source_dir'])),OPS_DIR=str(self.path(helper['helper_dir'])),REQUEST=pin['relative_path'],REQUEST_SHA256=pin['sha256'])
        argv=['sbatch','--parsable','--no-requeue','--export=ALL','--comment='+request['token'],'--job-name='+request['token'],
            '--account='+r['account'],'--partition='+r['partition'],'--time='+r['time'],'--nodes=1','--gpus-per-node=4',
            '--cpus-per-task='+str(r['cpus_per_task']),'--exclude='+r['exclude'],'--output='+str(self.path(r['output'])),str(self.path(helper['script']))]
        intent=dict(kind='production',token=request['token'],index=request['link_index'],argv=argv,env=env,created=time.time(),request=pin,work_dir=os.getcwd())
        self.save('production_submit_intent',intent=intent);self._fresh_intent=intent['token'];return self.j
    def execute_submission(self):
        intent=self.j['intent'];require(intent and self._fresh_intent==intent['token'],'only fresh intent creator may submit');self._fresh_intent=None
        self.pins();self.recheck_receipts(live_candidate=not self.j['links']);self.storage();self.no_writers();checked_pin(self.c['work'],intent['request'])
        try:output=self.scheduler.submit(intent['argv'],intent['env'])
        except (Wait,OSError,TimeoutError) as exc:return self.save('submission_uncertain',last_wait=str(exc))
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*',output['stdout'])
        if output['returncode']!=0 or match is None:return self.save('submission_uncertain',submission_output=output)
        return self.adopt(match.group(1),output)
    def adopt(self,job_id,evidence):
        entry=dict(job_id=job_id,intent=self.j['intent'],submission_evidence=evidence)
        return self.save('waiting_production',links=self.j['links']+[entry],intent=None,last_wait=None)
    def reconcile(self):
        intent=self.j['intent'];require(intent and intent['kind']=='production','no production intent')
        self.recheck_receipts();checked_pin(self.c['work'],intent['request'])
        matches=self.scheduler.discover(intent,self.c['owner'])
        if len(matches)!=1:return self.save('submission_uncertain',last_wait='zero or multiple submission matches; no resubmission')
        found=matches[0]
        require(found.get('owner')==self.c['owner'] and found.get('uid')==self.c['uid'] and found.get('token')==intent['token'] and
                type(found.get('created')) in (int,float) and intent['created']-1<=found['created']<=intent['created']+60,'reconciliation identity/time mismatch')
        observation=self.scheduler.observe(found['job_id'],self.c['owner'])
        if observation is None:raise Wait('matching allocation identity unavailable')
        expected=dict(job_id=found['job_id'],owner=self.c['owner'],uid=self.c['uid'],created=observation['created'],token=intent['token'],argv=intent['argv'],
            work_dir=intent['work_dir'],command=submission_command(intent['argv'],intent['work_dir']))
        validate_identity(observation,expected)
        timestamp=time.mktime(time.strptime(observation['created'],'%Y-%m-%dT%H:%M:%S'))
        require(intent['created']-1<=timestamp<=intent['created']+60,'accounting submit time mismatch')
        return self.adopt(found['job_id'],observation)
    def tick(self):
        with self.locked():
            try:
                self.pins();phase=self.j['phase']
                if phase=='waiting_validation':return self.accept_validation()
                if phase=='validation_accepted':return self.retire_prepare()
                if phase in ('retire_intent','retire_waiting'):return self.retire_observe()
                if phase in ('both_old_jobs_retired','activation_intent'):return self.activation()
                if phase in ('candidate_activated','link_accepted'):
                    self.prepare_submission(self.build_request());return self.execute_submission()
                if phase in ('production_submit_intent','submission_uncertain'):return self.reconcile()
                if phase=='waiting_production':return self.accept_link()
                if phase=='final_verified':
                    self.accept_link();return self.save('complete')
                return self.j
            except Wait as exc:return self.save(self.j['phase'],last_wait=str(exc))
            except (Blocked,OSError,KeyError,TypeError,ValueError) as exc:
                self.save(self.j['phase'],last_error=str(exc));raise Blocked(str(exc))

    def inventory(self,directory,files):
        require(type(files) is dict and bool(files),'missing checkpoint inventory')
        actual=set()
        for root,dirs,names in os.walk(str(directory),followlinks=False):
            for name in dirs+names: require(not (Path(root)/name).is_symlink(),'symlinked checkpoint payload')
            for name in names:actual.add(str((Path(root)/name).relative_to(directory)))
        require(actual==set(files),'checkpoint inventory file set differs')
        for name,item in files.items():
            path=safe_path(directory,name)
            require(type(item) is dict and set(item)=={'size','sha256'} and integer(item['size'],1) and
                    digest_ok(item['sha256']) and regular(path).stat().st_size==item['size'],'checkpoint inventory size/hash invalid')
            # Deliberately never read model/optimizer/RNG/pending-grad bytes on the host.
            if name in ('meta.json','identity.json','compile_history.json','PILOT_FINAL','PRUNE_EXEMPT','eval.json'):
                require(sha(path)==item['sha256'],'small checkpoint artifact changed: '+name)

    def epochs(self,events,expected,previous):
        require(type(events) is list and [e.get('epoch') for e in events]==list(range(1,expected+1)),
                'missing or extra completed epoch evaluation')
        require(events[:len(previous)]==previous,'previous epoch evidence changed')
        for event in events:
            require(integer(event.get('epoch'),1) and integer(event.get('global_step'),1) and
                    digest_ok(event.get('data_order_sha256')),'invalid epoch position/order')
            directory=self.path(self.active_run())/('epoch_%02d'%event['epoch'])
            for field,name in [('meta_sha256','meta.json'),('eval_sha256','eval.json'),
                               ('identity_sha256','identity.json'),('history_sha256','compile_history.json')]:
                require(digest_ok(event.get(field)) and sha(directory/name)==event[field],'epoch artifact binding changed')
            meta=read_json(directory/'meta.json');evaluation=read_json(directory/'eval.json')
            require(meta.get('epoch')==event['epoch'] and meta.get('step')==0 and
                    meta.get('global_step')==event['global_step'],'epoch checkpoint position changed')
            observation=event.get('epoch_end',{})
            require(observation.get('epoch_end')==event['epoch'] and observation.get('arm')=='B6' and
                    observation.get('data_order_first1000_sha256')==event['data_order_sha256'],'missing epoch_end provenance')
            for key,field in [('train','train_loss'),('de_novo','eval_loss'),('evidence','eval_evidence_loss')]:
                value=event.get(key)
                require(type(value) in (int,float) and math.isfinite(value) and evaluation.get(field)==value and
                        observation.get(key)==value,'invalid completed epoch loss')
            history=read_json(directory/'compile_history.json')
            require(meta.get('compile_history_sha256')==event['history_sha256'] and
                    history.get('fingerprint')==self.c['expected_runtime_fingerprint'],'epoch history changed')
            for entry in history.get('entries',[]):
                position=entry['position']
                if position['phase']=='eval':
                    require(position['training'] is False and position['grad_enabled'] is False and
                            position['view'] in ('de_novo','evidence') and position['epoch']<event['epoch'],
                            'invalid eval history context')

    def validate_link_receipt(self,entry,receipt):
        request_pin=entry['intent']['request'];checked_pin(self.c['work'],request_pin)
        request=read_json(self.path(request_pin['relative_path']))
        require(request['run_dir']==self.active_run(),'production request targets a different active run')
        self.recheck_receipts()
        expected=dict(version=1,kind='b6_production_link_ready',operation_id=self.c['operation_id'],token=request['token'],
            link_index=request['link_index'],job_id=entry['job_id'],request_sha256=request_pin['sha256'],
            source_commit=self.c['source_commit'],predecessor_sha256=request['predecessor']['sha256'],
            start_global_step=request['start_checkpoint']['global_step'],source_snapshot_git=self.c['source_snapshot_git'],
            source_manifest_sha256=self.c['source_manifest_sha256'],helper_manifest_sha256=request['helper_manifest_sha256'],
            config_sha256=self.c['config_sha256'],runtime_fingerprint_sha256=canonical_sha(self.c['expected_runtime_fingerprint']),
            image_sha256=self.c['image_sha256'],activation_sha256=request['activation']['sha256'])
        require(all(type(receipt.get(k)) is type(v) and receipt[k]==v for k,v in expected.items()),'production receipt envelope mismatch')
        require(encoded(receipt.get('profile'))==encoded(PROFILE),'production profile changed')
        verified=receipt.get('input_verification',{})
        require(verified.get('job_id')==entry['job_id'] and encoded(verified.get('ranks'))==encoded([0,1,2,3]) and
                verified.get('bindings_sha256')==self.c['bindings_sha256'] and digest_ok(verified.get('record_sha256')) and
                digest_ok(verified.get('identity_sha256')),'missing fresh four-rank input verification')
        output=self.path(request['receipt_dir'])/('execution_'+entry['job_id'])
        artifacts=receipt.get('execution_artifacts',{})
        require(set(artifacts)=={'preflight.json','rank0.json','rank1.json','rank2.json','rank3.json'},'missing allocation small artifacts')
        for name,digest in artifacts.items():require(digest_ok(digest) and sha(output/name)==digest,'allocation artifact changed: '+name)
        first=read_json(output/'preflight.json')
        require(first.get('request_sha256')==request_pin['sha256'] and first.get('job_id')==entry['job_id'],'preflight identity changed')
        producer=load('b6_pure_producer_validation',HERE/'b6_production_link.py')
        records=[read_json(output/('rank%d.json'%rank)) for rank in range(4)]
        record=producer.validate_rank_records(records,request,entry['job_id'],request_pin['sha256'])
        require(record['result']==receipt['result'] and record['counts']==receipt['counts'] and
                canonical_sha(record['input_verification']['record'])==verified['record_sha256'] and
                canonical_sha(record['identity'])==verified['identity_sha256'],'rank receipt differs from aggregate')
        result=receipt.get('result',{}); cp=receipt.get('checkpoint',{});position=cp.get('position',{})
        require(set(position)=={'epoch','step','global_step','pending_micro'} and all(integer(v) for v in position.values()),
                'invalid checkpoint position')
        step=result.get('global_step')
        require(integer(step) and request['start_checkpoint']['global_step']<step<=4099 and step==position['global_step'],
                'no progress or out-of-range production checkpoint')
        done=result.get('done') is True
        require((done and result.get('preempted',False) is False) or
                (result.get('preempted') is True and result.get('done',False) is False),'invalid trainer result')
        for field in ('meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256'):
            require(digest_ok(cp.get(field)),'invalid checkpoint digest')
        relative=cp.get('relative_dir')
        require(relative=='step_04099' if done else (relative=='latest_state' or
                type(relative) is str and re.fullmatch(r'epoch_[0-9]{2,}',relative) is not None),'invalid selected checkpoint directory')
        run=self.path(self.active_run()); directory=safe_path(run,relative)
        for marker in ('TRAINING_FAILED','MEMORY_ABORT'): require(not os.path.lexists(str(run/marker)),'production failure marker present')
        bindings=self.checkpoint_small(run,relative,position,payload=not done)
        require(all(bindings[k]==cp[k] for k in bindings),'production checkpoint small binding changed')
        history=read_json(directory/'compile_history.json');prefix=first['history_prefix']
        require(canonical_sha(history['bootstrap'])==prefix['bootstrap'] and
                [canonical_sha(e) for e in history['entries'][:len(prefix['entries'])]]==prefix['entries'],'production history prefix changed')
        require(read_json(directory/'identity.json')==record['identity']==first['identity'],'production identity changed')
        require(verified['identity_sha256']==canonical_sha(read_json(directory/'identity.json')),'fresh verifier identity differs')
        self.inventory(directory,cp.get('files'))
        counts=receipt.get('counts',{})
        require(set(counts)=={'train_rows','rank_batches','per_rank_accumulation'} and
                all(integer(v,1) for v in counts.values()) and counts['per_rank_accumulation']==24 and
                counts['rank_batches']==(counts['train_rows']+3)//4,'invalid actual sampler counts')
        total=position['epoch']*counts['rank_batches']+position['step']
        require(position['step']<=counts['rank_batches'] and position['pending_micro']<24 and
                total==step*24+position['pending_micro'],'checkpoint sampler/accumulation relation invalid')
        draws=receipt.get('draw_logs',{})
        require(set(draws)=={'0','1','2','3'} and all(integer(v.get('rows')) and v['rows']==total and
                digest_ok(v.get('sha256')) for v in draws.values()),'draw-prefix receipt incomplete')
        events=receipt.get('epoch_events'); self.epochs(events,position['epoch'],request['completed_epoch_events'])
        newevents=events[len(request['completed_epoch_events']):]
        history=result.get('history')
        keys=('epoch','global_step','data_order_sha256','train','de_novo','evidence')
        require(type(history) is list and history==[{k:e[k] for k in keys} for e in newevents], 'trainer returned epoch history mismatch')
        final=receipt.get('final')
        if done:
            require(step==4099 and position['pending_micro']==0 and result.get('reason')=='stop_after_updates' and
                    result.get('pilot_final')=='/work/'+self.active_run()+'/step_04099','not exact final4099')
            expected_epochs=(4099*24-1)//counts['rank_batches']
            require(type(final) is dict and type(final.get('expected_completed_epochs')) is int and
                    final['expected_completed_epochs']==expected_epochs==position['epoch'] and final.get('epoch_events')==events and
                    final.get('relative_dir')=='step_04099','final completed epochs differ')
            meta=read_json(directory/'meta.json')
            require(meta.get('arm')=='B6' and type(meta.get('seed')) is int and meta['seed']==123 and
                    meta.get('pilot_final') is True and type(meta.get('stop_after_updates')) is int and meta['stop_after_updates']==4099 and
                    meta.get('checkpoint_kind')=='fixed_step' and
                    meta.get('selection')=='fixed optimizer-update count; never a best-validation checkpoint', 'invalid final metadata')
            require(digest_ok(final.get('model_sha256')) and final['model_sha256']==meta.get('model_sha256')==
                    cp['files'].get('model.safetensors',{}).get('sha256'),'final model binding mismatch')
            for name,content in FINAL_MARKERS.items(): require(read_bytes(directory/name).decode()==content,'invalid final marker '+name)
            require(read_bytes(run/'TRAINING_DONE').strip() and digest_ok(final.get('done_sha256')) and
                    sha(run/'TRAINING_DONE')==final['done_sha256']==cp.get('done_sha256'),'missing or stale DONE')
        else:
            require(step<4099 and final is None and not os.path.lexists(str(run/'TRAINING_DONE')),'unexpected preempted final state')
        return done

    def accept_link(self):
        entry=self.j['links'][-1]
        self.observe_production(entry)
        self.pins();self.active_run();self.recheck_receipts();self.no_writers()
        request=read_json(self.path(entry['intent']['request']['relative_path']))
        path=self.path(request['receipt_dir'])/('execution_'+entry['job_id'])/'link_receipt.json'
        receipt=read_json(path);done=self.validate_link_receipt(entry,receipt)
        entry['receipt']=dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path))
        return self.save('final_verified' if done else 'link_accepted',epoch_events=receipt['epoch_events'])

    def observe_production(self,entry):
        intent=entry['intent'];observation=self.scheduler.observe(entry['job_id'],self.c['owner'])
        if observation is None:raise Wait('production accounting unavailable')
        expected=dict(job_id=entry['job_id'],owner=self.c['owner'],uid=self.c['uid'],created=observation['created'],token=intent['token'],argv=intent['argv'],
                      work_dir=intent['work_dir'],command=submission_command(intent['argv'],intent['work_dir']))
        validate_identity(observation,expected)
        timestamp=time.mktime(time.strptime(observation['created'],'%Y-%m-%dT%H:%M:%S'))
        require(intent['created']-1<=timestamp<=intent['created']+60,'production submission timestamp mismatch')
        if observation.get('active'):raise Wait('production pending/running')
        terminal_ok(observation,entry['job_id']);return observation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command')
    init=sub.add_parser('init');init.add_argument('--config',required=True);init.add_argument('--state-dir',required=True)
    for name in ('status','tick','reconcile','watch'):
        item=sub.add_parser(name);item.add_argument('--state-dir',required=True)
        if name=='watch':item.add_argument('--poll-seconds',type=int,default=45)
    args=parser.parse_args(argv)
    try:
        if args.command=='init':
            config=read_json(Path(args.config));require(os.getuid()==config['uid'],'process UID must match owner');Controller.initialize(config,args.state_dir)
        require(args.command is not None,'command required')
        controller=Controller(args.state_dir)
        if args.command in ('status','init'):print(json.dumps(controller.status(),sort_keys=True));return 0
        require(os.getuid()==controller.c['uid'],'process UID must match owner')
        if args.command=='reconcile':
            with controller.locked():result=controller.reconcile()
            print(json.dumps(result,sort_keys=True));return 0
        if args.command=='watch':require(1<=args.poll_seconds<=60,'poll seconds must be 1..60')
        while True:
            result=controller.tick();print(json.dumps(result,sort_keys=True),flush=True)
            if args.command!='watch' or result['phase']=='complete':return 0
            time.sleep(args.poll_seconds)
    except (Blocked,ValueError,KeyError,OSError) as exc:
        print('BLOCKED: '+str(exc),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
