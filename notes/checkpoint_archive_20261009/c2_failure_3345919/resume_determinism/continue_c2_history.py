#!/usr/bin/env python3
"""Receipt-bound C2 continuation. Python 3.6 standard library; no tensor reads."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import time


class Blocked(RuntimeError):
    pass


class Wait(RuntimeError):
    pass


def require(condition, reason):
    if not condition:
        raise Blocked(reason)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest_ok(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def safe_path(root, relative):
    require(type(relative) is str and relative and not relative.startswith('/') and
            all(p not in ('', '.', '..') for p in relative.split('/')), 'unsafe relative path')
    root = Path(root)
    current = Path(root.anchor)
    for part in root.parts[1:] + tuple(relative.split('/')):
        current = current / part
        require(not current.is_symlink(), 'symlinked control path: ' + str(current))
    return root / relative


def regular(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'missing regular file: ' + str(path))
    return path


def read_bytes(path, limit=16 * 1024 * 1024):
    path = regular(path)
    require(path.stat().st_size <= limit, 'control file exceeds size limit: ' + str(path))
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'control file grew beyond limit')
    return data


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')


def canonical_sha(value):
    return hashlib.sha256(encoded(value)[:-1]).hexdigest()


PROFILE = dict(arm='C2',target=4099,seed=123,world_size=4,batch_size=1,mixed_precision='bf16',deterministic=True)
FINAL_MARKERS = {'PILOT_FINAL':'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n',
                 'PRUNE_EXEMPT':'A28/plan 9: step checkpoint, never pruned\n'}


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad_constant(value):
        raise Blocked('nonfinite JSON: ' + value)
    try:
        return json.loads(read_bytes(path).decode('utf8'), object_pairs_hook=pairs, parse_constant=bad_constant)
    except (ValueError, UnicodeError) as exc:
        raise Blocked('invalid JSON: ' + str(exc))


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    require(not path.is_symlink(), 'symlinked destination')
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, str(path)); fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


class Lock:
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        require(not self.path.is_symlink(), 'symlinked lock')
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd); self.fd = None
            raise Blocked('controller lock busy or unsupported')
        return self

    def __exit__(self, *args):
        os.close(self.fd)


def exclusive_json(path,value):
    """Publish immutable control bytes without replacing any existing entry."""
    path=Path(path);raw=encoded(value)
    if os.path.lexists(str(path)):
        require(read_bytes(path)==raw,'conflicting immutable control receipt')
        fsync_dir(path.parent)
        return
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(path.parent))
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:
            os.link(temporary,str(path))
        except FileExistsError:
            require(read_bytes(path)==raw,'conflicting immutable control receipt')
        fsync_dir(path.parent)
    finally:
        os.unlink(temporary)


def validate_science(report, profile):
    spec = importlib.util.spec_from_file_location('c2_pure_history_proof', str(Path(__file__).with_name('history_proof.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only this pure function is permitted on the login host.
    numerical = dict(report)
    if profile == 'validate400': numerical['profile'] = 'resume210'
    try:
        module.validate_prior_proof(numerical)
    except (ValueError, TypeError) as exc:
        raise Blocked('scientific gate: ' + str(exc))
    if profile == 'validate400':
        require(report.get('profile') == profile, 'wrong validation profile')
        expected = dict(checkpoint_step=400, stop_step=410, rank3_specialization_slot=6718,
                        rank3_specialization_recorded=True)
        require(encoded(report.get('validation400')) == encoded(expected), 'invalid validation400 fields')
        digests = report.get('checkpoint_digests', {})
        require(set(digests) == {'continuous', 'resumed'} and
                all(digest_ok(x) for x in digests.values()) and
                digests['continuous'] == digests['resumed'], 'invalid canonical checkpoint equality')


def terminal_ok(observation, job_id):
    require(observation.get('active') is False, 'job is still active or requeued')
    rows = observation.get('rows', {})
    if job_id not in rows or job_id+'.batch' not in rows:
        raise Wait('parent/batch accounting not yet complete')
    require(rows.get(job_id) == ['COMPLETED', '0:0'] and
            rows.get(job_id + '.batch') == ['COMPLETED', '0:0'], 'parent/batch not COMPLETED 0:0')
    for name, row in rows.items():
        if name == job_id + '.extern' and row == ['OUT_OF_MEMORY', '0:125']: continue
        require(row == ['COMPLETED', '0:0'], 'nonzero allocation step: ' + name)
    return True


def check_test_log(path, expected):
    text = read_bytes(path).decode('utf8')
    counts = re.findall(r'^Ran (\d+) tests? in .+$', text, re.M)
    statuses = re.findall(r'^OK(?: \(skipped=(\d+)\))?$', text, re.M)
    require(counts == [str(expected['ran'])] and len(statuses) == 1 and
            int(statuses[0] or 0) == expected['skipped'] and
            re.search(r'^FAILED|^ERROR:|^FAIL:', text, re.M) is None,
            'unreviewed test result: ' + str(path))


def checked_pin(work, pin):
    require(type(pin) is dict and digest_ok(pin.get('sha256')), 'invalid artifact pin')
    path = safe_path(work, pin['relative_path'])
    require(sha(path) == pin['sha256'], 'artifact changed: ' + str(path))
    return path


def source_member_sha(path):
    """Stream only explicitly pinned source/helper members, never checkpoint payloads."""
    digest=hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def manifest_check(root, name, expected):
    manifest = safe_path(root, name)
    require(digest_ok(expected) and sha(manifest) == expected, 'manifest changed: ' + str(manifest))
    seen = set()
    for line in read_bytes(manifest).decode('utf8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        require(match is not None, 'invalid sha256 manifest line')
        digest, relative = match.groups()
        require(relative not in seen, 'duplicate manifest entry')
        seen.add(relative)
        require(source_member_sha(safe_path(root, relative)) == digest, 'manifest member changed: ' + relative)
    require(bool(seen), 'empty manifest')
    return seen


def canonical_controls(config):
    canonical = safe_path(config['work'], config['original_run'])
    return (canonical.parent / ('.' + canonical.name + '.history.lock'),
            canonical.parent / ('.' + canonical.name + '.history.owner.json'))


def submission_token(comment,argv):
    tokens=[]
    for index,arg in enumerate(argv):
        if arg.startswith('--comment='):tokens.append(arg[len('--comment='):])
        elif arg=='--comment':
            require(index+1<len(argv),'missing submission comment value')
            tokens.append(argv[index+1])
    require(len(tokens)<=1,'multiple submission comments')
    if tokens:
        require(not comment or comment==tokens[0],'accounting comment disagrees with SubmitLine')
        return tokens[0]
    return comment


def submission_command(argv,work_dir):
    require(bool(argv) and bool(argv[-1]),'missing submitted batch script')
    script=argv[-1]
    require(not any(c in script for c in '\x00\n\r|'),'ambiguous submitted batch script')
    if not posixpath.isabs(script):
        require(type(work_dir) is str and posixpath.isabs(work_dir) and
                not any(c in work_dir for c in '\x00\n\r|'),
                'relative submitted script requires an unambiguous absolute accounting WorkDir')
        script=posixpath.join(work_dir,script)
    return posixpath.normpath(script)


def validate_config(config):
    required = ('version operation_id owner uid work sif original_run diagnostic source_commit '
        'source_snapshot_git source_dir source_manifest_sha256 phases bindings bindings_sha256 config_relative '
        'config_sha256 expected_runtime_fingerprint original_pins diagnostic_pins original_checkpoint_receipt '
        'bootstrap_sha256 proof_job_id proof_identity proof_receipts obsolete_job protected_jobs resources '
        'validation_clone_peak_bytes storage_python storage_script').split()
    require(type(config) is dict and all(k in config for k in required), 'incomplete immutable config')
    require(type(config['version']) is int and config['version'] == 2 and
            not {'canonical','backup','active_run'} & set(config),'unsupported config or operator-selected active path')
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', config['operation_id']) is not None, 'invalid operation ID')
    require(integer(config['uid']) and config['uid'] == os.getuid(), 'controller owner UID mismatch')
    require(Path(config['work']).is_absolute() and Path(config['work']).is_dir(), 'WORK must exist')
    for field in ('original_run', 'diagnostic', 'source_dir', 'bindings', 'storage_script'):
        safe_path(config['work'], config[field])
    require(config['original_run']!=config['diagnostic'],'protected paths overlap')
    for field in ('source_commit','source_snapshot_git'): require(digest_ok(config[field],40), 'invalid commit')
    for field in ('source_manifest_sha256','bindings_sha256','config_sha256','bootstrap_sha256'):
        require(digest_ok(config[field]), 'invalid digest: ' + field)
    require(re.fullmatch(r'[0-9]+', config['proof_job_id']) is not None, 'invalid adopted proof job')
    require(integer(config['validation_clone_peak_bytes']), 'invalid clone budget')
    require(set(config['phases']) == {'proof','validation','production'}, 'missing phase configuration')
    for phase in config['phases'].values():
        for field in ('helper_dir','script','root'): safe_path(config['work'],phase[field])
        require(digest_ok(phase['helper_manifest_sha256']), 'invalid helper manifest pin')
    return config


class Slurm:
    """Only external boundary. All invocations use fixed argv and clean environments."""
    def command(self, argv, env=None):
        try:
            result = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, timeout=45)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Wait('scheduler command uncertain: ' + str(exc))
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def read(self, argv):
        result = self.command(argv)
        if result['returncode'] != 0: raise Wait('scheduler observation failed: ' + result['stderr'])
        return result['stdout']

    def active(self, owner):
        text = self.read(['squeue','-h','-u',owner,'-o','%i|%u|%T|%k|%o'])
        rows=[]
        for line in text.splitlines():
            parts=line.split('|')
            require(len(parts)==5, 'ambiguous squeue response')
            job,user,state,token,command=parts
            rows.append(dict(job_id=job,owner=user,state=state,token=token,command=command,active=True))
        return rows

    def observe(self, job_id, owner):
        live = [j for j in self.active(owner) if j['job_id']==job_id]
        raw=self.read(['sacct','-n','-P','-j',job_id,'--format=JobIDRaw,State,ExitCode'])
        rows={}
        for line in raw.splitlines():
            fields=line.split('|')
            if len(fields)<3: raise Wait('incomplete accounting')
            require(fields[0] not in rows, 'multiple executions or requeue in accounting')
            rows[fields[0]]=[fields[1].split()[0].rstrip('+'),fields[2]]
        identity=self.read(['sacct','-X','-n','-P','-j',job_id,
            '--format=JobIDRaw,User,UID,Submit,Comment%256,SubmitLine%4096,WorkDir%4096'])
        matches=[line.split('|') for line in identity.splitlines() if line.startswith(job_id+'|')]
        if not matches or not rows: return None
        require(len(matches)==1,'ambiguous job identity')
        fields=matches[0]
        if len(fields)==8 and fields[-1]=='':fields=fields[:-1]
        require(len(fields)==7,'ambiguous job identity or accounting WorkDir')
        _,user,uid,created,token,submit_line,work_dir=fields
        token=submission_token(token,shlex.split(submit_line))
        result=dict(job_id=job_id,owner=user,uid=int(uid),created=created,token=token,
                    submit_line=submit_line,argv=shlex.split(submit_line),work_dir=work_dir,rows=rows,active=bool(live))
        result['command']=submission_command(result['argv'],work_dir)
        if live:
            control=self.read(['scontrol','show','job',job_id,'-o'])
            values=dict(re.findall(r'(\w+)=([^ ]*)',control))
            command=values.get('Command','')
            require(posixpath.isabs(command) and posixpath.normpath(command)==result['command'],
                    'live Command disagrees with accounting submission path')
            result.update(reason=values.get('Reason'),state=values.get('JobState'),
                          nice=values.get('Nice'),raw=control)
            require(values.get('JobId')==job_id and values.get('UserId')=='%s(%s)'%(owner,uid),
                    'scontrol identity mismatch')
        return result

    def discover(self, intent, owner):
        since=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(intent['created']-60))
        raw=self.read(['sacct','-X','-n','-P','-u',owner,'-S',since,
            '--format=JobIDRaw,User,UID,Submit,Comment%256,SubmitLine%4096'])
        matches=[]
        for line in raw.splitlines():
            fields=line.split('|')
            require(len(fields)>=6,'incomplete discovery')
            job,user,uid,created,token,submit_line=fields[:6]
            token=submission_token(token,shlex.split(submit_line))
            if token != intent['token']: continue
            timestamp=time.mktime(time.strptime(created,'%Y-%m-%dT%H:%M:%S'))
            matches.append(dict(job_id=job,owner=user,uid=int(uid),created=timestamp,token=token,
                                argv=shlex.split(submit_line)))
        # Active jobs missing from accounting make an empty search uncertain, never safe to retry.
        return matches

    def submit(self, argv, env): return self.command(argv, env)
    def cancel(self, argv): return self.command(argv)
    def nice(self, argv): return self.command(argv)
    def storage(self, config, root):
        result=self.command([config['storage_python'],str(safe_path(config['work'],config['storage_script'])),str(root)])
        require(result['returncode']==0,'storage preflight failed: '+result['stderr'])
        try: return json.loads(result['stdout'])
        except ValueError: raise Blocked('invalid storage response')


class Controller:
    @classmethod
    def initialize(cls, config, state_dir):
        validate_config(config)
        state_dir=Path(state_dir)
        require(state_dir.is_absolute(), 'state directory must be absolute')
        safe_path(config['work'],str(state_dir.relative_to(config['work'])))
        protected=[config['original_run'],config['diagnostic'],config['source_dir'],
                   config['phases']['proof']['root'],config['phases']['validation']['root']]
        protected += [phase['helper_dir'] for phase in config['phases'].values()]
        for relative in protected:
            path=safe_path(config['work'],relative)
            require(path!=state_dir and path not in state_dir.parents and state_dir not in path.parents,
                    'operation controls overlap protected data or helpers')
        lock,claim=canonical_controls(config)
        with Lock(lock):
            require(not state_dir.exists(), 'operation directory already exists')
            require(not claim.exists(), 'canonical C2 already claimed by another operation')
            state_dir.mkdir(); fsync_dir(state_dir.parent)
            atomic_json(state_dir/'config.json',config)
            journal=dict(version=2,operation_id=config['operation_id'],config_sha256=sha(state_dir/'config.json'),
                phase='waiting_proof',revision=0,created=time.time(),intent=None,proof_job_id=config['proof_job_id'],
                validation_job_id=None,links=[],receipts={},epoch_events=[],restored=[],last_error=None,
                active_run=None,activation_receipt=None)
            atomic_json(state_dir/'journal.json',journal)
            atomic_json(claim,dict(operation_id=config['operation_id'],state_dir=str(state_dir),
                                   config_sha256=journal['config_sha256']))

    def __init__(self,state_dir,scheduler=None):
        self.state=Path(state_dir)
        self.c=validate_config(read_json(self.state/'config.json'))
        safe_path(self.c['work'],str(self.state.relative_to(self.c['work'])))
        self.scheduler=scheduler or Slurm()
        self.j=self.status()
        self._fresh_intent=None

    def path(self,relative): return safe_path(self.c['work'],relative)

    def status(self):
        journal=read_json(self.state/'journal.json')
        require(type(journal.get('version')) is int and journal['version']==2 and
                integer(journal.get('revision')) and journal.get('operation_id')==self.c['operation_id'] and
                journal.get('config_sha256')==sha(self.state/'config.json'), 'journal/config identity drift')
        require(journal.get('phase') in {'waiting_proof','proof_accepted','validation_submit_intent','waiting_validation',
            'validation_accepted','retire_intent','retire_waiting','old_job_retired','activation_intent','candidate_activated',
            'production_submit_intent','waiting_production','submission_uncertain','link_accepted','final_verified',
            'nice_intent','priorities_restored','complete'},'legacy or unsupported journal phase')
        return journal

    @contextlib.contextmanager
    def locked(self):
        canonical,claim=canonical_controls(self.c)
        with Lock(self.state/'controller.lock'):
            with Lock(canonical):
                claimed=read_json(claim)
                require(claimed.get('operation_id')==self.c['operation_id'] and
                        claimed.get('state_dir')==str(self.state) and
                        claimed.get('config_sha256')==sha(self.state/'config.json'),'canonical ownership changed')
                self.j=self.status()
                yield

    def save(self,phase,**values):
        if 'last_error' not in values: self.j['last_error']=None
        self.j.update(values); self.j.update(phase=phase,revision=self.j['revision']+1,updated=time.time())
        atomic_json(self.state/'journal.json',self.j)
        return self.j

    def pins(self):
        source=self.path(self.c['source_dir'])
        require(read_bytes(source/'SOURCE_COMMIT').decode().strip()==self.c['source_commit'],'source commit drift')
        result=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              universal_newlines=True,timeout=15)
        require(result.returncode==0 and result.stdout.strip()==self.c['source_snapshot_git'],'source snapshot Git drift')
        manifest_check(source,'validation_sources.sha256',self.c['source_manifest_sha256'])
        for phase in self.c['phases'].values():
            helper=self.path(phase['helper_dir'])
            members=manifest_check(helper,'manifest.sha256',phase['helper_manifest_sha256'])
            try:
                script=str(self.path(phase['script']).relative_to(helper))
            except ValueError:
                raise Blocked('phase script is outside its declared helper directory')
            require(script in members,'phase script is absent from its pinned helper manifest: '+script)
        require(sha(self.path(self.c['bindings']))==self.c['bindings_sha256'],'bindings drift')
        require(sha(safe_path(source,self.c['config_relative']))==self.c['config_sha256'],'training config drift')
        for pin in self.c['proof_receipts']+self.c['diagnostic_pins']+[self.c['original_checkpoint_receipt']]:
            checked_pin(self.c['work'],pin)

    def storage(self,clones=False):
        relative=self.derived_active_run() if 'validation' in self.j['receipts'] else self.c['original_run']
        root=self.path(relative)
        result=self.scheduler.storage(self.c,root)
        require(result.get('project_id')==22169 and result.get('pool')=='ddn_ssd','wrong storage project/pool')
        required=max(100*1024**3,self.c['validation_clone_peak_bytes'] if clones else 0)
        for key in ('quota_remaining_bytes','filesystem_available_bytes'):
            require(integer(result.get(key)) and result[key]>=required,'insufficient storage reserve')
        return result

    def no_writers(self,allowed=None):
        allowed=set(allowed or [])
        protected={p['job_id'] for p in self.c['protected_jobs']}
        for job in self.scheduler.active(self.c['owner']):
            if job['job_id'] in allowed: continue
            if job['job_id']==self.c['obsolete_job']['job_id']:
                require(self.j['phase'] in ('waiting_proof','proof_accepted','validation_submit_intent',
                    'waiting_validation','validation_accepted','retire_intent','retire_waiting','submission_uncertain'),
                    'retired old job became active again')
                self.held_old(self.scheduler.observe(job['job_id'],self.c['owner']))
                continue
            command=job.get('command','')
            targets=[self.c['original_run'],self.c['phases']['validation']['root']]
            if 'validation' in self.j['receipts']:targets.append(self.derived_active_run())
            require(not any(path in command for path in targets) and
                    not job.get('token','').startswith('c2-') and
                    (job['job_id'] in protected or 'c2' not in command.lower()) and
                    command != str(self.path(self.c['phases']['production']['script'])),
                    'competing canonical C2 writer: '+job['job_id'])

    def prepare_submission(self,kind,request=None):
        require(kind in ('validation','production') and self.j['intent'] is None,'pending mutation intent')
        self.pins(); self.storage(clones=kind=='validation'); self.no_writers()
        index=1 if kind=='validation' else len(self.j['links'])+1
        token='c2-%s-%s-%04d'%(self.c['operation_id'],kind,index)
        phase=self.c['phases'][kind]; resources=phase.get('resources',self.c['resources'])
        env={k:os.environ[k] for k in ('PATH','HOME','USER','LOGNAME','LANG') if k in os.environ}
        env.update(WORK=self.c['work'],SIF=self.c['sif'],REPO_DIR=str(self.path(self.c['source_dir'])),
                   OPS_DIR=str(self.path(phase['helper_dir'])))
        if kind=='validation': env['PROOF_JOB_ID']=self.c['proof_job_id']
        else:
            require(request is not None,'missing production request')
            env.update(REQUEST=request['relative_path'],REQUEST_SHA256=request['sha256'])
        argv=['sbatch','--parsable','--no-requeue','--export=ALL','--comment='+token,'--job-name='+token[:128],
              '--account='+resources['account'],'--partition='+resources['partition'],'--time='+resources['time'],
              '--nodes=1','--gpus-per-node=4','--cpus-per-task='+str(resources['cpus_per_task']),
              '--exclude='+resources['exclude'],'--output='+str(self.path(resources['output'])),
              str(self.path(phase['script']))]
        intent=dict(kind=kind,token=token,index=index,argv=argv,env=env,created=time.time(),request=request,
                    env_sha256=hashlib.sha256(encoded(env)).hexdigest())
        self._fresh_intent=token
        return self.save(kind+'_submit_intent',intent=intent)

    def execute_submission(self):
        intent=self.j['intent']
        require(intent and self._fresh_intent==intent['token'],'only the process that persisted the intent may submit')
        self._fresh_intent=None
        self.pins(); self.storage(clones=intent['kind']=='validation'); self.no_writers()
        if intent['kind']=='production':
            self.original();self.active_run();self.recheck_receipts(live_candidate=not self.j['links'])
            checked_pin(self.c['work'],intent['request'])
        try:
            output=self.scheduler.submit(intent['argv'],intent['env'])
        except (Wait,TimeoutError,OSError) as exc:
            return self.save('submission_uncertain',last_error=str(exc))
        match=re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?\s*',output['stdout'])
        if output['returncode']!=0 or match is None:
            return self.save('submission_uncertain',submission_output=output,last_error='submission response uncertain')
        return self.adopt(match.group(1),output)

    def adopt(self,job_id,evidence):
        intent=self.j['intent']; entry=dict(job_id=job_id,intent=intent,submission_evidence=evidence)
        if intent['kind']=='validation':
            return self.save('waiting_validation',validation_job_id=job_id,validation_submission=entry,intent=None)
        links=self.j['links']+[entry]
        return self.save('waiting_production',links=links,intent=None)

    def reconcile(self):
        intent=self.j['intent']
        require(intent is not None and intent['kind'] in ('validation','production'),'no submission to reconcile')
        if intent['kind']=='production':
            self.original();self.active_run();self.recheck_receipts();checked_pin(self.c['work'],intent['request'])
        matches=self.scheduler.discover(intent,self.c['owner'])
        if not matches:
            if self.j['phase']=='submission_uncertain': return self.j
            return self.save('submission_uncertain',last_error='no visible job does not establish non-submission')
        require(len(matches)==1,'multiple jobs match submission token')
        job=matches[0]
        require(job.get('owner')==self.c['owner'] and job.get('uid')==self.c['uid'] and
                job.get('token')==intent['token'] and job.get('argv')==intent['argv'] and
                type(job.get('created')) in (int,float) and job['created']>=intent['created']-1,
                'submission reconciliation identity mismatch')
        return self.adopt(job['job_id'],job)

    def tick(self):
        with self.locked():
            try:
                self.pins()
                phase=self.j['phase']
                if phase in ('validation_submit_intent','production_submit_intent','submission_uncertain'):
                    return self.reconcile()
                if phase=='waiting_proof':
                    observation=self.scheduler.observe(self.c['proof_job_id'],self.c['owner'])
                    if observation is None or observation.get('active'): return self.j
                    terminal_ok(observation,self.c['proof_job_id'])
                    return self.accept_science('proof',observation)
                return self.advance(phase)
            except Wait:
                return self.j
            except (Blocked, OSError, KeyError, TypeError, ValueError) as exc:
                reason=str(exc)
                if self.j.get('last_error')!=reason:self.save(self.j['phase'],last_error=reason)
                if isinstance(exc,Blocked):raise
                raise Blocked(reason)

    def original(self):
        base=self.c['original_run']
        for pin in self.c['original_pins']:
            prefix=base+'/'
            require(pin['relative_path'].startswith(prefix),'original pin outside original run')
            checked_pin(self.c['work'],pin)
        meta=read_json(self.path(base)/'latest_state/meta.json')
        require(all(type(meta.get(k)) is int and meta[k]==v for k,v in
                    dict(epoch=0,step=4800,global_step=200,pending_micro=0).items()),'original checkpoint is not200')

    def checkpoint_small(self,run,relative,position,specialization=False,payload=True):
        checkpoint=safe_path(run,relative)
        meta=read_json(checkpoint/'meta.json'); identity=read_json(checkpoint/'identity.json')
        require(all(integer(meta.get(k)) and meta[k]==v for k,v in position.items()),'checkpoint position drift')
        history=read_json(checkpoint/'compile_history.json'); history_sha=sha(checkpoint/'compile_history.json')
        require(meta.get('compile_history_sha256')==history_sha,'checkpoint history digest unbound')
        require(type(history.get('version')) is int and history['version']==1 and
                type(history.get('world_size')) is int and history['world_size']==4 and
                encoded(history.get('checkpoint'))==encoded(position) and
                encoded(history.get('identity'))==encoded(identity) and
                encoded(history.get('fingerprint'))==encoded(self.c['expected_runtime_fingerprint']),
                'checkpoint history identity/position/fingerprint mismatch')
        bootstrap=history.get('bootstrap',{})
        proof_root=self.path(self.c['phases']['proof']['root'])/self.c['proof_job_id']
        require(bootstrap.get('sha256')==self.c['bootstrap_sha256'] and
                sha(proof_root/'bootstrap.json')==self.c['bootstrap_sha256'] and
                encoded(bootstrap.get('manifest'))==encoded(read_json(proof_root/'bootstrap.json')),'checkpoint bootstrap drift')
        entries=history.get('entries')
        require(type(entries) is list and len(entries)>=16,'missing bootstrap history')
        slots=list(range(1,16))+[3915]
        for i,slot in enumerate(slots):
            expected=dict(phase='train',epoch=0,batch_index=slot-1,view='arm',training=True,
                          grad_enabled=True,optimizer_step=(slot-1)//24)
            require(entries[i].get('origin')=='reconstructed_legacy' and
                    encoded(entries[i].get('position'))==encoded(expected),'legacy history slots differ')
        require(all(e.get('origin')=='observed' for e in entries[16:]),'repeated reconstructed history')
        if specialization:
            expected=dict(phase='train',epoch=0,batch_index=6717,view='arm',training=True,
                          grad_enabled=True,optimizer_step=279)
            selected=[e for e in entries if encoded(e.get('position'))==encoded(expected) and e.get('origin')=='observed']
            require(len(selected)==1,'missing/duplicate observed slot6718')
            ranks=selected[0].get('ranks',[])
            require(len(ranks)==4 and [r.get('rank') for r in ranks]==list(range(4)),'specialization ranks invalid')
            rank=ranks[3]
            require(integer(rank.get('delta'),1) and rank.get('inputs',{}).get('input_ids',{}).get('shape')==[1,129024]
                    and rank.get('inputs',{}).get('labels',{}).get('shape')==[1,3924],'wrong observed specialization')
        if payload:
            for name in ['model.safetensors','optimizer.bin','scheduler.bin']+['random_states_%d.pkl'%r for r in range(4)]:
                require(regular(checkpoint/'accelerate_state'/name).stat().st_size>0,'empty checkpoint payload')
            if position['pending_micro']>0:
                for rank in range(4): regular(checkpoint/('pending_grads_rank%d.pt'%rank))
        return dict(meta_sha256=sha(checkpoint/'meta.json'),identity_sha256=sha(checkpoint/'identity.json'),
                    history_sha256=history_sha)

    def observe_success(self,job_id,identity=None):
        observation=self.scheduler.observe(job_id,self.c['owner'])
        if observation is None or observation.get('active'): raise Wait('job pending/running or accounting unavailable')
        terminal_ok(observation,job_id)
        require(observation.get('owner')==self.c['owner'] and observation.get('uid')==self.c['uid'],
                'job owner changed')
        if identity:
            require(observation.get('token')==identity['token'] and observation.get('argv')==identity['argv'],
                    'job submission identity changed')
        return observation

    def accept_science(self,kind,observation):
        job_id=self.c['proof_job_id'] if kind=='proof' else self.j['validation_job_id']
        terminal_ok(observation,job_id)
        require(observation.get('owner')==self.c['owner'] and observation.get('uid')==self.c['uid'] and
                observation.get('job_id')==job_id,'science allocation identity mismatch')
        if kind=='proof':
            expected=self.c['proof_identity']
            require(observation.get('token')==expected['comment'] and
                    observation.get('submit_line')==expected['submit_line'],'adopted proof submission mismatch')
        else:
            expected=self.j['validation_submission']['intent']
            require(observation.get('token')==expected['token'] and observation.get('argv')==expected['argv'],
                    'validation submission mismatch')
        self.pins(); self.original()
        phase=self.c['phases'][kind]; root=self.path(phase['root'])/job_id
        report=read_json(root/'comparison.json'); validate_science(report,'resume210' if kind=='proof' else 'validate400')
        require(read_json(root/'verified_record.json').get('job_id')==job_id,'stale verified allocation record')
        require(sha(root/'bootstrap.json')==self.c['bootstrap_sha256'],'bootstrap manifest drift')
        for filename,expectation in phase['tests'].items(): check_test_log(safe_path(root,filename),expectation)
        marker='HISTORY_CHECKPOINT_PROOF_PASS' if kind=='proof' else 'HISTORY_VALIDATION_PASS'
        log=self.path(phase['marker_log'].format(job_id=job_id))
        require(len(re.findall(r'^'+marker+r' .+$',read_bytes(log).decode('utf8'),re.M))==1,'missing terminal science marker')
        stop,snapshot=(220,210) if kind=='proof' else (410,400)
        artifacts=[]; bindings={}
        for name,step in [('continuous',stop),('resumed',stop),('at%d'%snapshot,snapshot)]:
            position=dict(epoch=0,step=step*24,global_step=step,pending_micro=0)
            bindings[name]=self.checkpoint_small(root/name,'latest_state',position,kind=='validation')
            for filename in ('meta.json','identity.json','compile_history.json'):
                path=root/name/'latest_state'/filename
                artifacts.append(dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path)))
        require(bindings['continuous']['history_sha256']==bindings['resumed']['history_sha256'],'unequal final histories')
        if 'checkpoint_history_sha256' in report:
            require(report['checkpoint_history_sha256']=={n:bindings[n]['history_sha256'] for n in ('continuous','resumed')},
                    'comparison history binding mismatch')
        for path in [root/'bootstrap.json',root/'verified_record.json',log]+[safe_path(root,name) for name in phase['tests']]:
            artifacts.append(dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path)))
        pin=dict(relative_path=str((root/'comparison.json').relative_to(self.c['work'])),sha256=sha(root/'comparison.json'),
                 artifacts=artifacts,job_id=job_id,root=str(root.relative_to(self.c['work'])))
        receipts=dict(self.j['receipts']);receipts[kind]=pin
        if kind=='validation':
            receipt=read_json(root/'validation_receipt.json')
            expected=dict(version=1,kind='c2_history_validation_ready',source_commit=self.c['source_commit'],
                validation_job_id=job_id,predecessor_job_id=self.c['proof_job_id'],
                predecessor_comparison_sha256=receipts['proof']['sha256'],comparison_sha256=pin['sha256'],
                original_checkpoint_receipt_sha256=self.c['original_checkpoint_receipt']['sha256'])
            require(all(type(receipt.get(k)) is type(v) and receipt[k]==v for k,v in expected.items()),'validation receipt envelope mismatch')
            candidate=receipt.get('candidate',{})
            expected=dict(relative_run='resumed',relative_checkpoint='latest_state',epoch=0,step=9840,global_step=410,
                pending_micro=0,meta_sha256=bindings['resumed']['meta_sha256'],history_sha256=bindings['resumed']['history_sha256'],
                canonical_checkpoint_sha256=report['checkpoint_digests']['resumed'])
            require(encoded(candidate)==encoded(expected),'validation candidate receipt mismatch')
            pin.update(ready_relative_path=str((root/'validation_receipt.json').relative_to(self.c['work'])),
                       ready_sha256=sha(root/'validation_receipt.json'),candidate=candidate,
                       identity_sha256=bindings['resumed']['identity_sha256'])
        return self.save('proof_accepted' if kind=='proof' else 'validation_accepted',receipts=receipts,intent=None)

    def derived_active_run(self):
        pin=self.j['receipts']['validation']
        expected=self.c['phases']['validation']['root']+'/'+self.j['validation_job_id']
        require(pin['root']==expected and pin['job_id']==self.j['validation_job_id'],'validation root/job drift')
        relative=expected+'/resumed';target=self.path(relative)
        protected=[self.c['original_run'],self.c['diagnostic'],self.c['source_dir'],
                   self.c['phases']['proof']['root'],str(self.state.relative_to(self.c['work'])),
                   expected+'/continuous',expected+'/at400']
        protected += [phase['helper_dir'] for phase in self.c['phases'].values()]
        for name in protected:
            other=self.path(name)
            require(target!=other and other not in target.parents and target not in other.parents,
                    'active run overlaps protected evidence or controls')
        require(target.is_dir(),'active candidate directory missing')
        return relative

    def production_started(self):
        return bool(self.j['links']) or (self.j.get('intent') or {}).get('kind')=='production'

    def activation_bindings(self):
        proof=self.j['receipts']['proof'];validation=self.j['receipts']['validation']
        pin=lambda value: {k:value[k] for k in ('relative_path','sha256')}
        candidate=validation['candidate']
        start={k:candidate[k] for k in ('epoch','step','global_step','pending_micro','meta_sha256',
                                       'history_sha256','canonical_checkpoint_sha256')}
        start.update(relative_dir='latest_state',identity_sha256=validation['identity_sha256'])
        return dict(version=2,kind='c2_history_activation',mode='in_place',operation_id=self.c['operation_id'],
            config_sha256=self.j['config_sha256'],owner=self.c['owner'],uid=self.c['uid'],
            original_run=self.c['original_run'],diagnostic=self.c['diagnostic'],active_run=self.derived_active_run(),
            pins=dict(source_commit=self.c['source_commit'],source_snapshot_git=self.c['source_snapshot_git'],
                source_dir=self.c['source_dir'],source_manifest_sha256=self.c['source_manifest_sha256'],
                helper_manifests={name:dict(helper_dir=phase['helper_dir'],sha256=phase['helper_manifest_sha256'])
                                  for name,phase in self.c['phases'].items()},
                config_relative=self.c['config_relative'],config_sha256=self.c['config_sha256'],
                bindings=self.c['bindings'],bindings_sha256=self.c['bindings_sha256'],
                runtime_fingerprint_sha256=canonical_sha(self.c['expected_runtime_fingerprint']),
                bootstrap_sha256=self.c['bootstrap_sha256'],original_checkpoint_receipt=self.c['original_checkpoint_receipt']),
            proof=dict(job_id=self.c['proof_job_id'],comparison=pin(proof)),
            validation=dict(job_id=self.j['validation_job_id'],comparison=pin(validation),
                ready=dict(relative_path=validation['ready_relative_path'],sha256=validation['ready_sha256'])),
            retirement=self.j['retirement'],start_checkpoint=start,
            semantics=dict(original='preserved in place',candidate='activated without relocation',
                           comparison='describes checkpoint at activation; active run subsequently evolves'))

    def initial_artifact_pins(self):
        root=self.derived_active_run()+'/latest_state/'
        selected={pin['relative_path'][len(root):]:pin for pin in self.j['receipts']['validation']['artifacts']
                  if pin['relative_path'].startswith(root)}
        require(set(selected)=={'meta.json','identity.json','compile_history.json'},'initial candidate artifact set differs')
        return selected

    def validate_activation_payload(self,payload):
        expected=self.activation_bindings()
        require(type(payload) is dict and set(payload)==set(expected)|{'created','directory_observation','initial_artifacts'},
                'activation receipt fields differ')
        require(all(encoded(payload[k])==encoded(v) for k,v in expected.items()),'activation binding differs')
        require(type(payload['created']) in (int,float) and math.isfinite(payload['created']) and payload['created']>0,
                'invalid activation creation time')
        audit=payload['directory_observation']
        require(type(audit) is dict and set(audit)=={'host','device','inode'} and
                type(audit['host']) is str and bool(audit['host']) and integer(audit['device']) and integer(audit['inode']),
                'invalid audit-only directory observation')
        initial=payload['initial_artifacts'];pins=self.initial_artifact_pins()
        require(type(initial) is dict and set(initial)==set(pins),'missing embedded initial artifacts')
        for name,pin in pins.items():
            item=initial[name]
            require(type(item) is dict and set(item)=={'relative_path','sha256','utf8'} and
                    item['relative_path']==pin['relative_path'] and item['sha256']==pin['sha256'] and
                    type(item['utf8']) is str and hashlib.sha256(item['utf8'].encode('utf8')).hexdigest()==pin['sha256'],
                    'embedded initial checkpoint bytes differ')
        return payload

    def activation_payload(self):
        pin=self.j.get('activation_receipt')
        expected=str((self.state/'activation_receipt.json').relative_to(self.c['work']))
        require(type(pin) is dict and set(pin)=={'relative_path','sha256'} and pin['relative_path']==expected,
                'missing or redirected activation receipt')
        payload=self.validate_activation_payload(read_json(checked_pin(self.c['work'],pin)))
        require(self.j.get('active_run')==payload['active_run'],'journal active path differs')
        return payload

    def active_run(self):
        return self.activation_payload()['active_run']

    def recheck_receipts(self,live_candidate=None):
        live_candidate=not self.production_started() if live_candidate is None else live_candidate
        initial=self.activation_payload()['initial_artifacts'] if self.j.get('activation_receipt') else None
        for kind,pin in self.j['receipts'].items():
            checked_pin(self.c['work'],pin)
            prefix=pin['root']+'/resumed/latest_state/'
            for artifact in pin['artifacts']:
                if kind=='validation' and artifact['relative_path'].startswith(prefix) and not live_candidate:
                    name=artifact['relative_path'][len(prefix):]
                    require(initial is not None and name in initial and initial[name]['sha256']==artifact['sha256'],
                            'historical candidate bytes lack activation binding')
                else:checked_pin(self.c['work'],artifact)
            if kind=='validation':
                checked_pin(self.c['work'],dict(relative_path=pin['ready_relative_path'],sha256=pin['ready_sha256']))

    def candidate(self):
        pin=self.j['receipts']['validation'];candidate=pin['candidate'];relative=self.derived_active_run()
        self.recheck_receipts(live_candidate=True)
        bindings=self.checkpoint_small(self.path(relative),'latest_state',dict(epoch=0,step=9840,global_step=410,pending_micro=0),True)
        require(bindings['meta_sha256']==candidate['meta_sha256'] and bindings['history_sha256']==candidate['history_sha256'] and
                bindings['identity_sha256']==pin['identity_sha256'],'candidate changed')
        for marker in ('TRAINING_FAILED','MEMORY_ABORT','TRAINING_DONE'):
            require(not os.path.lexists(str(self.path(relative)/marker)),'candidate has failure/final marker')
        return relative

    def held_old(self,observation):
        old=self.c['obsolete_job']
        require(observation and observation.get('job_id')==old['job_id'] and
                observation.get('owner')==self.c['owner'] and observation.get('uid')==self.c['uid'] and
                observation.get('command')==old['command'] and observation.get('active') is True and
                observation.get('state')=='PENDING' and observation.get('reason')=='JobHeldUser',
                'obsolete job is not the owned held PENDING allocation')

    def retire_prepare(self):
        self.candidate(); self.storage(); self.no_writers()
        old=self.c['obsolete_job']; observation=self.scheduler.observe(old['job_id'],self.c['owner'])
        if observation and observation.get('active') is False:
            require(observation.get('owner')==self.c['owner'] and observation.get('uid')==self.c['uid'] and
                    observation.get('command')==old['command'] and
                    observation.get('rows',{}).get(old['job_id'],[''])[0]=='CANCELLED','unexpected old terminal allocation')
            return self.save('old_job_retired',intent=None,retirement=observation)
        self.held_old(observation)
        intent=dict(kind='retire',created=time.time(),evidence=observation,
                    argv=['scancel','--state=PENDING','--user='+self.c['owner'],old['job_id']])
        self._fresh_intent='retire'
        return self.save('retire_intent',intent=intent)

    def retire(self):
        old=self.c['obsolete_job']; observation=self.scheduler.observe(old['job_id'],self.c['owner'])
        if observation is None: raise Wait('obsolete job accounting unavailable')
        if not observation.get('active'):
            rows=observation.get('rows',{})
            require(observation.get('owner')==self.c['owner'] and observation.get('uid')==self.c['uid'] and
                    observation.get('command')==old['command'] and
                    rows.get(old['job_id'],[''])[0]=='CANCELLED','obsolete job not terminal CANCELLED')
            return self.save('old_job_retired',intent=None,retirement=observation)
        self.held_old(observation)
        require(self._fresh_intent=='retire','uncertain cancellation: observe until terminal; no retry')
        self._fresh_intent=None
        self.scheduler.cancel(self.j['intent']['argv'])
        return self.save('retire_waiting')

    def retired_old(self):
        old=self.c['obsolete_job'];job=self.scheduler.observe(old['job_id'],self.c['owner'])
        if job is None:raise Wait('retired old job accounting unavailable')
        require(job.get('job_id')==old['job_id'] and job.get('owner')==self.c['owner'] and
                job.get('uid')==self.c['uid'] and job.get('command')==old['command'] and job.get('active') is False and
                job.get('rows',{}).get(old['job_id'],[''])[0]=='CANCELLED','retired old job identity/state changed')

    def activation_gates(self):
        self.pins();self.original();self.candidate();self.storage();self.no_writers();self.retired_old()
        self.observe_success(self.j['validation_job_id'],self.j['validation_submission']['intent'])
        claim=read_json(canonical_controls(self.c)[1])
        require(claim==dict(operation_id=self.c['operation_id'],state_dir=str(self.state),
                            config_sha256=self.j['config_sha256']),'logical C2 claim changed')

    def activation_prepare(self):
        require(self.j['phase']=='old_job_retired' and self.j['intent'] is None,'activation requires retired old job')
        self.activation_gates();relative=self.derived_active_run();observed=self.path(relative).stat()
        payload=self.activation_bindings()
        payload.update(created=time.time(),directory_observation=dict(host=socket.gethostname(),device=observed.st_dev,inode=observed.st_ino),
                       initial_artifacts={name:dict(pin,utf8=read_bytes(self.path(pin['relative_path'])).decode('utf8'))
                                          for name,pin in self.initial_artifact_pins().items()})
        self.validate_activation_payload(payload)
        return self.save('activation_intent',activation_intent=dict(payload=payload))

    def activate(self):
        require(self.j['phase']=='activation_intent' and not self.production_started(),'no recoverable activation intent')
        payload=self.validate_activation_payload(self.j['activation_intent']['payload'])
        self.activation_gates()
        path=self.state/'activation_receipt.json';exclusive_json(path,payload)
        return self.save('candidate_activated',activation_receipt=dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path)),
                         active_run=payload['active_run'])

    def advance(self,phase):
        if phase=='proof_accepted':
            self.recheck_receipts();self.observe_success(self.c['proof_job_id'])
            self.prepare_submission('validation');return self.execute_submission()
        if phase=='waiting_validation':
            job=self.j['validation_job_id']
            return self.accept_science('validation',self.observe_success(job,self.j['validation_submission']['intent']))
        if phase=='validation_accepted':
            self.retire_prepare()
            return self.j if self.j['phase']=='old_job_retired' else self.retire()
        if phase in ('retire_intent','retire_waiting'):return self.retire()
        if phase=='old_job_retired':return self.activation_prepare()
        if phase=='activation_intent':return self.activate()
        if phase in ('candidate_activated','link_accepted'):
            request=self.build_request();self.prepare_submission('production',request)
            return self.execute_submission()
        if phase=='waiting_production':return self.accept_link()
        if phase in ('final_verified','nice_intent','priorities_restored'):return self.restore_nice()
        if phase=='complete':return self.j
        raise Blocked('unsupported continuation phase: '+phase)

    def build_request(self):
        require(self.j['phase'] in ('candidate_activated','link_accepted') and self.j['intent'] is None,
                'pending submission requires reconciliation, never another request')
        self.pins();self.original();self.active_run();self.recheck_receipts();self.storage();self.no_writers()
        index=len(self.j['links'])+1
        if index==1:
            self.candidate()
            pin=self.j['receipts']['validation']; candidate=pin['candidate']
            start={k:candidate[k] for k in ('epoch','step','global_step','pending_micro','meta_sha256','history_sha256','canonical_checkpoint_sha256')}
            start.update(relative_dir='latest_state',identity_sha256=pin['identity_sha256'])
            predecessor=dict(relative_path=pin['ready_relative_path'],sha256=pin['ready_sha256'])
        else:
            prior=self.j['links'][-1]
            checked_pin(self.c['work'],prior['receipt'])
            receipt=read_json(self.path(prior['receipt']['relative_path']))
            require(receipt.get('final') is None and receipt['result'].get('done') is not True,'cannot continue final checkpoint')
            self.validate_link_receipt(prior,receipt)
            cp=receipt['checkpoint'];start=dict(cp['position'])
            start.update({k:cp[k] for k in ('relative_dir','meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256')})
            predecessor={k:prior['receipt'][k] for k in ('relative_path','sha256')}
        require(start['global_step']<4099,'production already reached target')
        helper=self.c['phases']['production']
        relative=str(self.state.relative_to(self.c['work']))+'/links/%04d'%index
        request=dict(version=1,operation_id=self.c['operation_id'],token='c2-%s-production-%04d'%(self.c['operation_id'],index),
            link_index=index,source_commit=self.c['source_commit'],source_snapshot_git=self.c['source_snapshot_git'],
            source_dir=self.c['source_dir'],source_manifest_sha256=self.c['source_manifest_sha256'],helper_dir=helper['helper_dir'],
            helper_manifest_sha256=helper['helper_manifest_sha256'],run_dir=self.active_run(),bindings=self.c['bindings'],
            bindings_sha256=self.c['bindings_sha256'],config_relative=self.c['config_relative'],config_sha256=self.c['config_sha256'],
            receipt_dir=relative,profile=dict(PROFILE),start_checkpoint=start,predecessor=predecessor,
            expected_runtime_fingerprint=self.c['expected_runtime_fingerprint'],completed_epoch_events=self.j['epoch_events'])
        directory=self.path(relative)
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'request.json'
        if path.exists(): require(read_json(path)==request,'existing production request conflicts')
        else:atomic_json(path,request)
        return dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path))

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
            require(observation.get('epoch_end')==event['epoch'] and observation.get('arm')=='C2' and
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
        self.original();self.recheck_receipts()
        expected=dict(version=1,kind='c2_production_link_ready',operation_id=self.c['operation_id'],token=request['token'],
            link_index=request['link_index'],job_id=entry['job_id'],request_sha256=request_pin['sha256'],
            source_commit=self.c['source_commit'],predecessor_sha256=request['predecessor']['sha256'],
            start_global_step=request['start_checkpoint']['global_step'],source_snapshot_git=self.c['source_snapshot_git'],
            source_manifest_sha256=self.c['source_manifest_sha256'],helper_manifest_sha256=request['helper_manifest_sha256'],
            config_sha256=self.c['config_sha256'],runtime_fingerprint_sha256=canonical_sha(self.c['expected_runtime_fingerprint']))
        require(all(type(receipt.get(k)) is type(v) and receipt[k]==v for k,v in expected.items()),'production receipt envelope mismatch')
        require(encoded(receipt.get('profile'))==encoded(PROFILE),'production profile changed')
        verified=receipt.get('input_verification',{})
        require(verified.get('job_id')==entry['job_id'] and encoded(verified.get('ranks'))==encoded([0,1,2,3]) and
                verified.get('bindings_sha256')==self.c['bindings_sha256'] and digest_ok(verified.get('record_sha256')) and
                digest_ok(verified.get('identity_sha256')),'missing fresh four-rank input verification')
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
            require(meta.get('arm')=='C2' and type(meta.get('seed')) is int and meta['seed']==123 and
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
        self.observe_success(entry['job_id'],entry['intent'])
        self.pins();self.original();self.active_run();self.recheck_receipts();self.no_writers()
        request=read_json(self.path(entry['intent']['request']['relative_path']))
        path=self.path(request['receipt_dir'])/('execution_'+entry['job_id'])/'link_receipt.json'
        receipt=read_json(path);done=self.validate_link_receipt(entry,receipt)
        entry['receipt']=dict(relative_path=str(path.relative_to(self.c['work'])),sha256=sha(path))
        return self.save('final_verified' if done else 'link_accepted',epoch_events=receipt['epoch_events'])

    def restore_nice(self):
        require(self.j['phase'] in ('final_verified','nice_intent','priorities_restored'),'final verification required before priority changes')
        entry=self.j['links'][-1]; checked_pin(self.c['work'],entry['receipt'])
        self.observe_success(entry['job_id'],entry['intent'])
        require(self.validate_link_receipt(entry,read_json(self.path(entry['receipt']['relative_path']))),'final receipt required')
        remaining=[p for p in self.c['protected_jobs'] if p['job_id'] not in self.j['restored']]
        if not remaining:
            return self.save('complete',intent=None,b6_followup='blocked: held B6 source/launcher/input review and replacement plan required')
        pinned=remaining[0];job=self.scheduler.observe(pinned['job_id'],self.c['owner'])
        if job is None:raise Wait('protected job accounting unavailable')
        require(job.get('owner')==self.c['owner'] and job.get('uid')==self.c['uid'],'protected job owner differs')
        if not job.get('active'):
            rows=job.get('rows',{}); require(pinned['job_id'] in rows,'protected job terminal accounting missing')
            return self.save('final_verified',restored=self.j['restored']+[pinned['job_id']],intent=None)
        require(job.get('command')==pinned['command'],'protected job command differs')
        if pinned['held']:
            require(job.get('state')=='PENDING' and job.get('reason')=='JobHeldUser','B6 no longer held')
        if str(job.get('nice'))=='0':
            return self.save('final_verified',restored=self.j['restored']+[pinned['job_id']],intent=None)
        if self.j['phase']=='nice_intent':
            require(self.j['intent'].get('kind')=='nice' and self.j['intent'].get('job_id')==pinned['job_id'],
                    'priority intent identity mismatch')
            self.scheduler.nice(self.j['intent']['argv'])
            return self.j
        intent=dict(kind='nice',job_id=pinned['job_id'],created=time.time(),
                    argv=['scontrol','update','JobId='+pinned['job_id'],'Nice=0'])
        self.save('nice_intent',intent=intent)
        self.scheduler.nice(intent['argv'])
        return self.j


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command')
    init=commands.add_parser('init');init.add_argument('--config',required=True);init.add_argument('--state-dir',required=True)
    for name in ('status','tick','reconcile','watch'):
        command=commands.add_parser(name);command.add_argument('--state-dir',required=True)
        if name=='watch':command.add_argument('--poll-seconds',type=int,default=60)
    args=parser.parse_args(argv)
    if args.command is None:parser.error('a command is required')
    try:
        if args.command=='init':
            Controller.initialize(read_json(args.config),Path(args.state_dir))
            print(json.dumps(dict(initialized=True,state_dir=args.state_dir),sort_keys=True));return 0
        controller=Controller(Path(args.state_dir))
        if args.command=='status':result=controller.status()
        elif args.command=='tick':result=controller.tick()
        elif args.command=='reconcile':
            with controller.locked():
                controller.pins();result=controller.reconcile()
        else:
            require(1<=args.poll_seconds<=60,'watch interval must be1..60 seconds')
            while True:
                try:result=controller.tick()
                except Blocked as exc:
                    print(json.dumps(dict(blocked=str(exc),phase=controller.j['phase'])),flush=True)
                    return 2
                print(json.dumps(dict(phase=result['phase'],revision=result['revision'],last_error=result['last_error'])),flush=True)
                if result['phase']=='complete':return 0
                time.sleep(args.poll_seconds)
        print(json.dumps(result,sort_keys=True,allow_nan=False));return 0
    except (Blocked,Wait,OSError,ValueError,KeyError,TypeError) as exc:
        print('BLOCKED: '+str(exc),file=sys.stderr);return 2


if __name__=='__main__':
    sys.exit(main())
