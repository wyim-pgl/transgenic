"""Real immutable control artifacts; only scheduler/storage are simulated."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    import continue_b6_history as host
except ImportError:
    host = None


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')


def manifest(root, names, filename='manifest.sha256'):
    (root / filename).write_text(''.join(hashlib.sha256((root/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
    return hashlib.sha256((root/filename).read_bytes()).hexdigest()


def rt_identity(binding):
    from transgenic.training import b6_runtime as rt
    return rt.identity_from_bindings(binding)


class Scheduler:
    def __init__(self, config):
        self.config=config; self.jobs={}; self.submissions=[]; self.cancellations=[]
        self.matches=[]; self.lose_submit=False; self.lose_cancel=False; self.reserve=101*1024**3
        for pin in [config['proof_identity']]+config['obsolete_jobs']:
            self.jobs[pin['job_id']]=dict(pin, active=False, rows={pin['job_id']:['COMPLETED','0:0'],pin['job_id']+'.batch':['COMPLETED','0:0']})
        for pin in config['obsolete_jobs']:
            self.jobs[pin['job_id']].update(active=True,state='PENDING',reason='JobHeldUser')
    def observe(self, job, owner): return copy.deepcopy(self.jobs.get(job))
    def active(self, owner): return [copy.deepcopy(j) for j in self.jobs.values() if j['active']]
    def storage(self, config, root):
        return dict(project_id=22169,pool='ddn_ssd',quota_remaining_bytes=self.reserve,filesystem_available_bytes=self.reserve)
    def cancel(self, argv):
        job=argv[-1];self.cancellations.append(job)
        self.jobs[job].update(active=False,rows={job:['CANCELLED','0:0']})
        if self.lose_cancel: raise host.Wait('lost cancel response')
        return dict(returncode=0,stdout='',stderr='')
    def submit(self, argv, env):
        self.submissions.append((argv,env))
        if self.lose_submit: raise host.Wait('lost submit response')
        return dict(returncode=0,stdout=str(8999+len(self.submissions))+'\n',stderr='')
    def discover(self, intent, owner): return copy.deepcopy(self.matches)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cached_proof=None
        cls.template=tempfile.TemporaryDirectory()
        cls.root=Path(cls.template.name)/'proof/1234'
        # Actual Task1 fixture and publisher: all real small CPU checkpoint state.
        from test_b6_history_proof import B6ProofTests
        cls.proofcase=B6ProofTests();cls.proofcase.setUp()
        cls.proofcase.fixture(cls.root)
        from transgenic.training import b6_runtime as rt
        sys.path.insert(0,str(Path(rt.__file__).parents[3]))
        from revision.scripts.b6 import wp4_synthetic as syn
        cls.binding=syn.binding_record()
        # Ordinary training identity includes accumulation and metadata includes arm.
        for phase in ('continuous','at400','resumed'):
            cp=cls.root/phase/'latest_state'
            identity=json.loads((cp/'identity.json').read_text())
            for key in rt.IDENTITY_REQUIRED_KEYS:identity.setdefault(key,'fixture')
            identity.update(rt.identity_from_bindings(cls.binding));identity['accumulation_steps']=96
            identity['recipe_sha256']=hashlib.sha256(b'{}\n').hexdigest()
            history=json.loads((cp/'compile_history.json').read_text());history['identity']=identity
            history['fingerprint']['recipe']['_sha256']=identity['recipe_sha256']
            put(cp/'identity.json',identity);put(cp/'compile_history.json',history)
            meta=json.loads((cp/'meta.json').read_text());meta.update(arm='B6',seed=123,
                identity_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                compile_history_sha256=hashlib.sha256((cp/'compile_history.json').read_bytes()).hexdigest())
            put(cp/'meta.json',meta)
            put(cls.root/phase/'run_config_b6.json',identity)
        # Snapshot diagnostic inventory must bind the updated files.
        snap=cls.root/'at400'
        put(snap/'DIAGNOSTIC_SOURCE.json',dict(sha256={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (snap/'latest_state').rglob('*') if p.is_file()}))
        # Refresh the one snapshot trace entry after metadata changed.
        trace=cls.root/'continuous/trace_rank0.jsonl'
        lines=trace.read_text().splitlines()
        for i,line in enumerate(lines):
            row=json.loads(line)
            if row['event']=='snapshot_published':
                row['receipt_sha256']=hashlib.sha256((snap/'DIAGNOSTIC_SOURCE.json').read_bytes()).hexdigest();lines[i]=json.dumps(row)
        trace.write_text('\n'.join(lines)+'\n')
    @classmethod
    def tearDownClass(cls): cls.template.cleanup()
    def setUp(self):
        self.assertIsNotNone(host, 'B6 controller is not implemented')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.work=Path(self.tmp.name);self.proof=self.work/'proof/1234'
        shutil.copytree(self.root,self.proof)
        source=self.work/'source';source.mkdir()
        required=['train/train_HyenaTransgenic_B6.py','src/transgenic/training/b6_runtime.py','src/transgenic/training/b6_compile_history.py','src/transgenic/training/b6_compile_replay.py','src/transgenic/training/b6_replay_state.py','revision/scripts/b6/storage_preflight.py','recipe.json']
        for n in required:
            p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}\n')
        (source/'SOURCE_COMMIT').write_text(host.SOURCE+'\n')
        source_sha=manifest(source,required,'validation_sources.sha256')
        subprocess.run(['git','init','-q',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'add','.'],check=True)
        subprocess.run(['git','-C',str(source),'-c','user.name=Fixture','-c','user.email=fixture@example.test','commit','-qm','fixture'],check=True)
        snapshot=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],universal_newlines=True).strip()
        phases={}
        for phase,names in [('proof',host.PROOF_MEMBERS),('production',host.PRODUCTION_MEMBERS)]:
            folder=self.work/('helpers/'+phase);folder.mkdir(parents=True)
            for n in names:
                p=folder/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# pinned fixture '+n+'\n')
            phases[phase]=dict(helper_dir='helpers/'+phase,script='helpers/'+phase+'/'+('b6_history_proof.sbatch' if phase=='proof' else 'b6_production_link.sbatch'),helper_manifest_sha256=manifest(folder,sorted(names)))
        put(self.work/'bindings.json',{'binding':'fixed'})
        verified=json.loads((self.proof/'verified_record.json').read_text())
        verified.update(bindings_sha256=host.sha(self.work/'bindings.json'),recipe_sha256=host.sha(source/'recipe.json'),identity_from_bindings=rt_identity(self.binding),record=self.binding)
        verified['recipe']['_sha256']=verified['recipe_sha256']
        verified['provenance'].update(source_manifest_sha256=source_sha,helper_manifest_sha256=phases['proof']['helper_manifest_sha256'])
        put(self.proof/'verified_record.json',verified)
        fresh=json.loads((self.proof/'fresh_start.json').read_text());fresh['verified_record_sha256']=host.sha(self.proof/'verified_record.json');put(self.proof/'fresh_start.json',fresh)
        oldenv=os.environ.get('SLURM_JOB_ID');os.environ['SLURM_JOB_ID']='1234'
        try:
            if self.__class__.cached_proof is None:
                self.proofcase.publish(self.proof)
                self.__class__.cached_proof={n:(self.proof/n).read_bytes() for n in ('comparison.json','validation_receipt.json')}
            else:
                for n,raw in self.__class__.cached_proof.items():(self.proof/n).write_bytes(raw)
        finally:
            if oldenv is None: os.environ.pop('SLURM_JOB_ID',None)
            else: os.environ['SLURM_JOB_ID']=oldenv
        def pin(job,script):
            argv=['sbatch','--comment=pin-'+job,str(self.work/script)]
            return dict(job_id=job,owner='wcyim',uid=97434,token='pin-'+job,argv=argv,work_dir=str(self.work),command=str(self.work/script),created='2026-10-20T01:00:00')
        self.config=dict(version=1,operation_id='test-b6',owner='wcyim',uid=97434,work=str(self.work),
            sif='/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif',image_sha256='4'*64,
            source_commit=host.SOURCE,source_snapshot_git=snapshot,source_dir='source',source_manifest_sha256=source_sha,
            phases=phases,proof_root='proof',proof_job_id='1234',proof_identity=pin('1234',phases['proof']['script']),
            obsolete_jobs=[pin('3345918','old/wp5_pilot.sbatch'),pin('3345917','old/wp5_pilot.sbatch')],
            bindings='bindings.json',bindings_sha256=host.sha(self.work/'bindings.json'),config_relative='recipe.json',config_sha256=host.sha(source/'recipe.json'),
            expected_runtime_fingerprint=json.loads((self.proof/'comparison.json').read_text())['checkpoint_bindings']['resumed']['fingerprint'],
            storage_python=sys.executable,storage_script='source/revision/scripts/b6/storage_preflight.py',
            resources=dict(account='bilv-dtai-gh',partition='ghx4',time='48:00:00',cpus_per_task=64,exclude='gh062,gh093',output='logs/%x-%j.out'))
        self.state=self.work/'operation';host.Controller.initialize(self.config,self.state)
        self.scheduler=Scheduler(self.config);self.controller=host.Controller(self.state,self.scheduler)
    def advance(self,phase):
        for _ in range(20):
            if self.controller.j['phase']==phase:return
            self.controller.tick()
        self.fail(self.controller.j)
    def test_actual_ready_activates_embedded_initial_history_and_request(self):
        self.advance('candidate_activated')
        self.assertEqual(self.scheduler.cancellations,['3345918','3345917'])
        activation=host.read_json(self.state/'activation.json')
        self.assertEqual(json.loads(activation['initial']['meta.json']['text'])['global_step'],410)
        request_pin=self.controller.build_request()
        from b6_production_link import validate_request
        request=host.read_json(self.work/request_pin['relative_path']);validate_request(request)
        self.assertEqual(request['predecessor']['sha256'],host.sha(self.proof/'validation_receipt.json'))
        self.assertEqual(request['profile']['arm'],'B6')
    def test_forged_false_comparison_and_missing_prepared_rank_refuse(self):
        report=host.read_json(self.proof/'comparison.json');report['model_equal']=False;put(self.proof/'comparison.json',report)
        ready=host.read_json(self.proof/'validation_receipt.json');ready['comparison_sha256']=host.sha(self.proof/'comparison.json');put(self.proof/'validation_receipt.json',ready)
        with self.assertRaises(host.Blocked):self.controller.tick()
        self.assertEqual(self.scheduler.cancellations,[])
    def test_prepared_file_and_trace_binding_refuse(self):
        (self.proof/'resumed/prepared_schema_rank3.json').unlink()
        with self.assertRaises(host.Blocked):self.controller.tick()
    def test_partial_retirement_lost_cancel_never_reissues(self):
        self.scheduler.lose_cancel=True
        self.advance('retire_waiting');self.assertEqual(self.scheduler.cancellations,['3345918'])
        self.controller=host.Controller(self.state,self.scheduler)
        self.advance('candidate_activated')
        self.assertEqual(self.scheduler.cancellations,['3345918','3345917'])
    def test_running_foreign_script_old_job_refuse(self):
        for field,value in [('state','RUNNING'),('uid',42),('command','/foreign/wp5_pilot.sbatch')]:
            with self.subTest(field=field):
                original=copy.deepcopy(self.scheduler.jobs['3345918'])
                self.scheduler.jobs['3345918'][field]=value
                with self.assertRaises(host.Blocked):self.controller.tick()
                self.scheduler.jobs['3345918']=original
        self.assertEqual(self.scheduler.cancellations,[])
    def test_reserve_wait_then_recovers(self):
        self.advance('both_old_jobs_retired');self.scheduler.reserve=99*1024**3
        self.controller.tick();self.assertEqual(self.controller.j['phase'],'both_old_jobs_retired')
        self.assertIn('reserve',self.controller.j['last_wait'])
        self.scheduler.reserve=101*1024**3;self.advance('candidate_activated')
    def test_submission_recovered_intent_never_submits(self):
        self.advance('candidate_activated');self.controller.prepare_submission(self.controller.build_request())
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'submission_uncertain');self.assertEqual(self.scheduler.submissions,[])
        self.scheduler.matches=[dict(job_id='8'),dict(job_id='9')]
        self.controller.tick();self.assertEqual(self.controller.j['phase'],'submission_uncertain')
        self.assertEqual(self.scheduler.submissions,[])
    def test_lost_submission_and_mutated_candidate_refuse(self):
        self.advance('candidate_activated');self.scheduler.lose_submit=True;self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'submission_uncertain');self.assertEqual(len(self.scheduler.submissions),1)
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick();self.assertEqual(len(self.scheduler.submissions),1)
    def test_mutated_candidate_before_first_link_refused(self):
        self.advance('candidate_activated');p=self.proof/'resumed/latest_state/meta.json'
        value=host.read_json(p);value['step']+=1;put(p,value)
        with self.assertRaises(host.Blocked):self.controller.tick()
        self.assertEqual(self.scheduler.submissions,[])
    def test_mutated_candidate_between_fresh_intent_and_sbatch_refuses(self):
        self.advance('candidate_activated');self.controller.prepare_submission(self.controller.build_request())
        p=self.proof/'resumed/latest_state/meta.json';value=host.read_json(p);value['global_step']=411;put(p,value)
        with self.assertRaisesRegex(host.Blocked,'position'):self.controller.execute_submission()
        self.assertEqual(self.scheduler.submissions,[])
    def test_exclusive_operation_and_concurrent_watcher(self):
        with self.assertRaises(host.Blocked):host.Controller.initialize(self.config,self.work/'second')
        with host.Lock(self.controller.lock_path):
            with self.assertRaises(host.Blocked):self.controller.tick()
    def test_large_manifest_member_streams_but_json_cap_remains(self):
        p=self.work/'large.py';p.write_bytes(b'x'*(17*1024**2))
        self.assertEqual(host.source_member_sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
        pin=manifest(self.work,['large.py'])
        self.assertEqual(host.manifest_check(self.work,'manifest.sha256',pin),{'large.py'})
        with self.assertRaises(host.Blocked):host.read_json(p)
    def test_real_storage_helper_insufficient_error_is_retryable(self):
        self.advance('both_old_jobs_retired')
        class Below(Scheduler):
            def storage(self,config,root):
                raise host.Blocked('storage preflight failed: RuntimeError: insufficient CP4 storage: quota_remaining=1, filesystem_available=2, required=107374182400')
        below=Below(self.config);below.jobs=copy.deepcopy(self.scheduler.jobs);self.controller.scheduler=below
        self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'both_old_jobs_retired')
        self.assertIn('storage',self.controller.j['last_wait'])
    def test_prepared_trace_and_helper_script_pins_refuse(self):
        path=self.proof/'continuous/trace_rank2.jsonl'
        raw=path.read_bytes();first,rest=raw.split(b'\n',1)
        row=json.loads(first);row['prepared_schema_sha256']='0'*64
        path.write_bytes(json.dumps(row).encode()+b'\n'+rest)
        with self.assertRaisesRegex(host.Blocked,'prepared trace'):self.controller.tick()
        path.write_bytes(raw)
        script=self.work/self.config['phases']['proof']['script'];script.write_text('# replaced\n')
        with self.assertRaisesRegex(host.Blocked,'manifest member'):self.controller.tick()
        self.assertEqual(self.scheduler.cancellations,[])
    def test_exact_proof_script_and_batch_accounting_required(self):
        job=self.scheduler.jobs['1234'];original=copy.deepcopy(job)
        job['argv'][-1]='/different/b6_history_proof.sbatch';job['command']=job['argv'][-1]
        with self.assertRaisesRegex(host.Blocked,'script'):self.controller.tick()
        self.scheduler.jobs['1234']=original
        self.scheduler.jobs['1234']['rows']['1234.batch']=['FAILED','1:0']
        with self.assertRaisesRegex(host.Blocked,'parent/batch'):self.controller.tick()
        self.assertEqual(self.scheduler.cancellations,[])
    def test_reconcile_adopts_one_exact_job_without_another_submission(self):
        self.advance('candidate_activated');self.controller.prepare_submission(self.controller.build_request())
        intent=self.controller.j['intent'];created=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(intent['created']))
        job=dict(job_id='9123',owner='wcyim',uid=97434,token=intent['token'],argv=intent['argv'],work_dir=intent['work_dir'],created=created,command=intent['argv'][-1],active=True,state='PENDING',rows={})
        self.scheduler.jobs['9123']=job
        self.scheduler.matches=[dict(job,created=intent['created'])]
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'waiting_production');self.assertEqual(self.controller.j['links'][-1]['job_id'],'9123')
        self.assertEqual(self.scheduler.submissions,[])
    def test_recovered_cancel_intent_never_reissues_unobserved_cancel(self):
        self.advance('validation_accepted');pin=self.config['obsolete_jobs'][0]
        self.controller.save('retire_intent',intent=dict(kind='retire',job_id=pin['job_id'],pin=pin,created=time.time(),argv=['scancel',pin['job_id']]))
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick();self.controller.tick()
        self.assertEqual(self.scheduler.cancellations,[])
        self.assertEqual(self.controller.j['phase'],'retire_intent')
        self.assertIn('never reissued',self.controller.j['last_wait'])
    def test_activation_crash_after_publication_reuses_exact_small_receipt(self):
        self.advance('candidate_activated');raw=(self.state/'activation.json').read_bytes()
        receipts=dict(self.controller.j['receipts']);receipts.pop('activation')
        self.controller.save('activation_intent',receipts=receipts)
        self.controller=host.Controller(self.state,self.scheduler);self.controller.tick()
        self.assertEqual(self.controller.j['phase'],'candidate_activated')
        self.assertEqual((self.state/'activation.json').read_bytes(),raw)
        self.assertEqual(self.scheduler.cancellations,['3345918','3345917'])
    def test_host_python36_pure_import(self):
        ast.parse((HERE/'continue_b6_history.py').read_text(),feature_version=(3,6))
        result=subprocess.run([sys.executable,'-S','-c',"import sys;sys.path.insert(0,%r);import continue_b6_history;assert 'torch' not in sys.modules"%str(HERE)],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__': unittest.main()
