"""Host controller contracts; filesystem effects are real, only Slurm is fake."""
import copy
import ctypes
import hashlib
import importlib.util
import json
import multiprocessing
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('continue_c2_history', str(HERE / 'continue_c2_history.py'))
if SPEC and (HERE / 'continue_c2_history.py').exists():
    ctl = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(ctl)
else:
    ctl = None


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode() if not isinstance(value, bytes) else value
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def proof():
    return dict(passed=True, model_equal=True, history_equal=True, different_model_keys=[],
        states_equal={k: True for k in ['optimizer.bin', 'scheduler.bin'] +
                      ['random_states_%d.pkl' % r for r in range(4)]},
        traces={str(r): {k: [] for k in ['inputs', 'shapes', 'rng_before', 'rng_after',
                'loss', 'context', 'gradients', 'boundary_rng']} for r in range(4)},
        historical210=dict(expected=3.2567708492279053, actual=3.2567708492279053, equal=True))


def lock_child(path, queue):
    try:
        with ctl.Lock(path):
            queue.put('acquired')
    except ctl.Blocked:
        queue.put('busy')


def init_child(config,path,queue):
    try:
        ctl.Controller.initialize(config,path);queue.put('created')
    except ctl.Blocked:queue.put('blocked')


def tick_child(path,queue,ready):
    scheduler=FakeSlurm();ready.wait(5)
    try:
        ctl.Controller(path,scheduler).tick()
    except ctl.Blocked:
        pass
    queue.put(sum(c[0]=='submit' for c in scheduler.calls))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ctl, 'new controller is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_strict_json_rejects_duplicate_nan_and_symlink(self):
        for data in [b'{"a":1,"a":2}', b'{"a":NaN}']:
            p = self.root / 'bad.json'; p.write_bytes(data)
            with self.assertRaises(ctl.Blocked): ctl.read_json(p)
        put(self.root / 'good', {})
        (self.root / 'link').symlink_to(self.root / 'good')
        with self.assertRaises(ctl.Blocked): ctl.read_json(self.root / 'link')

    def test_paths_reject_escape_symlink_and_empty_destination(self):
        for name in ['../bad', '/tmp/bad', 'a/../../b', '.', 'a//b']:
            with self.assertRaises(ctl.Blocked): ctl.safe_path(self.root, name)
        (self.root / 'outside').symlink_to('/tmp', target_is_directory=True)
        with self.assertRaises(ctl.Blocked): ctl.safe_path(self.root, 'outside/file')

    def test_immutable_receipt_racing_destination_is_never_replaced(self):
        self.assertTrue(hasattr(ctl,'exclusive_json'),'exclusive receipt publication missing')
        destination=self.root/'receipt.json';real_link=os.link;inserted=[]
        def publish_before_link(source,target,*args,**kwargs):
            destination.write_bytes(b'foreign evidence');inserted.append(destination.stat().st_ino)
            return real_link(source,target,*args,**kwargs)
        with patch.object(os,'link',new=publish_before_link):
            with self.assertRaises(ctl.Blocked):ctl.exclusive_json(destination,{'valid':True})
        self.assertEqual(destination.stat().st_ino,inserted[0])
        self.assertEqual(destination.read_bytes(),b'foreign evidence')

    def test_immutable_receipt_recovers_identical_bytes_and_rejects_directory_or_symlink(self):
        self.assertTrue(hasattr(ctl,'exclusive_json'),'exclusive receipt publication missing')
        destination=self.root/'receipt.json';ctl.exclusive_json(destination,{'valid':True})
        inode=destination.stat().st_ino;ctl.exclusive_json(destination,{'valid':True})
        self.assertEqual(destination.stat().st_ino,inode)
        with self.assertRaises(ctl.Blocked):ctl.exclusive_json(destination,{'valid':False})
        (self.root/'empty').mkdir();(self.root/'alias').symlink_to(destination)
        for name in ('empty','alias'):
            with self.assertRaises(ctl.Blocked):ctl.exclusive_json(self.root/name,{'valid':True})

    def test_lock_inode_persists_and_other_process_cannot_enter(self):
        lock = self.root / 'lock'; queue = multiprocessing.Queue()
        with ctl.Lock(lock):
            inode = lock.stat().st_ino
            p = multiprocessing.Process(target=lock_child, args=(lock, queue))
            p.start(); p.join(5)
            self.assertFalse(p.is_alive()); self.assertEqual(queue.get(timeout=2), 'busy')
        with ctl.Lock(lock): self.assertEqual(lock.stat().st_ino, inode)

    def test_atomic_journal_replacement_survives_reopen(self):
        path = self.root / 'journal.json'
        ctl.atomic_json(path, {'revision': 1})
        ctl.atomic_json(path, {'revision': 2})
        self.assertEqual(ctl.read_json(path), {'revision': 2})

    def test_science_rejects_passed_only_and_string_booleans(self):
        ctl.validate_science(proof(), 'resume210')
        for bad in [dict(passed=True), dict(proof(), passed='true'),
                    dict(proof(), traces={'0': {}})]:
            with self.assertRaises(ctl.Blocked): ctl.validate_science(bad, 'resume210')
        v = proof(); v.update(profile='validate400', checkpoint_digests={'continuous':'a'*64, 'resumed':'a'*64},
            validation400=dict(checkpoint_step=400, stop_step=410, rank3_specialization_slot=6718,
                               rank3_specialization_recorded=True))
        ctl.validate_science(v, 'validate400')
        v['validation400']['rank3_specialization_recorded'] = 1
        with self.assertRaises(ctl.Blocked): ctl.validate_science(v, 'validate400')

    def test_scheduler_requires_parent_batch_and_no_requeue(self):
        good = dict(active=False, rows={'77': ['COMPLETED','0:0'], '77.batch':['COMPLETED','0:0'],
                                       '77.extern':['OUT_OF_MEMORY','0:125']})
        self.assertTrue(ctl.terminal_ok(good, '77'))
        for bad in [dict(good, active=True), dict(good, rows={'77':['COMPLETED','0:0']}),
                    dict(good, rows={'77':['COMPLETED','0:0'], '77.batch':['OUT_OF_MEMORY','0:125']})]:
            with self.assertRaises((ctl.Blocked,ctl.Wait)): ctl.terminal_ok(bad, '77')

    def test_unittest_log_gate_requires_exact_counts_and_reviewed_skips(self):
        p = self.root / 'tests.log'
        p.write_text('Ran 12 tests in 0.12s\n\nOK (skipped=1)\n')
        ctl.check_test_log(p, {'ran': 12, 'skipped': 1})
        with self.assertRaises(ctl.Blocked): ctl.check_test_log(p, {'ran': 12, 'skipped': 0})
        p.write_text('Ran 12 tests in 0.12s\nFAILED (failures=1)\n')
        with self.assertRaises(ctl.Blocked): ctl.check_test_log(p, {'ran': 12, 'skipped': 1})

    def test_blank_sacct_comment_uses_unique_exact_submitline_token(self):
        self.assertTrue(hasattr(ctl,'submission_token'),'accounting blank-comment support missing')
        argv=['sbatch','--parsable','--comment=c2-history-198ce14-5d10eaad8598','/immutable/proof.sbatch']
        self.assertEqual(ctl.submission_token('',argv),'c2-history-198ce14-5d10eaad8598')
        with self.assertRaises(ctl.Blocked):ctl.submission_token('conflict',argv)
        with self.assertRaises(ctl.Blocked):ctl.submission_token('',argv+['--comment=second'])

    def test_pinned_source_manifest_streams_members_larger_than_control_json_limit(self):
        member=self.root/'source-example.db'
        with member.open('wb') as stream:stream.seek(17*1024*1024);stream.write(b'x')
        digest=hashlib.sha256()
        with member.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        manifest=put(self.root/'manifest.sha256',(digest.hexdigest()+'  source-example.db\n').encode())
        ctl.manifest_check(self.root,'manifest.sha256',manifest)
        with self.assertRaises(ctl.Blocked):ctl.read_json(member)


class FakeSlurm:
    def __init__(self):
        self.jobs = {}; self.calls = []; self.matches = []; self.uncertain = False
        self.free = 150 * 1024**3

    def submit(self, argv, env):
        self.calls.append(('submit', argv, env))
        if self.uncertain: raise TimeoutError('accepted but response lost')
        return {'returncode': 0, 'stdout': '9001\n', 'stderr': ''}

    def discover(self, intent, owner): return self.matches
    def observe(self, job_id, owner): return self.jobs.get(job_id)
    def active(self, owner): return [v for v in self.jobs.values() if v.get('active')]
    def storage(self, config, root):
        return dict(project_id=22169, pool='ddn_ssd', quota_remaining_bytes=self.free,
                    filesystem_available_bytes=self.free, required_bytes=100 * 1024**3)
    def cancel(self, argv): self.calls.append(('cancel', argv))
    def nice(self, argv): self.calls.append(('nice', argv))


class ControllerTests(unittest.TestCase):
    setUp=CoreTests.setUp
    def config(self):
        def file(name, data=b'fixture'):
            digest = put(self.root / name, data)
            return {'relative_path': name, 'sha256': digest}
        source = '1' * 40
        put(self.root / 'source/SOURCE_COMMIT', (source + '\n').encode())
        put(self.root / 'source/SNAPSHOT_GIT', ('2'*40 + '\n').encode())
        manifest = (hashlib.sha256((source+'\n').encode()).hexdigest() + '  SOURCE_COMMIT\n').encode()
        srcpin = file('source/validation_sources.sha256', manifest)
        phases = {}
        for name in ['proof', 'validation', 'production']:
            script = file(name + '/run.sbatch', b'#!/bin/bash\n')
            manifest = (script['sha256'] + '  run.sbatch\n').encode()
            pin = file(name + '/manifest.sha256', manifest)
            phases[name] = dict(helper_dir=name, helper_manifest_sha256=pin['sha256'],
                script=script['relative_path'], root='evidence/' + name,
                marker_log='logs/' + name + '-{job_id}.out', tests={'ngc_tests.log':dict(ran=3,skipped=0),
                       'harness_tests.log':dict(ran=2,skipped=0)})
        orig_meta=file('runs/c2/latest_state/meta.json', dict(epoch=0,step=4800,global_step=200,pending_micro=0))
        orig_ident=file('runs/c2/latest_state/identity.json', {})
        failure=file('runs/c2/TRAINING_FAILED', b'old failure')
        diagnostic=file('runs/diagnostic/latest_state/meta.json', dict(global_step=256))
        bindings=file('inputs/bindings.json', {})
        config=file('source/train.json', {})
        config_value=dict(version=2, operation_id='test-operation', owner=os.environ.get('USER','test'), uid=os.getuid(),
            work=str(self.root), sif='/image.sif', original_run='runs/c2',
            diagnostic='runs/diagnostic', source_commit=source,
            source_snapshot_git='2'*40, source_dir='source', source_manifest_sha256=srcpin['sha256'],
            phases=phases, bindings=bindings['relative_path'], bindings_sha256=bindings['sha256'],
            config_relative='train.json',config_sha256=config['sha256'], expected_runtime_fingerprint={'test':True},
            original_pins=[orig_meta,orig_ident,failure], diagnostic_pins=[diagnostic],
            original_checkpoint_receipt=file('proof/original_receipt.json', {'sha256':{}}),
            bootstrap_sha256=hashlib.sha256(json.dumps({'slots':list(range(1,16))+[3915]},sort_keys=True).encode()).hexdigest(), proof_job_id='3369338', proof_identity=dict(comment='pinned-proof',
                command='/proof.sbatch',submit_line='sbatch /proof.sbatch'), proof_receipts=[file('proof/staging.json',{}),
                file('proof/submission.json',{'job_id':'3369338'})], obsolete_job=dict(job_id='3345920',command='/old.sbatch'),
            protected_jobs=[dict(job_id='3345917',command='/b6.sbatch', held=True)],
            resources=dict(account='acct',partition='ghx4',time='02:00:00',cpus_per_task=64,exclude='gh062,gh093',
                           output='logs/%x-%j.out'), validation_clone_peak_bytes=20*1024**3,
            storage_python=sys.executable, storage_script='source/storage_preflight.py')
        source_dir=str(self.root/'source')
        for args in [['init','-q'],['add','.'],['-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture']]:
            subprocess.check_call(['git','-C',source_dir]+args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        config_value['source_snapshot_git']=subprocess.check_output(['git','-C',source_dir,'rev-parse','HEAD'],universal_newlines=True).strip()
        return config_value

    def start(self):
        self.c = self.config(); self.scheduler = FakeSlurm()
        self.assertTrue(hasattr(ctl, 'Controller'), 'durable controller is not implemented')
        self.state = self.root / 'operation'
        ctl.Controller.initialize(self.c, self.state)
        return ctl.Controller(self.state, self.scheduler)

    def test_init_is_exclusive_and_canonical_claim_blocks_second_operation(self):
        self.start()
        with self.assertRaises(ctl.Blocked): ctl.Controller.initialize(self.c, self.state)
        other=copy.deepcopy(self.c); other['operation_id']='other'
        with self.assertRaises(ctl.Blocked): ctl.Controller.initialize(other, self.root/'other')
        self.assertEqual(ctl.read_json(self.state/'journal.json')['phase'], 'waiting_proof')

    def test_concurrent_initialization_has_exactly_one_owner(self):
        config=self.config();queue=multiprocessing.Queue();state=self.root/'op'
        children=[multiprocessing.Process(target=init_child,args=(config,state,queue)) for _ in range(2)]
        for child in children:child.start()
        for child in children:child.join(15);self.assertFalse(child.is_alive())
        self.assertEqual(sorted(queue.get(timeout=2) for _ in children),['blocked','created'])

    def test_cli_status_is_read_only_and_host_import_is_stdlib(self):
        self.start();before=(self.state/'journal.json').read_bytes()
        result=subprocess.run([sys.executable,str(HERE/'continue_c2_history.py'),'status','--state-dir',str(self.state)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('waiting_proof',result.stdout)
        self.assertEqual((self.state/'journal.json').read_bytes(),before)

    def test_waiting_tick_does_not_touch_journal(self):
        controller=self.start(); before=(self.state/'journal.json').read_bytes()
        self.assertEqual(controller.tick()['phase'], 'waiting_proof')
        self.assertEqual((self.state/'journal.json').read_bytes(), before)

    def test_missing_batch_accounting_waits_without_journal_mutation(self):
        controller=self.start();before=(self.state/'journal.json').read_bytes()
        self.scheduler.jobs[self.c['proof_job_id']]=dict(active=False,rows={self.c['proof_job_id']:['COMPLETED','0:0']})
        controller.tick()
        self.assertEqual((self.state/'journal.json').read_bytes(),before)

    def test_legacy_config_and_promotion_states_cannot_migrate_implicitly(self):
        config=self.config();config.update(version=1,canonical=config.pop('original_run'),backup='runs/backup')
        with self.assertRaises(ctl.Blocked):ctl.Controller.initialize(config,self.root/'legacy')
        config.pop('canonical');config.pop('backup');config.update(version=2,original_run='runs/c2')
        ctl.Controller.initialize(config,self.root/'new')
        path=self.root/'new/journal.json';journal=ctl.read_json(path);journal['phase']='original_archived';put(path,journal)
        with self.assertRaises(ctl.Blocked):ctl.Controller(self.root/'new',FakeSlurm())

    def test_operation_controls_cannot_be_created_inside_protected_runs(self):
        config=self.config();state=self.root/'runs/c2/operation'
        with self.assertRaises(ctl.Blocked):ctl.Controller.initialize(config,state)
        self.assertFalse(state.exists())

    def test_pins_drift_blocks_without_losing_pending_intent(self):
        controller=self.start(); controller.prepare_submission('validation')
        intent=controller.status()['intent']
        (self.root/'validation/run.sbatch').write_text('changed')
        with self.assertRaises(ctl.Blocked): controller.tick()
        self.assertEqual(controller.status()['intent'], intent)
        self.assertEqual(self.scheduler.calls, [])

    def test_unmanifested_submitted_script_blocks_before_intent(self):
        config=self.config();script=self.root/'validation/unpinned.sbatch'
        script.write_text('#!/bin/bash\nexit 0\n')
        config['phases']['validation']['script']='validation/unpinned.sbatch'
        state=self.root/'op';ctl.Controller.initialize(config,state)
        scheduler=FakeSlurm();controller=ctl.Controller(state,scheduler)
        script.write_text('#!/bin/bash\necho altered\n')
        with self.assertRaises(ctl.Blocked):controller.prepare_submission('validation')
        self.assertIsNone(controller.status()['intent'])
        self.assertEqual(scheduler.calls,[])

    def test_same_named_script_outside_declared_helper_blocks_before_intent(self):
        config=self.config();put(self.root/'external/run.sbatch',b'#!/bin/bash\n')
        config['phases']['validation']['script']='external/run.sbatch'
        state=self.root/'op';ctl.Controller.initialize(config,state)
        scheduler=FakeSlurm();controller=ctl.Controller(state,scheduler)
        with self.assertRaises(ctl.Blocked):controller.prepare_submission('validation')
        self.assertIsNone(controller.status()['intent'])
        self.assertEqual(scheduler.calls,[])

    def test_changed_manifested_script_blocks_before_intent(self):
        controller=self.start();(self.root/'validation/run.sbatch').write_text('#!/bin/bash\necho altered\n')
        with self.assertRaises(ctl.Blocked):controller.prepare_submission('validation')
        self.assertIsNone(controller.status()['intent'])
        self.assertEqual(self.scheduler.calls,[])

    def test_snapshot_commit_drift_blocks_even_when_pinned_files_match(self):
        controller=self.start()
        subprocess.check_call(['git','-C',str(self.root/'source'),'-c','user.name=Fixture','-c',
            'user.email=fixture@example.invalid','commit','--allow-empty','-qm','different snapshot'])
        with self.assertRaises(ctl.Blocked):controller.tick()

    def test_requeued_old_job_blocks_activation(self):
        controller,_=self.accepted_validation();controller.save('old_job_retired')
        self.scheduler.jobs['3345920']=dict(job_id='3345920',active=True,owner=self.c['owner'],uid=self.c['uid'],
                                          state='RUNNING',reason='None',command='/old.sbatch')
        with self.assertRaises(ctl.Blocked):controller.activation_prepare()
        self.assertTrue((self.root/'runs/c2/latest_state').is_dir())

    def test_uncertain_submit_never_retries_even_after_empty_discovery(self):
        controller=self.start(); controller.prepare_submission('validation')
        self.scheduler.uncertain=True
        controller.execute_submission()
        self.assertEqual(len(self.scheduler.calls),1)
        for _ in range(3): controller.reconcile()
        self.assertEqual(len(self.scheduler.calls),1)
        self.assertEqual(controller.status()['phase'], 'submission_uncertain')

    def test_intent_crash_before_execution_is_also_uncertain(self):
        controller=self.start(); controller.prepare_submission('validation')
        ctl.Controller(self.state,self.scheduler).tick()
        self.assertEqual(self.scheduler.calls,[])
        self.assertEqual(controller.status()['phase'],'submission_uncertain')

    def test_reconcile_adopts_only_single_exact_submission(self):
        controller=self.start(); controller.prepare_submission('validation')
        intent=controller.status()['intent']
        match=dict(job_id='9001',owner=self.c['owner'],uid=self.c['uid'],token=intent['token'],
                   argv=intent['argv'],created=intent['created'])
        self.scheduler.matches=[match,dict(match,job_id='9002')]
        with self.assertRaises(ctl.Blocked): controller.reconcile()
        self.scheduler.matches=[dict(match,owner='foreign')]
        with self.assertRaises(ctl.Blocked): controller.reconcile()
        self.scheduler.matches=[match]; controller.reconcile()
        self.assertEqual(controller.status()['validation_job_id'],'9001')
        self.assertEqual(self.scheduler.calls,[])

    def test_storage_reserve_and_clone_peak_gate_submission(self):
        controller=self.start(); self.scheduler.free=99*1024**3
        with self.assertRaises(ctl.Blocked): controller.prepare_submission('validation')
        self.assertIsNone(controller.status()['intent'])
        self.scheduler.free=100*1024**3; controller.prepare_submission('validation')
        self.assertEqual(controller.status()['phase'],'validation_submit_intent')

    def test_clone_peak_must_fit_available_without_an_additional_reserve(self):
        config=self.config();config['validation_clone_peak_bytes']=121*1024**3
        state=self.root/'op';ctl.Controller.initialize(config,state);scheduler=FakeSlurm();scheduler.free=120*1024**3
        controller=ctl.Controller(state,scheduler)
        with self.assertRaises(ctl.Blocked):controller.prepare_submission('validation')
        scheduler.free=121*1024**3;controller.prepare_submission('validation')
        self.assertEqual(controller.status()['phase'],'validation_submit_intent')

    def test_submission_argv_and_environment_are_persisted_before_external_call(self):
        controller=self.start(); controller.prepare_submission('validation')
        intent=controller.status()['intent']; controller.execute_submission()
        call=self.scheduler.calls[0]
        self.assertEqual(call[1],intent['argv']); self.assertEqual(call[2],intent['env'])
        self.assertIn('--no-requeue',call[1]); self.assertNotIn('PYTHONOPTIMIZE',call[2])
        self.assertEqual(controller.status()['validation_job_id'],'9001')

    def test_explicit_phase_resource_override_is_used_in_recorded_argv(self):
        config=self.config();config['phases']['validation']['resources']=dict(config['resources'],partition='ghx4',time='08:00:00')
        state=self.root/'op';ctl.Controller.initialize(config,state);controller=ctl.Controller(state,FakeSlurm())
        controller.prepare_submission('validation')
        self.assertIn('--partition=ghx4',controller.status()['intent']['argv'])
        self.assertIn('--time=08:00:00',controller.status()['intent']['argv'])

    def science_fixture(self, controller, kind, job='9001'):
        root=self.root/self.c['phases'][kind]['root']/job
        report=proof(); stop,snap=(410,400) if kind=='validation' else (220,210)
        bootstrap={'slots':list(range(1,16))+[3915]}
        bootstrap_sha=put(root/'bootstrap.json',bootstrap)
        # An immutable configuration is created with the known bootstrap pin before init.
        self.assertEqual(bootstrap_sha,self.c['bootstrap_sha256'])
        report['checkpoint_history_sha256']={}
        for name,step in [('continuous',stop),('resumed',stop),('at%d'%snap,snap)]:
            position=dict(epoch=0,step=step*24,global_step=step,pending_micro=0)
            entries=[dict(ordinal=i,origin='reconstructed_legacy',position=dict(phase='train',epoch=0,
                batch_index=slot-1,view='arm',training=True,grad_enabled=True,optimizer_step=(slot-1)//24),
                ranks=[]) for i,slot in enumerate(bootstrap['slots'])]
            if kind=='validation':
                entries.append(dict(ordinal=16,origin='observed',position=dict(phase='train',epoch=0,
                    batch_index=6717,view='arm',training=True,grad_enabled=True,optimizer_step=279),
                    ranks=[dict(rank=r,delta=1,inputs={'input_ids':{'shape':[1,129024]},
                           'labels':{'shape':[1,3924]}}) for r in range(4)]))
            history=dict(version=1,identity={},fingerprint=self.c['expected_runtime_fingerprint'],world_size=4,
                         checkpoint=position,bootstrap=dict(manifest=bootstrap,sha256=bootstrap_sha),entries=entries)
            hd=put(root/name/'latest_state/compile_history.json',history)
            put(root/name/'latest_state/meta.json',dict(position,compile_history_sha256=hd))
            put(root/name/'latest_state/identity.json',{})
            for fn in ['model.safetensors','optimizer.bin','scheduler.bin']+['random_states_%d.pkl'%r for r in range(4)]:
                put(root/name/'latest_state/accelerate_state'/fn,b'tensor fixture')
            if name in ('continuous','resumed'): report['checkpoint_history_sha256'][name]=hd
        if kind=='validation':
            report.update(profile='validate400',checkpoint_digests={'continuous':'a'*64,'resumed':'a'*64},
                validation400=dict(checkpoint_step=400,stop_step=410,rank3_specialization_slot=6718,
                                   rank3_specialization_recorded=True))
        put(root/'comparison.json',report);put(root/'verified_record.json',{'job_id':job})
        for file,count in [('ngc_tests.log',3),('harness_tests.log',2)]:
            put(root/file,('Ran %d tests in 0.1s\n\nOK\n'%count).encode())
        marker='HISTORY_VALIDATION_PASS' if kind=='validation' else 'HISTORY_CHECKPOINT_PROOF_PASS'
        put(self.root/('logs/%s-%s.out'%(kind,job)),(marker+' now\n').encode())
        if kind=='validation':
            candidate=dict(relative_run='resumed',relative_checkpoint='latest_state',epoch=0,step=9840,
                global_step=410,pending_micro=0,meta_sha256=hashlib.sha256((root/'resumed/latest_state/meta.json').read_bytes()).hexdigest(),
                history_sha256=report['checkpoint_history_sha256']['resumed'],canonical_checkpoint_sha256='a'*64)
            receipt=dict(version=1,kind='c2_history_validation_ready',source_commit=self.c['source_commit'],
                validation_job_id=job,predecessor_job_id=self.c['proof_job_id'],
                predecessor_comparison_sha256=controller.j['receipts']['proof']['sha256'],
                comparison_sha256=hashlib.sha256((root/'comparison.json').read_bytes()).hexdigest(),
                original_checkpoint_receipt_sha256=self.c['original_checkpoint_receipt']['sha256'],candidate=candidate)
            put(root/'validation_receipt.json',receipt)
        observation=dict(job_id=job,owner=self.c['owner'],uid=self.c['uid'],active=False,
            rows={job:['COMPLETED','0:0'],job+'.batch':['COMPLETED','0:0']},token='pinned-proof',
            command='/proof.sbatch',submit_line='sbatch /proof.sbatch')
        return root,observation

    def accepted_validation(self):
        controller=self.start()
        self.assertTrue(hasattr(controller,'accept_science'),'science acceptance not implemented')
        _,obs=self.science_fixture(controller,'proof',self.c['proof_job_id'])
        self.scheduler.jobs[self.c['proof_job_id']]=obs
        controller.accept_science('proof',obs)
        controller.prepare_submission('validation');controller.execute_submission()
        root,obs=self.science_fixture(controller,'validation')
        submission=controller.j['validation_submission']['intent']
        obs.update(token=submission['token'],argv=submission['argv'])
        self.scheduler.jobs['9001']=obs
        controller.accept_science('validation',obs)
        return controller,root

    def test_strict_science_acceptance_binds_job_marker_history_and_candidate(self):
        controller,root=self.accepted_validation()
        self.assertEqual(controller.status()['phase'],'validation_accepted')
        (root/'resumed/latest_state/meta.json').write_text('{}')
        with self.assertRaises(ctl.Blocked): controller.candidate()

    def test_old_job_foreign_released_or_running_is_never_cancelled(self):
        controller,_=self.accepted_validation()
        good=dict(job_id='3345920',owner=self.c['owner'],uid=self.c['uid'],active=True,state='PENDING',
                  reason='JobHeldUser',command='/old.sbatch')
        for bad in [dict(good,owner='foreign'),dict(good,state='RUNNING'),dict(good,reason='Priority')]:
            self.scheduler.jobs['3345920']=bad
            with self.assertRaises(ctl.Blocked): controller.retire_prepare()
        self.assertFalse(any(c[0]=='cancel' for c in self.scheduler.calls))

    def test_cancel_response_loss_reconciles_terminal_without_second_cancel(self):
        controller,_=self.accepted_validation()
        held=dict(job_id='3345920',owner=self.c['owner'],uid=self.c['uid'],active=True,state='PENDING',
                  reason='JobHeldUser',command='/old.sbatch')
        self.scheduler.jobs['3345920']=held
        controller.retire_prepare();controller.retire()
        self.scheduler.jobs['3345920']=dict(held,active=False,rows={'3345920':['CANCELLED','0:0']})
        restarted=ctl.Controller(self.state,self.scheduler);restarted.retire()
        self.assertEqual(restarted.status()['phase'],'old_job_retired')
        self.assertEqual(sum(c[0]=='cancel' for c in self.scheduler.calls),1)

    def retired(self,controller):
        observation=dict(job_id='3345920',owner=self.c['owner'],uid=self.c['uid'],active=False,
                         command='/old.sbatch',rows={'3345920':['CANCELLED','0:0']})
        self.scheduler.jobs['3345920']=observation
        controller.save('old_job_retired',retirement=observation)

    def test_activation_preserves_all_run_inodes_and_bytes_without_native_rename(self):
        self.assertTrue(hasattr(ctl.Controller,'activation_prepare'),'activation workflow missing')
        controller,root=self.accepted_validation();self.retired(controller)
        roots=[self.root/'runs/c2',self.root/'runs/diagnostic',self.root/'evidence/proof',root]
        before={str(p):(p.stat().st_ino,p.read_bytes() if p.is_file() else None)
                for base in roots for p in [base]+list(base.rglob('*'))}
        with patch.object(ctypes,'CDLL',side_effect=AssertionError('native rename forbidden')), \
             patch.object(os,'rename',side_effect=AssertionError('directory rename forbidden')):
            controller.activation_prepare();controller.activate()
        after={str(p):(p.stat().st_ino,p.read_bytes() if p.is_file() else None)
               for base in roots for p in [base]+list(base.rglob('*'))}
        self.assertEqual(after,before)
        self.assertEqual(controller.active_run(),str((root/'resumed').relative_to(self.root)))
        self.assertEqual(controller.status()['phase'],'candidate_activated')
        self.assertTrue((self.root/'runs/c2/TRAINING_FAILED').exists())
        self.assertEqual(ctl.canonical_controls(self.c)[0],self.root/'runs/.c2.history.lock')

    def test_activation_receipt_publication_crash_rolls_forward_same_bytes(self):
        controller,root=self.accepted_validation();self.retired(controller);controller.activation_prepare()
        intended=copy.deepcopy(controller.j['activation_intent']['payload'])
        original_link=os.link
        def link_then_die(source,destination,*args,**kwargs):
            original_link(source,destination,*args,**kwargs);raise RuntimeError('crash after exclusive publication')
        with patch.object(os,'link',new=link_then_die):
            with self.assertRaisesRegex(RuntimeError,'crash after'):controller.activate()
        receipt=self.state/'activation_receipt.json';before=receipt.read_bytes();inode=receipt.stat().st_ino
        self.assertEqual(controller.status()['phase'],'activation_intent')
        restarted=ctl.Controller(self.state,self.scheduler);restarted.activate()
        self.assertEqual(receipt.read_bytes(),before);self.assertEqual(receipt.stat().st_ino,inode)
        self.assertEqual(ctl.read_json(receipt),intended)
        self.assertEqual(restarted.status()['phase'],'candidate_activated')
        self.assertEqual(sum(c[0]=='submit' for c in self.scheduler.calls),1)

    def test_activation_crash_before_journal_commit_uses_original_payload(self):
        controller,_=self.accepted_validation();self.retired(controller);controller.activation_prepare()
        intended=copy.deepcopy(controller.j['activation_intent']['payload'])
        real_atomic=ctl.atomic_json
        def crash_journal(path,value):
            if Path(path)==self.state/'journal.json':raise RuntimeError('journal crash')
            return real_atomic(path,value)
        with patch.object(ctl,'atomic_json',new=crash_journal):
            with self.assertRaisesRegex(RuntimeError,'journal crash'):controller.activate()
        restarted=ctl.Controller(self.state,self.scheduler);restarted.activate()
        self.assertEqual(ctl.read_json(self.state/'activation_receipt.json'),intended)
        self.assertEqual(restarted.status()['phase'],'candidate_activated')

    def test_activation_intent_and_stray_temp_do_not_authorize_early_submission(self):
        controller,_=self.accepted_validation();self.retired(controller);controller.activation_prepare()
        put(self.state/'.activation_receipt.abandoned',b'partial control bytes')
        restarted=ctl.Controller(self.state,self.scheduler);restarted.tick()
        self.assertEqual(restarted.status()['phase'],'candidate_activated')
        self.assertEqual(sum(c[0]=='submit' for c in self.scheduler.calls),1)
        self.assertEqual((self.state/'.activation_receipt.abandoned').read_bytes(),b'partial control bytes')

    def test_activation_conflicting_receipt_and_changed_original_block(self):
        controller,_=self.accepted_validation();self.retired(controller);controller.activation_prepare()
        receipt=self.state/'activation_receipt.json';receipt.write_bytes(b'foreign evidence');inode=receipt.stat().st_ino
        with self.assertRaises(ctl.Blocked):controller.activate()
        self.assertEqual(receipt.read_bytes(),b'foreign evidence');self.assertEqual(receipt.stat().st_ino,inode)
        receipt.unlink();(self.root/'runs/c2/latest_state/meta.json').write_text('{}')
        with self.assertRaises(ctl.Blocked):controller.activate()
        self.assertFalse(receipt.exists())

    def test_activation_low_storage_live_candidate_drift_and_foreign_writer_block(self):
        controller,root=self.accepted_validation();self.retired(controller)
        self.scheduler.free=99*1024**3
        with self.assertRaises(ctl.Blocked):controller.activation_prepare()
        self.scheduler.free=100*1024**3
        self.scheduler.jobs['999']=dict(job_id='999',active=True,command='train --run '+str(root/'resumed'),token='other')
        with self.assertRaises(ctl.Blocked):controller.activation_prepare()
        del self.scheduler.jobs['999']
        controller.activation_prepare();(root/'resumed/latest_state/meta.json').write_text('{}')
        with self.assertRaises(ctl.Blocked):controller.activate()
        self.assertFalse((self.state/'activation_receipt.json').exists())

    def test_activation_forged_path_and_embedded_bytes_block(self):
        controller=self.activated();path=self.state/'activation_receipt.json'
        original=ctl.read_json(path);raw=path.read_bytes();original_pin=copy.deepcopy(controller.j['activation_receipt'])
        embedded=copy.deepcopy(original);embedded['initial_artifacts']['meta.json']['utf8']='{}'
        for bad in [dict(original,active_run='runs/c2'),dict(original,owner='foreign'),embedded]:
            digest=put(path,bad)
            controller.j['activation_receipt']['sha256']=digest
            with self.assertRaises(ctl.Blocked):controller.active_run()
        path.write_bytes(raw);controller.j['activation_receipt']=original_pin
        self.assertEqual(controller.active_run(),'evidence/validation/9001/resumed')

    def test_activation_audit_device_numbers_are_not_cross_host_identity_gates(self):
        controller,_=self.accepted_validation();self.retired(controller);controller.activation_prepare()
        # A recovery on another client retains the original audit observation.
        payload=controller.j['activation_intent']['payload']
        payload['directory_observation']=dict(host='previous-login-client',device=999,inode=123)
        controller.save('activation_intent');controller.activate()
        self.assertEqual(controller.active_run(),'evidence/validation/9001/resumed')

    def test_failed_proof_never_submits_validation(self):
        controller=self.start()
        self.scheduler.jobs[self.c['proof_job_id']]=dict(active=False,rows={
            self.c['proof_job_id']:['FAILED','1:0'],self.c['proof_job_id']+'.batch':['FAILED','1:0']})
        with self.assertRaises(ctl.Blocked):controller.tick()
        self.assertEqual(self.scheduler.calls,[])
        self.assertIn('parent/batch',controller.status()['last_error'])

    def activated(self):
        self.assertTrue(hasattr(ctl.Controller,'activation_prepare'),'activation workflow missing')
        controller,_=self.accepted_validation();self.retired(controller)
        controller.activation_prepare();controller.activate()
        return controller

    def production_fixture(self,controller,step=420,done=False):
        self.assertTrue(hasattr(controller,'build_request'),'production request builder not implemented')
        request_pin=controller.build_request();request=ctl.read_json(self.root/request_pin['relative_path'])
        controller.prepare_submission('production',request_pin);controller.execute_submission()
        run=self.root/controller.active_run();relative='step_04099' if done else 'latest_state'
        position=dict(epoch=0,step=step*24,global_step=step,pending_micro=0)
        history=ctl.read_json(run/'latest_state/compile_history.json');history['checkpoint']=position
        hd=put(run/relative/'compile_history.json',history)
        meta=dict(position,compile_history_sha256=hd)
        if done:
            md=put(run/relative/'model.safetensors',b'final-model')
            meta.update(arm='C2',seed=123,checkpoint_kind='fixed_step',selection='fixed optimizer-update count; never a best-validation checkpoint',
                        pilot_final=True,stop_after_updates=4099,model_sha256=md)
            put(run/relative/'PILOT_FINAL',b'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n')
            put(run/relative/'PRUNE_EXEMPT',b'A28/plan 9: step checkpoint, never pruned\n')
            put(run/'TRAINING_DONE',b'done')
        meta_sha=put(run/relative/'meta.json',meta);ident_sha=put(run/relative/'identity.json',{})
        inventory={str(p.relative_to(run/relative)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in (run/relative).rglob('*') if p.is_file()}
        cp=dict(position=position,relative_dir=relative,meta_sha256=meta_sha,identity_sha256=ident_sha,
            history_sha256=hd,canonical_checkpoint_sha256='c'*64,files=inventory)
        result=dict(global_step=step,history=[])
        final=None
        if done:
            result.update(done=True,reason='stop_after_updates',pilot_final='/work/'+controller.active_run()+'/'+relative)
            final=dict(relative_dir=relative,model_sha256=md,done_sha256=hashlib.sha256(b'done').hexdigest(),
                       expected_completed_epochs=0,epoch_events=[])
            cp['done_sha256']=final['done_sha256']
        else:result['preempted']=True
        receipt=dict(version=1,kind='c2_production_link_ready',operation_id=self.c['operation_id'],token=request['token'],
            link_index=request['link_index'],job_id='9001',request_sha256=request_pin['sha256'],source_commit=self.c['source_commit'],
            predecessor_sha256=request['predecessor']['sha256'],start_global_step=request['start_checkpoint']['global_step'],result=result,profile=request['profile'],
            runtime_fingerprint_sha256=hashlib.sha256(json.dumps(self.c['expected_runtime_fingerprint'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            input_verification=dict(job_id='9001',ranks=[0,1,2,3],bindings_sha256=self.c['bindings_sha256'],record_sha256='d'*64,identity_sha256=hashlib.sha256(b'{}').hexdigest()),
            checkpoint=cp,epoch_events=[],final=final,counts=dict(train_rows=1000000,rank_batches=250000,per_rank_accumulation=24),
            draw_logs={str(r):dict(rows=step*24,sha256='f'*64) for r in range(4)},
            source_snapshot_git=self.c['source_snapshot_git'],source_manifest_sha256=self.c['source_manifest_sha256'],
            helper_manifest_sha256=self.c['phases']['production']['helper_manifest_sha256'],config_sha256=self.c['config_sha256'])
        receipt_path=self.root/request['receipt_dir']/'execution_9001/link_receipt.json'
        put(receipt_path,receipt)
        intent=controller.j['links'][-1]['intent']
        self.scheduler.jobs['9001']=dict(job_id='9001',active=False,owner=self.c['owner'],uid=self.c['uid'],
            token=intent['token'],argv=intent['argv'],rows={'9001':['COMPLETED','0:0'],'9001.batch':['COMPLETED','0:0']})
        return receipt_path,receipt,request

    def test_production_request_starts_from_receipt_bound_in_place410(self):
        controller=self.activated();_,_,request=self.production_fixture(controller)
        self.assertEqual(request['run_dir'],'evidence/validation/9001/resumed')
        self.assertEqual(request['predecessor']['relative_path'],'evidence/validation/9001/validation_receipt.json')
        self.assertEqual(request['start_checkpoint']['global_step'],410)
        self.assertEqual(request['start_checkpoint']['canonical_checkpoint_sha256'],'a'*64)
        self.assertEqual(request['completed_epoch_events'],[])
        controller.accept_link()
        self.assertEqual(controller.status()['phase'],'link_accepted')

    def test_evolving_active_run_keeps_initial_evidence_and_final_uses_actual_path(self):
        controller=self.activated();root=self.root/'evidence/validation/9001'
        preserved={str(p):p.read_bytes() for base in (self.root/'runs/c2',self.root/'runs/diagnostic',
                   self.root/'evidence/proof',root/'continuous',root/'at400') for p in base.rglob('*') if p.is_file()}
        for name in ('comparison.json','validation_receipt.json'):preserved[str(root/name)]=(root/name).read_bytes()
        initial=ctl.read_json(self.state/'activation_receipt.json')['initial_artifacts']['meta.json']['utf8']
        self.production_fixture(controller,420);controller.accept_link();controller.recheck_receipts()
        request_pin=controller.build_request();request=ctl.read_json(self.root/request_pin['relative_path'])
        self.assertEqual(request['start_checkpoint']['global_step'],420)
        self.assertEqual(request['run_dir'],'evidence/validation/9001/resumed')
        self.production_fixture(controller,430);controller.accept_link()
        path,receipt,_=self.production_fixture(controller,4099,True)
        bad=copy.deepcopy(receipt);bad['result']['pilot_final']='/work/runs/c2/step_04099';put(path,bad)
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(path,receipt);controller.accept_link()
        self.assertEqual(controller.status()['phase'],'final_verified')
        self.assertEqual({name:Path(name).read_bytes() for name in preserved},preserved)
        self.assertEqual(ctl.read_json(self.state/'activation_receipt.json')['initial_artifacts']['meta.json']['utf8'],initial)

    def test_pending_submission_reconciles_without_rechecking_historical_live410(self):
        controller=self.activated();pin=controller.build_request();controller.prepare_submission('production',pin)
        (self.root/controller.active_run()/'latest_state/meta.json').write_text('{}')
        restarted=ctl.Controller(self.state,self.scheduler);restarted.tick()
        self.assertEqual(restarted.status()['phase'],'submission_uncertain')
        self.assertEqual(sum(c[0]=='submit' for c in self.scheduler.calls),1)
        with self.assertRaises(ctl.Blocked):restarted.build_request()

    def test_retained_validation_sibling_drift_blocks_after_active_checkpoint_evolves(self):
        controller=self.activated();self.production_fixture(controller,420);controller.accept_link()
        sibling=self.root/'evidence/validation/9001/continuous/latest_state/meta.json';sibling.write_text('{}')
        with self.assertRaises(ctl.Blocked):controller.build_request()

    def test_concurrent_ticks_submit_only_one_production_link(self):
        self.activated();queue=multiprocessing.Queue();ready=multiprocessing.Event()
        children=[multiprocessing.Process(target=tick_child,args=(self.state,queue,ready)) for _ in range(2)]
        for child in children:child.start()
        ready.set()
        for child in children:child.join(20);self.assertFalse(child.is_alive())
        self.assertEqual(sum(queue.get(timeout=2) for _ in children),1)
        self.assertEqual(len(ctl.Controller(self.state,FakeSlurm()).status()['links']),1)

    def test_epoch_step_zero_requires_all_pending_gradients_and_preserves_epoch_events(self):
        controller=self.activated();path,receipt,_=self.production_fixture(controller)
        run=self.root/controller.active_run();directory=run/'epoch_01';os.rename(str(run/'latest_state'),str(directory))
        position=dict(epoch=1,step=0,global_step=420,pending_micro=3)
        history=ctl.read_json(directory/'compile_history.json');history['checkpoint']=position
        hd=put(directory/'compile_history.json',history)
        md=put(directory/'meta.json',dict(position,compile_history_sha256=hd))
        put(directory/'model.safetensors',b'epoch model')
        for rank in range(4):put(directory/('pending_grads_rank%d.pt'%rank),b'pending gradients')
        ed=put(directory/'eval.json',dict(epoch=1,global_step=420,train_loss=1.0,eval_loss=2.0,eval_evidence_loss=3.0))
        event=dict(epoch=1,global_step=420,data_order_sha256='a'*64,train=1.0,de_novo=2.0,evidence=3.0)
        receipt['result']['history']=[dict(event)]
        event.update(epoch_end=dict(epoch_end=1,arm='C2',data_order_first1000_sha256='a'*64,train=1.0,de_novo=2.0,evidence=3.0),
                     meta_sha256=md,eval_sha256=ed,identity_sha256=receipt['checkpoint']['identity_sha256'],history_sha256=hd)
        receipt['epoch_events']=[event]
        receipt['counts']=dict(train_rows=40332,rank_batches=10083,per_rank_accumulation=24)
        for draw in receipt['draw_logs'].values():draw['rows']=10083
        inventory={str(p.relative_to(directory)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                   for p in directory.rglob('*') if p.is_file()}
        receipt['checkpoint'].update(position=position,relative_dir='epoch_01',meta_sha256=md,history_sha256=hd,files=inventory)
        put(path,receipt)
        gradient=directory/'pending_grads_rank3.pt';gradient.unlink()
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(gradient,b'pending gradients');controller.accept_link()
        request_pin=controller.build_request();request=ctl.read_json(self.root/request_pin['relative_path'])
        self.assertEqual(request['start_checkpoint']['pending_micro'],3)
        self.assertEqual(request['start_checkpoint']['step'],0)
        self.assertEqual(request['completed_epoch_events'],[event])

    def test_production_no_progress_missing_rank_wrong_hash_and_timeout_block(self):
        controller=self.activated();path,receipt,_=self.production_fixture(controller)
        for bad in [dict(receipt,result=dict(preempted=True,global_step=410,history=[])),
                    dict(receipt,input_verification=dict(receipt['input_verification'],ranks=[0,1,2])),
                    dict(receipt,request_sha256='0'*64)]:
            put(path,bad)
            with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(path,receipt);self.scheduler.jobs['9001']['rows']['9001']=['TIMEOUT','0:0']
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        self.assertEqual(controller.status()['phase'],'waiting_production')

    def test_final_requires_exact4099_markers_done_and_epoch_evaluations(self):
        controller=self.activated();path,receipt,_=self.production_fixture(controller,4099,True)
        for step in [4098,4100]:
            bad=copy.deepcopy(receipt);bad['result']['global_step']=step;put(path,bad)
            with self.assertRaises(ctl.Blocked):controller.accept_link()
        bad=copy.deepcopy(receipt);bad['final']['expected_completed_epochs']=1;put(path,bad)
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(path,receipt);marker=self.root/controller.active_run()/'step_04099/PILOT_FINAL';marker.unlink()
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(marker,b'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n');controller.accept_link()
        self.assertEqual(controller.status()['phase'],'final_verified')

    def test_nice_restore_is_idempotent_and_b6_remains_held(self):
        controller=self.activated();self.production_fixture(controller,4099,True);controller.accept_link()
        self.scheduler.jobs['3345917']=dict(job_id='3345917',active=True,owner=self.c['owner'],uid=self.c['uid'],
            state='PENDING',reason='JobHeldUser',command='/b6.sbatch',nice='10000')
        controller.restore_nice()
        self.assertEqual(self.scheduler.calls[-1],('nice',['scontrol','update','JobId=3345917','Nice=0']))
        self.scheduler.jobs['3345917']['nice']='0'
        restarted=ctl.Controller(self.state,self.scheduler);restarted.restore_nice();restarted.restore_nice()
        self.assertEqual(restarted.status()['phase'],'complete')
        self.assertEqual(sum(c[0]=='nice' for c in self.scheduler.calls),1)
        self.assertEqual(self.scheduler.jobs['3345917']['reason'],'JobHeldUser')


if __name__ == '__main__':
    unittest.main()
