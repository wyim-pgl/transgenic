"""Host controller contracts; filesystem effects are real, only Slurm is fake."""
import copy
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
        (self.root / 'a').mkdir(); (self.root / 'b').mkdir()
        with self.assertRaises(ctl.Blocked): ctl.rename_new(self.root / 'a', self.root / 'b')
        self.assertTrue((self.root / 'a').is_dir())

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
        config_value=dict(version=1, operation_id='test-operation', owner=os.environ.get('USER','test'), uid=os.getuid(),
            work=str(self.root), sif='/image.sif', canonical='runs/c2',
            backup='runs/c2-original200', diagnostic='runs/diagnostic', source_commit=source,
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

    def test_cross_device_promotion_never_copies(self):
        if not Path('/dev/shm').is_dir():self.skipTest('requires Linux shared-memory filesystem')
        with tempfile.TemporaryDirectory(dir='/dev/shm') as destination:
            source=self.root/'candidate';source.mkdir();put(source/'evidence',b'preserve')
            if source.stat().st_dev==Path(destination).stat().st_dev:self.skipTest('same filesystem')
            with self.assertRaises(ctl.Blocked):ctl.rename_new(source,Path(destination)/'target')
            self.assertEqual((source/'evidence').read_bytes(),b'preserve')
            self.assertFalse((Path(destination)/'target').exists())

    def test_pins_drift_blocks_without_losing_pending_intent(self):
        controller=self.start(); controller.prepare_submission('validation')
        intent=controller.status()['intent']
        (self.root/'validation/run.sbatch').write_text('changed')
        with self.assertRaises(ctl.Blocked): controller.tick()
        self.assertEqual(controller.status()['intent'], intent)
        self.assertEqual(self.scheduler.calls, [])

    def test_snapshot_commit_drift_blocks_even_when_pinned_files_match(self):
        controller=self.start()
        subprocess.check_call(['git','-C',str(self.root/'source'),'-c','user.name=Fixture','-c',
            'user.email=fixture@example.invalid','commit','--allow-empty','-qm','different snapshot'])
        with self.assertRaises(ctl.Blocked):controller.tick()

    def test_requeued_old_job_blocks_promotion(self):
        controller,_=self.accepted_validation();controller.save('old_job_retired')
        self.scheduler.jobs['3345920']=dict(job_id='3345920',active=True,owner=self.c['owner'],uid=self.c['uid'],
                                          state='RUNNING',reason='None',command='/old.sbatch')
        with self.assertRaises(ctl.Blocked):controller.promotion_prepare()
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
        controller.accept_science('proof',obs)
        controller.prepare_submission('validation');controller.execute_submission()
        root,obs=self.science_fixture(controller,'validation')
        submission=controller.j['validation_submission']['intent']
        obs.update(token=submission['token'],argv=submission['argv'])
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

    def test_crash_after_candidate_rename_publishes_missing_provenance(self):
        controller,root=self.accepted_validation();controller.save('old_job_retired');controller.promotion_prepare()
        os.rename(str(self.root/'runs/c2'),str(self.root/'runs/c2-original200'))
        os.rename(str(root/'resumed'),str(self.root/'runs/c2'))
        restarted=ctl.Controller(self.state,self.scheduler);restarted.promote()
        self.assertEqual(restarted.status()['phase'],'candidate_promoted')
        self.assertTrue((self.state/'promotion_receipt.json').exists())

    def test_promotion_recovers_all_three_layouts_and_preserves_diagnostic(self):
        controller,root=self.accepted_validation()
        controller.save('old_job_retired')
        before=(self.root/'runs/diagnostic/latest_state/meta.json').read_bytes()
        controller.promotion_prepare()
        self.assertEqual(controller.status()['phase'],'promotion_intent')
        # Simulate death after first rename, before journal save.
        os.rename(str(self.root/'runs/c2'),str(self.root/'runs/c2-original200'))
        controller=ctl.Controller(self.state,self.scheduler); controller.promote()
        self.assertEqual(controller.status()['phase'],'candidate_promoted')
        self.assertFalse((root/'resumed').exists()); self.assertTrue((root/'continuous').is_dir())
        self.assertTrue((root/'at400').is_dir()); self.assertTrue((self.root/'runs/c2-original200/TRAINING_FAILED').exists())
        self.assertEqual((self.root/'runs/diagnostic/latest_state/meta.json').read_bytes(),before)
        provenance=ctl.read_json(self.root/'runs/c2/RECOVERY_PROMOTION.json')
        self.assertEqual(provenance['old_relative_run'],str((root/'resumed').relative_to(self.root)))
        # Simulate missing journal commit after second rename/provenance publication.
        controller.save('promotion_intent'); controller.promote()
        self.assertEqual(controller.status()['phase'],'candidate_promoted')

    def test_promotion_never_overwrites_empty_backup_or_changed_original(self):
        controller,_=self.accepted_validation();controller.save('old_job_retired')
        (self.root/'runs/c2-original200').mkdir()
        with self.assertRaises(ctl.Blocked): controller.promotion_prepare()
        self.assertTrue((self.root/'runs/c2/latest_state/meta.json').exists())

    def test_failed_proof_never_submits_validation(self):
        controller=self.start()
        self.scheduler.jobs[self.c['proof_job_id']]=dict(active=False,rows={
            self.c['proof_job_id']:['FAILED','1:0'],self.c['proof_job_id']+'.batch':['FAILED','1:0']})
        with self.assertRaises(ctl.Blocked):controller.tick()
        self.assertEqual(self.scheduler.calls,[])
        self.assertIn('parent/batch',controller.status()['last_error'])

    def promoted(self):
        controller,_=self.accepted_validation(); controller.save('old_job_retired')
        controller.promotion_prepare();controller.promote();controller.promote()
        return controller

    def production_fixture(self,controller,step=420,done=False):
        self.assertTrue(hasattr(controller,'build_request'),'production request builder not implemented')
        request_pin=controller.build_request();request=ctl.read_json(self.root/request_pin['relative_path'])
        controller.prepare_submission('production',request_pin);controller.execute_submission()
        run=self.root/'runs/c2';relative='step_04099' if done else 'latest_state'
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
            result.update(done=True,reason='stop_after_updates',pilot_final='/work/runs/c2/'+relative)
            final=dict(relative_dir=relative,model_sha256=md,done_sha256=hashlib.sha256(b'done').hexdigest(),
                       expected_completed_epochs=0,epoch_events=[])
            cp['done_sha256']=final['done_sha256']
        else:result['preempted']=True
        receipt=dict(version=1,kind='c2_production_link_ready',operation_id=self.c['operation_id'],token=request['token'],
            link_index=request['link_index'],job_id='9001',request_sha256=request_pin['sha256'],source_commit=self.c['source_commit'],
            predecessor_sha256=request['predecessor']['sha256'],start_global_step=410,result=result,profile=request['profile'],
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

    def test_production_request_starts_from_receipt_bound_relocated410(self):
        controller=self.promoted();_,_,request=self.production_fixture(controller)
        self.assertEqual(request['run_dir'],'runs/c2')
        self.assertEqual(request['start_checkpoint']['global_step'],410)
        self.assertEqual(request['start_checkpoint']['canonical_checkpoint_sha256'],'a'*64)
        self.assertEqual(request['completed_epoch_events'],[])
        controller.accept_link()
        self.assertEqual(controller.status()['phase'],'link_accepted')

    def test_concurrent_ticks_submit_only_one_production_link(self):
        self.promoted();queue=multiprocessing.Queue();ready=multiprocessing.Event()
        children=[multiprocessing.Process(target=tick_child,args=(self.state,queue,ready)) for _ in range(2)]
        for child in children:child.start()
        ready.set()
        for child in children:child.join(20);self.assertFalse(child.is_alive())
        self.assertEqual(sum(queue.get(timeout=2) for _ in children),1)
        self.assertEqual(len(ctl.Controller(self.state,FakeSlurm()).status()['links']),1)

    def test_epoch_step_zero_requires_all_pending_gradients_and_preserves_epoch_events(self):
        controller=self.promoted();path,receipt,_=self.production_fixture(controller)
        run=self.root/'runs/c2';directory=run/'epoch_01';os.rename(str(run/'latest_state'),str(directory))
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
        controller=self.promoted();path,receipt,_=self.production_fixture(controller)
        for bad in [dict(receipt,result=dict(preempted=True,global_step=410,history=[])),
                    dict(receipt,input_verification=dict(receipt['input_verification'],ranks=[0,1,2])),
                    dict(receipt,request_sha256='0'*64)]:
            put(path,bad)
            with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(path,receipt);self.scheduler.jobs['9001']['rows']['9001']=['TIMEOUT','0:0']
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        self.assertEqual(controller.status()['phase'],'waiting_production')

    def test_final_requires_exact4099_markers_done_and_epoch_evaluations(self):
        controller=self.promoted();path,receipt,_=self.production_fixture(controller,4099,True)
        for step in [4098,4100]:
            bad=copy.deepcopy(receipt);bad['result']['global_step']=step;put(path,bad)
            with self.assertRaises(ctl.Blocked):controller.accept_link()
        bad=copy.deepcopy(receipt);bad['final']['expected_completed_epochs']=1;put(path,bad)
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(path,receipt);marker=self.root/'runs/c2/step_04099/PILOT_FINAL';marker.unlink()
        with self.assertRaises(ctl.Blocked):controller.accept_link()
        put(marker,b'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n');controller.accept_link()
        self.assertEqual(controller.status()['phase'],'final_verified')

    def test_nice_restore_is_idempotent_and_b6_remains_held(self):
        controller=self.promoted();self.production_fixture(controller,4099,True);controller.accept_link()
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
