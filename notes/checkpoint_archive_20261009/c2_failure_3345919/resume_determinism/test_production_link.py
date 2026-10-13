"""Allocated helper tests: actual tensor fixtures and subprocess signal behavior."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch
from safetensors.torch import save_file
from transgenic.training import b6_runtime as rt
from transgenic.training.b6_compile_history import CompileHistory, encode_rng, input_signature
import probe

try:
    import production_link as p
except ImportError:
    p = None


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True))


class FixtureMixin:
    def setUp(self):
        self.assertIsNotNone(p, 'production allocation helper is not implemented')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.identity = {k: 'fixture' for k in rt.IDENTITY_REQUIRED_KEYS}
        self.identity.update(arm='C2', seed=123, batch_size=1, world_size=4,
                             accumulation_steps=96, stop_after_updates=4099,
                             mixed_precision='bf16', deterministic=True)
        self.fp = {'compiler': 'fixture'}
        write(self.run / 'run_config_b6.json', self.identity)
        self.ck = self.make_checkpoint()

    def make_checkpoint(self, name='latest_state', epoch=0, step=48, global_step=2, pending_micro=0):
        ck = self.run / name
        (ck / 'accelerate_state').mkdir(parents=True)
        weights = {'weight': torch.tensor([1., 2.])}
        save_file(weights, str(ck / 'accelerate_state/model.safetensors'))
        torch.save({'state': {0: {'step': torch.tensor(float(global_step)), 'exp_avg': torch.zeros(2),
                                      'exp_avg_sq': torch.ones(2)}},
                    'param_groups': [{'params': [0], 'lr': .01}]}, ck / 'accelerate_state/optimizer.bin')
        torch.save({'last_epoch': global_step, '_step_count': global_step + 1, '_last_lr': [.01]}, ck / 'accelerate_state/scheduler.bin')
        for rank in range(4):
            torch.save({'random_state': random.Random(rank).getstate(),
                        'numpy_random_seed': np.random.RandomState(rank).get_state(),
                        'torch_manual_seed': torch.Generator().manual_seed(rank).get_state(),
                        'torch_cuda_manual_seed': [torch.Generator().manual_seed(rank).get_state()]},
                       ck / ('accelerate_state/random_states_%d.pkl' % rank))
            if pending_micro:
                torch.save({'weight': torch.ones(2)}, ck / ('pending_grads_rank%d.pt' % rank))
        position = dict(epoch=epoch, step=step, global_step=global_step, pending_micro=pending_micro)
        rng = encode_rng({'python': random.Random(1).getstate(), 'numpy': np.random.RandomState(2).get_state(),
                          'torch_cpu': torch.Generator().manual_seed(3).get_state(), 'torch_cuda': None})
        entry = {'ordinal': 0, 'position': dict(phase='train', epoch=0, batch_index=0, view='arm', training=True,
                                               grad_enabled=True, optimizer_step=0), 'origin': 'observed',
                 'ranks': [dict(rank=r, inputs=input_signature({'labels': torch.tensor([[1, 2]])}),
                                meta={}, rng_before=rng, delta=1, activity_delta=1) for r in range(4)]}
        history = dict(version=1, identity=self.identity, fingerprint=self.fp, world_size=4,
                       entries=[entry], bootstrap=None, checkpoint=position)
        CompileHistory._validate(history)
        write(ck / 'compile_history.json', history)
        write(ck / 'identity.json', self.identity)
        write(ck / 'meta.json', dict(position, arm='C2', seed=123, identity_sha256=rt.canonical_sha256(self.identity),
                                   compile_history_sha256=probe.sha(ck / 'compile_history.json')))
        return ck

    def check(self, ck=None):
        return p.check_checkpoint(ck or self.ck, self.identity, self.fp)

class CheckpointTests(FixtureMixin, unittest.TestCase):
    def test_canonical_matches_reviewed_probe(self):
        self.assertEqual(self.check()['canonical_checkpoint_sha256'], probe.digest(probe.checkpoint(self.run)))

    def test_pending_epoch_uses_extended_value_tree(self):
        ck = self.make_checkpoint('epoch_01', epoch=1, step=0, global_step=2, pending_micro=3)
        result = self.check(ck)
        self.assertEqual(result['position']['pending_micro'], 3)
        (ck / 'pending_grads_rank3.pt').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.check(ck)

    def test_partial_state_and_nonfinite_tensors_refused(self):
        for name in ['optimizer.bin', 'scheduler.bin'] + ['random_states_%d.pkl' % r for r in range(4)]:
            path = self.ck / 'accelerate_state' / name
            content = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises((ValueError, FileNotFoundError)):
                self.check()
            path.write_bytes(content)
        torch.save({'state': {}, 'param_groups': []}, self.ck / 'accelerate_state/optimizer.bin')
        with self.assertRaises(ValueError): self.check()

    def test_forged_history_and_wrong_identity_refused(self):
        with self.assertRaises((ValueError, RuntimeError)):
            p.check_checkpoint(self.ck, dict(self.identity, seed=124), self.fp)
        with self.assertRaises((ValueError, RuntimeError)):
            p.check_checkpoint(self.ck, self.identity, {'compiler': 'changed'})
        history = json.loads((self.ck / 'compile_history.json').read_text())
        history['checkpoint']['global_step'] = 3
        write(self.ck / 'compile_history.json', history)
        meta = json.loads((self.ck / 'meta.json').read_text())
        meta['compile_history_sha256'] = probe.sha(self.ck / 'compile_history.json')
        write(self.ck / 'meta.json', meta)
        with self.assertRaises(ValueError): self.check()

    def test_finite_weights_and_grad_shapes_required(self):
        save_file({'weight': torch.tensor([float('nan'), 2.])}, str(self.ck / 'accelerate_state/model.safetensors'))
        with self.assertRaises(ValueError): self.check()
        ck = self.make_checkpoint('epoch_01', epoch=1, step=0, global_step=2, pending_micro=3)
        torch.save({'weight': torch.ones(3)}, ck / 'pending_grads_rank2.pt')
        with self.assertRaises(ValueError): self.check(ck)

    def test_partial_initialized_optimizer_state_cannot_disappear(self):
        state_path=self.ck/'accelerate_state/optimizer.bin'
        state=torch.load(state_path,weights_only=False)
        state['param_groups'][0]['params'].append(1)
        state['state'][1]=copy.deepcopy(state['state'][0])
        torch.save(state,state_path)
        contract={}
        self.assertIn('capture_contract', __import__('inspect').signature(p.check_checkpoint).parameters,
                      'checkpoint structural continuity is missing')
        p.check_checkpoint(self.ck,self.identity,self.fp,capture_contract=contract)
        del state['state'][1];torch.save(state,state_path)
        with self.assertRaises(ValueError):
            p.check_checkpoint(self.ck,self.identity,self.fp,expected_contract=contract)

    def test_final_requires_exact_publication(self):
        final = self.run / 'step_04099'
        final.mkdir()
        for n in ['identity.json', 'compile_history.json']:
            (final / n).write_bytes((self.ck / n).read_bytes())
        save_file({'weight': torch.ones(2)}, str(final / 'model.safetensors'))
        history = json.loads((final / 'compile_history.json').read_text())
        position = dict(epoch=0, step=98376, global_step=4099, pending_micro=0)
        history['checkpoint'] = position
        write(final / 'compile_history.json', history)
        meta = dict(position, arm='C2', seed=123, stop_after_updates=4099, checkpoint_kind='fixed_step',
                    selection=rt.FIXED_STEP_SELECTION, pilot_final=True,
                    model_sha256=probe.sha(final / 'model.safetensors'),
                    compile_history_sha256=probe.sha(final / 'compile_history.json'))
        write(final / 'meta.json', meta)
        (final / rt.PILOT_FINAL_MARKER).write_text('WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n')
        (final / rt.PRUNE_EXEMPT_MARKER).write_text('A28/plan 9: step checkpoint, never pruned\n')
        (self.run / 'TRAINING_DONE').write_text('2026-10-12T00:00:00\n')
        self.assertEqual(p.check_final(self.run, self.identity, self.fp)['position']['global_step'], 4099)
        self.assertIn('expected_model', __import__('inspect').signature(p.check_final).parameters, 'final model completeness missing')
        with self.assertRaises(ValueError):
            p.check_final(self.run,self.identity,self.fp,expected_model={'weight':{'shape':[2],'dtype':'torch.float32'},'missing':{'shape':[2],'dtype':'torch.float32'}})
        for key, value in [('global_step',4098), ('global_step',4100), ('pilot_final','true')]:
            write(final / 'meta.json', dict(meta, **{key:value}))
            with self.assertRaises(ValueError): p.check_final(self.run,self.identity,self.fp)
        write(final / 'meta.json',meta)
        (final / rt.PILOT_FINAL_MARKER).unlink()
        with self.assertRaises(ValueError): p.check_final(self.run,self.identity,self.fp)


class ControlTests(unittest.TestCase):
    def test_strict_json_and_paths(self):
        self.assertIsNotNone(p, 'production allocation helper is not implemented')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'x.json'
            for raw in ['{"a":1,"a":2}', '{"a":NaN}']:
                path.write_text(raw)
                with self.assertRaises(ValueError): p.read_json(path)
            for relative in ['/tmp/a', '../a', 'a/../b', './a']:
                with self.assertRaises(ValueError): p.beneath(root, relative)
            (Path(root) / 'link').symlink_to('/tmp')
            with self.assertRaises(ValueError): p.beneath(root, 'link/a')



class RequestTests(unittest.TestCase):
    def test_schema_strict_profile_and_checkpoint(self):
        request = dict(version=1, operation_id='operation-1', token='c2-prod-0001', link_index=1,
                       source_commit=p.SOURCE, source_snapshot_git='a'*40, source_dir='source',
                       source_manifest_sha256='a'*64, helper_dir='helpers', helper_manifest_sha256='b'*64,
                       run_dir='runs/C2', bindings='bindings.json', bindings_sha256='c'*64,
                       config_relative='configs/pilot.json', config_sha256='d'*64, receipt_dir='ops/links/0001',
                       profile=dict(p.PROFILE), start_checkpoint=dict(relative_dir='latest_state', epoch=0,step=9840,
                       global_step=410,pending_micro=0,meta_sha256='e'*64,identity_sha256='f'*64,
                       history_sha256='a'*64,canonical_checkpoint_sha256='b'*64),
                       predecessor=dict(relative_path='ops/ready.json',sha256='c'*64),
                       expected_runtime_fingerprint={'compiler':'fixture'}, completed_epoch_events=[])
        self.assertTrue(hasattr(p, 'validate_request'), 'request validation missing')
        p.validate_request(request)
        for key, value in [('version', True), ('link_index', True), ('source_commit','bad'),
                           ('source_manifest_sha256','xyz'),('run_dir','../escape')]:
            changed=copy.deepcopy(request); changed[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): p.validate_request(changed)
        for key, value in [('deterministic',1),('target',True),('world_size',2)]:
            changed=copy.deepcopy(request); changed['profile'][key]=value
            with self.assertRaises(ValueError): p.validate_request(changed)
        changed=copy.deepcopy(request); changed['start_checkpoint']['relative_dir']='step_04099'
        with self.assertRaises(ValueError): p.validate_request(changed)

    def test_manifest_drift_is_refused(self):
        self.assertTrue(hasattr(p, 'check_manifest'), 'manifest validation missing')
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); (root/'file.py').write_text('original')
            (root/'manifest.sha256').write_text(probe.sha(root/'file.py')+'  file.py\n')
            pin=probe.sha(root/'manifest.sha256')
            p.check_manifest(root,'manifest.sha256',pin)
            (root/'file.py').write_text('changed')
            with self.assertRaises(ValueError): p.check_manifest(root,'manifest.sha256',pin)

    def test_rank_results_require_four_fresh_matching_verifiers(self):
        self.assertTrue(hasattr(p,'validate_rank_records'), 'rank observations missing')
        request=dict(profile=dict(p.PROFILE),start_checkpoint={'global_step':410},completed_epoch_events=[],
                     expected_runtime_fingerprint={'compiler':'fixture'},bindings_sha256='b'*64)
        records=[dict(rank=r,job_id='123',request_sha256='a'*64,input_verification=dict(job_id='123',
                      bindings_sha256='b'*64,record={'identity':'fresh'}),runtime_fingerprint={'compiler':'fixture'},
                      identity={'fixture':1},counts={'train_rows':100,'rank_batches':25,'per_rank_accumulation':24},
                      result={'preempted':True,'global_step':411,'history':[]},epoch_end=[]) for r in range(4)]
        p.validate_rank_records(records,request,'123','a'*64)
        for change in [lambda rows:rows.pop(),lambda rows:rows[3].pop('input_verification'),
                       lambda rows:rows[2].update(job_id='122'),
                       lambda rows:rows[1]['result'].update(global_step=410),
                       lambda rows:rows[0]['result'].update(preempted='true')]:
            bad=copy.deepcopy(records);change(bad)
            with self.assertRaises(ValueError):p.validate_rank_records(bad,request,'123','a'*64)

class EpochTests(FixtureMixin, unittest.TestCase):
    def test_draw_prefix_cannot_hide_tail_or_short_rank(self):
        self.assertTrue(hasattr(p,'check_draws'))
        for rank in range(4):
            path=self.run/'draw_log'/('rank%02d.tsv'%rank);path.parent.mkdir(exist_ok=True)
            path.write_text(''.join('%d\t0\t%s\n'%(i,'a'*64) for i in range(48)))
        position={'epoch':0,'step':48,'global_step':2,'pending_micro':0}
        counts={'rank_batches':100,'per_rank_accumulation':24}
        p.check_draws(self.run,position,counts)
        with (self.run/'draw_log/rank03.tsv').open('a') as f:f.write('99\t0\t'+'a'*64+'\n')
        with self.assertRaises(ValueError):p.check_draws(self.run,position,counts)

    def test_epoch_receipts_bind_actual_evaluation(self):
        self.assertTrue(hasattr(p,'check_epochs'),'epoch acceptance missing')
        ck=self.make_checkpoint('epoch_01',epoch=1,step=0,global_step=2,pending_micro=3)
        row=dict(epoch=1,global_step=2,data_order_sha256='a'*64,train=1.0,de_novo=2.0,evidence=3.0)
        write(ck/'eval.json',dict(epoch=1,global_step=2,train_loss=1.0,eval_loss=2.0,eval_evidence_loss=3.0))
        obs=dict(epoch_end=1,arm='C2',data_order_first1000_sha256='a'*64,train=1.0,de_novo=2.0,evidence=3.0)
        records=[dict(result={'history':[row]},epoch_end=[obs] if rank==0 else []) for rank in range(4)]
        receipt=p.check_epochs(self.run,[],records,1,self.identity,self.fp)
        self.assertEqual(receipt[0]['epoch'],1)
        self.assertEqual(p.check_epochs(self.run,receipt,[dict(result={'history':[]},epoch_end=[]) for _ in range(4)],1,self.identity,self.fp),receipt)
        with self.assertRaises(ValueError):p.check_epochs(self.run,[],[dict(result={'history':[]},epoch_end=[]) for _ in range(4)],1,self.identity,self.fp)
        write(ck/'eval.json',dict(epoch=1,global_step=2,train_loss=1.0,eval_loss=float('nan'),eval_evidence_loss=3.0))
        with self.assertRaises(ValueError):p.check_epochs(self.run,[],records,1,self.identity,self.fp)


class SignalTests(unittest.TestCase):
    def test_inherited_compile_and_container_overrides_are_removed(self):
        shell=Path(__file__).with_name('production_link.sbatch')
        command='source "$1"; sanitize_environment; "$2" -c "import os,json; print(json.dumps(dict(os.environ)))"'
        env=dict(PATH='/usr/bin:/bin',APPTAINER_BINDPATH='/bad:/work/repo',APPTAINERENV_TORCHINDUCTOR_FORCE_DISABLE_CACHES='1',SINGULARITYENV_TRANSGENIC_NO_COMPILE='1',
                 TORCHINDUCTOR_CACHE_DIR='/bad',PYTHONOPTIMIZE='1',SBATCH_PARTITION='bad',UNCHANGED_VALUE='yes')
        proc=subprocess.run(['bash','-c',command,'_',str(shell),sys.executable],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertEqual(proc.returncode,0,proc.stderr)
        result=json.loads(proc.stdout)
        self.assertNotIn('APPTAINERENV_TORCHINDUCTOR_FORCE_DISABLE_CACHES',result)
        self.assertNotIn('APPTAINER_BINDPATH',result)
        self.assertNotIn('SINGULARITYENV_TRANSGENIC_NO_COMPILE',result)
        self.assertNotIn('TORCHINDUCTOR_CACHE_DIR',result)
        self.assertNotIn('PYTHONOPTIMIZE',result)
        self.assertNotIn('SBATCH_PARTITION',result)
        self.assertEqual(result['UNCHANGED_VALUE'],'yes')

    def test_wait_keeps_rank_and_launcher_exit_semantics(self):
        shell=Path(__file__).with_name('production_link.sbatch')
        self.assertTrue(shell.exists(),'signal supervisor missing')
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            worker=root/'production_link.py'
            worker.write_text('import signal,time,sys\nfrom pathlib import Path\np=Path(sys.argv[2])\nsignal.signal(signal.SIGTERM,lambda *a: (p.write_text("term"),sys.exit(0)))\np.with_suffix(".ready").write_text("ready")\nwhile True: time.sleep(.01)\n')
            launcher=root/'accelerate.py'
            launcher.write_text('import subprocess,sys\np=subprocess.Popen([sys.executable,sys.argv[1],"run",sys.argv[2]])\nr=p.wait()\nsys.exit(int(sys.argv[3]) or r)\n')
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            self.addCleanup(lambda: unrelated.poll() is None and unrelated.kill())
            for exitcode in (0,7):
                marker=root/('rank%d'%exitcode)
                command='source "$1"; launch_and_wait "$2" "$3" "$4" "$5" "$6"'
                proc=subprocess.Popen(['bash','-c',command,'_',str(shell),sys.executable,str(launcher),str(worker),str(marker),str(exitcode)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                try:
                    limit=time.monotonic()+10
                    while not marker.with_suffix('.ready').exists() and time.monotonic()<limit:
                        time.sleep(.01)
                    self.assertTrue(marker.with_suffix('.ready').exists())
                    os.kill(proc.pid,signal.SIGUSR1)
                    out,err=proc.communicate(timeout=10)
                    self.assertEqual(proc.returncode,exitcode,(out,err))
                    self.assertEqual(marker.read_text(),'term')
                    self.assertIsNone(unrelated.poll())
                finally:
                    if proc.poll() is None:proc.kill();proc.wait()
            unrelated.terminate();unrelated.wait()

class LinkTests(FixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        from revision.scripts.b6 import wp4_synthetic as syn
        self.binding=syn.binding_record()
        self.identity.update(rt.identity_from_bindings(self.binding))
        write(self.run/'run_config_b6.json',self.identity)
        write(self.ck/'identity.json',self.identity)
        hist=json.loads((self.ck/'compile_history.json').read_text());hist['identity']=self.identity
        write(self.ck/'compile_history.json',hist)
        meta=json.loads((self.ck/'meta.json').read_text())
        meta.update(identity_sha256=rt.canonical_sha256(self.identity),compile_history_sha256=probe.sha(self.ck/'compile_history.json'))
        write(self.ck/'meta.json',meta)
        self.source=self.root/'source'; self.source.mkdir()
        files=['train/train_HyenaTransgenic_B6.py','src/transgenic/training/b6_runtime.py',
               'src/transgenic/training/b6_compile_history.py','src/transgenic/training/b6_compile_replay.py']
        for name in files:
            path=self.source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture\n')
        write(self.source/'configs/pilot.json',{'fixture':1})
        (self.source/'validation_sources.sha256').write_text(''.join(probe.sha(self.source/n)+'  '+n+'\n' for n in files))
        def git(*args):return subprocess.check_output(['git','-C',str(self.source)]+list(args),stderr=subprocess.DEVNULL).decode().strip()
        git('init');git('add','.');git('-c','user.name=fixture','-c','user.email=fixture@local','commit','-m','fixture')
        helpers=self.root/'helpers';helpers.mkdir()
        for name in ['production_link.py','production_link.sbatch','probe.py']:(helpers/name).write_text('fixture')
        (helpers/'manifest.sha256').write_text(''.join(probe.sha(helpers/n)+'  '+n+'\n' for n in ['production_link.py','production_link.sbatch','probe.py']))
        write(self.root/'bindings.json',self.binding);write(self.root/'prior.json',{'ready':True})
        ck=p.check_checkpoint(self.ck,self.identity,self.fp)
        start=dict(ck['position'],**{k:ck[k] for k in ['relative_dir','meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256']})
        self.request=dict(version=1,operation_id='fixture',token='c2-fixture',link_index=1,source_commit='d'*40,
            source_snapshot_git=git('rev-parse','HEAD'),source_dir='source',source_manifest_sha256=probe.sha(self.source/'validation_sources.sha256'),
            helper_dir='helpers',helper_manifest_sha256=probe.sha(helpers/'manifest.sha256'),run_dir='run',bindings='bindings.json',
            bindings_sha256=probe.sha(self.root/'bindings.json'),config_relative='configs/pilot.json',config_sha256=probe.sha(self.source/'configs/pilot.json'),
            receipt_dir='links/0001',profile=dict(p.PROFILE),start_checkpoint=start,predecessor={'relative_path':'prior.json','sha256':probe.sha(self.root/'prior.json')},
            expected_runtime_fingerprint=self.fp,completed_epoch_events=[])
        write(self.root/'request.json',self.request);self.pin=probe.sha(self.root/'request.json')
        self.output=self.root/'links/0001/execution_123';self.output.mkdir(parents=True)
        self.draws(48)

    def draws(self,total):
        for rank in range(4):
            path=self.run/'draw_log'/('rank%02d.tsv'%rank);path.parent.mkdir(exist_ok=True)
            path.write_text(''.join('%d\t0\t%s\n'%(i,'a'*64) for i in range(total)))

    def preflight(self):return p.preflight(self.root,'request.json',self.pin,'123')

    def results(self):
        for rank in range(4):
            write(self.output/('rank%d.json'%rank),dict(rank=rank,job_id='123',request_sha256=self.pin,
                input_verification={'job_id':'123','bindings_sha256':self.request['bindings_sha256'],'record':self.binding},
                runtime_fingerprint=self.fp,identity=self.identity,counts={'train_rows':400,'rank_batches':100,'per_rank_accumulation':24},
                result={'preempted':True,'global_step':3,'history':[]},epoch_end=[]))
        meta=json.loads((self.ck/'meta.json').read_text());meta.update(global_step=3,step=72)
        hist=json.loads((self.ck/'compile_history.json').read_text());hist['checkpoint'].update(global_step=3,step=72)
        write(self.ck/'compile_history.json',hist);meta['compile_history_sha256']=probe.sha(self.ck/'compile_history.json');write(self.ck/'meta.json',meta)
        self.draws(72)

    def postflight(self,exitcode=0):return p.postflight(self.root,'request.json',self.pin,'123',exitcode)

    def test_host_paths_is_stdlib_only_and_rejects_changed_pinned_manifest(self):
        argv=[sys.executable,'-S',str(Path(__file__).with_name('production_link.py')),'paths','--work',str(self.root),
              '--request','request.json','--request-sha256',self.pin,'--job-id','123']
        good=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertEqual(len(good.stdout.decode().splitlines()),4)
        (self.source/'validation_sources.sha256').write_text('changed')
        bad=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        self.assertNotEqual(bad.returncode,0)

    def test_preflight_accepts_its_own_batch_log_but_never_existing_execution(self):
        (self.output/'preflight.log').touch()
        self.preflight()
        with self.assertRaises(ValueError):self.preflight()

    def test_full_preempt_receipt_and_exclusive_publication(self):
        self.preflight();self.results();receipt=self.postflight()
        self.assertEqual(receipt['checkpoint']['position']['global_step'],3)
        self.assertIsNone(receipt['final'])
        with self.assertRaises(FileExistsError):self.postflight()

    def test_uncommitted_checkpoint_directory_is_preserved_and_refused(self):
        temporary=self.run/'latest_state.tmp';temporary.mkdir()
        (temporary/'evidence').write_text('partial')
        with self.assertRaises(ValueError):self.preflight()
        self.assertEqual((temporary/'evidence').read_text(),'partial')

    def test_preflight_refuses_stale_done(self):
        (self.run/'TRAINING_DONE').write_text('old')
        with self.assertRaises(ValueError):self.preflight()

    def test_nonzero_after_done_cannot_emit_receipt(self):
        self.preflight();self.results();(self.run/'TRAINING_DONE').write_text('done')
        with self.assertRaises(ValueError):self.postflight(7)
        self.assertFalse((self.output/'link_receipt.json').exists())

    def test_mutated_source_or_missing_rank_blocks_receipt(self):
        self.preflight();self.results();(self.output/'rank3.json').unlink()
        with self.assertRaises(ValueError):self.postflight()
        self.results();(self.source/'train/train_HyenaTransgenic_B6.py').write_text('changed')
        with self.assertRaises(ValueError):self.postflight()

    def test_changed_predecessor_and_wrong_request_pin_block(self):
        with self.assertRaises(ValueError):p.preflight(self.root,'request.json','a'*64,'123')
        write(self.root/'prior.json',{'ready':'forged'})
        with self.assertRaises(ValueError):self.preflight()

    def test_history_prefix_cannot_be_rewritten_with_fresh_digest(self):
        self.preflight();self.results()
        history=json.loads((self.ck/'compile_history.json').read_text())
        history['entries'][0]['ranks'][0]['meta']={'forged':True}
        write(self.ck/'compile_history.json',history)
        meta=json.loads((self.ck/'meta.json').read_text())
        meta['compile_history_sha256']=probe.sha(self.ck/'compile_history.json');write(self.ck/'meta.json',meta)
        with self.assertRaises(ValueError):self.postflight()


    def test_final_receipt_exact4099_without_partial_epoch_evaluation(self):
        self.preflight()
        final=self.run/'step_04099';final.mkdir()
        save_file({'weight':torch.ones(2)},str(final/'model.safetensors'))
        write(final/'identity.json',self.identity)
        history=json.loads((self.ck/'compile_history.json').read_text())
        position=dict(epoch=0,step=98376,global_step=4099,pending_micro=0)
        history['checkpoint']=position;write(final/'compile_history.json',history)
        write(final/'meta.json',dict(position,arm='C2',seed=123,stop_after_updates=4099,checkpoint_kind='fixed_step',
              selection=rt.FIXED_STEP_SELECTION,pilot_final=True,model_sha256=probe.sha(final/'model.safetensors'),
              compile_history_sha256=probe.sha(final/'compile_history.json')))
        (final/rt.PILOT_FINAL_MARKER).write_text('WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n')
        (final/rt.PRUNE_EXEMPT_MARKER).write_text('A28/plan 9: step checkpoint, never pruned\n')
        (self.run/'TRAINING_DONE').write_text('2026-10-12T00:00:00\n')
        self.draws(98376)
        for rank in range(4):
            write(self.output/('rank%d.json'%rank),dict(rank=rank,job_id='123',request_sha256=self.pin,
                input_verification={'job_id':'123','bindings_sha256':self.request['bindings_sha256'],'record':self.binding},
                runtime_fingerprint=self.fp,identity=self.identity,counts={'train_rows':400000,'rank_batches':100000,'per_rank_accumulation':24},
                result={'done':True,'global_step':4099,'history':[],'reason':'stop_after_updates','pilot_final':str(final)},epoch_end=[]))
        receipt=self.postflight()
        self.assertEqual(receipt['final']['expected_completed_epochs'],0)
        self.assertEqual(receipt['result']['global_step'],4099)


class ObserverTests(FixtureMixin, unittest.TestCase):
    def test_observer_calls_originals_and_restores_even_on_missing_verification(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from accelerate import Accelerator
        calls=[]
        trainer=SimpleNamespace()
        def verified(*args,**kwargs):calls.append('verify');return {'fresh':probe.sha(self.ck/'meta.json')}
        def identity(*args,**kwargs):calls.append('identity');return self.identity
        def fingerprint(*args,**kwargs):calls.append('fingerprint');return self.fp
        trainer.verify_training_inputs=verified
        trainer.rt=SimpleNamespace(build_run_identity=identity,accumulation_steps=lambda cfg,batch:96)
        trainer.replay=SimpleNamespace(runtime_fingerprint=fingerprint)
        spec=SimpleNamespace(batch_size=1,cfg={'effective_batch_size':96},log=lambda msg:calls.append('log'))
        for rank in range(4):
            path=self.run/'draw_log'/('rank%02d.tsv'%rank);path.parent.mkdir(exist_ok=True)
            path.write_text(''.join('%d\t0\t%s\n'%(i,'a'*64) for i in range(48)))
        def original_run(spec,model,train_ds,eval_ds,**kwargs):
            acc=SimpleNamespace(num_processes=4)
            Accelerator.prepare(acc,None,None,list(range(100)),None,None)
            trainer.rt.build_run_identity();trainer.replay.runtime_fingerprint()
            return {'preempted':True,'global_step':3,'history':[]}
        trainer.run_b6=original_run
        def original_main(argv):
            calls.append(argv)
            trainer.verify_training_inputs()
            return trainer.run_b6(spec,None,None,None,train_sampler=list(range(400)))
        trainer.main=original_main
        request=dict(source_dir='source',config_relative='cfg.json',bindings='bindings.json',run_dir='run',
                     bindings_sha256='a'*64,expected_runtime_fingerprint=self.fp,
                     start_checkpoint=dict(epoch=0,step=48,global_step=2,pending_micro=0))
        with patch.dict(os.environ,{'RANK':'0','WORLD_SIZE':'4'}),patch.object(Accelerator,'prepare',lambda acc,*args,**kw:args):
            record=p.observe_training(trainer,request,self.root,'123','b'*64)
            self.assertEqual(calls.count('verify'),1)
            self.assertEqual(record['counts']['rank_batches'],100)
            self.assertIn('--resume',calls[0]);self.assertIn('auto',calls[0])
            self.assertIn('--stop-after-updates',calls[0]);self.assertIn('4099',calls[0])
            self.assertNotIn('--allow-ungated',calls[0])
            self.assertIs(trainer.verify_training_inputs,verified)
            self.assertIs(trainer.run_b6,original_run)
            trainer.main=lambda argv:trainer.run_b6(spec,None,None,None,train_sampler=list(range(400)))
            with self.assertRaises(ValueError):p.observe_training(trainer,request,self.root,'123','b'*64)
            self.assertIs(trainer.verify_training_inputs,verified)
            self.assertIs(trainer.run_b6,original_run)

    def test_symlink_ancestor_cannot_enter_checkpoint_tree(self):
        (self.root/'alias').symlink_to(self.run,target_is_directory=True)
        with self.assertRaises(ValueError):p.beneath(self.root/'alias'/'latest_state','meta.json')


if __name__=='__main__':unittest.main()
