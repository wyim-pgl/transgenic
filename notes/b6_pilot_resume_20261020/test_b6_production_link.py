"""Allocation validation on actual Task1 small CPU checkpoint artifacts."""
import copy
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
import sys
import unittest
import torch
from transgenic.training import b6_runtime as rt
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_continue_b6_history as fixtures
from test_continue_b6_history import put
try:
    import b6_production_link as producer
except ImportError:
    producer=None

class ProductionTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.ControllerTests.setUpClass.__func__)
    tearDownClass=classmethod(fixtures.ControllerTests.tearDownClass.__func__)
    setUp=fixtures.ControllerTests.setUp
    advance=fixtures.ControllerTests.advance
    def request(self):
        self.assertIsNotNone(producer,'B6 producer is not implemented')
        self.advance('candidate_activated')
        pin=self.controller.build_request()
        return json.loads((self.work/pin['relative_path']).read_text())
    def test_actual_initial_checkpoint_and_production_request_accepted(self):
        request=self.request();producer.validate_request(request)
        cp=self.proof/'resumed/latest_state';identity=producer.read_json(cp/'identity.json')
        checked=producer.check_checkpoint(cp,identity,request['expected_runtime_fingerprint'])
        self.assertEqual(checked['canonical_checkpoint_sha256'],request['start_checkpoint']['canonical_checkpoint_sha256'])
    def test_request_requires_explicit_existing_proof_mmap(self):
        request=self.request();request.pop('mmap_dir',None)
        with self.assertRaisesRegex(ValueError,'request fields'):producer.validate_request(request)
    def test_wrong_profile_source_cache_or_predecessor_rejected(self):
        request=self.request()
        for field,value in [('source_commit','a'*40),('profile',dict(request['profile'],arm='C2')),
                            ('expected_runtime_fingerprint',dict(request['expected_runtime_fingerprint'],world_size=2))]:
            changed=copy.deepcopy(request);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):producer.validate_request(changed)
    def test_four_device_rng_and_nonempty_optimizer_required(self):
        request=self.request();import torch
        cp=self.proof/'resumed/latest_state';identity=producer.read_json(cp/'identity.json')
        p=cp/'accelerate_state/random_states_3.pkl';value=torch.load(p,weights_only=False);value['torch_cuda_manual_seed']=value['torch_cuda_manual_seed'][:1];torch.save(value,p)
        with self.assertRaisesRegex(ValueError,'four CUDA'):producer.check_checkpoint(cp,identity,request['expected_runtime_fingerprint'])
    def test_nonzero_launcher_or_missing_rank_refuses_even_with_done(self):
        request=self.request()
        with self.assertRaisesRegex(ValueError,'nonzero'):producer.postflight(self.work,'not-opened','a'*64,'9',1)
        with self.assertRaises(ValueError):producer.validate_rank_records([],request,'9','a'*64)
    def test_compiled_namespace_and_alias_compatibility(self):
        self.request()
        self.assertEqual(producer.compatibility_model({'_orig_mod.weight':1}),{'weight':1})
        with self.assertRaises(ValueError):producer.compatibility_model({'_orig_mod.weight':1,'buffer':2})

    def compiled_model(self, checkpoint):
        from accelerate import Accelerator
        class TinyTied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.z_weight = torch.nn.Parameter(torch.tensor([1., 2.]))
                self.a_weight = self.z_weight
                self.bias = torch.nn.Parameter(torch.zeros(2))
            def forward(self, x):
                return x * self.z_weight + self.bias
        bare = TinyTied()
        compiled = torch.compile(bare, backend='eager')
        accelerator = Accelerator(cpu=True)
        compiled = accelerator.prepare(compiled)
        compiled(torch.ones(2)).sum().backward()
        saved = self.work / 'real_compiled_save'
        accelerator.save_state(str(saved))
        self.addCleanup(accelerator.free_memory)
        import shutil
        shutil.copyfile(saved / 'model.safetensors', checkpoint / 'accelerate_state/model.safetensors')
        actual = producer.load_weights(checkpoint / 'accelerate_state/model.safetensors')
        self.assertEqual(set(actual), {'_orig_mod.z_weight', '_orig_mod.bias'})
        return bare

    def make_final(self, model):
        final = self.run / 'step_04099'
        final.mkdir()
        aliases = rt.save_weights_deterministic(model, str(final / 'model.safetensors'))
        self.assertEqual(aliases, {'z_weight': 'a_weight'})
        history = json.loads((self.ck / 'compile_history.json').read_text())
        position = dict(epoch=0, step=98376, global_step=4099, pending_micro=0)
        history['checkpoint'] = position
        put(final / 'compile_history.json', history)
        put(final / 'identity.json', self.identity)
        put(final / 'meta.json', dict(position, arm='B6', seed=123, stop_after_updates=4099,
            checkpoint_kind='fixed_step', selection=rt.FIXED_STEP_SELECTION, pilot_final=True,
            tied_aliases=aliases, model_sha256=producer.sha(final / 'model.safetensors'),
            compile_history_sha256=producer.sha(final / 'compile_history.json')))
        (final / rt.PILOT_FINAL_MARKER).write_text('WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n')
        (final / rt.PRUNE_EXEMPT_MARKER).write_text('A28/plan 9: step checkpoint, never pruned\n')
        (self.run / 'TRAINING_DONE').write_text('2026-10-12T00:00:00\n')
        return final


    def test_real_compiled_tied_export_and_pending_gradients(self):
        request=self.request();self.run=self.proof/'resumed';self.ck=self.run/'latest_state'
        self.identity=producer.read_json(self.ck/'identity.json');self.fp=request['expected_runtime_fingerprint']
        model=self.compiled_model(self.ck)
        state_contract={};producer.check_checkpoint(self.ck,self.identity,self.fp,capture_contract=state_contract)
        self.assertIn('_orig_mod.z_weight',state_contract['model'])
        final=self.make_final(model)
        result=producer.check_final(self.run,self.identity,self.fp,expected_model=state_contract['model'])
        self.assertEqual(result['position']['global_step'],4099)
        meta=producer.read_json(final/'meta.json');meta['tied_aliases']={'z_weight':'missing'};put(final/'meta.json',meta)
        with self.assertRaises(ValueError):producer.check_final(self.run,self.identity,self.fp,expected_model=state_contract['model'])
        # Actual public source writes bare pending gradient names for compiled models.
        meta=producer.read_json(self.ck/'meta.json');meta.update(step=9841,pending_micro=1)
        history=producer.read_json(self.ck/'compile_history.json');history['checkpoint']=dict(epoch=0,step=9841,global_step=410,pending_micro=1)
        put(self.ck/'compile_history.json',history);meta['compile_history_sha256']=producer.sha(self.ck/'compile_history.json');put(self.ck/'meta.json',meta)
        for rank in range(4):rt.save_pending_grads(model,str(self.ck),rank)
        checked=producer.check_checkpoint(self.ck,self.identity,self.fp,expected_contract=state_contract)
        self.assertEqual(checked['position']['pending_micro'],1)
    def test_cuda_rng_malformed_encodings_and_rng_immutability(self):
        request=self.request();cp=self.proof/'resumed/latest_state';identity=producer.read_json(cp/'identity.json')
        path=cp/'accelerate_state/random_states_3.pkl';original=torch.load(path,weights_only=False);good=original['torch_cuda_manual_seed']
        cases=[good[:3],good+[good[0]],tuple(good),[torch.zeros(17,dtype=torch.uint8)]+good[1:],
               [good[0].reshape(2,8)]+good[1:],[good[0].to(torch.int8)]+good[1:],
               [torch.zeros(32,dtype=torch.uint8)[::2]]+good[1:]]
        for values in cases:
            changed=dict(original,torch_cuda_manual_seed=values);torch.save(changed,path)
            with self.assertRaises(ValueError):producer.check_checkpoint(cp,identity,request['expected_runtime_fingerprint'])
        torch.save(original,path)
        before=torch.get_rng_state().clone();cuda_before=torch.cuda.is_initialized()
        producer.check_checkpoint(cp,identity,request['expected_runtime_fingerprint'])
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(torch.cuda.is_initialized(),cuda_before)

class SignalTests(unittest.TestCase):
    def test_inherited_compile_and_container_overrides_are_removed(self):
        shell=Path(__file__).with_name('b6_production_link.sbatch')
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
        shell=Path(__file__).with_name('b6_production_link.sbatch')
        self.assertTrue(shell.exists(),'signal supervisor missing')
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            worker=root/'b6_production_link.py'
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


if __name__=='__main__':unittest.main()
