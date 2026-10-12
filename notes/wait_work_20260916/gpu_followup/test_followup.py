import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('followup', Path(__file__).with_name('followup.py'))
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binding = 'a' * 64
        self.value = dict(schema_version=1, preflight='p1_identity', bindings_sha256=self.binding,
                          command_sha256='b'*64, allocation='3163986', **{'pass': True},
                          resume_supported=False, exit_code=0, timed_out=False, signal=None,
                          measurement=dict(mode='real_tokenizer', bindings_sha256=self.binding,
                                           device='cuda', real_runtime=True, tokenizer_roundtrip=True,
                                           hyena_loaded=True, probe_identity=dict(schema_version=1,
                                           probe_path='/work/repo/revision/scripts/mvp/real_probes.py',
                                           imports={'/work/repo/revision/scripts/mvp/real_probes.py': 'c'*64},
                                           asset_roots={'/work/repo/HFmodels': 'd'*64})))
    def receipt(self):
        p=self.root/'real_tokenizer.json'
        p.write_bytes(f.canonical(self.value))
        p.with_suffix('.DONE').write_text(f.sha(p)+'\n')
        return p
    def test_real_wrapper_shape(self):
        self.assertEqual(f.wrapper(self.receipt(), 'real_tokenizer', self.binding)['allocation'], '3163986')
    def test_missing_p1(self):
        with self.assertRaises((ValueError, FileNotFoundError)): f.wrapper(self.root/'absent.json','real_tokenizer',self.binding)
    def test_failed(self):
        p=self.receipt(); p.with_name('real_tokenizer.FAILED.json').write_text('{}')
        with self.assertRaises(ValueError): f.wrapper(p,'real_tokenizer',self.binding)
    def test_missing_done(self):
        p=self.receipt(); p.with_suffix('.DONE').unlink()
        with self.assertRaises(FileNotFoundError): f.wrapper(p,'real_tokenizer',self.binding)
    def test_bad_done(self):
        p=self.receipt(); p.with_suffix('.DONE').write_text('0'*64)
        with self.assertRaises(ValueError): f.wrapper(p,'real_tokenizer',self.binding)
    def test_binding_mismatch(self):
        with self.assertRaises(ValueError): f.wrapper(self.receipt(),'real_tokenizer','e'*64)
    def test_cpu_boolean_proxy(self):
        self.value['measurement']['device']='cpu'
        with self.assertRaises(ValueError): f.wrapper(self.receipt(),'real_tokenizer',self.binding)
    def test_exit_failure(self):
        self.value['exit_code']=1
        with self.assertRaises(ValueError): f.wrapper(self.receipt(),'real_tokenizer',self.binding)
    def ledger(self):
        return dict(observed_at=time.time()-1, expires_at=time.time()+3600,
                    controller_record_reference='test fixture only', complete_diagnostic_history=True,
                    no_overlap_confirmed=True, jobs=[dict(job_id='3163986', state='COMPLETED',
                    nodes=1, charged_seconds=1200),dict(job_id='3163545', state='CANCELLED',nodes=1,charged_seconds=0)])
    def build_fixture(self):
        control=self.root/'control'; control.mkdir()
        source=f.HERE/'source/revision/scripts/mvp'
        names=('make_launch_readiness.py','stage_checkpoint.py','run_bounded.py','run_training.py','accelerate_mvp.json','real_probes.py','resource_probe.py')
        tooling={'/work/repo/revision/scripts/mvp/'+n:f.sha(source/n) for n in names}
        f.save(control/'consumed-code.json',{'files':tooling, 'external_packages':{}})
        bindings=dict(seed=1,tooling=tooling,consumed_code_manifest=dict(path=str(control/'consumed-code.json'),sha256=f.sha(control/'consumed-code.json')),
                      accelerate_config=dict(path='/work/repo/revision/scripts/mvp/accelerate_mvp.json',sha256=f.sha(source/'accelerate_mvp.json')))
        self.binding=hashlib.sha256(f.canonical(bindings)).hexdigest()
        f.save(control/'prepared.json',dict(bindings=bindings,bindings_sha256=self.binding))
        f.save(control/'assets.json',{'roots':{'/work/repo/HFmodels':'d'*64}})
        f.save(control/'tiles.json',{})
        f.save(control/'tile-availability.json',dict(status='available_and_actual_measured_batches_checked'))
        f.save(control/'s0-inputs.json',{n:f.sha(control/n) for n in ('prepared.json','consumed-code.json','assets.json','tiles.json','tile-availability.json')})
        argv=['/usr/bin/python3','/work/repo/revision/scripts/mvp/real_probes.py','--mode','real_tokenizer']
        options={'bindings':str(control/'prepared.json'),'bindings-file-sha256':f.sha(control/'prepared.json'),
                 'assets-manifest':str(control/'assets.json'),'assets-sha256':f.sha(control/'assets.json'),
                 'tile-rns':str(control/'tiles.json'),'tile-rns-sha256':f.sha(control/'tiles.json'),
                 'db':'/corpus.db','evidence-db':'/mvp/inputs/evidence.db','config':'/work/repo/config.json','scratch-root':'/scratch'}
        for k,v in options.items(): argv += ['--'+k,v]
        f.save(control/'p1.argv.json',argv)
        self.value['bindings_sha256']=self.binding
        self.value['measurement']['bindings_sha256']=self.binding
        self.value['command_sha256']=f.sha(control/'p1.argv.json')
        self.value['measurement']['probe_identity']['imports'][argv[1]]=f.sha(source/'real_probes.py')
        p1=self.receipt()
        f.save(self.root/'ledger.json',self.ledger())
        cfg=dict(commit=f.COMMIT,s0_control=str(control),ledger=str(self.root/'ledger.json'),
                 outer_seconds=600,timeout_seconds=400,receipts={'real_tokenizer':str(p1)})
        f.save(self.root/'inputs.json',cfg)
        return SimpleNamespace(inputs=str(self.root/'inputs.json'),mode='real_hyena_step0_three_tiers',output=str(self.root/'new'),execute=False)
    def test_build_bound_step0_command(self):
        args=self.build_fixture(); f.build(args)
        command=f.load(Path(args.output)/'command.json')
        self.assertEqual(command[command.index('--mode')+1],args.mode)
        self.assertEqual(command[command.index('--scratch-root')+1],str(Path(args.output)/'cache'))
        self.assertFalse(f.load(Path(args.output)/'plan.json')['measured_pass'])
    def check_timeout(self, timeout, accepted=False):
        args=self.build_fixture()
        cfg=f.load(args.inputs); cfg['timeout_seconds']=timeout
        Path(args.inputs).write_bytes(f.canonical(cfg))
        if accepted:
            f.build(args)
            runner=f.load(Path(args.output)/'runner.json')
            self.assertEqual(runner[runner.index('--timeout-seconds')+1],str(timeout))
        else:
            with self.assertRaises(ValueError): f.build(args)
            self.assertFalse(Path(args.output).exists())
    def test_timeout_zero(self): self.check_timeout(0)
    def test_timeout_negative(self): self.check_timeout(-1)
    def test_timeout_true(self): self.check_timeout(True)
    def test_timeout_false(self): self.check_timeout(False)
    def test_timeout_float(self): self.check_timeout(1.0)
    def test_timeout_minimum(self): self.check_timeout(1, accepted=True)
    def test_timeout_maximum(self): self.check_timeout(480, accepted=True)
    def test_timeout_exceeds_reserve(self): self.check_timeout(481)
    def test_build_refuses_p1_missing(self):
        args=self.build_fixture(); (self.root/'real_tokenizer.json').unlink()
        with self.assertRaises(FileNotFoundError): f.build(args)
        self.assertFalse(Path(args.output).exists())
    def test_build_refuses_modified_tooling(self):
        args=self.build_fixture()
        p=self.root/'control/consumed-code.json'; value=f.load(p)
        value['files']['/work/repo/revision/scripts/mvp/real_probes.py']='f'*64
        p.write_bytes(f.canonical(value))
        pins=self.root/'control/s0-inputs.json'; value=f.load(pins); value[p.name]=f.sha(p); pins.write_bytes(f.canonical(value))
        with self.assertRaisesRegex(ValueError,'source mismatch'): f.build(args)
    def test_source_binding_mismatch(self):
        with self.assertRaises(ValueError): f.source_binding({'files':{'/work/repo/x.py':'a'*64}}, {'/work/repo/x.py':'b'*64})
    def test_empty_source_binding(self):
        with self.assertRaises(ValueError): f.source_binding({'files':{}}, {})
    def test_budget(self): self.assertEqual(f.budget(self.ledger(), 600), 1200)
    def test_expired(self):
        l=self.ledger();l['expires_at']=time.time()-1
        with self.assertRaises(ValueError): f.budget(l,600)
    def test_insufficient(self):
        l=self.ledger();l['jobs'][0]['charged_seconds']=17900
        with self.assertRaises(ValueError): f.budget(l,600)
    def test_pending_reserved_p1(self):
        l=self.ledger();l['jobs'][0]['state']='PENDING'
        with self.assertRaises(ValueError): f.budget(l,600)
    def test_nan_budget(self):
        l=self.ledger();l['jobs'][0]['charged_seconds']=float('nan')
        with self.assertRaises(ValueError): f.budget(l,600)

if __name__=='__main__': unittest.main()
